"""Workspace confinement and permission policy.

This is *not* an operating-system sandbox.  It blocks paths that leave the
workspace, reparse points, protected historical assets and unapproved command
execution.  A determined process running as the same user can bypass any
in-process check, so executors are also gated by an explicit enable switch.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from ...source_policy import DEFAULT_PROTECTED_FILES, DEFAULT_PROTECTED_PREFIXES
from .errors import PathViolation

FILE_ATTRIBUTE_REPARSE_POINT = 0x400

# Historical or frozen assets a coding agent must never rewrite in place.
DEFAULT_PROTECTED_PREFIXES: tuple[str, ...] = (
    "eval",
    "models",
    "adapters",
    "data/yuzhu_python_coding_v1",
    "logs",
    ".git",
)


class PermissionLevel(str, Enum):
    READ = "READ"
    WRITE_WORKSPACE = "WRITE_WORKSPACE"
    EXECUTE_APPROVED = "EXECUTE_APPROVED"
    DESTRUCTIVE = "DESTRUCTIVE"


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    level: PermissionLevel
    reason: str


class WorkspaceGuard:
    def __init__(
        self,
        root: Path,
        *,
        protected_prefixes: tuple[str, ...] = DEFAULT_PROTECTED_PREFIXES,
        read_protected_prefixes: tuple[str, ...] = DEFAULT_PROTECTED_PREFIXES,
        read_protected_files: tuple[str, ...] = DEFAULT_PROTECTED_FILES,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise PathViolation(f"workspace is not a directory: {self.root}", code="WORKSPACE_INVALID")
        self.protected_prefixes = tuple(
            prefix.replace("\\", "/").strip("/") for prefix in protected_prefixes if prefix
        )
        self.read_protected_prefixes = tuple(
            prefix.replace("\\", "/").strip("/").casefold()
            for prefix in read_protected_prefixes
            if prefix
        )
        self.read_protected_files = frozenset(
            item.replace("\\", "/").strip("/").casefold()
            for item in read_protected_files
            if item
        )

    # ------------------------------------------------------------------ paths

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def is_protected(self, path: Path) -> bool:
        relative = self.relative(path).lower()
        for prefix in self.protected_prefixes:
            lowered = prefix.lower()
            if relative == lowered or relative.startswith(lowered + "/"):
                return True
        return False

    def resolve(
        self,
        raw: str,
        *,
        must_exist: bool = False,
        for_write: bool = False,
    ) -> Path:
        if not isinstance(raw, str) or not raw.strip():
            raise PathViolation("path must be a non-empty string", code="PATH_EMPTY")
        text = raw.strip()
        if "\x00" in text:
            raise PathViolation("path contains a NUL byte", code="PATH_INVALID")
        if text.startswith("\\\\") or text.startswith("//"):
            raise PathViolation("UNC paths are not allowed", code="PATH_UNC")
        if text.startswith("\\\\?\\") or text.startswith("\\\\.\\"):
            raise PathViolation("device paths are not allowed", code="PATH_DEVICE")
        # Reject drive-relative forms such as "C:file" and NTFS alternate data
        # streams ("file.txt:stream") without rejecting the drive colon itself.
        if len(text) >= 2 and text[1] == ":" and not (len(text) >= 3 and text[2] in "\\/"):
            raise PathViolation("drive-relative paths are not allowed", code="PATH_DRIVE_RELATIVE")
        if ":" in text[2:]:
            raise PathViolation("alternate data streams are not allowed", code="PATH_ADS")

        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        if must_exist or not for_write:
            resolved = candidate.resolve()
        else:
            # The file may not exist yet: resolve the parent for the boundary
            # check without requiring the leaf to exist.
            parent = candidate.parent.resolve()
            resolved = parent / candidate.name
        self._check_inside(resolved)
        self._check_reparse(resolved)
        if must_exist and not resolved.exists():
            raise PathViolation(f"path does not exist: {text}", code="PATH_NOT_FOUND")
        return resolved

    def _check_inside(self, resolved: Path) -> None:
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise PathViolation(
                f"path escapes the workspace: {resolved}", code="PATH_OUTSIDE_WORKSPACE"
            ) from exc

    def _check_reparse(self, resolved: Path) -> None:
        current = resolved
        while True:
            try:
                info = os.lstat(current)
            except OSError:
                current = current.parent
                if current == current.parent:
                    return
                continue
            attributes = getattr(info, "st_file_attributes", 0)
            if attributes & FILE_ATTRIBUTE_REPARSE_POINT:
                raise PathViolation(
                    f"reparse point (symlink/junction) is not allowed: {current}",
                    code="PATH_REPARSE_POINT",
                )
            if current == self.root:
                return
            parent = current.parent
            if parent == current:
                return
            current = parent

    def ensure_write_allowed(self, resolved: Path) -> None:
        if self.is_protected(resolved):
            raise PathViolation(
                f"path is protected by policy: {self.relative(resolved)}",
                code="PATH_PROTECTED",
            )

    def is_read_protected(self, path: Path) -> bool:
        try:
            relative = self.relative(path).casefold()
        except ValueError:
            return True
        if relative in self.read_protected_files:
            return True
        for prefix in self.read_protected_prefixes:
            if relative == prefix or relative.startswith(prefix + "/"):
                return True
        return False

    def ensure_read_allowed(self, resolved: Path) -> None:
        if self.is_read_protected(resolved):
            raise PathViolation(
                f"RETRIEVAL_SOURCE_DENIED: model reads of protected evaluation "
                f"assets are not allowed: {self.relative(resolved)}",
                code="RETRIEVAL_SOURCE_DENIED",
            )


ApprovalCallback = Callable[[str, dict], bool]


@dataclass
class PermissionPolicy:
    workspace: Path
    guard: WorkspaceGuard
    allow_execution: bool = False
    allow_destructive: bool = False
    approval_callback: ApprovalCallback | None = None

    @classmethod
    def create(
        cls,
        workspace: Path,
        *,
        allow_execution: bool = False,
        allow_destructive: bool = False,
        approval_callback: ApprovalCallback | None = None,
    ) -> "PermissionPolicy":
        return cls(
            workspace=Path(workspace).resolve(),
            guard=WorkspaceGuard(workspace),
            allow_execution=allow_execution,
            allow_destructive=allow_destructive,
            approval_callback=approval_callback,
        )

    def check(
        self,
        *,
        name: str,
        arguments: dict,
        level: PermissionLevel,
    ) -> PermissionDecision:
        if level is PermissionLevel.READ:
            raw_path = arguments.get("path")
            if isinstance(raw_path, str) and raw_path.strip():
                try:
                    resolved = self.guard.resolve(raw_path)
                    self.guard.ensure_read_allowed(resolved)
                except PathViolation as exc:
                    return PermissionDecision(False, level, f"{exc.code}: {exc}")
            return PermissionDecision(True, level, "read-only tool allowed")

        if level is PermissionLevel.WRITE_WORKSPACE:
            raw_path = arguments.get("path")
            if isinstance(raw_path, str):
                try:
                    resolved = self.guard.resolve(raw_path, for_write=True)
                    self.guard.ensure_write_allowed(resolved)
                except PathViolation as exc:
                    return PermissionDecision(False, level, f"{exc.code}: {exc}")
            return PermissionDecision(True, level, "workspace write allowed")

        if level is PermissionLevel.EXECUTE_APPROVED:
            if not self.allow_execution:
                return PermissionDecision(
                    False,
                    level,
                    "command execution is disabled; pass --allow-execute for an "
                    "isolated test workspace",
                )
            return PermissionDecision(True, level, "approved executor allowed")

        if level is PermissionLevel.DESTRUCTIVE:
            if self.allow_destructive:
                return PermissionDecision(True, level, "destructive operation pre-approved")
            if self.approval_callback is not None and self.approval_callback(name, arguments):
                return PermissionDecision(True, level, "destructive operation approved by callback")
            return PermissionDecision(
                False,
                level,
                "destructive operation requires explicit approval",
            )

        return PermissionDecision(False, level, f"unknown permission level: {level}")


def is_reparse_point(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return bool(getattr(info, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT)


def is_regular_file(path: Path) -> bool:
    try:
        mode = os.lstat(path).st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode)