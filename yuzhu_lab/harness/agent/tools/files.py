"""Workspace-confined file tools."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..permissions import PermissionLevel, is_reparse_point
from ..registry import ToolContext, ToolDefinition, ToolResult
from .common import (
    atomic_write_text,
    decode_text,
    ensure_dir,
    ensure_file,
    fail,
    format_size,
    ok,
    read_bytes,
    sha256_bytes,
    sha256_file,
    sha256_text,
)

_SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "models",
    ".cache",
    "unsloth_compiled_cache",
}

MAX_WRITE_CHARS = 1_000_000


def _relative(ctx: ToolContext, path: Path) -> str:
    return path.resolve().relative_to(ctx.guard.root).as_posix()


def _read_text(path: Path, ctx: ToolContext, *, tool: str) -> tuple[str, int, bool] | ToolResult:
    data, size_truncated = read_bytes(path, ctx.max_read_bytes)
    text, binary = decode_text(data)
    if binary:
        return fail(tool, "BINARY_FILE", f"refusing to read binary file: {path.name}")
    return text, path.stat().st_size, size_truncated


# ----------------------------------------------------------------- read_file


def read_file(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    path = ctx.guard.resolve(arguments["path"], must_exist=True)
    ctx.guard.ensure_read_allowed(path)
    check = ensure_file(path, tool="read_file")
    if check:
        return check
    read = _read_text(path, ctx, tool="read_file")
    if isinstance(read, ToolResult):
        return read
    text, size_bytes, size_truncated = read

    lines = text.splitlines()
    total_lines = len(lines)
    start_line = max(1, int(arguments.get("start_line") or 1))
    max_lines = min(int(arguments.get("max_lines") or 400), 2000)
    if total_lines == 0:
        return ok(
            ctx,
            "read_file",
            "",
            structured={
                "path": _relative(ctx, path),
                "start_line": 1,
                "end_line": 0,
                "total_lines": 0,
                "size_bytes": size_bytes,
                "sha256": sha256_bytes(b""),
                "truncated": size_truncated,
            },
        )
    end_line = min(total_lines, start_line - 1 + max_lines)
    if start_line > total_lines:
        return fail(
            "read_file",
            "LINE_OUT_OF_RANGE",
            f"start_line {start_line} is beyond end of file ({total_lines} lines)",
        )
    selected = lines[start_line - 1:end_line]
    content = "\n".join(selected)
    truncated = size_truncated or end_line < total_lines
    digest = None if size_truncated else sha256_file(path)
    return ok(
        ctx,
        "read_file",
        content,
        structured={
            "path": _relative(ctx, path),
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": total_lines,
            "size_bytes": size_bytes,
            "sha256": digest,
            "truncated": truncated,
        },
    )


# ----------------------------------------------------------------- list_files


def list_files(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    base = ctx.guard.resolve(arguments.get("path") or ".", must_exist=True)
    ctx.guard.ensure_read_allowed(base)
    check = ensure_dir(base, tool="list_files")
    if check:
        return check
    recursive = bool(arguments.get("recursive", False))
    max_entries = min(int(arguments.get("max_entries") or 200), 1000)

    entries: list[dict[str, Any]] = []
    lines: list[str] = []
    truncated = False

    def add(path: Path, kind: str, size: int | None) -> bool:
        nonlocal truncated
        if len(entries) >= max_entries:
            truncated = True
            return False
        relative = _relative(ctx, path)
        entries.append({"path": relative, "type": kind, "size_bytes": size})
        if kind == "directory":
            lines.append(f"d  {relative}/")
        else:
            lines.append(f"f  {relative} ({format_size(size or 0)})")
        return True

    def scan(directory: Path) -> bool:
        try:
            items = sorted(os.scandir(directory), key=lambda item: item.name.lower())
        except OSError:
            return True
        for item in items:
            ctx.check_deadline()
            item_path = Path(item.path)
            if ctx.guard.is_read_protected(item_path):
                continue
            if is_reparse_point(item_path):
                if not add(item_path, "reparse_point", None):
                    return False
                continue
            if item.is_dir(follow_symlinks=False):
                if item.name in _SKIP_DIRS:
                    continue
                if not add(item_path, "directory", None):
                    return False
                if recursive and not scan(item_path):
                    return False
            elif item.is_file(follow_symlinks=False):
                size = item.stat(follow_symlinks=False).st_size
                if not add(item_path, "file", size):
                    return False
        return True

    scan(base)
    if not entries:
        lines.append("(empty directory)")
    return ok(
        ctx,
        "list_files",
        "\n".join(lines),
        structured={
            "path": _relative(ctx, base),
            "entries": len(entries),
            "truncated": truncated,
            "recursive": recursive,
        },
    )


# ---------------------------------------------------------------- search_text


def search_text(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    query = arguments["query"]
    if not query.strip():
        return fail("search_text", "INVALID_ARGUMENTS", "query must not be empty")
    base = ctx.guard.resolve(arguments.get("path") or ".", must_exist=True)
    ctx.guard.ensure_read_allowed(base)
    max_results = min(int(arguments.get("max_results") or 50), 200)
    case_sensitive = bool(arguments.get("case_sensitive", False))
    if case_sensitive:
        needle = query
        def line_matches(line: str) -> bool:
            return needle in line
    else:
        lowered = query.lower()
        def line_matches(line: str) -> bool:
            return lowered in line.lower()

    files: list[Path] = []
    if base.is_file():
        files = [base]
    else:
        for root, dirs, names in os.walk(base):
            ctx.check_deadline()
            dirs[:] = [
                name for name in dirs if name not in _SKIP_DIRS
                and not is_reparse_point(Path(root) / name)
                and not ctx.guard.is_read_protected(Path(root) / name)
            ]
            for name in names:
                candidate = Path(root) / name
                if ctx.guard.is_read_protected(candidate):
                    continue
                files.append(candidate)
                if len(files) > 5000:
                    break

    results: list[str] = []
    matches = 0
    files_scanned = 0
    truncated = False
    for path in files:
        if matches >= max_results:
            truncated = True
            break
        ctx.check_deadline()
        try:
            if is_reparse_point(path) or not path.is_file():
                continue
            if path.stat().st_size > 1024 * 1024:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:4096]:
            continue
        files_scanned += 1
        text = data.decode("utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if line_matches(line):
                snippet = line.strip()[:300]
                results.append(f"{_relative(ctx, path)}:{number}: {snippet}")
                matches += 1
                if matches >= max_results:
                    truncated = True
                    break

    content = "\n".join(results) if results else "(no matches)"
    return ok(
        ctx,
        "search_text",
        content,
        structured={
            "query": query,
            "matches": matches,
            "files_scanned": files_scanned,
            "truncated": truncated,
        },
    )


# --------------------------------------------------------------- get_file_info


def get_file_info(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    path = ctx.guard.resolve(arguments["path"])
    ctx.guard.ensure_read_allowed(path)
    relative = _relative(ctx, path)
    if not path.exists():
        return ok(
            ctx,
            "get_file_info",
            f"{relative}: does not exist",
            structured={"path": relative, "exists": False},
        )
    stat = os.lstat(path)
    is_dir = path.is_dir()
    info: dict[str, Any] = {
        "path": relative,
        "exists": True,
        "type": "directory" if is_dir else "file",
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "reparse_point": is_reparse_point(path),
        "protected": ctx.guard.is_protected(path),
    }
    lines = [f"path: {relative}", f"type: {info['type']}", f"size: {format_size(stat.st_size)}"]
    if not is_dir and stat.st_size <= 32 * 1024 * 1024:
        info["sha256"] = sha256_file(path)
        lines.append(f"sha256: {info['sha256']}")
    if not is_dir and stat.st_size <= 2 * 1024 * 1024:
        data = path.read_bytes()
        if b"\x00" not in data[:4096]:
            info["lines"] = len(data.decode("utf-8", errors="replace").splitlines())
            lines.append(f"lines: {info['lines']}")
    return ok(ctx, "get_file_info", "\n".join(lines), structured=info)


# ----------------------------------------------------------------- write_file


def write_file(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    path = ctx.guard.resolve(arguments["path"], for_write=True)
    ctx.guard.ensure_write_allowed(path)
    content = arguments["content"]
    overwrite = bool(arguments.get("overwrite", False))
    create_parents = bool(arguments.get("create_parents", False))

    parent = path.parent
    if not parent.is_dir():
        if not create_parents:
            return fail(
                "write_file",
                "PARENT_NOT_FOUND",
                f"parent directory does not exist: {_relative(ctx, parent)} "
                "(set create_parents=true to create it)",
            )
        parent.mkdir(parents=True, exist_ok=True)

    existed = path.exists()
    if existed and not path.is_file():
        return fail("write_file", "NOT_A_FILE", f"path is not a file: {_relative(ctx, path)}")
    if existed and not overwrite:
        return fail(
            "write_file",
            "FILE_EXISTS",
            f"file already exists: {_relative(ctx, path)} (use edit_file, apply_patch, "
            "or overwrite=true)",
        )

    bytes_written = atomic_write_text(path, content)
    expected = sha256_text(content)
    actual = sha256_file(path)
    verified = expected == actual
    structured = {
        "path": _relative(ctx, path),
        "created": not existed,
        "bytes": bytes_written,
        "sha256": actual,
        "readback_verified": verified,
    }
    if not verified:
        return fail(
            "write_file",
            "WRITE_VERIFICATION_FAILED",
            f"written file hash mismatch for {_relative(ctx, path)}",
            structured=structured,
        )
    return ok(
        ctx,
        "write_file",
        f"wrote {bytes_written} bytes to {_relative(ctx, path)} (verified)",
        structured=structured,
    )


# ------------------------------------------------------------------ edit_file


def edit_file(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    path = ctx.guard.resolve(arguments["path"], must_exist=True)
    ctx.guard.ensure_write_allowed(path)
    check = ensure_file(path, tool="edit_file")
    if check:
        return check
    if path.stat().st_size > MAX_WRITE_CHARS:
        return fail("edit_file", "FILE_TOO_LARGE", "file exceeds the 1 MB edit limit")

    old_text = arguments["old_text"]
    new_text = arguments["new_text"]
    replace_all = bool(arguments.get("replace_all", False))
    if not old_text:
        return fail("edit_file", "INVALID_ARGUMENTS", "old_text must not be empty")

    try:
        original = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return fail(
            "edit_file",
            "ENCODING_UNSUPPORTED",
            "file is not valid UTF-8; refusing to rewrite it with replacement characters",
        )
    count = original.count(old_text)
    if count == 0:
        return fail(
            "edit_file",
            "OLD_TEXT_NOT_FOUND",
            "old_text was not found verbatim; read the file again before editing",
        )
    if count > 1 and not replace_all:
        return fail(
            "edit_file",
            "AMBIGUOUS_MATCH",
            f"old_text occurs {count} times; provide more context or set replace_all=true",
        )

    replaced = original.replace(old_text, new_text, -1 if replace_all else 1)
    atomic_write_text(path, replaced)
    verified = path.read_text(encoding="utf-8") == replaced
    structured = {
        "path": _relative(ctx, path),
        "replacements": count if replace_all else 1,
        "sha256": sha256_text(replaced),
        "readback_verified": verified,
    }
    if not verified:
        return fail(
            "edit_file",
            "WRITE_VERIFICATION_FAILED",
            "edited file did not read back as expected",
            structured=structured,
        )
    return ok(
        ctx,
        "edit_file",
        f"edited {_relative(ctx, path)} ({structured['replacements']} replacement(s), verified)",
        structured=structured,
    )


# ---------------------------------------------------------------- apply_patch

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchSyntaxError(ValueError):
    pass


def _parse_patch(patch: str) -> list[dict[str, Any]]:
    hunks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in patch.splitlines():
        line = raw.rstrip("\r")
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        match = _HUNK_RE.match(line)
        if match:
            current = {
                "old_start": int(match.group(1)),
                "old_count": int(match.group(2) or 1),
                "new_start": int(match.group(3)),
                "new_count": int(match.group(4) or 1),
                "lines": [],
            }
            hunks.append(current)
            continue
        if current is None:
            if line.strip():
                raise PatchSyntaxError("patch must start with a @@ hunk header")
            continue
        if line.startswith("\\"):
            continue
        if not line:
            raise PatchSyntaxError("empty line inside a hunk; use a single space for context")
        marker = line[0]
        if marker not in {" ", "+", "-"}:
            raise PatchSyntaxError(f"unsupported patch line: {line[:40]!r}")
        current["lines"].append((marker, line[1:]))
    if not hunks:
        raise PatchSyntaxError("patch contains no hunks")
    return hunks


def _apply_hunks(
    original_lines: list[str], hunks: list[dict[str, Any]]
) -> list[str]:
    output: list[str] = []
    position = 0
    for hunk in hunks:
        old_start = hunk["old_start"]
        start = max(0, old_start - 1) if hunk["old_count"] else max(0, old_start)
        if start < position:
            raise PatchSyntaxError("hunks overlap or are out of order")
        output.extend(original_lines[position:start])
        position = start
        consumed_old = 0
        produced_new = 0
        for marker, text in hunk["lines"]:
            if marker in {" ", "-"}:
                if position >= len(original_lines):
                    raise PatchSyntaxError("hunk runs past the end of the file")
                if original_lines[position] != text:
                    raise PatchSyntaxError(
                        f"context mismatch at line {position + 1}: "
                        f"expected {text!r}, found {original_lines[position]!r}"
                    )
                if marker == " ":
                    output.append(text)
                    produced_new += 1
                position += 1
                consumed_old += 1
            else:
                output.append(text)
                produced_new += 1
        expected_old = hunk["old_count"]
        expected_new = hunk["new_count"]
        if expected_old and consumed_old != expected_old:
            raise PatchSyntaxError(
                f"hunk expected {expected_old} old lines, consumed {consumed_old}"
            )
        if produced_new != expected_new:
            raise PatchSyntaxError(
                f"hunk expected {expected_new} new lines, produced {produced_new}"
            )
    output.extend(original_lines[position:])
    return output


def apply_patch(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    path = ctx.guard.resolve(arguments["path"], for_write=True)
    ctx.guard.ensure_write_allowed(path)
    patch = arguments["patch"]
    try:
        hunks = _parse_patch(patch)
    except PatchSyntaxError as exc:
        return fail("apply_patch", "PATCH_INVALID", str(exc))

    if path.exists():
        if not path.is_file():
            return fail("apply_patch", "NOT_A_FILE", f"path is not a file: {_relative(ctx, path)}")
        if path.stat().st_size > MAX_WRITE_CHARS:
            return fail("apply_patch", "FILE_TOO_LARGE", "file exceeds the 1 MB patch limit")
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return fail(
                "apply_patch",
                "ENCODING_UNSUPPORTED",
                "file is not valid UTF-8; refusing to rewrite it with replacement characters",
            )
        original_lines = original.splitlines()
        trailing_newline = original.endswith("\n")
    else:
        only_empty_hunk = (
            len(hunks) == 1 and hunks[0]["old_start"] == 0 and hunks[0]["old_count"] == 0
        )
        if not only_empty_hunk:
            return fail("apply_patch", "PATH_NOT_FOUND", f"file does not exist: {_relative(ctx, path)}")
        original_lines = []
        trailing_newline = True

    try:
        updated_lines = _apply_hunks(original_lines, hunks)
    except PatchSyntaxError as exc:
        return fail("apply_patch", "PATCH_CONFLICT", str(exc))

    updated = "\n".join(updated_lines)
    if updated_lines and (trailing_newline or not original_lines):
        updated += "\n"
    atomic_write_text(path, updated)
    verified = path.read_text(encoding="utf-8") == updated
    structured = {
        "path": _relative(ctx, path),
        "hunks_applied": len(hunks),
        "sha256": sha256_text(updated),
        "readback_verified": verified,
    }
    if not verified:
        return fail(
            "apply_patch",
            "WRITE_VERIFICATION_FAILED",
            "patched file did not read back as expected",
            structured=structured,
        )
    return ok(
        ctx,
        "apply_patch",
        f"applied {len(hunks)} hunk(s) to {_relative(ctx, path)} (verified)",
        structured=structured,
    )


# ---------------------------------------------------------------- definitions


def file_tool_definitions() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="read_file",
            description="Read a UTF-8 text file in the workspace. Returns raw text (no line numbers).",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "maxLength": 400},
                    "start_line": {"type": "integer", "minimum": 1},
                    "max_lines": {"type": "integer", "minimum": 1, "maximum": 2000},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            execute=read_file,
            timeout_seconds=20,
        ),
        ToolDefinition(
            name="list_files",
            description="List files and directories under a workspace path.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "maxLength": 400},
                    "recursive": {"type": "boolean"},
                    "max_entries": {"type": "integer", "minimum": 1, "maximum": 1000},
                },
                "additionalProperties": False,
            },
            execute=list_files,
            timeout_seconds=30,
        ),
        ToolDefinition(
            name="search_text",
            description="Search literal text in workspace files; returns path:line: text.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 200},
                    "path": {"type": "string", "maxLength": 400},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 200},
                    "case_sensitive": {"type": "boolean"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            execute=search_text,
            timeout_seconds=60,
        ),
        ToolDefinition(
            name="get_file_info",
            description="Return existence, type, size, mtime and sha256 for a workspace path.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 400}},
                "required": ["path"],
                "additionalProperties": False,
            },
            execute=get_file_info,
            timeout_seconds=30,
        ),
        ToolDefinition(
            name="write_file",
            description=(
                "Create a file in the workspace. Fails if the file exists unless "
                "overwrite=true (which requires approval)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "maxLength": 400},
                    "content": {"type": "string", "maxLength": MAX_WRITE_CHARS},
                    "overwrite": {"type": "boolean"},
                    "create_parents": {"type": "boolean"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            execute=write_file,
            permission_level=PermissionLevel.WRITE_WORKSPACE,
            permission_for=lambda args: (
                PermissionLevel.DESTRUCTIVE
                if args.get("overwrite")
                else PermissionLevel.WRITE_WORKSPACE
            ),
            idempotent=False,
            timeout_seconds=20,
        ),
        ToolDefinition(
            name="edit_file",
            description="Replace one exact text block in an existing file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "maxLength": 400},
                    "old_text": {"type": "string", "minLength": 1, "maxLength": 20000},
                    "new_text": {"type": "string", "maxLength": 100000},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
            execute=edit_file,
            permission_level=PermissionLevel.WRITE_WORKSPACE,
            idempotent=False,
            timeout_seconds=20,
        ),
        ToolDefinition(
            name="apply_patch",
            description="Apply a strict unified diff (hunks only) to a workspace file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "maxLength": 400},
                    "patch": {"type": "string", "minLength": 1, "maxLength": 200000},
                },
                "required": ["path", "patch"],
                "additionalProperties": False,
            },
            execute=apply_patch,
            permission_level=PermissionLevel.WRITE_WORKSPACE,
            idempotent=False,
            timeout_seconds=20,
        ),
    ]