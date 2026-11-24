"""Evaluation-source admission policy.

The policy is shared by RAG ingestion, RAG retrieval, the harness file tools,
the ContextBuilder and Hindsight recall so that every model-facing entry point
applies the same rules:

* protected benchmark assets (``eval/`` and the explicit generator scripts)
  are never admitted;
* a source must resolve inside the workspace, must not be a reparse-point
  escape and, when existence is required, must be a real file;
* missing or unresolvable provenance is denied (fail closed);
* normal project source, Markdown and engineering tests remain readable.

This is an in-process policy.  It is not an operating-system sandbox; see
``SECURITY.md`` for the residual execution risk.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

SOURCE_ALLOWED = "SOURCE_ALLOWED"
RETRIEVAL_SOURCE_DENIED = "RETRIEVAL_SOURCE_DENIED"
PROTECTED_EVAL_SOURCE = "PROTECTED_EVAL_SOURCE"
MISSING_PROVENANCE = "MISSING_PROVENANCE"
PATH_OUTSIDE_WORKSPACE = "PATH_OUTSIDE_WORKSPACE"
PATH_REPARSE_POINT = "PATH_REPARSE_POINT"
PATH_NOT_FOUND = "PATH_NOT_FOUND"
PATH_INVALID = "PATH_INVALID"

FILE_ATTRIBUTE_REPARSE_POINT = 0x400

# Directory names, relative to the workspace root, that contain frozen
# evaluation assets.  Matching is segment-based, so ``evaluation_notes.md`` is
# not treated as ``eval``.
DEFAULT_PROTECTED_PREFIXES: tuple[str, ...] = ("eval",)

# Explicit benchmark generators/scorers.  These are exact relative file names,
# not fuzzy substrings.
DEFAULT_PROTECTED_FILES: tuple[str, ...] = (
    "src/make_eval_v1.py",
    "src/make_eval_v2.py",
    "src/check_eval_v2_leakage.py",
    "src/blind_scoring_v2.py",
    "src/eval_base_v1.py",
    "src/eval_lora_v1.py",
    "src/eval_models_v2_1.py",
    "src/compare_eval_v1.py",
    "src/compare_eval_v2_1.py",
)

# The content fingerprint is deliberately narrower than path denial: it covers
# the benchmark-defining prompts, criteria, locked predictions and scoring
# results, not every JSON manifest under eval/ (hash-only metadata would cause
# false positives in ordinary documentation).
DEFAULT_FINGERPRINT_FILES: tuple[str, ...] = (
    "eval/sft_eval_v1.jsonl",
    "eval/sft_eval_v2.jsonl",
    "eval/base_predictions_v1_locked.jsonl",
    "eval/lora_predictions_v1_locked.jsonl",
    "eval/base_predictions_v1_final.jsonl",
    "eval/eval_v2_leakage_report.jsonl",
    "src/make_eval_v1.py",
    "src/make_eval_v2.py",
    "src/blind_scoring_v2.py",
)

DEFAULT_FINGERPRINT_PREFIXES: tuple[str, ...] = (
    "eval/results_v2_1",
    "eval/h5_3",
)

_FINGERPRINT_SUFFIXES = {".jsonl", ".md", ".txt", ".py", ".csv", ".sha256"}

_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".toml",
    ".yaml",
    ".yml",
    ".ps1",
    ".csv",
    ".sha256",
}

_CITATION_SYMBOL_RE = re.compile(r"\s+\([^()]*\)\s*$")
_CITATION_LINE_RE = re.compile(r":\d+(?:-\d+)?$")


@dataclass(frozen=True)
class SourceDecision:
    allowed: bool
    code: str
    reason: str
    normalized: str | None = None
    resolved: Path | None = None


def _deny(code: str, reason: str, *, normalized: str | None = None) -> SourceDecision:
    return SourceDecision(False, code, reason, normalized=normalized)


def _allow(normalized: str, resolved: Path) -> SourceDecision:
    return SourceDecision(True, SOURCE_ALLOWED, "source admitted", normalized, resolved)


def _clean_relative(value: str) -> str:
    text = value.replace("\\", "/")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    return "/".join(parts)


class SourcePolicy:
    """Fail-closed admission policy rooted at a workspace directory."""

    def __init__(
        self,
        root: Path | str,
        *,
        protected_prefixes: tuple[str, ...] = DEFAULT_PROTECTED_PREFIXES,
        protected_files: tuple[str, ...] = DEFAULT_PROTECTED_FILES,
        extra_protected_prefixes: tuple[str, ...] = (),
        extra_protected_files: tuple[str, ...] = (),
        fingerprint_prefixes: tuple[str, ...] = DEFAULT_FINGERPRINT_PREFIXES,
        fingerprint_files: tuple[str, ...] = DEFAULT_FINGERPRINT_FILES,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"source policy root is not a directory: {self.root}")
        self.protected_prefixes = tuple(
            _clean_relative(prefix).casefold()
            for prefix in (*protected_prefixes, *extra_protected_prefixes)
            if _clean_relative(prefix)
        )
        self.protected_files = frozenset(
            _clean_relative(item).casefold()
            for item in (*protected_files, *extra_protected_files)
            if _clean_relative(item)
        )
        self.fingerprint_prefixes = tuple(
            _clean_relative(item).casefold()
            for item in fingerprint_prefixes
            if _clean_relative(item)
        )
        self.fingerprint_files = frozenset(
            _clean_relative(item).casefold()
            for item in fingerprint_files
            if _clean_relative(item)
        )

    # ------------------------------------------------------------------ paths

    def normalize_relative(self, value: str | None) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        if not text or "\x00" in text:
            return None
        if text.startswith("\\\\") or text.startswith("//"):
            return None
        if text.startswith("\\\\?\\") or text.startswith("\\\\.\\"):
            return None
        if len(text) >= 2 and text[1] == ":" and not (
            len(text) >= 3 and text[2] in "\\/"
        ):
            return None
        if ":" in text[2:]:
            return None
        candidate = Path(text.replace("\\", "/"))
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            return None
        try:
            relative = resolved.relative_to(self.root)
        except ValueError:
            return None
        return _clean_relative(relative.as_posix())

    def is_protected_relative(self, relative: str) -> bool:
        normalized = _clean_relative(relative).casefold()
        if not normalized:
            return False
        if normalized in self.protected_files:
            return True
        for prefix in self.protected_prefixes:
            if normalized == prefix or normalized.startswith(prefix + "/"):
                return True
        return False

    def check_relative(
        self,
        source: str | None,
        *,
        require_exists: bool = False,
    ) -> SourceDecision:
        if not isinstance(source, str) or not source.strip():
            return _deny(MISSING_PROVENANCE, "source provenance is missing")
        text = source.strip()
        if "\x00" in text:
            return _deny(MISSING_PROVENANCE, "source contains a NUL byte")
        if text.startswith("\\\\") or text.startswith("//"):
            return _deny(PATH_INVALID, "UNC/device paths are not admissible sources")
        if len(text) >= 2 and text[1] == ":" and not (
            len(text) >= 3 and text[2] in "\\/"
        ):
            return _deny(PATH_INVALID, "drive-relative paths are not admissible")
        if ":" in text[2:]:
            return _deny(PATH_INVALID, "alternate data streams are not admissible")

        candidate = Path(text.replace("\\", "/"))
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            return _deny(PATH_INVALID, f"source cannot be resolved: {type(exc).__name__}")
        try:
            relative = resolved.relative_to(self.root)
        except ValueError:
            return _deny(
                PATH_OUTSIDE_WORKSPACE,
                "source resolves outside the workspace root",
            )
        normalized = _clean_relative(relative.as_posix())

        if self._has_reparse_component(resolved):
            return _deny(
                PATH_REPARSE_POINT,
                "source path contains a symlink/junction component",
                normalized=normalized,
            )
        if self.is_protected_relative(normalized):
            return _deny(
                PROTECTED_EVAL_SOURCE,
                "source is a protected evaluation asset",
                normalized=normalized,
            )
        if require_exists and not resolved.is_file():
            return _deny(
                PATH_NOT_FOUND,
                "source does not resolve to an existing file",
                normalized=normalized,
            )
        return _allow(normalized, resolved)

    def check_path(self, path: Path | str, *, require_exists: bool = True) -> SourceDecision:
        return self.check_relative(str(path), require_exists=require_exists)

    def check_citation(self, citation: str | None) -> SourceDecision:
        if not isinstance(citation, str) or not citation.strip():
            return _deny(MISSING_PROVENANCE, "citation provenance is missing")
        text = citation.strip()
        symbol = _CITATION_SYMBOL_RE.search(text)
        if symbol is not None:
            text = text[: symbol.start()]
        line = _CITATION_LINE_RE.search(text)
        if line is not None:
            text = text[: line.start()]
        if not text.strip():
            return _deny(MISSING_PROVENANCE, "citation has no path component")
        return self.check_relative(text, require_exists=False)

    def _has_reparse_component(self, resolved: Path) -> bool:
        current = resolved
        while True:
            if current == self.root:
                return False
            try:
                info = os.lstat(current)
            except OSError:
                parent = current.parent
                if parent == current:
                    return False
                current = parent
                continue
            attributes = getattr(info, "st_file_attributes", 0)
            if attributes & FILE_ATTRIBUTE_REPARSE_POINT:
                return True
            parent = current.parent
            if parent == current:
                return False
            current = parent

    # ------------------------------------------------------------ fingerprints

    def iter_protected_files(self, *, max_bytes: int = 2_000_000):
        seen: set[Path] = set()
        for prefix in self.protected_prefixes:
            base = self.root / prefix
            if base.is_file():
                candidates = [base]
            elif base.is_dir():
                candidates = base.rglob("*")
            else:
                candidates = []
            for path in candidates:
                try:
                    if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
                        continue
                    if path.stat().st_size > max_bytes:
                        continue
                except OSError:
                    continue
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield path
        for relative in self.protected_files:
            path = self.root / relative
            try:
                if not path.is_file() or path.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path

    def iter_fingerprint_files(self, *, max_bytes: int = 2_000_000):
        """Yield benchmark-defining assets used for memory content screening."""
        candidates: list[Path] = []
        for relative in self.fingerprint_files:
            path = self.root / relative
            if path.is_file():
                candidates.append(path)
        for prefix in self.fingerprint_prefixes:
            base = self.root / prefix
            if base.is_file():
                candidates.append(base)
            elif base.is_dir():
                candidates.extend(base.rglob("*"))
        seen: set[Path] = set()
        for path in candidates:
            try:
                if (
                    not path.is_file()
                    or path.suffix.lower() not in _FINGERPRINT_SUFFIXES
                ):
                    continue
                if path.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _hash_text(text: str) -> str:
    return hashlib.blake2b(text.casefold().encode("utf-8"), digest_size=16).hexdigest()


_WORD_RE = re.compile(r"[A-Za-z]{4,}|[一-鿿]{4,}")


def _is_semantic(text: str) -> bool:
    """Reject hash-only / structural windows so metadata cannot create false hits."""
    for token in _WORD_RE.findall(text):
        if len(token) >= 16 and all(char in "0123456789abcdefABCDEF" for char in token):
            continue
        return True
    return False


def _json_strings(value, out: list[str]) -> None:
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _json_strings(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _json_strings(item, out)


class ProtectedContentGuard:
    """Exact-fragment screening for memory text.

    It is a screening control, not proof of provenance: memory that contains a
    substantial exact fragment of a protected asset is rejected.  Path-based
    admission remains the primary control.
    """

    def __init__(
        self,
        policy: SourcePolicy,
        *,
        window: int = 48,
        stride: int = 24,
        min_line: int = 32,
    ) -> None:
        self.policy = policy
        self.window = max(16, int(window))
        self.stride = max(8, int(stride))
        self.min_line = max(16, int(min_line))
        self._hashes: set[str] | None = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._hashes is not None or self._load_error is not None:
            return
        with self._lock:
            if self._hashes is not None or self._load_error is not None:
                return
            hashes: set[str] = set()
            try:
                for path in self.policy.iter_fingerprint_files():
                    for text in self._file_texts(path):
                        normalized = _normalize_text(text)
                        if (
                            len(normalized) >= self.min_line
                            and _is_semantic(normalized)
                        ):
                            hashes.add(_hash_text(normalized))
                        if len(normalized) >= self.window:
                            for start in range(
                                0, len(normalized) - self.window + 1, self.stride
                            ):
                                window = normalized[start : start + self.window]
                                if _is_semantic(window):
                                    hashes.add(_hash_text(window))
            except Exception as exc:  # noqa: BLE001 - screening must fail closed
                self._load_error = f"{type(exc).__name__}: {exc}"
                return
            self._hashes = hashes

    @staticmethod
    def _file_texts(path: Path) -> list[str]:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        if path.suffix.lower() == ".jsonl":
            texts: list[str] = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    texts.append(line)
                    continue
                _json_strings(value, texts)
            return texts
        return raw.splitlines()

    @property
    def available(self) -> bool:
        self._load()
        return self._hashes is not None

    def matches(self, text: str | None) -> bool:
        if not text:
            return False
        self._load()
        if self._load_error is not None or self._hashes is None:
            # Fail closed: if protected assets cannot be screened, model-visible
            # memory is not admitted.
            return True
        normalized = _normalize_text(text)
        if not normalized:
            return False
        for line in normalized.splitlines():
            if (
                len(line) >= self.min_line
                and _is_semantic(line)
                and _hash_text(line) in self._hashes
            ):
                return True
        if len(normalized) >= self.window:
            for start in range(0, len(normalized) - self.window + 1, self.stride):
                window = normalized[start : start + self.window]
                if _is_semantic(window) and _hash_text(window) in self._hashes:
                    return True
        return False

    @property
    def load_error(self) -> str | None:
        self._load()
        return self._load_error


_GUARD_CACHE: dict[tuple[str, tuple[str, ...], tuple[str, ...]], ProtectedContentGuard] = {}
_GUARD_LOCK = threading.Lock()


def protected_content_guard(policy: SourcePolicy) -> ProtectedContentGuard:
    key = (
        str(policy.root),
        policy.fingerprint_prefixes,
        tuple(sorted(policy.fingerprint_files)),
    )
    with _GUARD_LOCK:
        guard = _GUARD_CACHE.get(key)
        if guard is None:
            guard = ProtectedContentGuard(policy)
            _GUARD_CACHE[key] = guard
        return guard