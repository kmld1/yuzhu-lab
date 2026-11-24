"""Shared helpers for local coding tools: bounded IO and process control."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..errors import ToolExecutionError
from ..registry import ToolContext, ToolResult, ToolStatus

if os.name == "nt":
    from .windows_job import WindowsProcessJob, pipe_available

_CLEANUP_FAILED = threading.Event()
_PROCESS_CLEANUP_SECONDS = 2.0
_PIPE_DRAIN_SECONDS = 1.0

# Only variables a build/test process normally needs.  API keys and tokens from
# the harness process are deliberately not inherited.
_SAFE_ENV_KEYS = (
    "PATH",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "COMSPEC",
    "PATHEXT",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "PROGRAMDATA",
    "LOCALAPPDATA",
    "APPDATA",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "VIRTUAL_ENV",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "PYTHONHASHSEED",
    "PYTHONDONTWRITEBYTECODE",
)

# Never persist bytecode from tool runs: a stale same-second .pyc can make a
# test pass against the previous version of a file.
_CHILD_EXTRAS = {"PYTHONDONTWRITEBYTECODE": "1"}
_SENSITIVE = re.compile(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH)", re.IGNORECASE)


def child_environment() -> dict[str, str]:
    env: dict[str, str] = {}
    for name in _SAFE_ENV_KEYS:
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    for name, value in os.environ.items():
        if name.upper() in env:
            continue
        # Keep locale settings but never credentials.
        if name.upper().startswith(("LC_", "LANG")):
            env[name] = value
    env.update(_CHILD_EXTRAS)
    return {name: value for name, value in env.items() if not _SENSITIVE.search(name)}


def truncate_output(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + f"\n[...truncated {len(text) - limit} chars]", True


def ok(
    ctx: ToolContext,
    name: str,
    content: str,
    *,
    structured: dict[str, Any] | None = None,
    exit_code: int | None = None,
) -> ToolResult:
    text, truncated = truncate_output(content, ctx.max_output_chars)
    return ToolResult(
        call_id="",
        name=name,
        status=ToolStatus.OK,
        content=text,
        structured=structured or {},
        truncated=truncated,
        exit_code=exit_code,
    )


def fail(
    name: str,
    code: str,
    message: str,
    *,
    structured: dict[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        call_id="",
        name=name,
        status=ToolStatus.ERROR,
        content=message,
        structured=structured or {},
        error_code=code,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bytes(path: Path, max_bytes: int) -> tuple[bytes, bool]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        data = handle.read(max_bytes)
    return data, size > len(data)


def decode_text(data: bytes) -> tuple[str, bool]:
    """Decode UTF-8 with replacement; report whether the payload looked binary."""
    binary = b"\x00" in data[:4096]
    return data.decode("utf-8", errors="replace"), binary


def atomic_write_text(path: Path, text: str) -> int:
    """Write UTF-8 bytes atomically.  Returns the byte count written."""
    payload = text.encode("utf-8")
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        with temp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass
    return len(payload)


def format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


@dataclass
class ProcessResult:
    argv: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    cancelled: bool = False
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    read_incomplete: bool = False
    raw_returncode: int | None = None
    reader_errors: str = ""

    @property
    def output_truncated(self) -> bool:
        return self.stdout_truncated or self.stderr_truncated

    def combined(self) -> str:
        parts = []
        if self.stdout:
            parts.append(self.stdout.rstrip())
        if self.stderr:
            parts.append("[stderr]\n" + self.stderr.rstrip())
        return "\n".join(parts)


class _ByteBudget:
    """Shared byte budget across stdout and stderr."""

    def __init__(self, total: int) -> None:
        self._remaining = max(0, total)
        self._lock = threading.Lock()

    def take(self, requested: int) -> int:
        with self._lock:
            granted = min(max(0, requested), self._remaining)
            self._remaining -= granted
            return granted


class _PipeReader(threading.Thread):
    """Drain a pipe continuously, retaining only a bounded prefix."""

    def __init__(
        self,
        stream: Any,
        *,
        stream_cap: int,
        budget: _ByteBudget,
        name: str,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self._stream = stream
        self._stop_reading = threading.Event()
        self._stream_cap = max(0, stream_cap)
        self._budget = budget
        self._chunks: list[bytes] = []
        self.total_bytes = 0
        self.stored_bytes = 0
        self.truncated = False
        self.error: str | None = None

    def run(self) -> None:
        try:
            while not self._stop_reading.is_set():
                size = 8192
                if os.name == "nt":
                    available = pipe_available(self._stream)
                    if available is None:
                        break
                    if available == 0:
                        self._stop_reading.wait(0.005)
                        continue
                    size = min(size, available)
                chunk = self._stream.read(size)
                if not chunk:
                    break
                self.total_bytes += len(chunk)
                own_remaining = max(0, self._stream_cap - self.stored_bytes)
                requested = min(len(chunk), own_remaining)
                granted = self._budget.take(requested)
                if granted > 0:
                    self._chunks.append(chunk[:granted])
                    self.stored_bytes += granted
                if granted < len(chunk):
                    self.truncated = True
        except (OSError, ValueError) as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                self._stream.close()
            except OSError:
                pass

    def stop_reading(self) -> None:
        self._stop_reading.set()

    def text(self) -> str:
        return b"".join(self._chunks).decode("utf-8", errors="replace")


def _terminate_process(process: subprocess.Popen, job: WindowsProcessJob | None) -> None:
    deadline = time.monotonic() + _PROCESS_CLEANUP_SECONDS
    errors: list[str] = []
    if job is not None:
        try:
            if job.assigned:
                job.finish(deadline)
        except OSError as exc:
            errors.append(str(exc))
        finally:
            try:
                job.close()
            except OSError as exc:
                errors.append(str(exc))
    if process.poll() is None:
        try:
            process.kill()
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except (OSError, subprocess.TimeoutExpired) as exc:
            if process.poll() is None:
                errors.append(f"owned process did not exit: {exc}")
    if errors:
        _block_process_launches("; ".join(errors))


def _block_process_launches(message: str) -> None:
    _CLEANUP_FAILED.set()
    raise ToolExecutionError("PROCESS_CLEANUP_FAILED", message)


def _drain_pipes(process: subprocess.Popen, readers: tuple[_PipeReader, ...]) -> bool:
    draining = [reader for reader in readers if reader.ident is not None]
    deadline = time.monotonic() + _PIPE_DRAIN_SECONDS
    for reader in draining:
        reader.join(max(0, deadline - time.monotonic()))
    unfinished = [reader for reader in draining if reader.is_alive()]
    if os.name == "nt":
        for reader in unfinished:
            reader.stop_reading()
        stop_deadline = time.monotonic() + 0.2
        for reader in unfinished:
            reader.join(max(0, stop_deadline - time.monotonic()))
    # Live non-Windows reader threads own their pipe until the kernel reports EOF.
    for stream, reader in zip((process.stdout, process.stderr), (*readers, None, None)):
        if reader is None or reader.ident is None:
            stream.close()
    return bool(unfinished)


def require_process_cleanup() -> None:
    if _CLEANUP_FAILED.is_set():
        raise ToolExecutionError("PROCESS_CLEANUP_FAILED", "previous process cleanup could not be verified")


def run_process(
    argv: list[str],
    *,
    cwd: Path,
    timeout: float,
    max_output_chars: int = 20000,
    cancel: Callable[[], bool] | None = None,
    max_stdout_bytes: int = 200_000,
    max_stderr_bytes: int = 100_000,
    max_total_bytes: int = 300_000,
) -> ProcessResult:
    require_process_cleanup()
    started = time.perf_counter()
    try:
        job = WindowsProcessJob() if os.name == "nt" else None
    except OSError as exc:
        raise ToolExecutionError("PROCESS_SETUP_FAILED", str(exc)) from exc
    try:
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=child_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            close_fds=True,
            creationflags=job.creation_flags if job is not None else 0,
        )
    except BaseException:
        if job is not None:
            job.close()
        raise

    readers: tuple[_PipeReader, ...] = ()
    timed_out = False
    cancelled = False
    deadline = time.monotonic() + max(0, timeout - (time.perf_counter() - started))
    try:
        if job is not None:
            try:
                job.start(process)
            except OSError as exc:
                raise ToolExecutionError("PROCESS_SETUP_FAILED", str(exc)) from exc
        budget = _ByteBudget(max_total_bytes)
        readers = (
            _PipeReader(process.stdout, stream_cap=max_stdout_bytes, budget=budget, name="yuzhu-stdout"),
            _PipeReader(process.stderr, stream_cap=max_stderr_bytes, budget=budget, name="yuzhu-stderr"),
        )
        for reader in readers:
            reader.start()
        while process.poll() is None:
            cancelled = cancel is not None and cancel()
            remaining = deadline - time.monotonic()
            if cancelled or remaining <= 0:
                timed_out = not cancelled
                break
            time.sleep(min(0.02, remaining))
    finally:
        try:
            _terminate_process(process, job)
        finally:
            read_incomplete = _drain_pipes(process, readers)
            if job is not None and read_incomplete:
                _block_process_launches("process output pipes did not reach EOF after job cleanup")

    stdout_reader, stderr_reader = readers
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    stdout_text, stdout_char_truncated = truncate_output(stdout_reader.text(), max_output_chars)
    stderr_text, stderr_char_truncated = truncate_output(stderr_reader.text(), max_output_chars)
    reader_errors = "; ".join(reader.error for reader in readers if reader.error)
    return ProcessResult(
        argv=argv,
        exit_code=None if (timed_out or cancelled) else process.returncode,
        stdout=stdout_text,
        stderr=stderr_text,
        duration_ms=duration_ms,
        timed_out=timed_out,
        cancelled=cancelled,
        stdout_bytes=stdout_reader.total_bytes,
        stderr_bytes=stderr_reader.total_bytes,
        stdout_truncated=stdout_reader.truncated or stdout_char_truncated,
        stderr_truncated=stderr_reader.truncated or stderr_char_truncated,
        read_incomplete=read_incomplete or bool(reader_errors),
        raw_returncode=process.returncode,
        reader_errors=reader_errors,
    )


def default_interpreter(ctx: ToolContext) -> str:
    if ctx.interpreter is not None:
        return str(ctx.interpreter)
    return sys.executable


def ensure_file(path: Path, *, tool: str) -> ToolResult | None:
    if not path.exists():
        return fail(tool, "PATH_NOT_FOUND", f"file does not exist: {path}")
    if not path.is_file():
        return fail(tool, "NOT_A_FILE", f"path is not a file: {path}")
    return None


def ensure_dir(path: Path, *, tool: str) -> ToolResult | None:
    if not path.exists():
        return fail(tool, "PATH_NOT_FOUND", f"directory does not exist: {path}")
    if not path.is_dir():
        return fail(tool, "NOT_A_DIRECTORY", f"path is not a directory: {path}")
    return None


def env_summary(env: Mapping[str, str]) -> dict[str, int]:
    return {"env_keys": len(env)}