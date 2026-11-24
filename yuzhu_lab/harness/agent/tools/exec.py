"""Approved command execution tools.

Only workspace scripts are executed; the model never supplies a raw shell
command.  Execution is disabled unless the permission policy explicitly allows
it, and this layer is not an OS sandbox (see the H2 reports).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from ..errors import PathViolation, ToolExecutionError
from ..permissions import PermissionLevel
from ..registry import ToolContext, ToolDefinition, ToolResult, ToolStatus
from ..verification import VERDICT_PASS, build_test_evidence
from .common import ProcessResult, default_interpreter, ensure_file, fail, ok, run_process

MAX_TIMEOUT_SECONDS = 300.0

MAX_STDOUT_BYTES = 200_000
MAX_STDERR_BYTES = 100_000
MAX_TOTAL_BYTES = 300_000


def classify_failure(
    name: str, text: str, exit_code: int | None
) -> tuple[str | None, str | None]:
    """Map common Windows/Python failures to a stable class and safe advice."""
    if exit_code in (0, None):
        return None, None
    lowered = text.lower()
    if name == "run_powershell":
        if (
            "commandnotfoundexception" in lowered
            or "is not recognized as the name of a cmdlet" in lowered
            or "无法将" in text and "识别为" in text
        ):
            return (
                "COMMAND_NOT_FOUND",
                "Use an existing workspace script path such as .\\name.ps1; do not "
                "pass a bare command name. Verify the file with list_files first.",
            )
        if "pathnotfound" in lowered or "cannot find path" in lowered or "找不到路径" in text:
            return (
                "PATH_NOT_FOUND",
                "Verify the path with list_files or get_file_info and use a "
                "workspace-relative path.",
            )
        if "unauthorizedaccess" in lowered or "access is denied" in lowered or "拒绝访问" in text:
            return (
                "ACCESS_DENIED",
                "The command lacks permission. Do not attempt to elevate privileges; "
                "use a workspace file or stop and report the blocker.",
            )
        if (
            "parameterbindingexception" in lowered
            or "a parameter cannot be found" in lowered
            or "无法找到接受" in text
        ):
            return (
                "INVALID_PARAMETERS",
                "Read the script to confirm its parameter names before retrying.",
            )
    if name == "run_python":
        if "modulenotfounderror" in lowered:
            return (
                "MISSING_MODULE",
                "The module is not installed in the harness interpreter; remove the "
                "import or record the dependency as a blocker.",
            )
        if "syntaxerror" in lowered:
            return "SYNTAX_ERROR", "Fix the reported syntax error before running again."
        if "permissionerror" in lowered:
            return (
                "ACCESS_DENIED",
                "The script lacks permission. Do not elevate privileges; use a "
                "workspace file or stop and report.",
            )
    return "NONZERO_EXIT", (
        "Read stdout/stderr, fix the cause, and do not repeat the same command unchanged."
    )


def _effective_timeout(requested: Any, ctx: ToolContext) -> float:
    try:
        value = float(requested) if requested is not None else 60.0
    except (TypeError, ValueError):
        value = 60.0
    value = max(1.0, min(value, MAX_TIMEOUT_SECONDS))
    if ctx.deadline is not None:
        remaining = ctx.deadline - time.monotonic()
        if remaining <= 0:
            raise ToolExecutionError("TIMEOUT", "tool deadline already expired")
        value = min(value, remaining)
    return value


def _process_result(
    ctx: ToolContext,
    name: str,
    argv: list[str],
    timeout: float,
    *,
    note: str = "",
) -> ToolResult:
    ctx.check_deadline()
    result = run_process(
        argv,
        cwd=ctx.workspace,
        timeout=timeout,
        max_output_chars=ctx.max_output_chars,
        cancel=ctx.cancelled,
        max_stdout_bytes=MAX_STDOUT_BYTES,
        max_stderr_bytes=MAX_STDERR_BYTES,
        max_total_bytes=MAX_TOTAL_BYTES,
    )
    output = _output_text(result, timeout)
    base_structured = {
        "exit_code": None if (result.cancelled or result.timed_out) else result.exit_code,
        "timed_out": result.timed_out,
        "cancelled": result.cancelled,
        "duration_ms": result.duration_ms,
        "stdout_bytes": result.stdout_bytes,
        "stderr_bytes": result.stderr_bytes,
        "output_truncated": result.output_truncated,
        "read_incomplete": result.read_incomplete,
        "raw_returncode": result.raw_returncode,
        "argv": [Path(argv[0]).name, *[str(item) for item in argv[1:4]]],
    }
    if result.cancelled:
        base_structured.update({"error_class": "CANCELLED", "suggestion": None})
        return ToolResult(
            call_id="",
            name=name,
            status=ToolStatus.CANCELLED,
            content=_join(note, output, "[cancelled by user; process tree terminated]"),
            structured=base_structured,
            error_code="CANCELLED",
            duration_ms=result.duration_ms,
        )
    if result.timed_out:
        base_structured.update({"error_class": "TIMEOUT", "suggestion": None})
        return ToolResult(
            call_id="",
            name=name,
            status=ToolStatus.TIMEOUT,
            content=_join(
                note,
                output,
                f"[timeout after {timeout:.0f}s; process tree terminated]",
            ),
            structured=base_structured,
            error_code="TIMEOUT",
            duration_ms=result.duration_ms,
        )

    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    error_class, suggestion = classify_failure(name, combined, result.exit_code)
    base_structured["error_class"] = error_class
    base_structured["suggestion"] = suggestion
    lines = [note] if note else []
    if output:
        lines.append(output)
    lines.append(f"[exit code: {result.exit_code}]")
    if error_class:
        lines.append(f"[failure: {error_class}]")
    if suggestion:
        lines.append(f"[suggestion: {suggestion}]")
    return ok(
        ctx,
        name,
        "\n".join(lines),
        structured=base_structured,
        exit_code=result.exit_code,
    )


def _output_text(result: ProcessResult, timeout: float) -> str:
    # Warnings go first so that a later character-limit truncation cannot hide
    # the fact that the output itself was truncated.
    parts: list[str] = []
    if result.stdout_truncated or result.stderr_truncated:
        parts.append(
            "[output truncated: kept "
            f"{len(result.stdout)} chars stdout / {len(result.stderr)} chars stderr; "
            f"the process wrote {result.stdout_bytes} bytes stdout and "
            f"{result.stderr_bytes} bytes stderr]"
        )
    if result.read_incomplete:
        parts.append("[output read incomplete: a pipe reader did not finish]")
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.stderr:
        parts.append("[stderr]\n" + result.stderr.rstrip())
    return "\n".join(parts)


def _join(*parts: str) -> str:
    return "\n".join(part for part in parts if part)


# ----------------------------------------------------------------- run_python


def run_python(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    try:
        script = ctx.guard.resolve(arguments["script"], must_exist=True)
    except PathViolation as exc:
        if exc.code == "PATH_NOT_FOUND":
            suggestion = (
                "Verify the script path with list_files or get_file_info; use a "
                "workspace-relative path."
            )
            return fail(
                "run_python",
                "SCRIPT_NOT_FOUND",
                f"{exc}. {suggestion}",
                structured={"suggestion": suggestion},
            )
        raise
    ctx.guard.ensure_read_allowed(script)
    check = ensure_file(script, tool="run_python")
    if check:
        return check
    if script.suffix.lower() != ".py":
        return fail("run_python", "NOT_A_SCRIPT", "script must have a .py extension")
    args = [str(item) for item in arguments.get("args") or []]
    if len(args) > 32:
        return fail("run_python", "TOO_MANY_ARGS", "at most 32 arguments are allowed")
    timeout = _effective_timeout(arguments.get("timeout_seconds"), ctx)
    argv = [default_interpreter(ctx), str(script), *args]
    return _process_result(
        ctx,
        "run_python",
        argv,
        timeout,
        note=f"python {script.relative_to(ctx.guard.root).as_posix()}",
    )


# -------------------------------------------------------------- run_powershell


def run_powershell(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    try:
        script = ctx.guard.resolve(arguments["script"], must_exist=True)
    except PathViolation as exc:
        if exc.code == "PATH_NOT_FOUND":
            suggestion = (
                "Use a workspace-relative script path such as .\\name.ps1 and verify "
                "it exists with list_files; do not pass a bare command name."
            )
            return fail(
                "run_powershell",
                "SCRIPT_NOT_FOUND",
                f"{exc}. {suggestion}",
                structured={"suggestion": suggestion},
            )
        raise
    ctx.guard.ensure_read_allowed(script)
    check = ensure_file(script, tool="run_powershell")
    if check:
        return check
    if script.suffix.lower() != ".ps1":
        return fail("run_powershell", "NOT_A_SCRIPT", "script must have a .ps1 extension")
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        return fail("run_powershell", "SHELL_NOT_FOUND", "PowerShell executable not found")
    args = [str(item) for item in arguments.get("args") or []]
    if len(args) > 32:
        return fail("run_powershell", "TOO_MANY_ARGS", "at most 32 arguments are allowed")
    timeout = _effective_timeout(arguments.get("timeout_seconds"), ctx)
    argv = [shell, "-NoProfile", "-NonInteractive", "-File", str(script), *args]
    return _process_result(
        ctx,
        "run_powershell",
        argv,
        timeout,
        note=f"pwsh -File {script.relative_to(ctx.guard.root).as_posix()}",
    )


# ------------------------------------------------------------------ run_tests


def run_tests(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    target = ctx.guard.resolve(arguments.get("path") or ".", must_exist=True)
    ctx.guard.ensure_read_allowed(target)
    framework = arguments.get("framework") or "unittest"
    extra = [str(item) for item in arguments.get("args") or []]
    if len(extra) > 16:
        return fail("run_tests", "TOO_MANY_ARGS", "at most 16 arguments are allowed")
    timeout = _effective_timeout(arguments.get("timeout_seconds") or 180, ctx)
    interpreter = default_interpreter(ctx)

    relative = target.relative_to(ctx.guard.root).as_posix()
    if framework == "unittest":
        if target.is_dir():
            argv = [
                interpreter,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(target),
                "-p",
                "test*.py",
                "-v",
                *extra,
            ]
        else:
            # Never execute the test file as a plain script: a missing
            # ``if __name__ == '__main__'`` guard would define the test class
            # and exit 0 without running a single test (H5.2 D3).
            module = ".".join(target.relative_to(ctx.workspace).with_suffix("").parts)
            argv = [interpreter, "-m", "unittest", "-v", module, *extra]
        note = f"unittest {relative}"
    elif framework == "pytest":
        argv = [interpreter, "-m", "pytest", str(target), "-q", *extra]
        note = f"pytest {relative}"
    else:
        return fail("run_tests", "INVALID_ARGUMENTS", "framework must be unittest or pytest")

    result = _process_result(ctx, "run_tests", argv, timeout, note=note)
    if result.status is ToolStatus.OK:
        evidence = build_test_evidence(
            ctx.workspace,
            target=relative,
            framework=framework,
            output=result.content,
            exit_code=result.exit_code,
        )
        result.structured["verification"] = evidence
        result.structured["framework"] = framework
        result.structured["path"] = relative
        result.structured["tests_run"] = evidence["tests_executed"]
        result.structured["passed"] = evidence["verdict"] == VERDICT_PASS
        if evidence["verdict"] != VERDICT_PASS:
            result.status = ToolStatus.ERROR
            result.error_code = evidence["verdict"]
            result.content = f"{evidence['verdict']}\n{result.content}"
        else:
            result.content = f"TESTS_PASSED\n{result.content}"
    return result


# ---------------------------------------------------------------- definitions


def exec_tool_definitions() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="run_python",
            description=(
                "Run an existing .py script inside the workspace with the harness "
                "interpreter. Returns stdout, stderr and the real exit code."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "script": {"type": "string", "minLength": 1, "maxLength": 400},
                    "args": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 500},
                        "maxItems": 32,
                    },
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
                },
                "required": ["script"],
                "additionalProperties": False,
            },
            execute=run_python,
            permission_level=PermissionLevel.EXECUTE_APPROVED,
            idempotent=False,
            timeout_seconds=MAX_TIMEOUT_SECONDS,
        ),
        ToolDefinition(
            name="run_powershell",
            description=(
                "Run an existing .ps1 script inside the workspace. Inline commands "
                "are not accepted."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "script": {"type": "string", "minLength": 1, "maxLength": 400},
                    "args": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 500},
                        "maxItems": 32,
                    },
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
                },
                "required": ["script"],
                "additionalProperties": False,
            },
            execute=run_powershell,
            permission_level=PermissionLevel.EXECUTE_APPROVED,
            idempotent=False,
            timeout_seconds=MAX_TIMEOUT_SECONDS,
        ),
        ToolDefinition(
            name="run_tests",
            description=(
                "Run unittest or pytest for a workspace file or directory. "
                "Reports the real exit code and PASS/FAIL."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "maxLength": 400},
                    "framework": {"type": "string", "enum": ["unittest", "pytest"]},
                    "args": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 500},
                        "maxItems": 16,
                    },
                    "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 300},
                },
                "additionalProperties": False,
            },
            execute=run_tests,
            permission_level=PermissionLevel.EXECUTE_APPROVED,
            idempotent=False,
            repeatable=True,
            timeout_seconds=MAX_TIMEOUT_SECONDS,
        ),
    ]