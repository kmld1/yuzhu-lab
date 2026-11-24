"""Read-only git inspection tools."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..registry import ToolContext, ToolDefinition, ToolResult
from .common import fail, ok, run_process

MAX_DIFF_CHARS = 60000


def _git(ctx: ToolContext, name: str, args: list[str], timeout: float = 30.0) -> ToolResult:
    executable = shutil.which("git")
    if executable is None:
        return fail(name, "GIT_NOT_FOUND", "git executable is not available on PATH")
    result = run_process(
        [executable, "-C", str(ctx.workspace), *args],
        cwd=ctx.workspace,
        timeout=timeout,
        max_output_chars=ctx.max_output_chars,
    )
    if result.timed_out:
        return fail(name, "TIMEOUT", "git command timed out")
    if result.exit_code != 0:
        message = (result.stderr or result.stdout or "").strip()
        code = "GIT_NOT_REPOSITORY" if "not a git repository" in message.lower() else "GIT_ERROR"
        return fail(name, code, message or f"git exited with {result.exit_code}")
    return ok(
        ctx,
        name,
        result.stdout.rstrip() or "(no output)",
        structured={"exit_code": 0},
        exit_code=0,
    )


def git_status(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    return _git(ctx, "git_status", ["status", "--porcelain=v1", "-b"])


def git_diff(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    staged = bool(arguments.get("staged", False))
    args = ["diff", "--staged"] if staged else ["diff"]
    result = _git(ctx, "git_diff", args)
    if result.status.value == "OK" and len(result.content) > MAX_DIFF_CHARS:
        result.content = result.content[:MAX_DIFF_CHARS] + "\n[...diff truncated]"
        result.truncated = True
    return result


def git_log(ctx: ToolContext, arguments: dict[str, Any]) -> ToolResult:
    limit = max(1, min(int(arguments.get("limit") or 10), 50))
    return _git(ctx, "git_log", ["log", f"-n{limit}", "--oneline"])


def git_tool_definitions() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="git_status",
            description="Show porcelain git status for the workspace (read-only).",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            execute=git_status,
            timeout_seconds=30,
        ),
        ToolDefinition(
            name="git_diff",
            description="Show the git diff of the workspace (read-only).",
            input_schema={
                "type": "object",
                "properties": {"staged": {"type": "boolean"}},
                "additionalProperties": False,
            },
            execute=git_diff,
            timeout_seconds=30,
        ),
        ToolDefinition(
            name="git_log",
            description="Show recent git commits (read-only).",
            input_schema={
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}},
                "additionalProperties": False,
            },
            execute=git_log,
            timeout_seconds=30,
        ),
    ]