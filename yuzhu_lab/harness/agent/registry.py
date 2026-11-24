"""Tool registry: schema validation, permission policy, execution and audit.

Every model-originated call goes through this object.  A tool cannot be
executed because a model asked for it: the name must be registered, the
arguments must validate, and the permission policy must allow it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from ..context_policy import sanitize_untrusted
from .errors import PathViolation, ToolExecutionError, ToolValidationError
from .permissions import PermissionLevel, PermissionPolicy
from .schema import validate


class ToolOrigin(str, Enum):
    """How a tool call came into existence.

    MODEL_SELECTED_TOOL: parsed from the model's native tool-use blocks.
    HARNESS_SELECTED_TOOL: injected by harness policy/test code, not the model.
    DETERMINISTIC_FALLBACK: recovered from a strict structured action block
    after native tool syntax produced nothing.
    """

    MODEL_SELECTED_TOOL = "MODEL_SELECTED_TOOL"
    HARNESS_SELECTED_TOOL = "HARNESS_SELECTED_TOOL"
    DETERMINISTIC_FALLBACK = "DETERMINISTIC_FALLBACK"


class ToolStatus(str, Enum):
    OK = "OK"
    ERROR = "ERROR"
    DENIED = "DENIED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    DUPLICATE = "DUPLICATE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ToolRequest:
    call_id: str
    name: str
    arguments: dict[str, Any]
    origin: ToolOrigin = ToolOrigin.MODEL_SELECTED_TOOL
    turn: int = 0


@dataclass
class ToolResult:
    call_id: str
    name: str
    status: ToolStatus
    content: str
    structured: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    permission: str = PermissionLevel.READ.value
    origin: str = ToolOrigin.MODEL_SELECTED_TOOL.value
    duration_ms: float = 0.0
    truncated: bool = False
    exit_code: int | None = None

    @property
    def ok(self) -> bool:
        return self.status in {ToolStatus.OK, ToolStatus.DUPLICATE}

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "status": self.status.value,
            "content": self.content,
            "structured": self.structured,
            "error_code": self.error_code,
            "permission": self.permission,
            "origin": self.origin,
            "duration_ms": self.duration_ms,
            "truncated": self.truncated,
            "exit_code": self.exit_code,
        }


@dataclass
class ToolContext:
    workspace: Path
    guard: Any
    interpreter: Path | str | None = None
    max_output_chars: int = 20000
    max_read_bytes: int = 262144
    audit: Callable[[dict[str, Any]], None] | None = None
    deadline: float | None = None
    cancelled: Callable[[], bool] | None = None

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise ToolExecutionError("TIMEOUT", "tool exceeded its execution deadline")
        if self.cancelled is not None and self.cancelled():
            raise ToolExecutionError("CANCELLED", "tool cancelled by the user")

    def with_deadline(
        self, deadline: float | None, cancelled: Callable[[], bool] | None = None
    ) -> "ToolContext":
        return replace(self, deadline=deadline, cancelled=cancelled or self.cancelled)


Executor = Callable[[ToolContext, dict[str, Any]], ToolResult]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    execute: Executor
    output_schema: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    permission_level: PermissionLevel = PermissionLevel.READ
    idempotent: bool = True
    repeatable: bool = False
    permission_for: Callable[[dict[str, Any]], PermissionLevel] | None = None

    def level_for(self, arguments: dict[str, Any]) -> PermissionLevel:
        if self.permission_for is not None:
            return self.permission_for(arguments)
        return self.permission_level

    def anthropic_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(
        self,
        policy: PermissionPolicy,
        *,
        context: ToolContext | None = None,
        audit: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.policy = policy
        self.context = context or ToolContext(workspace=policy.workspace, guard=policy.guard)
        self.audit_sink = audit
        self.cancel_check: Callable[[], bool] | None = None
        self._tools: dict[str, ToolDefinition] = {}

    # ------------------------------------------------------------- registration

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"tool already registered: {definition.name}")
        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def definitions(self) -> list[ToolDefinition]:
        return [self._tools[name] for name in self.names()]

    def anthropic_tools(self) -> list[dict[str, Any]]:
        return [definition.anthropic_tool() for definition in self.definitions()]

    # ---------------------------------------------------------------- execution

    def execute(self, request: ToolRequest) -> ToolResult:
        started = time.perf_counter()
        definition = self._tools.get(request.name)
        base = ToolResult(
            call_id=request.call_id,
            name=request.name,
            status=ToolStatus.ERROR,
            content="",
            origin=request.origin.value,
        )
        if definition is None:
            base.error_code = "UNKNOWN_TOOL"
            base.content = (
                f"Unknown tool '{request.name}'. Available tools: "
                + ", ".join(self.names())
            )
            return self._finish(request, base, started, arguments=request.arguments)

        errors = validate(request.arguments, definition.input_schema)
        if errors:
            base.error_code = "INVALID_ARGUMENTS"
            base.content = "Tool arguments failed validation: " + "; ".join(errors)
            base.structured = {"validation_errors": errors}
            return self._finish(request, base, started, arguments=request.arguments)

        level = definition.level_for(request.arguments)
        decision = self.policy.check(
            name=definition.name, arguments=request.arguments, level=level
        )
        base.permission = level.value
        if not decision.allowed:
            base.status = ToolStatus.DENIED
            base.error_code = "PERMISSION_DENIED"
            base.content = f"Permission denied ({level.value}): {decision.reason}"
            return self._finish(request, base, started, arguments=request.arguments)

        context = self.context.with_deadline(
            time.monotonic() + definition.timeout_seconds,
            cancelled=self.cancel_check,
        )
        try:
            result = definition.execute(context, request.arguments)
        except PathViolation as exc:
            result = _error_result(request, exc.code, str(exc), level)
        except ToolValidationError as exc:
            result = _error_result(request, "INVALID_ARGUMENTS", str(exc), level)
        except ToolExecutionError as exc:
            result = _error_result(request, exc.code, str(exc), level)
            if exc.code == "CANCELLED":
                result.status = ToolStatus.CANCELLED
        except TimeoutError:
            result = _error_result(request, "TIMEOUT", "tool exceeded its time budget", level)
            result.status = ToolStatus.TIMEOUT
        except Exception as exc:  # noqa: BLE001 - a tool crash must not kill the loop
            result = _error_result(
                request, "TOOL_CRASH", f"{type(exc).__name__}: {exc}", level
            )

        result.permission = level.value
        result.origin = request.origin.value
        return self._finish(request, result, started, arguments=request.arguments)

    # ------------------------------------------------------------------ helpers

    def _finish(
        self,
        request: ToolRequest,
        result: ToolResult,
        started: float,
        *,
        arguments: dict[str, Any],
    ) -> ToolResult:
        result.duration_ms = round((time.perf_counter() - started) * 1000, 2)
        result.call_id = request.call_id
        result.name = request.name
        if self.audit_sink is not None:
            self.audit_sink(
                {
                    "kind": "tool_call",
                    "call_id": request.call_id,
                    "turn": request.turn,
                    "tool": request.name,
                    "origin": request.origin.value,
                    "permission": result.permission,
                    "status": result.status.value,
                    "error_code": result.error_code,
                    "exit_code": result.exit_code,
                    "duration_ms": result.duration_ms,
                    "arguments_digest": _digest(arguments),
                    "arguments_summary": _summarize_arguments(arguments),
                    "result_chars": len(result.content or ""),
                    "truncated": result.truncated,
                }
            )
        return result


def _digest(arguments: dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_CONTENT_KEYS = {"content", "new_text", "old_text", "patch"}


def _summarize_arguments(arguments: dict[str, Any]) -> dict[str, str]:
    summary: dict[str, str] = {}
    for key, value in arguments.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        text = text.replace("\n", " ")
        if str(key) in _CONTENT_KEYS:
            summary[str(key)] = f"<redacted {len(text)} chars sha256:{_digest_text(text)[:16]}>"
        else:
            summary[str(key)] = text[:120]
    return summary


def _digest_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _error_result(
    request: ToolRequest, code: str, message: str, level: PermissionLevel
) -> ToolResult:
    sanitized, _ = sanitize_untrusted(message)
    return ToolResult(
        call_id=request.call_id,
        name=request.name,
        status=ToolStatus.ERROR,
        content=sanitized,
        error_code=code,
        permission=level.value,
        origin=request.origin.value,
    )