"""Shared fixtures for H2 agent tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from yuzhu_lab.harness.agent.audit import AgentAuditLog
from yuzhu_lab.harness.agent.checkpoint import CheckpointStore
from yuzhu_lab.harness.agent.loop import AgentLoop, AgentLoopConfig
from yuzhu_lab.harness.agent.model_client import ModelResponse, ModelToolCall
from yuzhu_lab.harness.agent.permissions import PermissionPolicy
from yuzhu_lab.harness.agent.registry import ToolContext, ToolRegistry
from yuzhu_lab.harness.agent.tools import build_default_registry, default_tool_context


class TempWorkspace:
    def __init__(self) -> None:
        self._temp = tempfile.TemporaryDirectory(prefix="yuzhu-h2-")
        self.path = Path(self._temp.name)

    def close(self) -> None:
        self._temp.cleanup()

    def __enter__(self) -> "TempWorkspace":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def make_policy(
    workspace: Path,
    *,
    allow_execution: bool = False,
    allow_destructive: bool = False,
    approval=None,
) -> PermissionPolicy:
    return PermissionPolicy.create(
        workspace,
        allow_execution=allow_execution,
        allow_destructive=allow_destructive,
        approval_callback=approval,
    )


def make_registry(
    workspace: Path,
    *,
    allow_execution: bool = False,
    allow_destructive: bool = False,
    audit=None,
) -> tuple[ToolRegistry, ToolContext]:
    policy = make_policy(
        workspace,
        allow_execution=allow_execution,
        allow_destructive=allow_destructive,
    )
    context = default_tool_context(policy)
    registry = build_default_registry(policy, context=context, audit=audit)
    return registry, context


class ScriptedModel:
    """Deterministic model double: returns canned responses in order."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, system, messages, tools, max_tokens, temperature=0.0):
        self.calls.append(
            {
                "system": system,
                "messages": [dict(message) for message in messages],
                "tools": [tool["name"] for tool in tools or []],
                "max_tokens": max_tokens,
            }
        )
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of responses")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def text_response(text: str, *, stop_reason: str = "end_turn") -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=[],
        request_id="req-text",
        stop_reason=stop_reason,
        usage={"input_tokens": 10, "output_tokens": 5},
    )


def tool_response(
    calls: list[tuple[str, dict]],
    *,
    text: str = "",
    request_id: str = "req-tool",
) -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=[ModelToolCall(name=name, arguments=args) for name, args in calls],
        request_id=request_id,
        stop_reason="tool_use",
        usage={"input_tokens": 10, "output_tokens": 5},
    )


def make_loop(
    workspace: Path,
    model,
    *,
    task_id: str = "task-test",
    allow_execution: bool = False,
    allow_destructive: bool = False,
    context_builder=None,
    memory_writer=None,
    cancel_token=None,
    **config_overrides,
) -> tuple[AgentLoop, CheckpointStore, AgentAuditLog]:
    state_dir = workspace / ".agent"
    state_dir.mkdir(parents=True, exist_ok=True)
    store = CheckpointStore(state_dir / "state.db")
    audit = AgentAuditLog(state_dir / "audit.jsonl")
    policy = make_policy(
        workspace,
        allow_execution=allow_execution,
        allow_destructive=allow_destructive,
    )
    context = default_tool_context(policy)
    registry = build_default_registry(policy, context=context, audit=audit)
    config = AgentLoopConfig(
        project_id="qianyuqwen",
        session_id="session-test",
        workspace=workspace,
        checkpoint_db=state_dir / "state.db",
        audit_log=state_dir / "audit.jsonl",
        task_id=task_id,
        allow_execution=allow_execution,
        allow_destructive=allow_destructive,
        **config_overrides,
    )
    loop = AgentLoop(
        config=config,
        model=model,
        registry=registry,
        checkpoints=store,
        audit=audit,
        context_builder=context_builder,
        memory_writer=memory_writer,
        cancel_token=cancel_token,
    )
    return loop, store, audit