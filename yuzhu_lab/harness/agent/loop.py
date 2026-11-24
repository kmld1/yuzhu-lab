"""Single-agent execution loop with checkpoints, limits and recovery."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..context_budget import estimate_tokens
from ..context_builder import ContextBuilder
from .actions import ActionParseError, extract_action
from .audit import AgentAuditLog
from .checkpoint import CheckpointStore
from .completion import CompletionAssessment, assess_completion
from .context import (
    AgentContextConfig,
    context_metadata,
    context_warnings,
    make_context_request,
    render_system_prompt,
)
from .errors import CheckpointError, ModelUnavailable
from .memory_writer import AgentMemoryWriter
from .model_client import ModelPort, ModelResponse, ModelToolCall, response_summary
from .registry import (
    ToolOrigin,
    ToolRegistry,
    ToolRequest,
    ToolResult,
    ToolStatus,
)
from .states import AgentState, TERMINAL_STATES, ensure_transition
from .tools.common import require_process_cleanup
from .verification import run_acceptance_oracle


_PROMPT_SAFETY_MARGIN_TOKENS = 64
# Room kept for the fixed agent rules, tool schemas, the current task and the
# newest tool result while the builder sizes the optional RAG/memory block.
_BUILDER_RESERVE_TOKENS = 2048
_OLD_TOOL_RESULT_PLACEHOLDER = (
    "[older tool result omitted to fit the context budget]"
)
_OLD_ASSISTANT_TEXT_PLACEHOLDER = (
    "[older assistant text omitted to fit the context budget]"
)
_LATEST_TOOL_RESULT_MARKER = "\n[...tool result truncated to fit context budget]"


class CancelledError(RuntimeError):
    """Raised internally when the cancellation token is set."""


class ContextBudgetExceeded(RuntimeError):
    """Raised before sending a prompt that the serving guard would reject."""

    def __init__(
        self, *, prompt_tokens: int, budget: int, detail: str = ""
    ) -> None:
        message = (
            f"CONTEXT_BUDGET_EXCEEDED: prompt is ~{prompt_tokens} tokens against "
            f"a {budget}-token input budget"
        )
        if detail:
            message = f"{message}; {detail}"
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.budget = budget


class IncompleteCompletion(RuntimeError):
    """Raised when the completion gate refuses to record TASK_COMPLETED."""

    def __init__(self, assessment: CompletionAssessment, final_text: str) -> None:
        super().__init__(assessment.summary)
        self.assessment = assessment
        self.final_text = final_text


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason: str | None = None

    def cancel(self, reason: str = "user") -> None:
        if self.reason is None:
            self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def __call__(self) -> bool:
        return self._event.is_set()


@dataclass
class AgentLoopConfig:
    project_id: str
    session_id: str
    workspace: Path
    checkpoint_db: Path
    audit_log: Path
    task_id: str = field(default_factory=lambda: f"task-{uuid.uuid4().hex[:12]}")
    max_turns: int = 12
    max_tool_calls: int = 24
    max_tools_per_turn: int = 4
    task_timeout_seconds: float = 900.0
    max_tokens: int = 1024
    temperature: float = 0.0
    model_context_budget: int = 8192
    observation_char_limit: int = 4000
    context_char_limit: int = 60000
    allow_execution: bool = False
    allow_destructive: bool = False
    retain_memory: bool = True
    include_action_fallback: bool = True
    max_model_retries: int = 2
    max_repeat_failures: int = 1
    max_repeat_successes: int = 2
    max_consecutive_failures: int = 6
    require_test_evidence_for_code_change: bool = False
    acceptance_script: Path | None = None
    acceptance_sha256: str | None = None
    acceptance_expected_checks: int | None = None
    acceptance_timeout_seconds: float = 300.0
    require_external_acceptance: bool = False


@dataclass
class AgentRunResult:
    task_id: str
    session_id: str
    status: str
    state: str
    final_text: str
    turns: int
    tool_calls: int
    warnings: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    retained_memory: dict[str, Any] | None = None
    duration_seconds: float = 0.0
    events: int = 0
    completion: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "status": self.status,
            "state": self.state,
            "final_text": self.final_text,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "warnings": list(self.warnings),
            "error": self.error,
            "retained_memory": self.retained_memory,
            "duration_seconds": self.duration_seconds,
            "events": self.events,
            "completion": self.completion,
        }


class AgentLoop:
    def __init__(
        self,
        *,
        config: AgentLoopConfig,
        model: ModelPort,
        registry: ToolRegistry,
        checkpoints: CheckpointStore,
        audit: AgentAuditLog,
        context_builder: ContextBuilder | None = None,
        memory_writer: AgentMemoryWriter | None = None,
        cancel_token: CancelToken | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.registry = registry
        self.checkpoints = checkpoints
        self.audit = audit
        self.context_builder = context_builder
        self.memory_writer = memory_writer
        self.cancel = cancel_token or CancelToken()
        self.registry.cancel_check = self.cancel

        self.state = AgentState.IDLE
        self.task_id = config.task_id
        self.turn = 0
        self.messages: list[dict[str, Any]] = []
        self.system_prompt = ""
        self.warnings: list[dict[str, Any]] = []
        self.tool_call_count = 0
        self.retry_count = 0
        self.deadline = 0.0
        self._last_response: ModelResponse | None = None
        self._empty_answers = 0
        self._context_info: dict[str, Any] = {}
        self._retained_memory: dict[str, Any] | None = None
        self._task_text = ""
        self._consecutive_failures = 0
        self._unparsed_tool_answers = 0
        self._truncated_retry_done = False
        self._completion: CompletionAssessment | None = None
        self.last_context_bundle: Any | None = None
        self._last_context_request: Any | None = None
        self._input_limits: Any | None = None
        self._input_limits_loaded = False
        self._tool_schemas_cache: list[dict[str, Any]] | None = None
        self._prompt_budget_info: dict[str, Any] = {}
        self._count_fallback_warned = False

    # ------------------------------------------------------------------ public

    def run(self, task: str, *, resume: bool = False) -> AgentRunResult:
        started = time.perf_counter()
        self.deadline = started + self.config.task_timeout_seconds
        status = "FAILED"
        final_text = ""
        error: str | None = None

        try:
            if resume:
                self._load_existing_task()
            else:
                self._accept_task(task)

            if self.state in TERMINAL_STATES:
                return self._result(
                    started, final_text, None, status=self.state.value
                )

            require_process_cleanup()
            if not self.system_prompt:
                self._build_context(self._task_text or task)
            self._recover_interrupted_calls()

            final_text = self._conversation_loop()
            status = "COMPLETED"
        except CancelledError:
            status = "CANCELLED"
            error = self.cancel.reason or "cancelled"
            self._cancel_task(error)
        except IncompleteCompletion as exc:
            status = "FAILED"
            error = f"completion gate: {exc}"
            final_text = exc.final_text
            self._fail_task(error)
        except ContextBudgetExceeded as exc:
            status = "FAILED"
            error = str(exc)
            self.warnings.append(
                {"code": "CONTEXT_BUDGET_EXCEEDED", "message": str(exc)}
            )
            self._fail_task(error)
        except Exception as exc:  # noqa: BLE001 - always persist a terminal state
            status = "FAILED"
            error = f"{type(exc).__name__}: {exc}"
            self._fail_task(error)
        return self._result(started, final_text, error, status=status)

    # ------------------------------------------------------------- task setup

    def _accept_task(self, task: str) -> None:
        if not task.strip():
            raise ValueError("task must not be empty")
        self._task_text = task
        if self.checkpoints.task_exists(self.task_id):
            raise CheckpointError(
                f"task {self.task_id} already exists; use resume instead of run"
            )
        self.checkpoints.create_task(
            task_id=self.task_id,
            session_id=self.config.session_id,
            project_id=self.config.project_id,
            workspace=self.config.workspace,
            task_description=task,
        )
        self._set_state(AgentState.ACCEPTING_TASK, 0)
        self.messages = [{"role": "user", "content": task}]
        self._check_cancel()

    def _load_existing_task(self) -> None:
        record = self.checkpoints.get_task(self.task_id)
        self._task_text = str(record["task_description"])
        stored_state = AgentState(record["state"])
        status = record["status"]
        self._restore_state(stored_state)
        self.turn = int(record["turn"] or 0)
        self.retry_count = int(record["retry_count"] or 0)
        self.messages = list(record["messages"] or [])
        context = record["context"] or {}
        self.system_prompt = str(context.get("system_prompt") or "")
        self._context_info = dict(context.get("context_metadata") or {})
        self.warnings = list(context.get("warnings") or [])
        self._event(
            "RESUME",
            {"status": status, "stored_state": stored_state.value, "turn": self.turn},
        )
        if status in {"COMPLETED", "FAILED", "CANCELLED"} and not self.messages:
            self.state = AgentState(status)

    def _build_context(self, task: str) -> None:
        if self.state not in {AgentState.IDLE, AgentState.ACCEPTING_TASK}:
            self._restore_state(AgentState.ACCEPTING_TASK)
        self._set_state(AgentState.BUILDING_CONTEXT, 0)
        bundle = None
        if self.context_builder is not None:
            config = AgentContextConfig(
                project_id=self.config.project_id,
                session_id=self.config.session_id,
                model_context_budget=self._builder_budget(),
            )
            summary = self._history_summary()
            try:
                request = make_context_request(
                    config,
                    task=task,
                    workspace=self.config.workspace,
                    conversation_summary=summary,
                )
                self._last_context_request = request
                bundle = self.context_builder.build(request)
                self.last_context_bundle = bundle
                self.warnings.extend(context_warnings(bundle))
                self._context_info = context_metadata(bundle)
            except Exception as exc:  # noqa: BLE001 - context is optional
                self.warnings.append(
                    {"code": "CONTEXT_BUILT_FAILED", "message": f"{type(exc).__name__}: {exc}"}
                )
        self.system_prompt = self._render_prompt(bundle)
        context_refs: list[dict[str, Any]] = []
        memory_refs: list[str] = []
        rag_refs: list[str] = []
        if bundle is not None:
            context_refs = [dict(ref.__dict__) for ref in bundle.source_references]
            memory_refs = [item.memory_id for item in bundle.recalled_memories]
            rag_refs = [item.citation for item in bundle.rag_evidence]
        self.checkpoints.save_task(
            self.task_id,
            state=self.state.value,
            turn=0,
            messages=self.messages,
            context={
                "system_prompt": self.system_prompt,
                "context_metadata": self._context_info,
                "warnings": self.warnings,
                "tool_names": self.registry.names(),
            },
            context_refs=context_refs,
            memory_refs=memory_refs,
            rag_refs=rag_refs,
        )

    def _history_summary(self) -> str:
        calls = self.checkpoints.tool_calls(self.task_id) if self.checkpoints.task_exists(self.task_id) else []
        if not calls:
            return ""
        lines = []
        for call in calls[-12:]:
            result = call.get("result") or {}
            lines.append(
                f"{call['turn']}: {call['name']} -> {call.get('status')}"
                f" {result.get('error_code') or ''}".strip()
            )
        return "已完成工具调用摘要:\n" + "\n".join(lines)

    # ------------------------------------------------------------ recover/resume

    def _recover_interrupted_calls(self) -> None:
        if not self.messages:
            return
        last = self.messages[-1]
        if last.get("role") != "assistant" or not isinstance(last.get("content"), list):
            return
        tool_blocks = [
            block
            for block in last["content"]
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        if not tool_blocks:
            return
        self._restore_state(AgentState.OBSERVING_RESULT)
        self._event("RECOVERY", {"unresolved_tool_blocks": len(tool_blocks)})
        results = self._recover_results(tool_blocks)
        self._append_tool_results(results)
        self._push_messages()

    def _recover_results(self, tool_blocks: list[dict[str, Any]]) -> list[ToolResult]:
        stored = {call["call_id"]: call for call in self.checkpoints.tool_calls(self.task_id)}
        results: list[ToolResult] = []
        for block in tool_blocks:
            call_id = str(block.get("id") or "")
            name = str(block.get("name") or "")
            arguments = block.get("input") or {}
            record = stored.get(call_id)
            if record is None:
                results.append(
                    ToolResult(
                        call_id=call_id,
                        name=name,
                        status=ToolStatus.UNKNOWN,
                        content="No checkpoint exists for this tool call; it was not executed.",
                        error_code="CHECKPOINT_MISSING",
                    )
                )
                continue
            status = record["status"]
            if status not in {"STARTED", "PENDING"} and record["result"]:
                results.append(_result_from_dict(record["result"]))
                continue
            if status == "STARTED":
                if record["idempotent"]:
                    self._event(
                        "RECOVERY_RERUN",
                        {"call_id": call_id, "tool": name, "reason": "idempotent read-only tool"},
                    )
                    request = ToolRequest(
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                        origin=ToolOrigin(record["origin"]),
                        turn=int(record["turn"]),
                    )
                    result = self._execute_with_checkpoint(request, idempotent=True)
                    results.append(result)
                else:
                    message = (
                        f"Tool '{name}' was interrupted after it started and before its "
                        "result was persisted. It has side effects and was NOT re-run. "
                        "Verify the workspace state (read_file) before retrying."
                    )
                    self.checkpoints.finish_tool_call(
                        call_id,
                        "UNKNOWN",
                        {
                            "call_id": call_id,
                            "name": name,
                            "status": "UNKNOWN",
                            "content": message,
                            "error_code": "INTERRUPTED_SIDE_EFFECT",
                        },
                    )
                    results.append(
                        ToolResult(
                            call_id=call_id,
                            name=name,
                            status=ToolStatus.UNKNOWN,
                            content=message,
                            error_code="INTERRUPTED_SIDE_EFFECT",
                        )
                    )
                continue
            # PENDING: it never started, so a normal execution is safe.
            request = ToolRequest(
                call_id=call_id,
                name=name,
                arguments=arguments,
                origin=ToolOrigin(record["origin"]),
                turn=int(record["turn"]),
            )
            results.append(self._execute_with_checkpoint(request, idempotent=record["idempotent"]))
        return results

    # --------------------------------------------------------------- main loop

    def _conversation_loop(self) -> str:
        if self.state not in {
            AgentState.BUILDING_CONTEXT,
            AgentState.REQUESTING_MODEL,
            AgentState.OBSERVING_RESULT,
        }:
            self._restore_state(AgentState.OBSERVING_RESULT)
        while True:
            self._check_limits()
            self._set_state(AgentState.REQUESTING_MODEL, self.turn)
            response = self._request_model()
            # H5.2: honour a cancellation that arrived while the model call was
            # in flight; otherwise a complete-looking answer after cancellation
            # would be recorded as COMPLETED.
            self._check_cancel()
            self._last_response = response
            self._set_state(AgentState.PARSING_RESPONSE, self.turn)
            text, calls, origin = self._parse_response(response)

            if not calls:
                final_text = text.strip()
                looks_like_tool_syntax = (
                    "<function" in final_text.lower()
                    or "<param" in final_text.lower()
                    or final_text.startswith("```yuzhu-action")
                )
                if looks_like_tool_syntax:
                    self._unparsed_tool_answers += 1
                    self._event(
                        "UNPARSED_TOOL_SYNTAX",
                        {
                            "text_chars": len(final_text),
                            "truncated": response.truncated,
                            "attempt": self._unparsed_tool_answers,
                        },
                    )
                    if self._unparsed_tool_answers <= 1:
                        self.messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Your previous reply contained an incomplete or unparsed "
                                    "tool call. It was NOT executed. Re-issue a complete tool "
                                    "call with a smaller payload (for example, write the file "
                                    "in two steps), or answer without tool syntax."
                                ),
                            }
                        )
                        self._push_messages()
                        self._advance_turn()
                        continue
                    raise RuntimeError(
                        "model repeatedly emitted incomplete or unparsed tool syntax"
                    )
                if response.truncated and not self._truncated_retry_done:
                    self._truncated_retry_done = True
                    self._event(
                        "TRUNCATED_ANSWER_RETRY",
                        {"text_chars": len(final_text)},
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous reply was truncated at max_tokens. Continue "
                                "from where it stopped, or give a shorter complete final answer."
                            ),
                        }
                    )
                    self._push_messages()
                    self._advance_turn()
                    continue
                if not final_text:
                    self._empty_answers += 1
                    self._event("EMPTY_ANSWER", {"turn": self.turn})
                    if self._empty_answers <= 1:
                        self.messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Your previous reply was empty. Either call a tool with "
                                    "the native function syntax, or answer the task in text."
                                ),
                            }
                        )
                        self._push_messages()
                        self._advance_turn()
                        continue
                    raise RuntimeError("model produced two consecutive empty replies")
                acceptance = None
                acceptance_configured = (
                    self.config.acceptance_script is not None
                    or self.config.require_external_acceptance
                )
                if self.config.acceptance_script is not None:
                    acceptance = run_acceptance_oracle(
                        self.config.workspace,
                        script=self.config.acceptance_script,
                        expected_sha256=self.config.acceptance_sha256,
                        timeout_seconds=self.config.acceptance_timeout_seconds,
                        expected_checks=self.config.acceptance_expected_checks,
                    )
                    self._event(
                        "ACCEPTANCE_ORACLE",
                        {
                            "passed": acceptance.get("passed"),
                            "sha256": acceptance.get("sha256"),
                            "exit_code": acceptance.get("exit_code"),
                            "duration_ms": acceptance.get("duration_ms"),
                            "reasons": list(acceptance.get("reasons") or []),
                        },
                    )
                assessment = assess_completion(
                    response,
                    self.checkpoints.tool_calls(self.task_id),
                    final_text=final_text,
                    require_verification_for_code_change=self.config.require_test_evidence_for_code_change,
                    workspace=self.config.workspace,
                    acceptance=acceptance,
                    acceptance_configured=acceptance_configured,
                )
                self._completion = assessment
                self._event("COMPLETION_ASSESSMENT", assessment.to_dict())
                if not assessment.task_completed:
                    raise IncompleteCompletion(assessment, final_text)
                self._finish_task(final_text, response)
                return final_text

            self._set_state(AgentState.VALIDATING_TOOL_CALL, self.turn)
            requests = self._build_requests(calls, origin)
            self.messages.append(
                {
                    "role": "assistant",
                    "content": [
                        *(
                            [{"type": "text", "text": text}]
                            if text
                            else []
                        ),
                        *[
                            {
                                "type": "tool_use",
                                "id": request.call_id,
                                "name": request.name,
                                "input": request.arguments,
                            }
                            for request in requests
                        ],
                    ],
                }
            )
            self._push_messages()

            results: list[ToolResult] = []
            for index, request in enumerate(requests):
                self._check_cancel()
                if index >= self.config.max_tools_per_turn:
                    result = ToolResult(
                        call_id=request.call_id,
                        name=request.name,
                        status=ToolStatus.ERROR,
                        content=(
                            f"Tool call skipped: at most {self.config.max_tools_per_turn} "
                            "calls are allowed per turn."
                        ),
                        error_code="TURN_TOOL_LIMIT",
                    )
                    self.checkpoints.record_tool_call(
                        self.task_id, request, idempotent=False
                    )
                    self.checkpoints.finish_tool_call(
                        request.call_id, "FAILED", result.to_dict()
                    )
                    results.append(result)
                    continue
                if self.tool_call_count >= self.config.max_tool_calls:
                    result = ToolResult(
                        call_id=request.call_id,
                        name=request.name,
                        status=ToolStatus.ERROR,
                        content="Tool call skipped: task tool-call budget exhausted.",
                        error_code="TOOL_BUDGET_EXHAUSTED",
                    )
                    self.checkpoints.record_tool_call(
                        self.task_id, request, idempotent=False
                    )
                    self.checkpoints.finish_tool_call(
                        request.call_id, "FAILED", result.to_dict()
                    )
                    results.append(result)
                    continue

                self._set_state(AgentState.VALIDATING_TOOL_CALL, self.turn)
                definition = self.registry.get(request.name)
                duplicate = self._duplicate_result(request)
                if duplicate is not None:
                    result = duplicate
                    self.checkpoints.record_tool_call(
                        self.task_id,
                        request,
                        idempotent=bool(definition.idempotent) if definition else False,
                    )
                    self.checkpoints.finish_tool_call(
                        request.call_id, result.status.value, result.to_dict()
                    )
                    results.append(result)
                    continue

                idempotent = bool(definition.idempotent) if definition else False
                self.checkpoints.record_tool_call(
                    self.task_id, request, idempotent=idempotent
                )
                self._set_state(AgentState.CHECKING_PERMISSION, self.turn)
                self._set_state(AgentState.EXECUTING_TOOL, self.turn)
                result = self._execute_with_checkpoint(request, idempotent=idempotent)
                self._set_state(AgentState.OBSERVING_RESULT, self.turn)
                results.append(result)
                if result.error_code == "PROCESS_CLEANUP_FAILED":
                    raise RuntimeError(f"PROCESS_CLEANUP_FAILED: {result.content}")

            for result in results:
                if result.ok and result.exit_code in (0, None):
                    self._consecutive_failures = 0
                else:
                    self._consecutive_failures += 1
            if self._consecutive_failures >= self.config.max_consecutive_failures:
                raise RuntimeError(
                    "tool failure budget exhausted "
                    f"({self._consecutive_failures} consecutive failures)"
                )
            if self.state is not AgentState.OBSERVING_RESULT:
                self._set_state(AgentState.OBSERVING_RESULT, self.turn)
            self._append_tool_results(results)
            self._push_messages()
            self._advance_turn()

    # ------------------------------------------------------------ prompt budget

    def _render_prompt(self, bundle: Any | None) -> str:
        return render_system_prompt(
            bundle,
            tool_names=self.registry.names(),
            max_tool_calls=self.config.max_tools_per_turn,
            execution_allowed=self.config.allow_execution,
            include_action_fallback=self.config.include_action_fallback,
        )

    def _tool_schemas(self) -> list[dict[str, Any]]:
        if self._tool_schemas_cache is None:
            self._tool_schemas_cache = self.registry.anthropic_tools()
        return self._tool_schemas_cache

    def _limits(self) -> Any | None:
        if not self._input_limits_loaded:
            getter = getattr(self.model, "input_limits", None)
            if callable(getter):
                try:
                    self._input_limits = getter()
                    self._input_limits_loaded = True
                except Exception:  # noqa: BLE001 - configured budget remains the cap
                    self._input_limits = None
            else:
                self._input_limits_loaded = True
        return self._input_limits

    def _prompt_cap(self) -> int:
        """Whole-request input cap, below both the 413 guard and the model window."""
        configured = max(256, int(self.config.model_context_budget))
        cap = configured
        limits = self._limits()
        if limits is not None:
            reject = int(getattr(limits, "prompt_reject_tokens", 0) or 0)
            window = int(getattr(limits, "model_context_limit", 0) or 0)
            output_limit = int(
                getattr(limits, "max_new_tokens_limit", 0) or 0
            )
            output = (
                min(int(self.config.max_tokens), output_limit)
                if output_limit
                else int(self.config.max_tokens)
            )
            runtime_caps = [value for value in (reject, window) if value > 0]
            if window > 0:
                runtime_caps = [
                    reject if reject > 0 else window,
                    max(0, window - max(0, output)),
                ]
            runtime_caps = [value for value in runtime_caps if value > 0]
            if runtime_caps:
                cap = min([cap, *runtime_caps])
        return max(256, cap - _PROMPT_SAFETY_MARGIN_TOKENS)

    def _builder_budget(self) -> int:
        configured = max(256, int(self.config.model_context_budget))
        return max(256, min(configured, self._prompt_cap() - _BUILDER_RESERVE_TOKENS))

    def _count_prompt(self, system: str, messages: list[dict[str, Any]]) -> int:
        counter = getattr(self.model, "count_input_tokens", None)
        if callable(counter):
            try:
                return int(
                    counter(
                        system=system,
                        messages=messages,
                        tools=self._tool_schemas(),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - fall back to safe estimate
                if not self._count_fallback_warned:
                    self._count_fallback_warned = True
                    self.warnings.append(
                        {
                            "code": "PROMPT_TOKEN_COUNT_UNAVAILABLE",
                            "message": (
                                f"{type(exc).__name__}: {exc}; using the "
                                "conservative character estimate"
                            ),
                        }
                    )
        payload = (
            system
            + json.dumps(messages, ensure_ascii=False, default=str)
            + json.dumps(self._tool_schemas(), ensure_ascii=False)
        )
        return estimate_tokens(payload)

    def _shrink_context_once(self, info: dict[str, Any]) -> Any | None:
        bundle = self.last_context_bundle
        request = self._last_context_request
        if bundle is None or request is None or self.context_builder is None:
            return None
        if getattr(bundle, "rag_evidence", ()):
            bundle = self.context_builder.shrink(bundle, request, drop_rag=1)
            info["dropped_rag_items"] += 1
        elif getattr(bundle, "recalled_memories", ()):
            bundle = self.context_builder.shrink(bundle, request, drop_memory=1)
            info["dropped_memory_items"] += 1
        else:
            return None
        self.last_context_bundle = bundle
        self.system_prompt = self._render_prompt(bundle)
        return bundle

    def _budgeted_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Fit the whole request under the serving guard before it is sent."""
        cap = self._prompt_cap()
        tokens = self._count_prompt(self.system_prompt, messages)
        info: dict[str, Any] = {
            "prompt_cap": cap,
            "initial_prompt_tokens": tokens,
            "prompt_tokens": tokens,
            "trimmed_old_tool_results": 0,
            "trimmed_old_assistant_text": 0,
            "dropped_rag_items": 0,
            "dropped_memory_items": 0,
            "truncated_latest_tool_results": 0,
            "dropped_turn_pairs": 0,
        }
        if tokens <= cap:
            self._prompt_budget_info = info
            self._context_info["prompt_budget"] = info
            return messages

        fitted = json.loads(json.dumps(messages, ensure_ascii=False, default=str))

        # 1. Historical tool output and prose are the largest stale payloads.
        for message in fitted[:-2]:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if (
                    block.get("type") == "tool_result"
                    and block.get("content") != _OLD_TOOL_RESULT_PLACEHOLDER
                ):
                    block["content"] = _OLD_TOOL_RESULT_PLACEHOLDER
                    info["trimmed_old_tool_results"] += 1
                elif (
                    block.get("type") == "text"
                    and len(str(block.get("text") or "")) > 200
                ):
                    block["text"] = _OLD_ASSISTANT_TEXT_PLACEHOLDER
                    info["trimmed_old_assistant_text"] += 1
        tokens = self._count_prompt(self.system_prompt, fitted)

        # 2. RAG first, then verified memory: the lowest-ranked supplementary
        #    context is removed before the active task's history.
        while tokens > cap:
            if self._shrink_context_once(info) is None:
                break
            tokens = self._count_prompt(self.system_prompt, fitted)

        # 3. Keep the newest observation, but only the part that still fits.
        attempts = 0
        while tokens > cap and attempts < 4:
            last_content = fitted[-1].get("content")
            if not isinstance(last_content, list):
                break
            blocks = [
                block
                for block in last_content
                if isinstance(block, dict)
                and block.get("type") == "tool_result"
                and len(str(block.get("content") or "")) > 240
            ]
            if not blocks:
                break
            block = max(
                blocks, key=lambda item: len(str(item.get("content") or ""))
            )
            current = str(block.get("content") or "")
            overshoot = tokens - cap
            target = max(200, len(current) - overshoot * 3)
            if target >= len(current):
                target = max(200, len(current) // 2)
            block["content"] = (
                current[:target].rstrip() + _LATEST_TOOL_RESULT_MARKER
            )
            info["truncated_latest_tool_results"] += 1
            attempts += 1
            tokens = self._count_prompt(self.system_prompt, fitted)

        # 4. Drop complete old turn pairs; the first user task and the newest
        #    tool_use/tool_result pair are never removed.
        while tokens > cap and len(fitted) > 3:
            if fitted[1].get("role") != "assistant":
                break
            del fitted[1:3]
            info["dropped_turn_pairs"] += 1
            tokens = self._count_prompt(self.system_prompt, fitted)

        info["prompt_tokens"] = tokens
        self._prompt_budget_info = info
        self._context_info["prompt_budget"] = info
        self._event("PROMPT_BUDGET", info)
        if tokens > cap:
            raise ContextBudgetExceeded(
                prompt_tokens=tokens,
                budget=cap,
                detail=(
                    "the fixed system rules, tool schema and current task do "
                    "not fit; no request was sent"
                ),
            )
        self.warnings.append(
            {
                "code": "CONTEXT_BUDGET_TRIMMED",
                "message": (
                    f"prompt trimmed from ~{info['initial_prompt_tokens']} to "
                    f"{tokens} tokens (budget {cap})"
                ),
            }
        )
        return fitted

    # ---------------------------------------------------------------- model IO

    def _request_model(self) -> ModelResponse:
        last_error: Exception | None = None
        for attempt in range(self.config.max_model_retries + 1):
            self._check_cancel()
            self._check_deadline()
            try:
                response = self.model.complete(
                    system=self.system_prompt,
                    messages=self._budgeted_messages(self._trimmed_messages()),
                    tools=self._tool_schemas(),
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                self._event("MODEL_RESPONSE", response_summary(response))
                self.checkpoints.save_task(
                    self.task_id,
                    state=self.state.value,
                    turn=self.turn,
                    model_request_id=response.request_id,
                    model_response_chars=len(response.text),
                )
                return response
            except ModelUnavailable as exc:
                message = str(exc)
                if (
                    "HTTP 413" in message
                    or "prompt_resource_risk" in message
                    or "model_context_limit_exceeded" in message
                ):
                    # A deterministic resource rejection is not retryable; the
                    # budget pass must make it impossible before this point.
                    raise ContextBudgetExceeded(
                        prompt_tokens=0,
                        budget=self._prompt_cap(),
                        detail=message,
                    ) from exc
                last_error = exc
                self.retry_count += 1
                self.checkpoints.save_task(
                    self.task_id, state=self.state.value, turn=self.turn,
                    retry_count=self.retry_count,
                )
                self._event(
                    "MODEL_RETRY",
                    {"attempt": attempt + 1, "error": str(exc)},
                )
                if attempt < self.config.max_model_retries:
                    time.sleep(min(4.0, 2.0 ** attempt))
        raise RuntimeError(f"model request failed after retries: {last_error}")

    def _parse_response(
        self, response: ModelResponse
    ) -> tuple[str, list[ModelToolCall], ToolOrigin]:
        if response.tool_calls:
            return response.text, response.tool_calls, ToolOrigin.MODEL_SELECTED_TOOL
        if response.truncated:
            self.warnings.append(
                {
                    "code": "MODEL_OUTPUT_TRUNCATED",
                    "message": "model hit max_tokens before finishing; answer may be incomplete",
                }
            )
            self._event("MODEL_TRUNCATED", {"text_chars": len(response.text)})
        if self.config.include_action_fallback:
            try:
                remaining, action = extract_action(response.text)
            except ActionParseError as exc:
                self.warnings.append(
                    {"code": "ACTION_PARSE_FAILED", "message": str(exc)}
                )
                self._event("ACTION_PARSE_FAILED", {"error": str(exc)})
                return response.text, [], ToolOrigin.MODEL_SELECTED_TOOL
            if action is not None:
                self._event(
                    "FALLBACK_ACTION", {"tool": action.tool, "text_chars": len(remaining)}
                )
                return (
                    remaining,
                    [ModelToolCall(name=action.tool, arguments=action.arguments)],
                    ToolOrigin.DETERMINISTIC_FALLBACK,
                )
        return response.text, [], ToolOrigin.MODEL_SELECTED_TOOL

    def _trimmed_messages(self) -> list[dict[str, Any]]:
        limit = self.config.context_char_limit
        serialized = json.dumps(self.messages, ensure_ascii=False, default=str)
        if len(serialized) <= limit or len(self.messages) <= 6:
            return self.messages
        trimmed = json.loads(serialized)
        for message in trimmed[:-6]:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    block["content"] = "[older tool result omitted to fit the context budget]"
                elif block.get("type") == "text" and len(str(block.get("text", ""))) > 1000:
                    block["text"] = "[older assistant text omitted]"
        self._event("CONTEXT_TRIMMED", {"messages": len(self.messages)})
        return trimmed

    # ---------------------------------------------------------------- tool calls

    def _build_requests(
        self, calls: list[ModelToolCall], origin: ToolOrigin
    ) -> list[ToolRequest]:
        requests = []
        for index, call in enumerate(calls):
            requests.append(
                ToolRequest(
                    call_id=f"call-{self.task_id}-t{self.turn}-{index}",
                    name=call.name,
                    arguments=call.arguments,
                    origin=origin,
                    turn=self.turn,
                )
            )
        return requests

    def _duplicate_result(self, request: ToolRequest) -> ToolResult | None:
        key = _canonical(request.arguments)
        definition = self.registry.get(request.name)
        previous_calls = self.checkpoints.tool_calls(self.task_id)
        last_modification = _last_successful_modification(previous_calls)
        if definition is not None and definition.repeatable:
            successes = [
                previous
                for index, previous in enumerate(previous_calls)
                if index > last_modification
                and previous["name"] == request.name
                and _canonical(previous["arguments"]) == key
                and previous.get("status") == "COMPLETED"
                and (previous.get("result") or {}).get("exit_code") == 0
            ]
            if len(successes) >= self.config.max_repeat_successes:
                self._event(
                    "REPEATED_SUCCESS_BLOCKED",
                    {"tool": request.name, "success_count": len(successes)},
                )
                return ToolResult(
                    call_id=request.call_id,
                    name=request.name,
                    status=ToolStatus.ERROR,
                    content=(
                        f"The exact call to {request.name} already succeeded "
                        f"{len(successes)} times without any file change. Stop repeating it "
                        "and report the verified result."
                    ),
                    structured={"success_count": len(successes)},
                    error_code="REPEATED_SUCCESS_CALL",
                    origin=request.origin.value,
                )
            return None
        failures = [
            previous
            for index, previous in enumerate(previous_calls)
            if index > last_modification
            and previous["name"] == request.name
            and _canonical(previous["arguments"]) == key
            and _is_failed_call(previous)
        ]
        if len(failures) >= self.config.max_repeat_failures:
            last = failures[-1]
            last_result = last.get("result") or {}
            last_structured = last_result.get("structured") or {}
            suggestion = last_structured.get("suggestion") or (
                "Change the arguments or stop; identical retries are blocked."
            )
            self._event(
                "REPEATED_CALL_BLOCKED",
                {
                    "tool": request.name,
                    "failure_count": len(failures),
                    "last_error_code": last_result.get("error_code"),
                },
            )
            return ToolResult(
                call_id=request.call_id,
                name=request.name,
                status=ToolStatus.ERROR,
                content=(
                    f"The exact call to {request.name} already failed "
                    f"{len(failures)} times with these arguments. "
                    f"Last error: {last_result.get('error_code') or last.get('status')}. "
                    f"{suggestion} Do not repeat it unchanged."
                ),
                structured={
                    "failure_count": len(failures),
                    "last_error_code": last_result.get("error_code"),
                    "suggestion": suggestion,
                },
                error_code="REPEATED_FAILED_CALL",
                origin=request.origin.value,
            )
        for previous in previous_calls:
            if previous["name"] != request.name:
                continue
            if _canonical(previous["arguments"]) != key:
                continue
            if _is_failed_call(previous):
                continue
            if previous["status"] not in {"COMPLETED", "DUPLICATE"}:
                continue
            if previous["result"] is None:
                continue
            if definition is not None and not definition.idempotent:
                return ToolResult(
                    call_id=request.call_id,
                    name=request.name,
                    status=ToolStatus.ERROR,
                    content=(
                        f"Duplicate side-effecting call blocked: {request.name} with identical "
                        "arguments already ran in this task. Verify the current state before retrying."
                    ),
                    error_code="DUPLICATE_CALL_BLOCKED",
                )
            cached = _result_from_dict(previous["result"])
            return ToolResult(
                call_id=request.call_id,
                name=request.name,
                status=ToolStatus.DUPLICATE,
                content=(
                    "[duplicate call: previous result reused]\n" + cached.content
                ),
                structured=cached.structured,
                exit_code=cached.exit_code,
                origin=request.origin.value,
            )
        return None

    def _execute_with_checkpoint(self, request: ToolRequest, *, idempotent: bool) -> ToolResult:
        self.checkpoints.mark_tool_started(request.call_id)
        self.tool_call_count += 1
        result = self.registry.execute(request)
        status = "COMPLETED" if result.status is ToolStatus.OK else result.status.value
        self.checkpoints.finish_tool_call(request.call_id, status, result.to_dict())
        if result.status is ToolStatus.CANCELLED:
            raise CancelledError(result.content)
        return result

    def _append_tool_results(self, results: list[ToolResult]) -> None:
        blocks = []
        for result in results:
            text = self._observation_text(result)
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result.call_id,
                    "content": text,
                }
            )
        self.messages.append({"role": "user", "content": blocks})

    def _observation_text(self, result: ToolResult) -> str:
        payload: dict[str, Any] = {
            "tool": result.name,
            "status": result.status.value,
        }
        if result.error_code:
            payload["error_code"] = result.error_code
        if result.exit_code is not None:
            payload["exit_code"] = result.exit_code
        if result.structured:
            payload["structured"] = result.structured
        content = result.content or ""
        limit = self.config.observation_char_limit
        if len(content) > limit:
            content = content[:limit] + f"\n[...observation truncated {len(content) - limit} chars]"
            payload["truncated"] = True
        payload["content"] = content
        return json.dumps(payload, ensure_ascii=False, default=str)

    # -------------------------------------------------------------- completion

    def _finish_task(self, final_text: str, response: ModelResponse) -> None:
        retained: dict[str, Any] | None = None
        if self.memory_writer is not None and self.config.retain_memory:
            self._set_state(AgentState.RETAINING_MEMORY, self.turn)
            evidence, files_touched, tests_passed = self._collect_evidence()
            outcome = self.memory_writer.retain_handoff(
                task_id=self.task_id,
                session_id=self.config.session_id,
                project_id=self.config.project_id,
                task_description=self.checkpoints.get_task(self.task_id)["task_description"],
                evidence=evidence,
                files_touched=files_touched,
                tests_passed=tests_passed,
            )
            retained = outcome.to_dict()
            self._retained_memory = retained
            self._event("MEMORY_RETAIN", retained)
        self._set_state(AgentState.CHECKPOINTING, self.turn)
        self.checkpoints.save_task(
            self.task_id,
            state=AgentState.COMPLETED.value,
            turn=self.turn,
            messages=self.messages,
            status="COMPLETED",
            final_text=final_text,
        )
        self._event(
            "TASK_COMPLETED",
            {
                "final_text_chars": len(final_text),
                "stop_reason": response.stop_reason,
                "retained_memory": retained,
                "completion": self._completion.to_dict() if self._completion else None,
            },
        )
        self._set_state(AgentState.COMPLETED, self.turn)

    def _collect_evidence(self) -> tuple[list[str], list[str], bool | None]:
        evidence: list[str] = []
        files_touched: list[str] = []
        tests_passed: bool | None = None
        for call in self.checkpoints.tool_calls(self.task_id):
            result = call.get("result") or {}
            if result.get("status") not in {"OK", "DUPLICATE"}:
                continue
            name = call["name"]
            structured = result.get("structured") or {}
            if name == "run_tests":
                exit_code = result.get("exit_code")
                evidence.append(f"{call['call_id']} run_tests exit={exit_code}")
                if exit_code is not None:
                    tests_passed = exit_code == 0
            elif name in {"write_file", "edit_file", "apply_patch"}:
                path = structured.get("path")
                if path:
                    files_touched.append(str(path))
                    evidence.append(
                        f"{call['call_id']} {name} verified={structured.get('readback_verified')} path={path}"
                    )
            elif name == "run_python" and result.get("exit_code") == 0:
                evidence.append(f"{call['call_id']} run_python exit=0")
        return evidence, files_touched, tests_passed

    # ------------------------------------------------------------------ states

    def _set_state(self, target: AgentState, turn: int) -> None:
        if self.state is not target:
            ensure_transition(self.state, target)
        self.state = target
        if self.checkpoints.task_exists(self.task_id):
            self.checkpoints.save_task(self.task_id, state=target.value, turn=turn)
        self._event("STATE", {"state": target.value})

    def _restore_state(self, state: AgentState) -> None:
        # Resume resets the state machine to the checkpointed state; this is the
        # only place a transition check does not apply.
        self.state = state
        self._event("STATE_RESTORED", {"state": state.value})

    def _advance_turn(self) -> None:
        self.turn += 1
        self._event("TURN", {"turn": self.turn})
        if self.turn >= self.config.max_turns:
            raise RuntimeError(f"maximum turn limit reached ({self.config.max_turns})")

    def _check_limits(self) -> None:
        self._check_cancel()
        self._check_deadline()

    def _check_cancel(self) -> None:
        if self.cancel.cancelled:
            raise CancelledError(self.cancel.reason or "cancelled")

    def _check_deadline(self) -> None:
        if time.monotonic() > self.deadline:
            raise RuntimeError(
                f"task timeout exceeded ({self.config.task_timeout_seconds:.0f}s)"
            )

    def _event(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        try:
            self.checkpoints.append_event(
                self.task_id,
                turn=self.turn,
                state=self.state.value,
                kind=kind,
                payload=payload or {},
            )
        except Exception:  # noqa: BLE001 - event logging must not kill the task
            pass
        try:
            self.audit.event(
                task_id=self.task_id,
                session_id=self.config.session_id,
                turn=self.turn,
                state=self.state.value,
                kind=kind,
                payload=payload or {},
            )
        except Exception:  # noqa: BLE001
            pass

    def _cancel_task(self, reason: str) -> None:
        self.state = AgentState.CANCELLED
        try:
            self.checkpoints.save_task(
                self.task_id,
                state=AgentState.CANCELLED.value,
                turn=self.turn,
                messages=self.messages,
                status="CANCELLED",
                last_error=reason,
            )
        except Exception:  # noqa: BLE001
            pass
        self._event("TASK_CANCELLED", {"reason": reason})

    def _fail_task(self, error: str) -> None:
        self.state = AgentState.FAILED
        try:
            self.checkpoints.save_task(
                self.task_id,
                state=AgentState.FAILED.value,
                turn=self.turn,
                messages=self.messages,
                status="FAILED",
                last_error=error,
            )
        except Exception:  # noqa: BLE001
            pass
        self._event("TASK_FAILED", {"error": error})

    def _push_messages(self) -> None:
        self.checkpoints.save_task(
            self.task_id,
            state=self.state.value,
            turn=self.turn,
            messages=self.messages,
        )

    def _result(
        self,
        started: float,
        final_text: str,
        error: str | None,
        *,
        status: str,
    ) -> AgentRunResult:
        try:
            events = len(self.checkpoints.events(self.task_id))
        except Exception:  # noqa: BLE001
            events = 0
        record = None
        try:
            record = self.checkpoints.get_task(self.task_id)
        except Exception:  # noqa: BLE001
            pass
        return AgentRunResult(
            task_id=self.task_id,
            session_id=self.config.session_id,
            status=status,
            state=self.state.value,
            final_text=final_text if final_text else (record or {}).get("final_text") or "",
            turns=self.turn,
            tool_calls=self.tool_call_count,
            warnings=self.warnings,
            error=error,
            retained_memory=self._retained_memory,
            duration_seconds=round(time.perf_counter() - started, 3),
            events=events,
            completion=self._completion.to_dict() if self._completion else None,
        )


def _canonical(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)


_MODIFYING_TOOLS = {"write_file", "edit_file", "apply_patch"}


def _last_successful_modification(previous_calls: list[dict[str, Any]]) -> int:
    """Index of the last verified file write; failures before it are stale."""
    last = -1
    for index, call in enumerate(previous_calls):
        if call.get("name") not in _MODIFYING_TOOLS or call.get("status") != "COMPLETED":
            continue
        result = call.get("result") or {}
        structured = result.get("structured") or {}
        if structured.get("readback_verified"):
            last = index
    return last


def _is_failed_call(previous: dict[str, Any]) -> bool:
    status = previous.get("status")
    if status in {"ERROR", "DENIED", "TIMEOUT", "CANCELLED"}:
        return True
    if status == "COMPLETED":
        result = previous.get("result") or {}
        return result.get("exit_code") not in (0, None)
    return False


def _result_from_dict(data: dict[str, Any]) -> ToolResult:
    return ToolResult(
        call_id=str(data.get("call_id") or ""),
        name=str(data.get("name") or ""),
        status=ToolStatus(str(data.get("status") or "UNKNOWN")),
        content=str(data.get("content") or ""),
        structured=dict(data.get("structured") or {}),
        error_code=data.get("error_code"),
        permission=str(data.get("permission") or "READ"),
        origin=str(data.get("origin") or ToolOrigin.MODEL_SELECTED_TOOL.value),
        duration_ms=float(data.get("duration_ms") or 0.0),
        truncated=bool(data.get("truncated", False)),
        exit_code=data.get("exit_code"),
    )