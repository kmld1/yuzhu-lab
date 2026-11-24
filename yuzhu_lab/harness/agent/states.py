"""Resumable agent state machine.

The loop persists every transition.  A checkpoint is only meaningful if the
allowed transitions are explicit, so the table below is the single source of
truth for what the loop may do next.
"""

from __future__ import annotations

from enum import Enum


class AgentState(str, Enum):
    IDLE = "IDLE"
    ACCEPTING_TASK = "ACCEPTING_TASK"
    BUILDING_CONTEXT = "BUILDING_CONTEXT"
    REQUESTING_MODEL = "REQUESTING_MODEL"
    PARSING_RESPONSE = "PARSING_RESPONSE"
    VALIDATING_TOOL_CALL = "VALIDATING_TOOL_CALL"
    CHECKING_PERMISSION = "CHECKING_PERMISSION"
    EXECUTING_TOOL = "EXECUTING_TOOL"
    OBSERVING_RESULT = "OBSERVING_RESULT"
    RETAINING_MEMORY = "RETAINING_MEMORY"
    CHECKPOINTING = "CHECKPOINTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = frozenset(
    {AgentState.COMPLETED, AgentState.FAILED, AgentState.CANCELLED}
)

ALLOWED_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    AgentState.IDLE: frozenset({AgentState.ACCEPTING_TASK}),
    AgentState.ACCEPTING_TASK: frozenset(
        {AgentState.BUILDING_CONTEXT, AgentState.FAILED, AgentState.CANCELLED}
    ),
    AgentState.BUILDING_CONTEXT: frozenset(
        {AgentState.REQUESTING_MODEL, AgentState.FAILED, AgentState.CANCELLED}
    ),
    AgentState.REQUESTING_MODEL: frozenset(
        {AgentState.PARSING_RESPONSE, AgentState.FAILED, AgentState.CANCELLED}
    ),
    AgentState.PARSING_RESPONSE: frozenset(
        {
            AgentState.REQUESTING_MODEL,
            AgentState.VALIDATING_TOOL_CALL,
            AgentState.RETAINING_MEMORY,
            AgentState.CHECKPOINTING,
            AgentState.FAILED,
            AgentState.CANCELLED,
        }
    ),
    AgentState.VALIDATING_TOOL_CALL: frozenset(
        {AgentState.CHECKING_PERMISSION, AgentState.OBSERVING_RESULT, AgentState.FAILED}
    ),
    AgentState.CHECKING_PERMISSION: frozenset(
        {AgentState.EXECUTING_TOOL, AgentState.OBSERVING_RESULT, AgentState.FAILED}
    ),
    AgentState.EXECUTING_TOOL: frozenset(
        {AgentState.OBSERVING_RESULT, AgentState.FAILED, AgentState.CANCELLED}
    ),
    AgentState.OBSERVING_RESULT: frozenset(
        {
            AgentState.REQUESTING_MODEL,
            AgentState.CHECKPOINTING,
            AgentState.VALIDATING_TOOL_CALL,
            AgentState.FAILED,
            AgentState.CANCELLED,
        }
    ),
    AgentState.RETAINING_MEMORY: frozenset(
        {AgentState.CHECKPOINTING, AgentState.FAILED}
    ),
    AgentState.CHECKPOINTING: frozenset(
        {AgentState.REQUESTING_MODEL, AgentState.COMPLETED, AgentState.FAILED, AgentState.CANCELLED}
    ),
    AgentState.COMPLETED: frozenset(),
    AgentState.FAILED: frozenset(),
    AgentState.CANCELLED: frozenset(),
}


class IllegalTransition(RuntimeError):
    """Raised when the loop asks for a transition the state machine forbids."""


def ensure_transition(current: AgentState, target: AgentState) -> None:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise IllegalTransition(f"{current.value} -> {target.value} is not allowed")