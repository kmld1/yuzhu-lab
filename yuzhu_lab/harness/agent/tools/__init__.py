"""Default local coding tool set for the agent harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..permissions import PermissionPolicy
from ..registry import ToolContext, ToolRegistry
from .exec import exec_tool_definitions
from .files import file_tool_definitions
from .git import git_tool_definitions

__all__ = [
    "build_default_registry",
    "default_tool_context",
    "exec_tool_definitions",
    "file_tool_definitions",
    "git_tool_definitions",
]


def default_tool_context(
    policy: PermissionPolicy,
    *,
    interpreter: Path | str | None = None,
    max_output_chars: int = 20000,
) -> ToolContext:
    return ToolContext(
        workspace=policy.workspace,
        guard=policy.guard,
        interpreter=interpreter,
        max_output_chars=max_output_chars,
    )


def build_default_registry(
    policy: PermissionPolicy,
    *,
    context: ToolContext | None = None,
    audit: Callable[[dict[str, Any]], None] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry(
        policy,
        context=context or default_tool_context(policy),
        audit=audit,
    )
    for definition in [
        *file_tool_definitions(),
        *exec_tool_definitions(),
        *git_tool_definitions(),
    ]:
        registry.register(definition)
    return registry