"""Error taxonomy shared by the agent loop, registry and tools."""

from __future__ import annotations


class AgentError(RuntimeError):
    """Base class for agent failures that should end the current task."""


class PathViolation(ValueError):
    """A requested path escapes or subverts the workspace boundary."""

    def __init__(self, message: str, *, code: str = "PATH_OUTSIDE_WORKSPACE") -> None:
        super().__init__(message)
        self.code = code


class ToolValidationError(ValueError):
    """Tool arguments failed schema validation."""

    def __init__(self, messages: list[str]) -> None:
        super().__init__("; ".join(messages) or "invalid arguments")
        self.messages = list(messages)


class ToolExecutionError(RuntimeError):
    """The tool ran but could not complete its operation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class PermissionDenied(RuntimeError):
    """The permission policy refused the tool call."""

    def __init__(self, message: str, *, level: str) -> None:
        super().__init__(message)
        self.level = level


class CheckpointError(RuntimeError):
    """Persisted state is missing or inconsistent."""


class ModelUnavailable(RuntimeError):
    """The model endpoint is unreachable or returned an invalid response."""