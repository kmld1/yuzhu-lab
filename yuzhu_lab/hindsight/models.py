"""Value objects exchanged with the memory policy, service and harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MemoryRecord:
    """A candidate long-term memory.

    ``verification`` is deliberately explicit.  A model guess must never enter
    a project or global bank as a verified fact.
    """

    content: str
    project_id: str
    category: str
    scope: str = "project"
    verification: str = "verified"
    evidence: tuple[str, ...] = ()
    source: str = "session"
    session_id: str | None = None
    memory_key: str | None = None
    occurred_at: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    code: str
    reason: str
    bank_id: str
    normalized_content: str
    tags: tuple[str, ...]
    metadata: dict[str, str]


@dataclass(frozen=True)
class RetainResult:
    accepted: bool
    bank_id: str
    memory_id: str | None
    code: str
    reason: str
    conflict: bool = False
    duplicate: bool = False
    warnings: tuple[str, ...] = ()
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecallMemory:
    memory_id: str
    bank_id: str
    text: str
    memory_type: str
    category: str
    verification: str
    confidence: float
    source: str
    occurred_at: str | None
    mentioned_at: str | None
    document_id: str | None
    metadata: dict[str, str]
    tags: tuple[str, ...]
    scores: dict[str, Any] = field(default_factory=dict)
    conflict: bool = False


@dataclass(frozen=True)
class ReflectAnswer:
    text: str
    bank_id: str
    based_on: tuple[dict[str, Any], ...] = ()
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bank_id": self.bank_id,
            "based_on": list(self.based_on),
            "usage": dict(self.usage),
        }