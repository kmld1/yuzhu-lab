"""Typed inputs and outputs of the harness context builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ContextRequest:
    user_message: str
    session_id: str
    project_id: str
    workspace: Path | None = None
    conversation_summary: str = ""
    model_context_budget: int = 8192
    memory_token_budget: int | None = None
    rag_token_budget: int | None = None
    final_k: int = 5
    allow_memory: bool = True
    allow_rag: bool = True
    include_unverified: bool = False


@dataclass(frozen=True)
class ContextWarning:
    code: str
    message: str
    severity: str = "warning"


@dataclass(frozen=True)
class MemoryContextItem:
    memory_id: str
    text: str
    category: str
    verification: str
    confidence: float
    source: str
    occurred_at: str | None
    bank_id: str
    trust: str
    conflict: bool = False
    injection_flags: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RagEvidenceItem:
    chunk_id: str
    citation: str
    source: str
    symbol: str | None
    start_line: int | None
    end_line: int | None
    content: str
    score: float
    rrf_score: float
    trust: str = "EVIDENCE"
    injection_flags: tuple[str, ...] = ()
    project_id: str = ""


@dataclass(frozen=True)
class SourceReference:
    kind: str
    identifier: str
    citation: str
    trust: str
    bank_id: str | None = None
    verification: str | None = None


@dataclass(frozen=True)
class TokenEstimate:
    budget: int
    system_context: int
    memories: int
    rag: int
    summary: int
    total: int
    truncated: bool
    dropped_items: int


@dataclass(frozen=True)
class ContextBundle:
    system_context: str
    recalled_memories: tuple[MemoryContextItem, ...]
    rag_evidence: tuple[RagEvidenceItem, ...]
    source_references: tuple[SourceReference, ...]
    warnings: tuple[ContextWarning, ...]
    token_estimate: TokenEstimate
    context_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_context": self.system_context,
            "recalled_memories": [item.__dict__ for item in self.recalled_memories],
            "rag_evidence": [item.__dict__ for item in self.rag_evidence],
            "source_references": [item.__dict__ for item in self.source_references],
            "warnings": [item.__dict__ for item in self.warnings],
            "token_estimate": self.token_estimate.__dict__,
            "context_metadata": dict(self.context_metadata),
        }