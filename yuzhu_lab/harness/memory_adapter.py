"""Memory port used by the context builder.

Hindsight memory describes history: decisions, verified fixes, prior failures
and user preferences.  It is never a substitute for the current contents of a
file, so the adapter labels every item with its verification state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from ..hindsight.models import RecallMemory, ReflectAnswer
from ..hindsight.service import MemoryService
from ..source_policy import (
    PROTECTED_EVAL_SOURCE,
    ProtectedContentGuard,
    SourcePolicy,
    protected_content_guard,
)
from .context_models import MemoryContextItem
from .context_policy import sanitize_untrusted


@dataclass
class MemorySearchOutcome:
    items: list[MemoryContextItem] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    missing_banks: list[str] = field(default_factory=list)
    banks_queried: list[str] = field(default_factory=list)
    excluded_unverified: int = 0
    excluded_protected: int = 0
    latency_ms: float = 0.0


class MemorySearchPort(Protocol):
    def recall(
        self,
        *,
        project_id: str,
        query: str,
        max_tokens: int,
        include_unverified: bool = False,
    ) -> MemorySearchOutcome: ...


def trust_label(memory: RecallMemory) -> str:
    if memory.conflict:
        return "MEMORY_CONFLICT"
    if memory.verification == "verified":
        return "MEMORY_VERIFIED"
    if memory.verification == "unverified":
        return "MEMORY_UNVERIFIED"
    return "MEMORY_UNKNOWN"


class HindsightMemoryAdapter:
    def __init__(
        self,
        service: MemoryService,
        *,
        source_policy: SourcePolicy | None = None,
        content_guard: ProtectedContentGuard | bool | None = None,
    ) -> None:
        self.service = service
        self.source_policy = source_policy
        if isinstance(content_guard, ProtectedContentGuard):
            self.content_guard: ProtectedContentGuard | None = content_guard
        elif content_guard and source_policy is not None:
            self.content_guard = protected_content_guard(source_policy)
        else:
            self.content_guard = None

    def _memory_is_protected(self, memory: RecallMemory) -> bool:
        if self.content_guard is not None and self.content_guard.matches(memory.text):
            return True
        if self.source_policy is None:
            return False
        candidates = [
            memory.source,
            memory.metadata.get("source"),
            memory.metadata.get("source_path"),
            memory.metadata.get("code_refs"),
        ]
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            decision = self.source_policy.check_relative(
                candidate, require_exists=False
            )
            if decision.code == PROTECTED_EVAL_SOURCE:
                return True
        return False

    def recall(
        self,
        *,
        project_id: str,
        query: str,
        max_tokens: int,
        include_unverified: bool = False,
    ) -> MemorySearchOutcome:
        started = time.perf_counter()
        try:
            outcome = self.service.recall(
                project_id=project_id,
                query=query,
                max_tokens=max_tokens,
                include_global=True,
                include_experiments=include_unverified,
            )
        except Exception as exc:  # noqa: BLE001 - degrade to memoryless context
            return MemorySearchOutcome(
                degraded=[f"hindsight: {type(exc).__name__}: {exc}"],
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        items: list[MemoryContextItem] = []
        excluded = 0
        excluded_protected = 0
        for memory in outcome.memories:
            if self._memory_is_protected(memory):
                excluded_protected += 1
                continue
            if memory.verification != "verified" and not include_unverified:
                excluded += 1
                continue
            text, flags = sanitize_untrusted(memory.text)
            items.append(
                MemoryContextItem(
                    memory_id=memory.memory_id,
                    text=text,
                    category=memory.category,
                    verification=memory.verification,
                    confidence=memory.confidence,
                    source=memory.source,
                    occurred_at=memory.occurred_at,
                    bank_id=memory.bank_id,
                    trust=trust_label(memory),
                    conflict=memory.conflict,
                    injection_flags=flags,
                    metadata=memory.metadata,
                )
            )
        return MemorySearchOutcome(
            items=items,
            degraded=list(outcome.degraded),
            missing_banks=list(outcome.missing_banks),
            banks_queried=list(outcome.banks_queried),
            excluded_unverified=excluded,
            excluded_protected=excluded_protected,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def reflect(self, *, project_id: str, query: str, max_tokens: int = 512) -> ReflectAnswer:
        return self.service.reflect(
            project_id=project_id, query=query, max_tokens=max_tokens
        )


class NullMemoryAdapter:
    """Explicit no-memory mode; produces a visible degradation, never fake data."""

    def recall(
        self,
        *,
        project_id: str,
        query: str,
        max_tokens: int,
        include_unverified: bool = False,
    ) -> MemorySearchOutcome:
        return MemorySearchOutcome(degraded=["hindsight: adapter disabled"])

    def reflect(self, *, project_id: str, query: str, max_tokens: int = 512) -> ReflectAnswer:
        raise RuntimeError("memory adapter is disabled")