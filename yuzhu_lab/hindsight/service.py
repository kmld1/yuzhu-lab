"""Retain / recall / reflect orchestration with admission and project isolation."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .audit import MemoryAuditLog, text_digest
from .client import HindsightAPIError, HindsightHttpClient
from .config import HindsightConfig, _slug, bank_id_for, project_tag
from .models import (
    AdmissionDecision,
    MemoryRecord,
    RecallMemory,
    ReflectAnswer,
    RetainResult,
)
from .policy import MemoryAdmissionPolicy


class MemoryRejected(ValueError):
    def __init__(self, decision: AdmissionDecision) -> None:
        super().__init__(f"{decision.code}: {decision.reason}")
        self.decision = decision


@dataclass
class RecallOutcome:
    memories: list[RecallMemory] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    missing_banks: list[str] = field(default_factory=list)
    banks_queried: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.degraded or bool(self.memories)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _safe_id(value: str, *, max_length: int = 120) -> str:
    value = re.sub(r"[^A-Za-z0-9._:-]+", "-", value.strip())
    return value[:max_length].strip("-._:") or "memory"


class MemoryService:
    """Project-aware facade over the official Hindsight API."""

    def __init__(
        self,
        client: HindsightHttpClient,
        *,
        policy: MemoryAdmissionPolicy | None = None,
        audit: MemoryAuditLog | None = None,
        config: HindsightConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or client.config
        self.policy = policy or MemoryAdmissionPolicy(self.config)
        self.audit = audit or MemoryAuditLog(self.config.audit_log_path)

    # -------------------------------------------------------------------- retain

    def retain(self, record: MemoryRecord) -> RetainResult:
        started = time.perf_counter()
        decision = self.policy.evaluate(record)
        if not decision.allowed:
            self._audit_retain(record, decision, started, accepted=False, code=decision.code)
            raise MemoryRejected(decision)

        digest = text_digest(decision.normalized_content)
        existing = self._same_key_memories(record, decision)
        duplicate = next(
            (
                memory
                for memory in existing
                if memory.metadata.get("content_sha256") == digest
            ),
            None,
        )
        if duplicate is not None:
            result = RetainResult(
                accepted=True,
                bank_id=decision.bank_id,
                memory_id=duplicate.memory_id,
                code="DUPLICATE",
                reason="an identical memory already exists",
                duplicate=True,
            )
            self._audit_retain(record, decision, started, accepted=True, code="DUPLICATE")
            return result

        conflict = bool(existing)
        warnings: list[str] = []
        metadata = dict(decision.metadata)
        metadata["content_sha256"] = digest
        metadata["status"] = "conflict" if conflict else "active"
        if conflict:
            metadata["supersedes"] = ",".join(memory.memory_id for memory in existing)
            warnings.append(
                "A verified memory with the same memory_key already exists; "
                "the new record is stored as a conflict instead of overwriting it."
            )

        short_digest = digest[:12]
        key_part = record.memory_key or short_digest
        document_id = _safe_id(
            f"yuzhu-{record.project_id}-{record.category}-{key_part}"
            + (f"-{short_digest}" if conflict else "")
        )

        response = self.client.retain(
            bank_id=decision.bank_id,
            content=decision.normalized_content,
            document_id=document_id,
            context=f"{record.category} / {record.source}",
            metadata=metadata,
            tags=decision.tags,
            timestamp=record.occurred_at,
        )
        if not response.get("success", False):
            raise RuntimeError(f"Hindsight retain did not succeed: {response}")

        result = RetainResult(
            accepted=True,
            bank_id=decision.bank_id,
            memory_id=document_id,
            code="CONFLICT_STORED" if conflict else "STORED",
            reason="stored",
            conflict=conflict,
            warnings=tuple(warnings),
            usage=dict(response.get("usage") or {}),
        )
        self._audit_retain(
            record,
            decision,
            started,
            accepted=True,
            code=result.code,
            memory_id=document_id,
            conflict=conflict,
            usage=result.usage,
        )
        return result

    def _same_key_memories(
        self, record: MemoryRecord, decision: AdmissionDecision
    ) -> list[RecallMemory]:
        if not record.memory_key:
            return []
        key_tag = "memory_key:" + _slug(record.memory_key, max_length=64)
        try:
            response = self.client.recall(
                bank_id=decision.bank_id,
                query=decision.normalized_content[:300],
                max_tokens=1024,
                tags=[key_tag],
                tags_match="all_strict",
            )
        except HindsightAPIError as exc:
            if exc.status_code == 404:
                return []
            raise
        return [self._to_memory(item, decision.bank_id) for item in response.get("results") or []]

    # -------------------------------------------------------------------- recall

    def recall(
        self,
        *,
        project_id: str,
        query: str,
        max_tokens: int = 1200,
        include_global: bool = True,
        include_experiments: bool = False,
    ) -> RecallOutcome:
        targets: list[tuple[str, list[str]]] = [
            (bank_id_for(project_id, "project"), [project_tag(project_id)])
        ]
        if include_global:
            targets.append((bank_id_for(project_id, "global"), ["scope:global"]))
        if include_experiments:
            targets.append((bank_id_for(project_id, "experiment"), ["scope:experiment"]))

        outcome = RecallOutcome()
        for bank_id, tags in targets:
            outcome.banks_queried.append(bank_id)
            try:
                response = self.client.recall(
                    bank_id=bank_id,
                    query=query,
                    max_tokens=max_tokens,
                    tags=tags,
                    tags_match="any_strict",
                )
            except HindsightAPIError as exc:
                if exc.status_code == 404:
                    outcome.missing_banks.append(bank_id)
                    continue
                outcome.degraded.append(f"{bank_id}: {type(exc).__name__}")
                continue
            except Exception as exc:  # network/timeout: degrade, never fabricate
                outcome.degraded.append(f"{bank_id}: {type(exc).__name__}")
                continue
            for item in response.get("results") or []:
                outcome.memories.append(self._to_memory(item, bank_id))

        self._mark_conflicts(outcome.memories)
        outcome.memories = self._deduplicate(outcome.memories)
        return outcome

    def _deduplicate(self, memories: list[RecallMemory]) -> list[RecallMemory]:
        best: dict[str, RecallMemory] = {}
        for memory in memories:
            key = _normalize(memory.text)
            current = best.get(key)
            if current is None or memory.confidence > current.confidence:
                best[key] = memory
        return sorted(best.values(), key=lambda item: item.confidence, reverse=True)

    @staticmethod
    def _mark_conflicts(memories: list[RecallMemory]) -> None:
        by_key: dict[str, list[RecallMemory]] = {}
        for memory in memories:
            key = memory.metadata.get("memory_key")
            if key:
                by_key.setdefault(key, []).append(memory)
        conflict_ids: set[str] = set()
        for group in by_key.values():
            # Several facts extracted from the same retained document share one
            # document id and content hash; they are complementary, not a
            # conflict.  A conflict needs two distinct source documents.
            identities = {
                item.metadata.get("content_sha256")
                or item.document_id
                or item.memory_id
                for item in group
            }
            if len(identities) > 1 or any(
                item.metadata.get("status") == "conflict" for item in group
            ):
                conflict_ids.update(item.memory_id for item in group)
        if not conflict_ids:
            return
        for index, memory in enumerate(memories):
            if memory.memory_id in conflict_ids:
                memories[index] = RecallMemory(
                    memory_id=memory.memory_id,
                    bank_id=memory.bank_id,
                    text=memory.text,
                    memory_type=memory.memory_type,
                    category=memory.category,
                    verification=memory.verification,
                    confidence=memory.confidence,
                    source=memory.source,
                    occurred_at=memory.occurred_at,
                    mentioned_at=memory.mentioned_at,
                    document_id=memory.document_id,
                    metadata=memory.metadata,
                    tags=memory.tags,
                    scores=memory.scores,
                    conflict=True,
                )

    # ------------------------------------------------------------------- reflect

    def reflect(
        self,
        *,
        project_id: str,
        query: str,
        max_tokens: int = 512,
    ) -> ReflectAnswer:
        bank_id = bank_id_for(project_id, "project")
        response = self.client.reflect(
            bank_id=bank_id,
            query=query,
            max_tokens=max_tokens,
            tags=[project_tag(project_id)],
            include_facts=True,
        )
        based_on = tuple((response.get("based_on") or {}).get("memories") or [])
        return ReflectAnswer(
            text=response.get("text") or "",
            bank_id=bank_id,
            based_on=based_on,
            usage=dict(response.get("usage") or {}),
        )

    # ------------------------------------------------------------------- helpers

    @staticmethod
    def _to_memory(item: dict[str, Any], bank_id: str) -> RecallMemory:
        metadata = {
            str(key): str(value)
            for key, value in (item.get("metadata") or {}).items()
            if value is not None
        }
        scores = dict(item.get("scores") or {})
        return RecallMemory(
            memory_id=str(item.get("id") or ""),
            bank_id=bank_id,
            text=str(item.get("text") or ""),
            memory_type=str(item.get("type") or "unknown"),
            category=metadata.get("category", "uncategorized"),
            verification=metadata.get("verification", "unknown"),
            confidence=float(scores.get("final") or 0.0),
            source=metadata.get("source", "unknown"),
            occurred_at=item.get("occurred_start"),
            mentioned_at=item.get("mentioned_at"),
            document_id=item.get("document_id"),
            metadata=metadata,
            tags=tuple(item.get("tags") or ()),
            scores=scores,
        )

    def _audit_retain(
        self,
        record: MemoryRecord,
        decision: AdmissionDecision,
        started: float,
        **fields: Any,
    ) -> None:
        self.audit.record(
            "retain",
            project_id=record.project_id,
            bank_id=decision.bank_id,
            category=record.category,
            verification=record.verification,
            scope=record.scope,
            content_sha256=text_digest(decision.normalized_content),
            content_length=len(decision.normalized_content),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            **fields,
        )