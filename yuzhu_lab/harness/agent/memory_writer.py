"""End-of-task memory retention policy.

Only one compact, evidence-backed session handoff is retained.  Model prose is
never stored: the record is a template filled from structured tool evidence.

A retain failure is not silently ignored and does not block the task forever.
The HTTP client already bounds its own retries; this writer makes one bounded
attempt and, on failure, appends the record to a durable JSONL pending queue.
The pending count is returned in the task result and can be inspected later by
``scripts/h4_hindsight_pending.py`` or retried with ``retry_pending``.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ...hindsight.config import PROJECT_ROOT
from ...hindsight.models import MemoryRecord, RetainResult
from ...hindsight.service import MemoryRejected, MemoryService
from ...source_policy import PROTECTED_EVAL_SOURCE, ProtectedContentGuard, SourcePolicy

DEFAULT_PENDING_PATH = PROJECT_ROOT / "logs" / "hindsight" / "retain_pending.jsonl"


@dataclass
class RetainOutcome:
    retained: bool
    reason: str
    result: RetainResult | None = None
    warnings: list[str] = field(default_factory=list)
    pending: bool = False
    attempts: int = 1
    pending_id: str | None = None
    pending_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "retained": self.retained,
            "reason": self.reason,
            "memory_id": self.result.memory_id if self.result else None,
            "code": self.result.code if self.result else None,
            "warnings": list(self.warnings),
            "pending": self.pending,
            "attempts": self.attempts,
            "pending_id": self.pending_id,
            "pending_count": self.pending_count,
        }


class AgentMemoryWriter:
    def __init__(
        self,
        service: MemoryService | None,
        *,
        enabled: bool = True,
        category: str = "session_handoff",
        pending_path: Path | None = None,
        attempts: int = 1,
        retry_delay_seconds: float = 0.0,
        source_policy: SourcePolicy | None = None,
        content_guard: ProtectedContentGuard | None = None,
    ) -> None:
        self.service = service
        self.enabled = enabled
        self.category = category
        self.pending_path = Path(pending_path) if pending_path else DEFAULT_PENDING_PATH
        self.attempts = max(1, int(attempts))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self.source_policy = source_policy
        self.content_guard = content_guard
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ retain

    def retain_handoff(
        self,
        *,
        task_id: str,
        session_id: str,
        project_id: str,
        task_description: str,
        evidence: Sequence[str],
        files_touched: Sequence[str],
        tests_passed: bool | None,
    ) -> RetainOutcome:
        if not self.enabled:
            return RetainOutcome(False, "memory writing disabled")
        if self.service is None:
            return RetainOutcome(False, "hindsight service unavailable")
        evidence_list = [item.strip() for item in evidence if item.strip()]
        if not evidence_list:
            return RetainOutcome(
                False, "no verified tool evidence; model output is not retainable"
            )
        if self.source_policy is not None:
            for path in files_touched:
                decision = self.source_policy.check_relative(
                    path, require_exists=False
                )
                if decision.code == PROTECTED_EVAL_SOURCE:
                    return RetainOutcome(
                        False,
                        "rejected: task evidence references a protected "
                        "evaluation asset",
                    )
        if self.content_guard is not None and self.content_guard.matches(
            "\n".join([task_description, *evidence_list, *files_touched])
        ):
            return RetainOutcome(
                False,
                "rejected: task evidence contains protected evaluation content",
            )

        record = self._build_record(
            task_id=task_id,
            session_id=session_id,
            project_id=project_id,
            task_description=task_description,
            evidence_list=evidence_list,
            files_touched=files_touched,
            tests_passed=tests_passed,
        )

        last_error: Exception | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                result = self.service.retain(record)
            except MemoryRejected as exc:
                # A policy rejection is deterministic; retrying cannot help and
                # must not enter the pending queue.
                return RetainOutcome(
                    False,
                    f"rejected by memory policy: {exc.decision.code}",
                    attempts=attempt,
                )
            except Exception as exc:  # noqa: BLE001 - memory is optional but must be visible
                last_error = exc
                if attempt < self.attempts and self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds * attempt)
                continue
            return RetainOutcome(
                True,
                result.code,
                result=result,
                warnings=list(result.warnings),
                attempts=attempt,
            )

        pending_id = self._append_pending(
            record,
            error=f"{type(last_error).__name__}: {last_error}",
            attempts=self.attempts,
        )
        return RetainOutcome(
            False,
            f"retain failed after {self.attempts} attempt(s); queued for retry: "
            f"{type(last_error).__name__}: {last_error}",
            pending=True,
            attempts=self.attempts,
            pending_id=pending_id,
            pending_count=self.pending_count(),
        )

    def _build_record(
        self,
        *,
        task_id: str,
        session_id: str,
        project_id: str,
        task_description: str,
        evidence_list: list[str],
        files_touched: Sequence[str],
        tests_passed: bool | None,
    ) -> MemoryRecord:
        task_text = " ".join(task_description.split())[:200]
        parts = [
            f"Agent 任务 {task_id} 在项目 {project_id} 完成。",
            f"任务：{task_text}",
            "证据：" + "; ".join(evidence_list[:6]),
        ]
        if files_touched:
            parts.append("涉及文件：" + ", ".join(sorted(set(files_touched))[:6]))
        if tests_passed is not None:
            parts.append("测试结果：" + ("PASS" if tests_passed else "FAIL"))
        content = " ".join(parts)

        return MemoryRecord(
            content=content,
            project_id=project_id,
            category=self.category,
            scope="project",
            verification="verified",
            evidence=tuple(evidence_list[:6]),
            source="agent-harness",
            session_id=session_id,
            memory_key=f"agent-handoff:{task_id}",
            tags=("agent:h2",),
            metadata={
                "task_id": task_id,
                "evidence_count": str(len(evidence_list)),
                "tests_passed": "" if tests_passed is None else str(tests_passed).lower(),
            },
        )

    # ----------------------------------------------------------------- pending

    def _append(self, payload: dict[str, Any]) -> None:
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock:
            with self.pending_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _append_pending(
        self, record: MemoryRecord, *, error: str, attempts: int
    ) -> str:
        pending_id = uuid.uuid4().hex
        self._append(
            {
                "schema": "yuzhu-hindsight-pending-1",
                "status": "PENDING",
                "pending_id": pending_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "attempts": attempts,
                "error": error,
                "record": {
                    "content": record.content,
                    "project_id": record.project_id,
                    "category": record.category,
                    "scope": record.scope,
                    "verification": record.verification,
                    "evidence": list(record.evidence),
                    "source": record.source,
                    "session_id": record.session_id,
                    "memory_key": record.memory_key,
                    "occurred_at": record.occurred_at,
                    "tags": list(record.tags),
                    "metadata": dict(record.metadata),
                },
            }
        )
        return pending_id

    def _pending_records(self) -> list[dict[str, Any]]:
        if not self.pending_path.is_file():
            return []
        pending: dict[str, dict[str, Any]] = {}
        resolved: set[str] = set()
        for line in self.pending_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            pending_id = event.get("pending_id")
            if not pending_id:
                continue
            if event.get("status") == "PENDING" and isinstance(event.get("record"), dict):
                pending[pending_id] = event
            elif event.get("status") == "RESOLVED":
                resolved.add(pending_id)
        return [event for key, event in pending.items() if key not in resolved]

    def pending_count(self) -> int:
        return len(self._pending_records())

    def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._pending_records()[: max(0, limit)]

    def retry_pending(self, *, max_items: int = 20) -> dict[str, Any]:
        """Retry a bounded number of pending records once each.

        The caller decides when to run this; the agent loop never performs an
        unbounded retry loop.  The HTTP client still applies its own bounded
        retries and timeout.
        """
        if not self.enabled or self.service is None:
            return {
                "attempted": 0,
                "resolved": 0,
                "failed": 0,
                "pending_count": self.pending_count(),
                "reason": "memory writer or service is unavailable",
            }
        items = self.list_pending(limit=max_items)
        resolved = 0
        failed = 0
        for event in items:
            pending_id = str(event["pending_id"])
            payload = event["record"]
            record = MemoryRecord(
                content=str(payload["content"]),
                project_id=str(payload["project_id"]),
                category=str(payload["category"]),
                scope=str(payload.get("scope") or "project"),
                verification=str(payload.get("verification") or "verified"),
                evidence=tuple(payload.get("evidence") or ()),
                source=str(payload.get("source") or "agent-harness"),
                session_id=payload.get("session_id"),
                memory_key=payload.get("memory_key"),
                occurred_at=payload.get("occurred_at"),
                tags=tuple(payload.get("tags") or ()),
                metadata=dict(payload.get("metadata") or {}),
            )
            try:
                self.service.retain(record)
            except Exception as exc:  # noqa: BLE001 - keep pending and record the failure
                failed += 1
                self._append(
                    {
                        "status": "RETRY_FAILED",
                        "pending_id": pending_id,
                        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            else:
                resolved += 1
                self._append(
                    {
                        "status": "RESOLVED",
                        "pending_id": pending_id,
                        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                )
        return {
            "attempted": len(items),
            "resolved": resolved,
            "failed": failed,
            "pending_count": self.pending_count(),
        }