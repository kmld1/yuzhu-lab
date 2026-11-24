"""Append-only, redacted agent audit log.

File contents, tool observations and model text never enter this log.  Only
identifiers, digests, lengths, permission decisions and exit codes are stored.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class AgentAuditLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, action: str, **fields: Any) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "action": action,
            **fields,
        }
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def __call__(self, payload: dict[str, Any]) -> None:
        action = str(payload.get("kind") or "tool_call")
        self.record(action, **{key: value for key, value in payload.items() if key != "kind"})

    def event(
        self,
        *,
        task_id: str,
        session_id: str,
        turn: int,
        state: str,
        kind: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.record(
            "agent_event",
            task_id=task_id,
            session_id=session_id,
            turn=turn,
            state=state,
            event=kind,
            payload=payload or {},
        )