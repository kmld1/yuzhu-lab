"""SQLite-backed task checkpoints.

Durability rules:
- A tool call is marked STARTED and committed *before* it runs.
- A side-effecting call found STARTED after a crash is never re-run
  automatically; the loop receives an UNKNOWN observation instead.
- Messages are only persisted after their tool results have been recorded.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import CheckpointError
from .registry import ToolRequest

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    workspace TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    workspace TEXT NOT NULL,
    task_description TEXT NOT NULL,
    status TEXT NOT NULL,
    state TEXT NOT NULL,
    turn INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    messages_json TEXT NOT NULL DEFAULT '[]',
    context_json TEXT NOT NULL DEFAULT '{}',
    final_text TEXT,
    last_error TEXT,
    model_request_id TEXT,
    model_response_chars INTEGER,
    context_refs_json TEXT,
    memory_refs_json TEXT,
    rag_refs_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tool_calls (
    call_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    origin TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT,
    idempotent INTEGER NOT NULL DEFAULT 1,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_task ON tool_calls(task_id, turn);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    ts TEXT NOT NULL,
    state TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id, event_id);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


_TASK_MIGRATIONS = (
    ("model_request_id", "TEXT"),
    ("model_response_chars", "INTEGER"),
    ("context_refs_json", "TEXT"),
    ("memory_refs_json", "TEXT"),
    ("rag_refs_json", "TEXT"),
)


class CheckpointStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        try:
            self._connection = sqlite3.connect(
                str(self.path), check_same_thread=False, timeout=30.0
            )
            self._connection.row_factory = sqlite3.Row
            with self._lock:
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute("PRAGMA synchronous=FULL")
                self._connection.executescript(SCHEMA)
                self._migrate_tasks_table()
                self._connection.commit()
        except sqlite3.DatabaseError as exc:
            try:
                self._connection.close()
            except Exception:  # noqa: BLE001 - already unusable
                pass
            raise CheckpointError(
                f"checkpoint database is not usable: {self.path}: {exc}"
            ) from exc

    def _migrate_tasks_table(self) -> None:
        columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(tasks)").fetchall()
        }
        for name, sql_type in _TASK_MIGRATIONS:
            if name not in columns:
                self._connection.execute(
                    f"ALTER TABLE tasks ADD COLUMN {name} {sql_type}"
                )

    def integrity_check(self) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute("PRAGMA integrity_check").fetchone()
        return {"ok": row is not None and row[0] == "ok", "result": row[0] if row else None}

    def wal_checkpoint(self) -> None:
        """Fold the WAL into the main database; used by recovery verification."""
        with self._lock:
            self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ tasks

    def create_task(
        self,
        *,
        task_id: str,
        session_id: str,
        project_id: str,
        workspace: Path,
        task_description: str,
    ) -> None:
        now = utc_now()
        with self._lock:
            self._connection.execute(
                "INSERT OR IGNORE INTO sessions(session_id, project_id, workspace, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, project_id, str(workspace), now, now),
            )
            self._connection.execute(
                "INSERT INTO tasks(task_id, session_id, project_id, workspace, task_description, "
                "status, state, turn, retry_count, messages_json, context_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'RUNNING', 'ACCEPTING_TASK', 0, 0, '[]', '{}', ?, ?)",
                (
                    task_id,
                    session_id,
                    project_id,
                    str(workspace),
                    task_description,
                    now,
                    now,
                ),
            )
            self._connection.commit()

    def task_exists(self, task_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return row is not None

    def get_task(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            raise CheckpointError(f"task not found: {task_id}")
        data = dict(row)
        return {
            **data,
            "messages": json.loads(data.get("messages_json") or "[]"),
            "context": json.loads(data.get("context_json") or "{}"),
            "context_refs": json.loads(data.get("context_refs_json") or "[]"),
            "memory_refs": json.loads(data.get("memory_refs_json") or "[]"),
            "rag_refs": json.loads(data.get("rag_refs_json") or "[]"),
        }

    def list_tasks(self, session_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        query = "SELECT task_id, session_id, project_id, status, state, turn, task_description, updated_at FROM tasks"
        params: list[Any] = []
        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def find_incomplete(self, session_id: str | None = None) -> dict[str, Any] | None:
        query = "SELECT * FROM tasks WHERE status = 'RUNNING'"
        params: list[Any] = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY updated_at DESC LIMIT 1"
        with self._lock:
            row = self._connection.execute(query, params).fetchone()
        return dict(row) if row else None

    def save_task(
        self,
        task_id: str,
        *,
        state: str,
        turn: int,
        messages: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        retry_count: int | None = None,
        status: str | None = None,
        final_text: str | None = None,
        last_error: str | None = None,
        model_request_id: str | None = None,
        model_response_chars: int | None = None,
        context_refs: list[dict[str, Any]] | None = None,
        memory_refs: list[str] | None = None,
        rag_refs: list[str] | None = None,
    ) -> None:
        fields = ["state = ?", "turn = ?", "updated_at = ?"]
        values: list[Any] = [state, turn, utc_now()]
        if messages is not None:
            fields.append("messages_json = ?")
            values.append(json.dumps(messages, ensure_ascii=False, default=str))
        if context is not None:
            fields.append("context_json = ?")
            values.append(json.dumps(context, ensure_ascii=False, default=str))
        if retry_count is not None:
            fields.append("retry_count = ?")
            values.append(retry_count)
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if final_text is not None:
            fields.append("final_text = ?")
            values.append(final_text)
        if last_error is not None:
            fields.append("last_error = ?")
            values.append(last_error)
        if model_request_id is not None:
            fields.append("model_request_id = ?")
            values.append(model_request_id)
        if model_response_chars is not None:
            fields.append("model_response_chars = ?")
            values.append(model_response_chars)
        if context_refs is not None:
            fields.append("context_refs_json = ?")
            values.append(json.dumps(context_refs, ensure_ascii=False, default=str))
        if memory_refs is not None:
            fields.append("memory_refs_json = ?")
            values.append(json.dumps(memory_refs, ensure_ascii=False, default=str))
        if rag_refs is not None:
            fields.append("rag_refs_json = ?")
            values.append(json.dumps(rag_refs, ensure_ascii=False, default=str))
        values.append(task_id)
        with self._lock:
            self._connection.execute(
                f"UPDATE tasks SET {', '.join(fields)} WHERE task_id = ?", values
            )
            self._connection.commit()

    # -------------------------------------------------------------- tool calls

    def record_tool_call(
        self, task_id: str, request: ToolRequest, *, idempotent: bool
    ) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO tool_calls(call_id, task_id, turn, name, arguments_json, "
                "origin, status, result_json, idempotent) VALUES (?, ?, ?, ?, ?, ?, 'PENDING', NULL, ?)",
                (
                    request.call_id,
                    task_id,
                    request.turn,
                    request.name,
                    json.dumps(request.arguments, ensure_ascii=False, default=str),
                    request.origin.value,
                    1 if idempotent else 0,
                ),
            )
            self._connection.commit()

    def mark_tool_started(self, call_id: str) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE tool_calls SET status = 'STARTED', started_at = ? WHERE call_id = ?",
                (utc_now(), call_id),
            )
            self._connection.commit()

    def finish_tool_call(self, call_id: str, status: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE tool_calls SET status = ?, result_json = ?, finished_at = ? WHERE call_id = ?",
                (
                    status,
                    json.dumps(result, ensure_ascii=False, default=str),
                    utc_now(),
                    call_id,
                ),
            )
            self._connection.commit()

    def tool_calls(self, task_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM tool_calls WHERE task_id = ? ORDER BY rowid", (task_id,)
            ).fetchall()
        calls = []
        for row in rows:
            item = dict(row)
            item["arguments"] = json.loads(item.pop("arguments_json") or "{}")
            result_json = item.pop("result_json")
            item["result"] = json.loads(result_json) if result_json else None
            item["idempotent"] = bool(item["idempotent"])
            calls.append(item)
        return calls

    def turn_tool_calls(self, task_id: str, turn: int) -> list[dict[str, Any]]:
        return [call for call in self.tool_calls(task_id) if call["turn"] == turn]

    def calls_for_turn(self, task_id: str, turn: int) -> list[dict[str, Any]]:
        return self.turn_tool_calls(task_id, turn)

    # ----------------------------------------------------------------- events

    def append_event(
        self,
        task_id: str,
        *,
        turn: int,
        state: str,
        kind: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT INTO events(task_id, turn, ts, state, kind, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    turn,
                    utc_now(),
                    state,
                    kind,
                    json.dumps(payload or {}, ensure_ascii=False, default=str),
                ),
            )
            self._connection.commit()

    def events(self, task_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM events WHERE task_id = ? ORDER BY event_id", (task_id,)
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json") or "{}")
            items.append(item)
        return items