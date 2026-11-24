"""SQLite checkpoint persistence and resume-token tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from yuzhu_lab.harness.agent.checkpoint import CheckpointStore
from yuzhu_lab.harness.agent.errors import CheckpointError
from yuzhu_lab.harness.agent.registry import ToolOrigin, ToolRequest


class CheckpointStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="yuzhu-ckpt-")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "state.db"
        self.store = CheckpointStore(self.path)
        self.addCleanup(lambda: self.store.close())

    def create(self, task_id: str = "task-1") -> None:
        self.store.create_task(
            task_id=task_id,
            session_id="session-1",
            project_id="qianyuqwen",
            workspace=Path(self.temp.name),
            task_description="fix the bug",
        )

    def test_create_and_load_task(self) -> None:
        self.create()
        task = self.store.get_task("task-1")
        self.assertEqual(task["status"], "RUNNING")
        self.assertEqual(task["state"], "ACCEPTING_TASK")
        self.assertEqual(task["task_description"], "fix the bug")
        self.assertEqual(task["messages"], [])

    def test_missing_task_raises(self) -> None:
        with self.assertRaises(CheckpointError):
            self.store.get_task("missing")

    def test_save_task_messages_and_context(self) -> None:
        self.create()
        messages = [{"role": "user", "content": "hello"}]
        self.store.save_task(
            "task-1",
            state="REQUESTING_MODEL",
            turn=2,
            messages=messages,
            context={"system_prompt": "sys"},
            retry_count=1,
            model_request_id="msg-123",
            model_response_chars=42,
            context_refs=[{"kind": "rag", "identifier": "chunk-1"}],
            memory_refs=["mem-1"],
            rag_refs=["src/app.py:1-2"],
        )
        task = self.store.get_task("task-1")
        self.assertEqual(task["state"], "REQUESTING_MODEL")
        self.assertEqual(task["turn"], 2)
        self.assertEqual(task["retry_count"], 1)
        self.assertEqual(task["messages"], messages)
        self.assertEqual(task["context"]["system_prompt"], "sys")
        self.assertEqual(task["model_request_id"], "msg-123")
        self.assertEqual(task["model_response_chars"], 42)
        self.assertEqual(task["memory_refs"], ["mem-1"])
        self.assertEqual(task["rag_refs"], ["src/app.py:1-2"])
        self.assertEqual(task["context_refs"][0]["identifier"], "chunk-1")

    def test_old_schema_is_migrated(self) -> None:
        import sqlite3

        path = Path(self.temp.name) / "old.db"
        connection = sqlite3.connect(path)
        connection.executescript(
            """
            CREATE TABLE tasks (
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
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        connection.commit()
        connection.close()
        store = CheckpointStore(path)
        try:
            store.create_task(
                task_id="task-old",
                session_id="s",
                project_id="p",
                workspace=Path(self.temp.name),
                task_description="old",
            )
            store.save_task(
                "task-old",
                state="REQUESTING_MODEL",
                turn=1,
                model_request_id="msg-old",
                memory_refs=["m1"],
            )
            task = store.get_task("task-old")
            self.assertEqual(task["model_request_id"], "msg-old")
            self.assertEqual(task["memory_refs"], ["m1"])
            self.assertEqual(task["rag_refs"], [])
        finally:
            store.close()

    def test_tool_call_lifecycle(self) -> None:
        self.create()
        request = ToolRequest(
            call_id="call-1",
            name="read_file",
            arguments={"path": "a.py"},
            origin=ToolOrigin.MODEL_SELECTED_TOOL,
            turn=0,
        )
        self.store.record_tool_call("task-1", request, idempotent=True)
        call = self.store.tool_calls("task-1")[0]
        self.assertEqual(call["status"], "PENDING")
        self.assertTrue(call["idempotent"])
        self.store.mark_tool_started("call-1")
        self.assertEqual(self.store.tool_calls("task-1")[0]["status"], "STARTED")
        self.store.finish_tool_call(
            "call-1", "COMPLETED", {"content": "ok", "status": "OK"}
        )
        call = self.store.tool_calls("task-1")[0]
        self.assertEqual(call["status"], "COMPLETED")
        self.assertEqual(call["result"]["content"], "ok")
        self.assertIsNotNone(call["finished_at"])

    def test_events_are_ordered(self) -> None:
        self.create()
        self.store.append_event("task-1", turn=0, state="IDLE", kind="A", payload={})
        self.store.append_event("task-1", turn=1, state="IDLE", kind="B", payload={})
        events = self.store.events("task-1")
        self.assertEqual([event["kind"] for event in events], ["A", "B"])

    def test_find_incomplete(self) -> None:
        self.create("task-1")
        self.create("task-2")
        self.store.save_task("task-1", state="COMPLETED", turn=1, status="COMPLETED")
        found = self.store.find_incomplete("session-1")
        self.assertEqual(found["task_id"], "task-2")

    def test_persistence_across_reopen(self) -> None:
        self.create()
        self.store.save_task("task-1", state="EXECUTING_TOOL", turn=3)
        self.store.close()
        self.store = CheckpointStore(self.path)
        task = self.store.get_task("task-1")
        self.assertEqual(task["state"], "EXECUTING_TOOL")
        self.assertEqual(task["turn"], 3)

    def test_list_tasks(self) -> None:
        self.create("task-1")
        self.create("task-2")
        tasks = self.store.list_tasks("session-1")
        self.assertEqual({task["task_id"] for task in tasks}, {"task-1", "task-2"})

    def test_multi_session_isolation(self) -> None:
        self.create("task-a")
        self.store.create_task(
            task_id="task-b",
            session_id="session-2",
            project_id="other-project",
            workspace=Path(self.temp.name),
            task_description="other",
        )
        session_a = {task["task_id"] for task in self.store.list_tasks("session-1")}
        session_b = {task["task_id"] for task in self.store.list_tasks("session-2")}
        self.assertEqual(session_a, {"task-a"})
        self.assertEqual(session_b, {"task-b"})
        self.assertEqual(self.store.find_incomplete("session-2")["task_id"], "task-b")

    def test_wal_checkpoint_and_integrity(self) -> None:
        self.create()
        self.store.save_task("task-1", state="REQUESTING_MODEL", turn=1)
        self.assertTrue(self.store.integrity_check()["ok"])
        self.store.wal_checkpoint()
        self.store.close()
        self.store = CheckpointStore(self.path)
        self.assertTrue(self.store.integrity_check()["ok"])
        self.assertEqual(self.store.get_task("task-1")["turn"], 1)

    def test_corrupt_database_raises_checkpoint_error(self) -> None:
        path = Path(self.temp.name) / "corrupt.db"
        path.write_bytes(b"not-a-sqlite-database" * 100)
        with self.assertRaises(CheckpointError):
            CheckpointStore(path)


if __name__ == "__main__":
    unittest.main()