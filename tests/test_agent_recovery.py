"""Crash, checkpoint and resume behaviour for side-effecting tools."""

from __future__ import annotations

import unittest

from tests.agent_test_utils import (
    ScriptedModel,
    TempWorkspace,
    make_loop,
    text_response,
    tool_response,
)
from yuzhu_lab.harness.agent.permissions import PermissionLevel
from yuzhu_lab.harness.agent.registry import ToolDefinition, ToolRequest


class AgentRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.root = self.workspace.path

    def test_crash_during_side_effect_not_rerun_on_resume(self) -> None:
        model = ScriptedModel(
            [tool_response([("crash_write", {"path": "x.txt", "content": "v1"})])]
        )
        loop, store, _ = make_loop(self.root, model, task_id="crash-task")

        def crash_write(ctx, arguments):
            path = ctx.guard.resolve(arguments["path"], for_write=True)
            path.write_text(arguments["content"], encoding="utf-8")
            raise SystemExit("simulated process crash")

        loop.registry.register(
            ToolDefinition(
                name="crash_write",
                description="test crash tool",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                execute=crash_write,
                permission_level=PermissionLevel.WRITE_WORKSPACE,
                idempotent=False,
            )
        )
        with self.assertRaises(SystemExit):
            loop.run("write x.txt then crash")
        self.assertEqual((self.root / "x.txt").read_text(encoding="utf-8"), "v1")
        task = store.get_task("crash-task")
        self.assertIn(task["state"], {"EXECUTING_TOOL", "OBSERVING_RESULT"})
        call = store.tool_calls("crash-task")[0]
        self.assertEqual(call["status"], "STARTED")
        store.close()

        model2 = ScriptedModel([text_response("Verified state and finished.")])
        loop2, store2, _ = make_loop(self.root, model2, task_id="crash-task")
        self.addCleanup(store2.close)
        result = loop2.run("", resume=True)
        self.assertEqual(result.status, "COMPLETED", result.error)
        # The side effect must not run twice.
        self.assertEqual((self.root / "x.txt").read_text(encoding="utf-8"), "v1")
        call = store2.tool_calls("crash-task")[0]
        self.assertEqual(call["status"], "UNKNOWN")
        self.assertEqual(call["result"]["error_code"], "INTERRUPTED_SIDE_EFFECT")
        observation = model2.calls[0]["messages"][-1]["content"][0]["content"]
        self.assertIn("INTERRUPTED_SIDE_EFFECT", observation)
        self.assertIn("NOT re-run", observation)

    def test_started_idempotent_tool_is_rerun(self) -> None:
        (self.root / "a.txt").write_text("actual-file-content", encoding="utf-8")
        task_id = "idem-task"
        loop, store, _ = make_loop(self.root, ScriptedModel([]), task_id=task_id)
        store.create_task(
            task_id=task_id,
            session_id="session-test",
            project_id="qianyuqwen",
            workspace=self.root,
            task_description="read a.txt",
        )
        messages = [
            {"role": "user", "content": "read a.txt"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-x",
                        "name": "read_file",
                        "input": {"path": "a.txt"},
                    }
                ],
            },
        ]
        store.save_task(task_id, state="EXECUTING_TOOL", turn=0, messages=messages)
        store.record_tool_call(
            task_id,
            ToolRequest(call_id="call-x", name="read_file", arguments={"path": "a.txt"}),
            idempotent=True,
        )
        store.mark_tool_started("call-x")
        store.close()

        model = ScriptedModel([text_response("read done")])
        loop2, store2, _ = make_loop(self.root, model, task_id=task_id)
        self.addCleanup(store2.close)
        result = loop2.run("", resume=True)
        self.assertEqual(result.status, "COMPLETED", result.error)
        call = store2.tool_calls(task_id)[0]
        self.assertEqual(call["status"], "COMPLETED")
        self.assertIn("actual-file-content", call["result"]["content"])
        observation = model.calls[0]["messages"][-1]["content"][0]["content"]
        self.assertIn("actual-file-content", observation)

    def test_completed_result_is_reused_not_re_executed(self) -> None:
        (self.root / "a.txt").write_text("ACTUAL-CURRENT-CONTENT", encoding="utf-8")
        task_id = "reuse-task"
        _, store, _ = make_loop(self.root, ScriptedModel([]), task_id=task_id)
        store.create_task(
            task_id=task_id,
            session_id="session-test",
            project_id="qianyuqwen",
            workspace=self.root,
            task_description="read a.txt",
        )
        messages = [
            {"role": "user", "content": "read a.txt"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-x",
                        "name": "read_file",
                        "input": {"path": "a.txt"},
                    }
                ],
            },
        ]
        store.save_task(task_id, state="EXECUTING_TOOL", turn=0, messages=messages)
        store.record_tool_call(
            task_id,
            ToolRequest(call_id="call-x", name="read_file", arguments={"path": "a.txt"}),
            idempotent=True,
        )
        store.finish_tool_call(
            "call-x",
            "COMPLETED",
            {
                "call_id": "call-x",
                "name": "read_file",
                "status": "OK",
                "content": "STORED-RESULT",
                "structured": {},
            },
        )
        store.close()

        model = ScriptedModel([text_response("done")])
        loop2, store2, _ = make_loop(self.root, model, task_id=task_id)
        self.addCleanup(store2.close)
        result = loop2.run("", resume=True)
        self.assertEqual(result.status, "COMPLETED", result.error)
        observation = model.calls[0]["messages"][-1]["content"][0]["content"]
        self.assertIn("STORED-RESULT", observation)
        self.assertNotIn("ACTUAL-CURRENT-CONTENT", observation)

    def test_resume_terminal_task_does_not_call_model(self) -> None:
        task_id = "done-task"
        _, store, _ = make_loop(self.root, ScriptedModel([]), task_id=task_id)
        store.create_task(
            task_id=task_id,
            session_id="session-test",
            project_id="qianyuqwen",
            workspace=self.root,
            task_description="done",
        )
        store.save_task(
            task_id,
            state="COMPLETED",
            turn=1,
            messages=[{"role": "user", "content": "done"}],
            status="COMPLETED",
            final_text="already done",
        )
        store.close()
        model = ScriptedModel([])
        loop2, store2, _ = make_loop(self.root, model, task_id=task_id)
        self.addCleanup(store2.close)
        result = loop2.run("", resume=True)
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.final_text, "already done")
        self.assertEqual(model.calls, [])

    def test_missing_task_fails_cleanly(self) -> None:
        model = ScriptedModel([])
        loop, store, _ = make_loop(self.root, model, task_id="ghost-task")
        self.addCleanup(store.close)
        result = loop.run("", resume=True)
        self.assertEqual(result.status, "FAILED")
        self.assertIn("task not found", result.error or "")


if __name__ == "__main__":
    unittest.main()