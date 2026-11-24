"""State-machine and control-flow tests for the agent loop (scripted model)."""

from __future__ import annotations

import json
import time
import unittest

from tests.agent_test_utils import (
    ScriptedModel,
    TempWorkspace,
    make_loop,
    text_response,
    tool_response,
)
from yuzhu_lab.harness.agent.errors import ModelUnavailable
from yuzhu_lab.harness.agent.loop import CancelToken, CancelledError
from yuzhu_lab.harness.agent.registry import ToolStatus
from yuzhu_lab.harness.agent.states import AgentState


class AgentLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.root = self.workspace.path

    def run_task(self, model, task="do the task", **kwargs):
        loop, store, audit = make_loop(self.root, model, **kwargs)
        self.addCleanup(store.close)
        result = loop.run(task)
        return result, loop, store, audit

    # -------------------------------------------------------------- happy paths

    def test_direct_final_answer(self) -> None:
        result, loop, store, _ = self.run_task(ScriptedModel([text_response("All done.")]))
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.final_text, "All done.")
        self.assertEqual(result.turns, 0)
        self.assertEqual(result.tool_calls, 0)
        task = store.get_task(loop.task_id)
        self.assertEqual(task["status"], "COMPLETED")
        self.assertEqual(task["state"], "COMPLETED")

    def test_native_tool_call_then_answer(self) -> None:
        (self.root / "a.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        model = ScriptedModel(
            [
                tool_response([("read_file", {"path": "a.py"})], text="reading"),
                text_response("The file defines add(a, b)."),
            ]
        )
        result, loop, store, _ = self.run_task(model)
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.tool_calls, 1)
        calls = store.tool_calls(loop.task_id)
        self.assertEqual(calls[0]["status"], "COMPLETED")
        self.assertEqual(calls[0]["origin"], "MODEL_SELECTED_TOOL")
        # The second model request must contain the tool result.
        second = model.calls[1]["messages"]
        self.assertEqual(second[-1]["role"], "user")
        tool_result = second[-1]["content"][0]
        self.assertEqual(tool_result["type"], "tool_result")
        self.assertEqual(tool_result["tool_use_id"], calls[0]["call_id"])
        self.assertIn("def add", tool_result["content"])

    def test_invalid_arguments_become_observation(self) -> None:
        model = ScriptedModel(
            [
                tool_response([("read_file", {"wrong": 1})]),
                text_response("I need the correct path argument."),
            ]
        )
        result, loop, store, _ = self.run_task(model)
        self.assertEqual(result.status, "COMPLETED")
        calls = store.tool_calls(loop.task_id)
        self.assertEqual(calls[0]["status"], "ERROR")
        self.assertEqual(calls[0]["result"]["error_code"], "INVALID_ARGUMENTS")
        tool_result = model.calls[1]["messages"][-1]["content"][0]
        self.assertIn("INVALID_ARGUMENTS", tool_result["content"])

    def test_unknown_tool_becomes_observation(self) -> None:
        model = ScriptedModel(
            [
                tool_response([("rm_rf", {"path": "/"})]),
                text_response("That tool does not exist."),
            ]
        )
        result, loop, store, _ = self.run_task(model)
        self.assertEqual(result.status, "COMPLETED")
        calls = store.tool_calls(loop.task_id)
        self.assertEqual(calls[0]["result"]["error_code"], "UNKNOWN_TOOL")

    def test_plain_text_with_function_syntax_is_not_a_tool_call(self) -> None:
        text = '<function name="read_file"><param name="path">a.py</param></function>'
        model = ScriptedModel([text_response(text), text_response("Recovered answer.")])
        result, loop, store, _ = self.run_task(model)
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(store.tool_calls(loop.task_id), [])
        self.assertEqual(result.final_text, "Recovered answer.")
        self.assertEqual(len(model.calls), 2)

    def test_repeated_unparsed_tool_syntax_fails(self) -> None:
        text = '<function name="read_file"><param name="path">a.py</param></function>'
        result, loop, store, _ = self.run_task(
            ScriptedModel([text_response(text), text_response(text)])
        )
        self.assertEqual(result.status, "FAILED")
        self.assertIn("unparsed tool syntax", result.error or "")

    def test_truncated_final_answer_is_retried(self) -> None:
        model = ScriptedModel(
            [
                text_response("partial answer", stop_reason="max_tokens"),
                text_response("complete answer"),
            ]
        )
        result, loop, store, _ = self.run_task(model)
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.final_text, "complete answer")
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(
            any(w["code"] == "MODEL_OUTPUT_TRUNCATED" for w in result.warnings)
        )

    def test_fallback_json_action(self) -> None:
        (self.root / "a.txt").write_text("fallback-content", encoding="utf-8")
        text = (
            "I will inspect the file.\n"
            '```yuzhu-action\n{"tool": "read_file", "arguments": {"path": "a.txt"}}\n```'
        )
        model = ScriptedModel(
            [text_response(text), text_response("It contains fallback-content.")]
        )
        result, loop, store, _ = self.run_task(model)
        self.assertEqual(result.status, "COMPLETED")
        calls = store.tool_calls(loop.task_id)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["origin"], "DETERMINISTIC_FALLBACK")
        self.assertNotIn("yuzhu-action", model.calls[1]["messages"][1]["content"][0]["text"])

    def test_empty_answer_is_retried_once(self) -> None:
        model = ScriptedModel(
            [text_response(""), text_response("Recovered answer.")]
        )
        result, _, _, _ = self.run_task(model)
        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.final_text, "Recovered answer.")

    # ------------------------------------------------------------------ limits

    def test_turn_limit_fails_without_infinite_loop(self) -> None:
        responses = [
            tool_response([("list_files", {"path": "."})]) for _ in range(10)
        ]
        result, loop, store, _ = self.run_task(
            ScriptedModel(responses),
            max_turns=2,
        )
        self.assertEqual(result.status, "FAILED")
        self.assertIn("maximum turn limit", result.error or "")
        self.assertEqual(store.get_task(loop.task_id)["status"], "FAILED")

    def test_tool_call_budget_skips_execution(self) -> None:
        responses = [
            tool_response([("list_files", {"path": "."})]),
            tool_response([("list_files", {"path": ".", "max_entries": 5})]),
            text_response("done"),
        ]
        result, loop, store, _ = self.run_task(
            ScriptedModel(responses),
            max_tool_calls=1,
        )
        self.assertEqual(result.status, "COMPLETED")
        calls = store.tool_calls(loop.task_id)
        self.assertEqual(calls[0]["status"], "COMPLETED")
        self.assertEqual(calls[1]["result"]["error_code"], "TOOL_BUDGET_EXHAUSTED")

    def test_duplicate_side_effect_blocked(self) -> None:
        responses = [
            tool_response([("write_file", {"path": "x.txt", "content": "one"})]),
            tool_response([("write_file", {"path": "x.txt", "content": "one"})]),
            text_response("done"),
        ]
        model = ScriptedModel(
            responses
        )
        result, loop, store, _ = self.run_task(model)
        self.assertEqual(result.status, "COMPLETED")
        calls = store.tool_calls(loop.task_id)
        self.assertEqual(calls[0]["status"], "COMPLETED")
        self.assertEqual(calls[1]["result"]["error_code"], "DUPLICATE_CALL_BLOCKED")
        self.assertEqual((self.root / "x.txt").read_text(encoding="utf-8"), "one")

    def test_duplicate_read_only_reuses_result(self) -> None:
        (self.root / "a.txt").write_text("cached", encoding="utf-8")
        responses = [
            tool_response([("read_file", {"path": "a.txt"})]),
            tool_response([("read_file", {"path": "a.txt"})]),
            text_response("done"),
        ]
        result, loop, store, _ = self.run_task(
            ScriptedModel(responses)
        )
        self.assertEqual(result.status, "COMPLETED")
        calls = store.tool_calls(loop.task_id)
        self.assertEqual(calls[1]["status"], "DUPLICATE")
        self.assertIn("cached", calls[1]["result"]["content"])

    def test_execution_denied_is_observation(self) -> None:
        (self.root / "ok.py").write_text("print(1)\n", encoding="utf-8")
        responses = [
            tool_response([("run_python", {"script": "ok.py"})]),
            text_response("Execution is disabled."),
        ]
        result, loop, store, _ = self.run_task(
            ScriptedModel(responses)
        )
        self.assertEqual(result.status, "COMPLETED")
        calls = store.tool_calls(loop.task_id)
        self.assertEqual(calls[0]["status"], "DENIED")
        self.assertEqual(calls[0]["result"]["error_code"], "PERMISSION_DENIED")

    # ------------------------------------------------------------ cancellation

    def test_cancel_before_run(self) -> None:
        token = CancelToken()
        token.cancel("test")
        result, loop, store, _ = self.run_task(
            ScriptedModel(
                [text_response("never")]
            ),
            cancel_token=token,
        )
        self.assertEqual(result.status, "CANCELLED")
        self.assertEqual(store.get_task(loop.task_id)["status"], "CANCELLED")

    def test_cancel_during_tool_execution(self) -> None:
        (self.root / "sleep.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
        token = CancelToken()

        import threading

        def cancel_later() -> None:
            time.sleep(0.6)
            token.cancel("user")

        threading.Thread(target=cancel_later, daemon=True).start()
        responses = [tool_response([("run_python", {"script": "sleep.py"})])]
        result, loop, store, _ = self.run_task(
            ScriptedModel(responses),
            allow_execution=True,
            cancel_token=token,
        )
        self.assertEqual(result.status, "CANCELLED")
        call = store.tool_calls(loop.task_id)[0]
        self.assertEqual(call["status"], "CANCELLED")
        self.assertEqual(store.get_task(loop.task_id)["status"], "CANCELLED")

    # ----------------------------------------------------------------- errors

    def test_model_unavailable_fails_cleanly(self) -> None:
        result, loop, store, _ = self.run_task(
            ScriptedModel(
                [ModelUnavailable("down")]
            ),
            max_model_retries=0,
        )
        self.assertEqual(result.status, "FAILED")
        self.assertIn("model request failed", result.error or "")
        self.assertEqual(store.get_task(loop.task_id)["status"], "FAILED")

    def test_events_record_state_transitions(self) -> None:
        result, loop, store, _ = self.run_task(ScriptedModel([text_response("ok")]))
        events = store.events(loop.task_id)
        states = [event["state"] for event in events if event["kind"] == "STATE"]
        self.assertEqual(states[0], "ACCEPTING_TASK")
        self.assertIn("BUILDING_CONTEXT", states)
        self.assertIn("REQUESTING_MODEL", states)
        self.assertIn("COMPLETED", states)

    def test_audit_log_written(self) -> None:
        result, loop, store, audit = self.run_task(ScriptedModel([text_response("ok")]))
        self.assertTrue(audit.path.is_file())
        lines = audit.path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreaterEqual(len(lines), 1)
        self.assertIn("agent_event", lines[0])


if __name__ == "__main__":
    unittest.main()