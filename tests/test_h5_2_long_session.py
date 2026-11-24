"""Deterministic H5.2 long-session, resume, truncation and cancellation tests."""

from __future__ import annotations

import unittest

from tests.agent_test_utils import (
    ScriptedModel,
    TempWorkspace,
    make_loop,
    text_response,
    tool_response,
)
from yuzhu_lab.harness.agent.loop import CancelToken
from yuzhu_lab.harness.agent.model_client import ModelResponse


class CancellingModel:
    """Returns canned responses and cancels the task on a chosen call."""

    def __init__(self, responses, token: CancelToken, cancel_on_call: int) -> None:
        self.responses = list(responses)
        self.token = token
        self.cancel_on_call = cancel_on_call
        self.calls = 0

    def complete(self, *, system, messages, tools, max_tokens, temperature=0.0):
        self.calls += 1
        if self.calls >= self.cancel_on_call:
            self.token.cancel("test-cancel")
        if not self.responses:
            raise AssertionError("CancellingModel ran out of responses")
        return self.responses.pop(0)


def _write_response(path: str, content: str) -> ModelResponse:
    return tool_response([("write_file", {"path": path, "content": content})])


class CrashingModel:
    """Simulates a hard process interruption between turns (BaseException)."""

    def __init__(self, responses, crash_on_call: int) -> None:
        self.responses = list(responses)
        self.crash_on_call = crash_on_call
        self.calls = 0

    def complete(self, *, system, messages, tools, max_tokens, temperature=0.0):
        self.calls += 1
        if self.calls >= self.crash_on_call:
            raise KeyboardInterrupt("simulated process interruption")
        return self.responses.pop(0)


class LongSessionTest(unittest.TestCase):
    def test_sequential_tasks_share_session_history(self) -> None:
        with TempWorkspace() as workspace:
            first, store, _ = make_loop(
                workspace.path,
                ScriptedModel(
                    [
                        _write_response("hello.py", "print('H5_2_LONG_SESSION')\n"),
                        text_response("created hello.py"),
                    ]
                ),
                task_id="task-a",
            )
            try:
                result_a = first.run("create hello.py")
                self.assertEqual(result_a.status, "COMPLETED")
            finally:
                store.close()

            second, store, _ = make_loop(
                workspace.path,
                ScriptedModel([text_response("hello.py prints the marker")]),
                task_id="task-b",
            )
            try:
                result_b = second.run("explain hello.py")
                self.assertEqual(result_b.status, "COMPLETED")
                tasks = store.list_tasks("session-test")
                self.assertEqual({task["task_id"] for task in tasks}, {"task-a", "task-b"})
                self.assertTrue((workspace.path / "hello.py").is_file())
            finally:
                store.close()

    def test_resume_after_interruption_does_not_repeat_side_effect(self) -> None:
        with TempWorkspace() as workspace:
            model = CrashingModel(
                [_write_response("state.txt", "H5_2_STATE\n")], crash_on_call=2
            )
            loop, store, _ = make_loop(
                workspace.path, model, task_id="task-resume"
            )
            try:
                with self.assertRaises(KeyboardInterrupt):
                    loop.run("write state.txt")
                state = store.get_task("task-resume")["state"]
                self.assertNotIn(state, {"COMPLETED", "FAILED", "CANCELLED"})
                self.assertEqual(len(store.tool_calls("task-resume")), 1)
            finally:
                store.close()

            resumed, store, _ = make_loop(
                workspace.path,
                ScriptedModel([text_response("resumed and finished")]),
                task_id="task-resume",
            )
            try:
                second = resumed.run("", resume=True)
                self.assertEqual(second.status, "COMPLETED")
                calls = [
                    call
                    for call in store.tool_calls("task-resume")
                    if call["name"] == "write_file"
                ]
                self.assertEqual(len(calls), 1)
                self.assertTrue((workspace.path / "state.txt").is_file())
            finally:
                store.close()

    def test_truncation_blocks_completion(self) -> None:
        with TempWorkspace() as workspace:
            loop, store, _ = make_loop(
                workspace.path,
                ScriptedModel(
                    [
                        text_response("partial answer", stop_reason="max_tokens"),
                        text_response("still partial", stop_reason="max_tokens"),
                    ]
                ),
                task_id="task-truncated",
            )
            try:
                result = loop.run("answer fully")
                self.assertEqual(result.status, "FAILED")
                self.assertIsNotNone(result.completion)
                self.assertEqual(result.completion["level"], "MODEL_TRUNCATED")
                self.assertIn("MODEL_TRUNCATED", result.completion["blockers"])
            finally:
                store.close()

    def test_cancellation_returns_cancelled(self) -> None:
        with TempWorkspace() as workspace:
            token = CancelToken()
            model = CancellingModel([text_response("never used")], token, cancel_on_call=1)
            loop, store, _ = make_loop(
                workspace.path, model, task_id="task-cancel", cancel_token=token
            )
            try:
                result = loop.run("long task")
                self.assertEqual(result.status, "CANCELLED")
                self.assertEqual(result.error, "test-cancel")
            finally:
                store.close()

    def test_tool_allow_list_filters_registry(self) -> None:
        from yuzhu_lab.harness.agent.factory import (
            AgentFactoryConfig,
            build_agent_runtime,
        )

        with TempWorkspace() as workspace:
            state = workspace.path / ".agent"
            state.mkdir()
            config = AgentFactoryConfig(
                workspace=workspace.path,
                project_id="qianyuqwen",
                session_id="s",
                task_id="task-tools",
                checkpoint_db=state / "state.db",
                audit_log=state / "audit.jsonl",
                use_memory=False,
                use_rag=False,
                allowed_tools=("read_file", "search_text"),
            )
            runtime = build_agent_runtime(config, model=ScriptedModel([]))
            try:
                self.assertEqual(set(runtime.registry.names()), {"read_file", "search_text"})
            finally:
                runtime.close()
            bad = AgentFactoryConfig(
                workspace=workspace.path,
                project_id="qianyuqwen",
                session_id="s2",
                task_id="task-tools-bad",
                checkpoint_db=state / "state2.db",
                audit_log=state / "audit2.jsonl",
                use_memory=False,
                use_rag=False,
                allowed_tools=("not_a_tool",),
            )
            with self.assertRaises(ValueError):
                build_agent_runtime(bad, model=ScriptedModel([]))

    def test_context_budget_truncates_oversized_items(self) -> None:
        from yuzhu_lab.harness.context_builder import ContextBuilder, ContextBuilderConfig
        from yuzhu_lab.harness.context_models import (
            ContextRequest,
            MemoryContextItem,
            RagEvidenceItem,
        )
        from yuzhu_lab.harness.memory_adapter import MemorySearchOutcome
        from yuzhu_lab.harness.rag_adapter import RagSearchOutcome

        class Memory:
            def recall(self, *, project_id, query, max_tokens, include_unverified=False):
                return MemorySearchOutcome(
                    items=[
                        MemoryContextItem(
                            memory_id=f"m{index}",
                            text="memory " + "x" * 500,
                            category="test",
                            verification="verified",
                            confidence=0.9,
                            source="test",
                            occurred_at=None,
                            bank_id="bank",
                            trust="MEMORY_VERIFIED",
                        )
                        for index in range(10)
                    ]
                )

        class Rag:
            def search(self, *, project_id, query, token_budget, final_k=5):
                return RagSearchOutcome(
                    items=[
                        RagEvidenceItem(
                            chunk_id=f"c{index}",
                            citation=f"file.py:{index}-{index + 5}",
                            source="file.py",
                            symbol=None,
                            start_line=index,
                            end_line=index + 5,
                            content="evidence " + "y" * 500,
                            score=0.1,
                            rrf_score=0.1,
                        )
                        for index in range(10)
                    ]
                )

        builder = ContextBuilder(
            memory=Memory(),
            rag=Rag(),
            config=ContextBuilderConfig(workspace=None, default_budget=600),
        )
        bundle = builder.build(
            ContextRequest(
                user_message="question",
                session_id="s",
                project_id="qianyuqwen",
                model_context_budget=600,
            )
        )
        self.assertTrue(bundle.token_estimate.truncated)
        self.assertLessEqual(bundle.token_estimate.total, 600)


if __name__ == "__main__":
    unittest.main()