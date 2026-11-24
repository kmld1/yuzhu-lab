"""Regression tests for whole-request prompt budgeting and 413 prevention."""

from __future__ import annotations

import asyncio
import json
import unittest

from tests.agent_test_utils import (
    ScriptedModel,
    TempWorkspace,
    make_loop,
    text_response,
    tool_response,
)
from yuzhu_lab.harness.agent.errors import ModelUnavailable
from yuzhu_lab.harness.agent.model_client import InputLimits
from yuzhu_lab.harness.context_builder import ContextBuilder, ContextBuilderConfig
from yuzhu_lab.harness.context_models import (
    ContextRequest,
    MemoryContextItem,
    RagEvidenceItem,
)
from yuzhu_lab.harness.memory_adapter import MemorySearchOutcome
from yuzhu_lab.harness.rag_adapter import RagSearchOutcome
from yuzhu_lab.runtime.engine import CancellationToken

try:  # The qwen35 environment intentionally does not install fastapi.
    from yuzhu_lab.runtime.server import _CancellableStreamingResponse

    _HAS_FASTAPI = True
except Exception:  # noqa: BLE001 - runtime tests are skipped, not failed
    _CancellableStreamingResponse = None  # type: ignore[assignment]
    _HAS_FASTAPI = False


class CountingModel:
    """Deterministic token model: exact-count contract plus scripted answers."""

    def __init__(self, responses, *, reject_tokens: int) -> None:
        self.responses = list(responses)
        self.reject_tokens = reject_tokens
        self.calls: list[dict] = []
        self.prompt_tokens_seen: list[int] = []

    def input_limits(self) -> InputLimits:
        return InputLimits(
            prompt_reject_tokens=self.reject_tokens,
            prompt_warning_tokens=max(1, self.reject_tokens - 100),
            model_context_limit=131072,
            max_new_tokens_limit=2048,
        )

    @staticmethod
    def _count_text(text: str) -> int:
        return max(1, (len(text) + 1) // 2)

    def count_text_tokens(self, text: str) -> int:
        return self._count_text(text)

    def count_input_tokens(self, *, system, messages, tools) -> int:
        payload = (
            system
            + json.dumps(messages, ensure_ascii=False, default=str)
            + json.dumps(tools or [], ensure_ascii=False)
        )
        return self._count_text(payload)

    def complete(self, *, system, messages, tools, max_tokens, temperature=0.0):
        tokens = self.count_input_tokens(
            system=system, messages=messages, tools=tools
        )
        self.prompt_tokens_seen.append(tokens)
        self.calls.append(
            {
                "system": system,
                "messages": [dict(message) for message in messages],
                "tokens": tokens,
            }
        )
        if not self.responses:
            raise AssertionError("CountingModel ran out of responses")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class AlwaysRejectModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *, system, messages, tools, max_tokens, temperature=0.0):
        self.calls += 1
        raise ModelUnavailable(
            'model request returned HTTP 413: {"detail":'
            '{"error":"prompt_resource_risk","prompt_tokens":6436}}'
        )


class BudgetGuardTest(unittest.TestCase):
    def test_oversized_tool_result_is_trimmed_before_send(self) -> None:
        with TempWorkspace() as workspace:
            (workspace.path / "big.txt").write_text(
                "x" * 9000, encoding="utf-8"
            )
            task = "read big.txt then answer"
            model = CountingModel(
                [
                    tool_response([("read_file", {"path": "big.txt"})]),
                    text_response("done"),
                ],
                reject_tokens=4500,
            )
            loop, store, _ = make_loop(
                workspace.path,
                model,
                max_tokens=256,
                model_context_budget=8192,
            )
            try:
                result = loop.run(task)
            finally:
                store.close()

            self.assertEqual(result.status, "COMPLETED")
            cap = loop._prompt_cap()
            self.assertTrue(model.prompt_tokens_seen)
            self.assertLessEqual(max(model.prompt_tokens_seen), cap)
            budget = loop._prompt_budget_info
            self.assertGreater(
                budget["truncated_latest_tool_results"], 0, budget
            )
            for call in model.calls:
                self.assertEqual(call["messages"][0]["content"], task)

    def test_impossible_budget_never_sends_a_request(self) -> None:
        with TempWorkspace() as workspace:
            model = CountingModel(
                [text_response("never sent")], reject_tokens=1200
            )
            loop, store, _ = make_loop(
                workspace.path,
                model,
                max_tokens=256,
                model_context_budget=8192,
            )
            try:
                result = loop.run("do the task")
            finally:
                store.close()

            self.assertEqual(result.status, "FAILED")
            self.assertIn("CONTEXT_BUDGET_EXCEEDED", result.error or "")
            self.assertIn(
                "CONTEXT_BUDGET_EXCEEDED",
                {warning["code"] for warning in result.warnings},
            )
            self.assertEqual(model.calls, [])

    def test_http_413_is_not_retried(self) -> None:
        with TempWorkspace() as workspace:
            model = AlwaysRejectModel()
            loop, store, _ = make_loop(workspace.path, model)
            try:
                result = loop.run("do the task")
            finally:
                store.close()

            self.assertEqual(result.status, "FAILED")
            self.assertIn("CONTEXT_BUDGET_EXCEEDED", result.error or "")
            self.assertEqual(model.calls, 1)

    def test_loop_keeps_small_prompt_untouched(self) -> None:
        model = CountingModel(
            [text_response("ok")], reject_tokens=6000
        )
        with TempWorkspace() as workspace:
            loop, store, _ = make_loop(workspace.path, model)
            try:
                result = loop.run("say ok")
            finally:
                store.close()
            self.assertEqual(result.status, "COMPLETED")
            budget = loop._prompt_budget_info
            self.assertEqual(budget["prompt_tokens"], budget["initial_prompt_tokens"])
            self.assertEqual(budget["trimmed_old_tool_results"], 0)


def _memory(memory_id: str, confidence: float = 0.9) -> MemoryContextItem:
    return MemoryContextItem(
        memory_id=memory_id,
        text=f"verified memory {memory_id} " + "m" * 400,
        category="test",
        verification="verified",
        confidence=confidence,
        source="test",
        occurred_at=None,
        bank_id="bank",
        trust="MEMORY_VERIFIED",
    )


def _evidence(chunk_id: str) -> RagEvidenceItem:
    return RagEvidenceItem(
        chunk_id=chunk_id,
        citation=f"{chunk_id}.py:1-2",
        source=f"{chunk_id}.py",
        symbol=None,
        start_line=1,
        end_line=2,
        content=f"evidence {chunk_id} " + "e" * 400,
        score=0.5,
        rrf_score=0.5,
    )


class _Memory:
    def recall(self, *, project_id, query, max_tokens, include_unverified=False):
        return MemorySearchOutcome(items=[_memory("m1"), _memory("m2", 0.8)])


class _Rag:
    def search(self, *, project_id, query, token_budget, final_k=5):
        return RagSearchOutcome(items=[_evidence("c1"), _evidence("c2")])


class BuilderExactCounterTest(unittest.TestCase):
    def test_builder_uses_supplied_exact_counter(self) -> None:
        builder = ContextBuilder(
            memory=_Memory(),
            rag=_Rag(),
            config=ContextBuilderConfig(
                workspace=None,
                token_counter=lambda text: max(1, len(text)),
            ),
        )
        bundle = builder.build(
            ContextRequest(
                user_message="question",
                session_id="s",
                project_id="qianyuqwen",
                model_context_budget=900,
            )
        )
        self.assertLessEqual(len(bundle.system_context), 900)
        self.assertLessEqual(bundle.token_estimate.total, 900)
        self.assertTrue(bundle.token_estimate.truncated)

    def test_shrink_keeps_references_in_sync(self) -> None:
        builder = ContextBuilder(
            memory=_Memory(),
            rag=_Rag(),
            config=ContextBuilderConfig(workspace=None),
        )
        request = ContextRequest(
            user_message="question",
            session_id="s",
            project_id="qianyuqwen",
            model_context_budget=4096,
        )
        bundle = builder.build(request)
        self.assertEqual(len(bundle.source_references), 4)
        shrunk = builder.shrink(bundle, request, drop_rag=1, drop_memory=1)
        self.assertEqual(len(shrunk.rag_evidence), 1)
        self.assertEqual(len(shrunk.recalled_memories), 1)
        self.assertEqual(len(shrunk.source_references), 2)
        citations = {ref.identifier for ref in shrunk.source_references}
        self.assertEqual(citations, {"m1", "c1"})
        self.assertNotIn("c2.py", shrunk.system_context)
        self.assertNotIn("hindsight://bank/m2", shrunk.system_context)


@unittest.skipUnless(_HAS_FASTAPI, "fastapi runtime environment required")
class CancellationTest(unittest.TestCase):
    def test_cancel_token_records_first_request_time(self) -> None:
        token = CancellationToken()
        self.assertIsNone(token.requested_at)
        token.cancel("client")
        requested = token.requested_at
        token.cancel("timeout")
        self.assertEqual(token.requested_at, requested)
        self.assertEqual(token.reason, "client")

    def test_response_finally_cancels_generation(self) -> None:
        cancel = CancellationToken()
        response = _CancellableStreamingResponse(
            iter(["a", "b", "c"]),
            media_type="text/event-stream",
            cancel=cancel,
        )
        sent: list[dict] = []

        async def receive():
            await asyncio.sleep(3600)

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body":
                raise OSError("client disconnected")

        async def scenario():
            scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
            try:
                await response(scope, receive, send)
            except BaseException:  # noqa: BLE001 - anyio may group the OSError
                pass

        asyncio.run(scenario())
        self.assertTrue(sent)
        self.assertTrue(cancel.cancelled)


if __name__ == "__main__":
    unittest.main()