"""HTTP contract tests for explicit output-budget and truncation behaviour."""

from __future__ import annotations

import json
import unittest
from typing import Any

from fastapi.testclient import TestClient

from yuzhu_lab.runtime.engine import GenerationResult, StreamDelta, Usage
from yuzhu_lab.runtime.server import create_app
from yuzhu_lab.runtime.settings import RuntimeSettings


class BudgetFakeEngine:
    loaded = True
    metadata = {"fake": True}

    def __init__(
        self,
        *,
        text: str = "Hello world",
        finish_reason: str = "stop",
        prompt_tokens: int = 7,
        completion_tokens: int = 2,
    ) -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.last_params: Any = None

    def load(self) -> dict[str, Any]:
        return self.metadata

    def unload(self) -> None:
        pass

    def count_prompt_tokens(self, messages, params=None) -> int:
        return self.prompt_tokens

    def generate(self, messages, params, cancel=None) -> GenerationResult:
        self.last_params = params
        return GenerationResult(
            text=self.text,
            finish_reason=self.finish_reason,
            usage=Usage(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.prompt_tokens + self.completion_tokens,
            ),
            generated_token_ids=[1] * self.completion_tokens,
            elapsed_seconds=0.1,
        )

    def stream(self, messages, params, cancel=None):
        self.last_params = params
        yield StreamDelta(self.text)
        yield StreamDelta(
            "",
            finish_reason=self.finish_reason,
            usage=Usage(self.prompt_tokens, self.completion_tokens, self.prompt_tokens + self.completion_tokens),
        )


def make_client(engine: BudgetFakeEngine, **settings_kwargs) -> TestClient:
    settings = RuntimeSettings(verify_adapter=False, **settings_kwargs)
    return TestClient(create_app(settings=settings, engine=engine))


def anthropic_payload(max_tokens: int, tools: list[dict] | None = None) -> dict:
    payload = {
        "model": "local",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if tools:
        payload["tools"] = tools
    return payload


READ_TOOL = {
    "name": "read_file",
    "description": "read",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}


class TokenBudgetTest(unittest.TestCase):
    def test_anthropic_request_is_explicitly_clamped(self) -> None:
        engine = BudgetFakeEngine()
        with make_client(engine) as client:
            response = client.post("/v1/messages", json=anthropic_payload(32000))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-yuzhu-requested-max-tokens"], "32000")
        self.assertEqual(response.headers["x-yuzhu-effective-max-tokens"], "2048")
        self.assertEqual(response.headers["x-yuzhu-token-budget-limited"], "true")
        self.assertEqual(response.headers["x-yuzhu-truncated"], "false")
        self.assertEqual(engine.last_params.max_new_tokens, 2048)

    def test_anthropic_length_sets_max_tokens_stop_reason(self) -> None:
        engine = BudgetFakeEngine(finish_reason="length", text="partial answer")
        with make_client(engine) as client:
            response = client.post("/v1/messages", json=anthropic_payload(32000))
        body = response.json()
        self.assertEqual(body["stop_reason"], "max_tokens")
        self.assertEqual(response.headers["x-yuzhu-truncated"], "true")
        self.assertEqual(body["usage"]["output_tokens"], 2)

    def test_openai_clamp_and_truncation(self) -> None:
        engine = BudgetFakeEngine(finish_reason="length")
        with make_client(engine) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "local",
                    "max_tokens": 5000,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-yuzhu-effective-max-tokens"], "2048")
        self.assertEqual(response.headers["x-yuzhu-token-budget-limited"], "true")
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "length")

    def test_reject_policy_returns_400(self) -> None:
        engine = BudgetFakeEngine()
        with make_client(engine, max_tokens_policy="reject") as client:
            response = client.post("/v1/messages", json=anthropic_payload(32000))
        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assertEqual(detail["error"], "max_tokens_exceeds_supported_limit")
        self.assertEqual(detail["effective_max_tokens"], 2048)
        self.assertIsNone(engine.last_params)

    def test_model_context_limit_is_enforced(self) -> None:
        engine = BudgetFakeEngine(prompt_tokens=130000)
        with make_client(engine, model_context_limit=131072) as client:
            response = client.post("/v1/messages", json=anthropic_payload(2048))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["error"], "model_context_limit_exceeded")

    def test_incomplete_tool_call_is_not_executed(self) -> None:
        engine = BudgetFakeEngine(
            text='<function name="read_file"><param name="path">a.py',
            finish_reason="length",
        )
        with make_client(engine) as client:
            response = client.post(
                "/v1/messages", json=anthropic_payload(2048, [READ_TOOL])
            )
        body = response.json()
        self.assertEqual(body["stop_reason"], "max_tokens")
        self.assertEqual([block["type"] for block in body["content"]], ["text"])
        self.assertIn("<function", body["content"][0]["text"])

    def test_complete_tool_call_survives_later_truncation(self) -> None:
        engine = BudgetFakeEngine(
            text=(
                '<function name="read_file"><param name="path">a.py</param></function>'
                "\n<function name=\"read_file\"><param name=\"path\">b.py"
            ),
            finish_reason="length",
        )
        with make_client(engine) as client:
            response = client.post(
                "/v1/messages", json=anthropic_payload(2048, [READ_TOOL])
            )
        body = response.json()
        tool_blocks = [block for block in body["content"] if block["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), 1)
        self.assertEqual(tool_blocks[0]["input"], {"path": "a.py"})
        self.assertEqual(body["stop_reason"], "tool_use")

    def test_sse_reports_budget_headers_and_stop_reason(self) -> None:
        engine = BudgetFakeEngine(finish_reason="length", text="partial")
        with make_client(engine) as client:
            with client.stream(
                "POST",
                "/v1/messages",
                json={**anthropic_payload(32000), "stream": True},
            ) as response:
                body = "".join(response.iter_text())
                headers = response.headers
        self.assertEqual(headers["x-yuzhu-effective-max-tokens"], "2048")
        self.assertEqual(headers["x-yuzhu-token-budget-limited"], "true")
        self.assertIn('"stop_reason": "max_tokens"', body)
        self.assertIn("message_stop", body)

    def test_health_reports_limits(self) -> None:
        engine = BudgetFakeEngine()
        with make_client(engine) as client:
            health = client.get("/health").json()
        self.assertEqual(health["limits"]["max_new_tokens_limit"], 2048)
        self.assertEqual(health["limits"]["model_context_limit"], 131072)
        self.assertEqual(health["limits"]["max_tokens_policy"], "clamp")


if __name__ == "__main__":
    unittest.main()