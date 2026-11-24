"""HTTP contract tests for the runtime service using a fake engine (no GPU)."""

import unittest
from typing import Any

from fastapi.testclient import TestClient

from yuzhu_lab.runtime.engine import GenerationResult, StreamDelta, Usage
from yuzhu_lab.runtime.server import create_app
from yuzhu_lab.runtime.settings import RuntimeSettings


class FakeEngine:
    loaded = True
    metadata = {"fake": True}

    def load(self) -> dict[str, Any]:
        return self.metadata

    def unload(self) -> None:
        pass

    def count_prompt_tokens(self, messages, params=None) -> int:
        return 7

    def generate(self, messages, params, cancel=None) -> GenerationResult:
        if params.tools:
            text = '<function name="read_file"><param name="path">a.py</param></function>'
        else:
            text = "Hello world"
        return GenerationResult(
            text=text,
            finish_reason="stop",
            usage=Usage(prompt_tokens=7, completion_tokens=2, total_tokens=9),
            generated_token_ids=[1, 2],
            elapsed_seconds=0.1,
        )

    def stream(self, messages, params, cancel=None):
        text = (
            '<function name="read_file"><param name="path">a.py</param></function>'
            if params.tools
            else "Hello world"
        )
        result = GenerationResult(
            text=text,
            finish_reason="stop",
            usage=Usage(prompt_tokens=7, completion_tokens=2, total_tokens=9),
            generated_token_ids=[1, 2],
            elapsed_seconds=0.1,
        )
        yield StreamDelta("Hello")
        yield StreamDelta(" world")
        yield StreamDelta(
            "",
            finish_reason="stop",
            usage=Usage(7, 2, 9),
            result=result,
        )


def make_client() -> TestClient:
    app = create_app(settings=RuntimeSettings(verify_adapter=False), engine=FakeEngine())
    return TestClient(app)


class RuntimeHttpTest(unittest.TestCase):
    def test_health_and_models(self):
        with make_client() as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["status"], "ok")
            models = client.get("/v1/models")
            self.assertEqual(models.status_code, 200)
            self.assertEqual(models.json()["object"], "list")

    def test_openai_non_stream(self):
        with make_client() as client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["choices"][0]["message"]["content"], "Hello world")
            self.assertEqual(body["usage"]["total_tokens"], 9)

    def test_openai_stream_sse(self):
        with make_client() as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
            ) as response:
                raw = "".join(response.iter_text())
            self.assertIn('"content": "Hello"', raw)
            self.assertIn('"finish_reason": "stop"', raw)
            self.assertIn("data: [DONE]", raw)

    def test_anthropic_text_stream(self):
        with make_client() as client:
            with client.stream(
                "POST",
                "/v1/messages",
                json={
                    "model": "claude",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                    "stream": True,
                },
            ) as response:
                raw = "".join(response.iter_text())
            self.assertIn("event: message_start", raw)
            self.assertIn("event: content_block_delta", raw)
            self.assertIn('"text_delta"', raw)
            self.assertIn("event: message_stop", raw)
            self.assertIn('"input_tokens": 7', raw)

    def test_anthropic_tool_use_parsed(self):
        with make_client() as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude",
                    "messages": [{"role": "user", "content": "read a.py"}],
                    "max_tokens": 32,
                    "tools": [
                        {
                            "name": "read_file",
                            "description": "read a file",
                            "input_schema": {"type": "object"},
                        }
                    ],
                },
            )
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["stop_reason"], "tool_use")
            self.assertEqual(body["content"][0]["type"], "tool_use")
            self.assertEqual(body["content"][0]["name"], "read_file")
            self.assertEqual(body["content"][0]["input"], {"path": "a.py"})

    def test_anthropic_tool_stream(self):
        with make_client() as client:
            with client.stream(
                "POST",
                "/v1/messages",
                json={
                    "model": "claude",
                    "messages": [{"role": "user", "content": "read a.py"}],
                    "max_tokens": 32,
                    "stream": True,
                    "tools": [
                        {
                            "name": "read_file",
                            "description": "read a file",
                            "input_schema": {"type": "object"},
                        }
                    ],
                },
            ) as response:
                raw = "".join(response.iter_text())
            self.assertIn('"type": "tool_use"', raw)
            self.assertIn('"input_json_delta"', raw)
            self.assertIn('"stop_reason": "tool_use"', raw)

    def test_anthropic_count_tokens(self):
        with make_client() as client:
            response = client.post(
                "/v1/messages/count_tokens",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"input_tokens": 7})

    def test_image_rejected(self):
        with make_client() as client:
            response = client.post(
                "/v1/messages",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "image", "source": {"type": "base64"}}],
                        }
                    ],
                    "max_tokens": 8,
                },
            )
            self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()