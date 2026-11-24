"""H5.3 runtime resource guard tests (fake engine, no GPU model copy)."""

from __future__ import annotations

import unittest
from typing import Any

from fastapi.testclient import TestClient

from yuzhu_lab.runtime.engine import GenerationResult, StreamDelta, Usage
from yuzhu_lab.runtime.server import create_app
from yuzhu_lab.runtime.settings import RuntimeSettings


class CountingEngine:
    loaded = True
    metadata = {"fake": True}

    def __init__(self, prompt_tokens: int = 7) -> None:
        self.prompt_tokens = prompt_tokens

    def load(self) -> dict[str, Any]:
        return self.metadata

    def unload(self) -> None:
        pass

    def count_prompt_tokens(self, messages, params=None) -> int:
        return self.prompt_tokens

    def generate(self, messages, params, cancel=None) -> GenerationResult:
        return GenerationResult(
            text="ok",
            finish_reason="stop",
            usage=Usage(self.prompt_tokens, 1, self.prompt_tokens + 1),
            generated_token_ids=[1],
            elapsed_seconds=0.01,
        )

    def stream(self, messages, params, cancel=None):
        yield StreamDelta("ok")
        yield StreamDelta(
            "",
            finish_reason="stop",
            usage=Usage(self.prompt_tokens, 1, self.prompt_tokens + 1),
        )


def make_client(
    *,
    prompt_tokens: int,
    warning: int = 100,
    reject: int = 1000,
    free_gib: float | None = 2.0,
) -> TestClient:
    settings = RuntimeSettings(
        verify_adapter=False,
        prompt_warning_tokens=warning,
        prompt_reject_tokens=reject,
        min_free_gib=0.35,
    )
    app = create_app(
        settings=settings,
        engine=CountingEngine(prompt_tokens),
        gpu_free_gib=lambda: free_gib,
    )
    return TestClient(app)


class RuntimeResourceGuardTest(unittest.TestCase):
    def test_warning_header_for_long_prompt(self) -> None:
        with make_client(prompt_tokens=150) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.headers.get("x-yuzhu-resource-warning"),
                "prompt_tokens_above_warning_threshold",
            )
            self.assertEqual(response.headers.get("x-yuzhu-prompt-tokens"), "150")

    def test_high_risk_prompt_is_rejected(self) -> None:
        with make_client(prompt_tokens=1500) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(response.status_code, 413)
            detail = response.json()["detail"]
            self.assertEqual(detail["error"], "prompt_resource_risk")
            self.assertEqual(detail["prompt_tokens"], 1500)
            self.assertEqual(detail["prompt_reject_tokens"], 1000)

    def test_anthropic_high_risk_prompt_is_rejected(self) -> None:
        with make_client(prompt_tokens=1500) as client:
            response = client.post(
                "/v1/messages",
                json={
                    "model": "claude",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                },
            )
            self.assertEqual(response.status_code, 413)
            self.assertEqual(
                response.json()["detail"]["error"], "prompt_resource_risk"
            )

    def test_low_free_vram_is_rejected(self) -> None:
        with make_client(prompt_tokens=10, free_gib=0.1) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            self.assertEqual(response.status_code, 503)
            detail = response.json()["detail"]
            self.assertEqual(detail["error"], "insufficient_gpu_memory")
            self.assertEqual(detail["free_gib"], 0.1)

    def test_health_reports_resource_limits(self) -> None:
        with make_client(prompt_tokens=10) as client:
            body = client.get("/health").json()
            self.assertEqual(body["limits"]["prompt_warning_tokens"], 100)
            self.assertEqual(body["limits"]["prompt_reject_tokens"], 1000)
            self.assertEqual(body["limits"]["min_free_gib"], 0.35)


if __name__ == "__main__":
    unittest.main()