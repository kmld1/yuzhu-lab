"""Client for the local MiniCPM5 Anthropic-compatible runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .errors import ModelUnavailable


@dataclass(frozen=True)
class InputLimits:
    """Serving limits that decide whether a prompt can be sent at all."""

    prompt_reject_tokens: int
    prompt_warning_tokens: int
    model_context_limit: int
    max_new_tokens_limit: int


@dataclass(frozen=True)
class ModelToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelResponse:
    text: str
    tool_calls: list[ModelToolCall]
    request_id: str
    stop_reason: str
    usage: dict[str, int] = field(default_factory=dict)
    raw_content: list[dict[str, Any]] = field(default_factory=list)

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"


class ModelPort(Protocol):
    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> ModelResponse: ...


class RuntimeModelClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8710",
        *,
        model: str | None = None,
        timeout_seconds: float = 300.0,
        api_key: str | None = None,
        max_tokens_limit: int = 2048,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens_limit = max_tokens_limit
        self._client = httpx.Client(
            base_url=self.base_url,
            trust_env=False,
            timeout=httpx.Timeout(timeout_seconds, connect=5.0),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
        )
        self._limits: InputLimits | None = None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "RuntimeModelClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ health

    def health(self, timeout: float = 5.0) -> dict[str, Any]:
        try:
            response = self._client.get("/health", timeout=timeout)
        except httpx.HTTPError as exc:
            raise ModelUnavailable(f"runtime is unreachable: {exc}") from exc
        if response.status_code != 200:
            raise ModelUnavailable(f"runtime health returned HTTP {response.status_code}")
        payload = response.json()
        if not payload.get("model_loaded"):
            raise ModelUnavailable("runtime is up but the model is not loaded")
        return payload

    def input_limits(self) -> InputLimits:
        """Fetch the runtime's prompt guard once; these limits are process-static."""
        if self._limits is None:
            payload = self.health(timeout=5.0)
            limits = payload.get("limits") or {}
            self._limits = InputLimits(
                prompt_reject_tokens=int(
                    limits.get("prompt_reject_tokens") or 6144
                ),
                prompt_warning_tokens=int(
                    limits.get("prompt_warning_tokens") or 3500
                ),
                model_context_limit=int(
                    limits.get("model_context_limit") or 131072
                ),
                max_new_tokens_limit=int(
                    limits.get("max_new_tokens_limit") or 2048
                ),
            )
        return self._limits

    def count_input_tokens(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """Exact chat-template token count, counted by the serving tokenizer."""
        payload: dict[str, Any] = {
            "model": self.model or "yuzhu-minicpm5",
            "system": system,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
        try:
            response = self._client.post(
                "/v1/messages/count_tokens",
                json=payload,
                timeout=httpx.Timeout(60.0, connect=5.0),
            )
        except httpx.HTTPError as exc:
            raise ModelUnavailable(f"token count failed: {exc}") from exc
        if response.status_code != 200:
            body = response.text[:400].replace("\n", " ")
            raise ModelUnavailable(
                f"token count returned HTTP {response.status_code}: {body}"
            )
        try:
            return int(response.json()["input_tokens"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ModelUnavailable("token count response was not valid") from exc

    def count_text_tokens(self, text: str) -> int:
        """Count free text as one user turn (context-block budgeting)."""
        return self.count_input_tokens(
            system="",
            messages=[{"role": "user", "content": text}],
            tools=None,
        )

    # ---------------------------------------------------------------- complete

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> ModelResponse:
        max_tokens = max(1, min(int(max_tokens), self.max_tokens_limit))
        payload: dict[str, Any] = {
            "model": self.model or "yuzhu-minicpm5",
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        try:
            response = self._client.post("/v1/messages", json=payload)
        except httpx.TimeoutException as exc:
            raise ModelUnavailable(f"model request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ModelUnavailable(f"model request failed: {exc}") from exc
        if response.status_code != 200:
            body = response.text[:400].replace("\n", " ")
            raise ModelUnavailable(f"model request returned HTTP {response.status_code}: {body}")
        try:
            data = response.json()
        except ValueError as exc:
            raise ModelUnavailable("model response was not valid JSON") from exc
        return self._parse(data)

    @staticmethod
    def _parse(data: dict[str, Any]) -> ModelResponse:
        content = data.get("content")
        if not isinstance(content, list):
            raise ModelUnavailable("model response has no content list")
        text_parts: list[str] = []
        tool_calls: list[ModelToolCall] = []
        for block in content:
            if not isinstance(block, dict):
                raise ModelUnavailable("model response contains a non-object content block")
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text", "")))
            elif block_type == "tool_use":
                name = block.get("name")
                arguments = block.get("input")
                if not isinstance(name, str) or not name:
                    raise ModelUnavailable("tool_use block has no tool name")
                if not isinstance(arguments, dict):
                    raise ModelUnavailable("tool_use input must be an object")
                tool_calls.append(ModelToolCall(name=name, arguments=arguments))
            else:
                raise ModelUnavailable(f"unsupported content block type: {block_type}")
        usage_raw = data.get("usage") or {}
        usage = {
            "input_tokens": int(usage_raw.get("input_tokens") or 0),
            "output_tokens": int(usage_raw.get("output_tokens") or 0),
        }
        return ModelResponse(
            text="\n".join(part for part in text_parts if part).strip(),
            tool_calls=tool_calls,
            request_id=str(data.get("id") or ""),
            stop_reason=str(data.get("stop_reason") or ""),
            usage=usage,
            raw_content=content,
        )


def response_summary(response: ModelResponse) -> dict[str, Any]:
    return {
        "request_id": response.request_id,
        "stop_reason": response.stop_reason,
        "text_chars": len(response.text),
        "tool_calls": [call.name for call in response.tool_calls],
        "usage": response.usage,
    }


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)