"""Small HTTP client for the official Hindsight REST API.

The official ``hindsight-client`` package is used in the deployment probe.  The
project runtime installs only the standard library plus ``httpx``, so the
harness talks to the same documented endpoints directly.  Retain, recall and
reflect are the only operations the memory layer needs.
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import quote

import httpx

from .config import HindsightConfig


class HindsightError(RuntimeError):
    """Base class for memory service failures."""


class HindsightUnavailable(HindsightError):
    """The service could not be reached."""


class HindsightTimeout(HindsightError):
    """The request exceeded its deadline."""


class HindsightAPIError(HindsightError):
    """The service returned a non-success response."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Hindsight HTTP {status_code}: {message}")
        self.status_code = status_code


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class HindsightHttpClient:
    def __init__(self, config: HindsightConfig | None = None) -> None:
        self.config = config or HindsightConfig()
        self._client = httpx.Client(
            base_url=self.config.base_url.rstrip("/"),
            # The service is loopback-only.  A Windows system proxy must not
            # intercept localhost traffic (it answers 502 before Hindsight sees
            # the request), so environment proxy discovery is disabled.
            trust_env=False,
            timeout=httpx.Timeout(
                self.config.request_timeout_seconds,
                connect=self.config.connect_timeout_seconds,
            ),
        )

    # ------------------------------------------------------------------ transport

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
        retry: bool = True,
    ) -> dict[str, Any]:
        attempts = self.config.max_retries + 1 if retry else 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._client.request(
                    method,
                    path,
                    content=(
                        json.dumps(payload, ensure_ascii=True).encode("ascii")
                        if payload is not None
                        else None
                    ),
                    headers=(
                        {"Content-Type": "application/json"}
                        if payload is not None
                        else None
                    ),
                    timeout=timeout or self.config.request_timeout_seconds,
                )
            except httpx.TimeoutException as exc:
                last_error = HindsightTimeout(f"Hindsight request timed out: {exc}")
            except httpx.TransportError as exc:
                last_error = HindsightUnavailable(f"Hindsight is unreachable: {exc}")
            else:
                if response.status_code < 400:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise HindsightAPIError(
                            response.status_code, "response was not valid JSON"
                        ) from exc
                body = response.text[:300].replace("\n", " ")
                error = HindsightAPIError(response.status_code, body)
                if not (retry and response.status_code in _RETRYABLE_STATUS):
                    raise error
                last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(2.0, 0.5 * (2**attempt)))
        assert last_error is not None
        raise last_error

    # -------------------------------------------------------------------- health

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", timeout=10.0, retry=False)

    # -------------------------------------------------------------------- retain

    def retain(
        self,
        *,
        bank_id: str,
        content: str,
        document_id: str,
        context: str,
        metadata: dict[str, str],
        tags: tuple[str, ...] | list[str],
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {
            "content": content,
            "document_id": document_id,
            "context": context,
            "metadata": metadata,
            "tags": list(tags),
        }
        if timestamp:
            item["timestamp"] = timestamp
        return self._request(
            "POST",
            f"/v1/default/banks/{quote(bank_id, safe='')}/memories",
            payload={"items": [item], "async": False},
            timeout=self.config.request_timeout_seconds,
        )

    # -------------------------------------------------------------------- recall

    def recall(
        self,
        *,
        bank_id: str,
        query: str,
        max_tokens: int = 1200,
        tags: tuple[str, ...] | list[str] | None = None,
        tags_match: str = "any",
        types: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "max_tokens": max_tokens,
            "budget": "mid",
        }
        if tags:
            payload["tags"] = list(tags)
            payload["tags_match"] = tags_match
        if types:
            payload["types"] = types
        return self._request(
            "POST",
            f"/v1/default/banks/{quote(bank_id, safe='')}/memories/recall",
            payload=payload,
            timeout=self.config.recall_timeout_seconds,
        )

    # ------------------------------------------------------------------- reflect

    def reflect(
        self,
        *,
        bank_id: str,
        query: str,
        max_tokens: int = 512,
        tags: tuple[str, ...] | list[str] | None = None,
        tags_match: str = "any",
        include_facts: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "budget": "low",
            "max_tokens": max_tokens,
        }
        if tags:
            payload["tags"] = list(tags)
            payload["tags_match"] = tags_match
        if include_facts:
            # The API models ``facts`` as an options object, not a boolean.
            payload["include"] = {"facts": {}}
        return self._request(
            "POST",
            f"/v1/default/banks/{quote(bank_id, safe='')}/reflect",
            payload=payload,
            timeout=self.config.reflect_timeout_seconds,
        )

    # --------------------------------------------------------------------- close

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HindsightHttpClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()