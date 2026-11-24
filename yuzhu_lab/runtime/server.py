"""FastAPI service exposing OpenAI and Anthropic compatible endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .engine import CancellationToken, GenerationParams, ModelEngine
from .protocol import (
    AnthropicCountTokensRequest,
    AnthropicMessagesRequest,
    ChatCompletionRequest,
    UnsupportedContentError,
    anthropic_content_block_start,
    anthropic_content_block_stop,
    anthropic_message,
    anthropic_message_delta,
    anthropic_message_start,
    anthropic_message_stop,
    anthropic_input_json_delta,
    anthropic_messages_to_internal,
    anthropic_text_delta,
    anthropic_tool_use_block_start,
    anthropic_tools_to_template,
    new_anthropic_id,
    new_openai_id,
    new_tool_use_id,
    openai_chunk,
    openai_completion,
    openai_messages_to_internal,
    openai_tools_to_template,
    parse_tool_calls,
    stop_list,
)
from .settings import RuntimeSettings

logger = logging.getLogger("yuzhu.runtime")


def _gpu_snapshot(*, safe: bool = False) -> dict[str, Any]:
    if safe:
        # Torch CUDA calls can block behind an active generation kernel.  While
        # the engine is busy, use the driver query with a hard timeout instead.
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return {"cuda_available": True, "source": "nvidia-smi", "error": "not found"}
        try:
            completed = subprocess.run(
                [
                    executable,
                    "--query-gpu=name,memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3,
                check=False,
            )
            lines = (completed.stdout or "").strip().splitlines()
            parts = [part.strip() for part in lines[0].split(",")] if lines else []
            if completed.returncode == 0 and len(parts) >= 4:
                total, used, free = (float(parts[1]), float(parts[2]), float(parts[3]))
                return {
                    "cuda_available": True,
                    "source": "nvidia-smi",
                    "engine_busy": True,
                    "device": parts[0],
                    "total_gib": round(total / 1024, 3),
                    "used_gib": round(used / 1024, 3),
                    "free_gib": round(free / 1024, 3),
                }
        except Exception as exc:  # noqa: BLE001 - health must not fail
            return {"cuda_available": True, "source": "nvidia-smi", "error": str(exc)}
        return {"cuda_available": True, "source": "nvidia-smi", "error": "unexpected output"}
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        free, total = torch.cuda.mem_get_info()
        return {
            "cuda_available": True,
            "device": torch.cuda.get_device_name(0),
            "free_gib": round(free / (1024**3), 3),
            "total_gib": round(total / (1024**3), 3),
            "allocated_gib": round(torch.cuda.memory_allocated() / (1024**3), 3),
            "reserved_gib": round(torch.cuda.memory_reserved() / (1024**3), 3),
        }
    except Exception as exc:  # noqa: BLE001 - health must not fail
        return {"cuda_available": False, "error": str(exc)}


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class _CancellableStreamingResponse(StreamingResponse):
    """Cancel the generation token whenever a streaming response stops.

    Starlette consumes the ASGI ``http.disconnect`` message itself and may
    abandon a sync body iterator without closing it.  Setting the token in
    ``__call__``'s finally covers normal completion, client disconnect and send
    failure, so the producer stops at the next token boundary instead of
    running to ``max_tokens`` and holding the GPU lock.
    """

    def __init__(self, *args: Any, cancel: CancellationToken, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cancel_token = cancel

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._cancel_token.cancel("client")


def _resolve_output_budget(
    requested: int | None, settings: RuntimeSettings
) -> tuple[int, int, bool]:
    """Return (requested, effective, limited) without pretending full support."""
    requested_value = (
        int(requested)
        if requested is not None and int(requested) > 0
        else settings.default_max_new_tokens
    )
    effective = max(1, min(requested_value, settings.max_new_tokens_limit))
    limited = requested_value > settings.max_new_tokens_limit
    if limited and settings.max_tokens_policy == "reject":
        raise HTTPException(
            status_code=400,
            detail={
                "error": "max_tokens_exceeds_supported_limit",
                "requested_max_tokens": requested_value,
                "effective_max_tokens": effective,
                "model_context_limit": settings.model_context_limit,
                "policy": settings.max_tokens_policy,
            },
        )
    return requested_value, effective, limited


async def _watch_disconnect(
    request: Request,
    cancel: CancellationToken,
    generator: Iterator[str] | None = None,
) -> None:
    """Cancel generation when the HTTP client goes away.

    Sync generators are iterated in a threadpool by Starlette.  A client
    disconnect does not reliably close that generator, so the engine's producer
    thread must be told to stop; otherwise it could hold the GPU lock forever.
    """
    while not cancel.cancelled:
        try:
            disconnected = await request.is_disconnected()
        except Exception:  # noqa: BLE001 - request may already be closed
            disconnected = True
        if disconnected:
            cancel.cancel("client")
            close = getattr(generator, "close", None)
            if callable(close):
                try:
                    close()
                except (ValueError, RuntimeError):
                    pass
            return
        await asyncio.sleep(0.25)


def _budget_headers(
    *, requested: int, effective: int, limited: bool, settings: RuntimeSettings
) -> dict[str, str]:
    return {
        "X-Yuzhu-Requested-Max-Tokens": str(requested),
        "X-Yuzhu-Effective-Max-Tokens": str(effective),
        "X-Yuzhu-Token-Budget-Limited": "true" if limited else "false",
        "X-Yuzhu-Model-Context-Limit": str(settings.model_context_limit),
        "X-Yuzhu-Max-Tokens-Policy": settings.max_tokens_policy,
    }


def create_app(
    settings: RuntimeSettings | None = None,
    engine: ModelEngine | None = None,
    gpu_free_gib: Any | None = None,
) -> FastAPI:
    settings = settings or RuntimeSettings()
    engine = engine or ModelEngine(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("loading model from %s", settings.model_dir)
        loaded = engine.load()
        logger.info("model loaded: %s", loaded)
        yield
        engine.unload()
        logger.info("model unloaded")

    app = FastAPI(title="YUZHU LAB Runtime", version="1.0.0", lifespan=lifespan)
    app.state.engine = engine
    app.state.settings = settings

    def probe_gpu_free_gib() -> float | None:
        if gpu_free_gib is not None:
            value = gpu_free_gib()
            return float(value) if value is not None else None
        snapshot = _gpu_snapshot(safe=bool(getattr(engine, "busy", False)))
        value = snapshot.get("free_gib")
        return float(value) if isinstance(value, (int, float)) else None

    def enforce_resource_guard(prompt_tokens: int, headers: dict[str, str]) -> None:
        headers["X-Yuzhu-Prompt-Warning-Tokens"] = str(
            settings.prompt_warning_tokens
        )
        headers["X-Yuzhu-Prompt-Reject-Tokens"] = str(
            settings.prompt_reject_tokens
        )
        free = probe_gpu_free_gib()
        if free is not None:
            headers["X-Yuzhu-GPU-Free-GiB"] = str(round(free, 3))
        if prompt_tokens >= settings.prompt_reject_tokens:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "prompt_resource_risk",
                    "prompt_tokens": prompt_tokens,
                    "prompt_warning_tokens": settings.prompt_warning_tokens,
                    "prompt_reject_tokens": settings.prompt_reject_tokens,
                    "suggestion": (
                        "Reduce the prompt or split the task; a prompt of this "
                        "size can exhaust the 8 GiB device during prefill."
                    ),
                },
            )
        if free is not None and free < settings.min_free_gib:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "insufficient_gpu_memory",
                    "free_gib": round(free, 3),
                    "min_free_gib": settings.min_free_gib,
                    "suggestion": "Wait for the current request to finish or free GPU memory.",
                },
            )
        if prompt_tokens >= settings.prompt_warning_tokens:
            headers["X-Yuzhu-Resource-Warning"] = (
                "prompt_tokens_above_warning_threshold"
            )

    def require_auth(
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> None:
        if not settings.api_key:
            return
        token = x_api_key
        if token is None and authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if token != settings.api_key:
            raise HTTPException(status_code=401, detail="invalid API key")

    # ------------------------------------------------------------------ health

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok" if engine.loaded else "loading",
            "model": settings.model_name,
            "model_loaded": engine.loaded,
            "runtime": "yuzhu-minicpm5",
            "metadata": engine.metadata,
            "gpu": _gpu_snapshot(safe=bool(getattr(engine, "busy", False))),
            "busy": bool(getattr(engine, "busy", False)),
            "last_cancel": getattr(engine, "last_cancel", {}) or {},
            "limits": {
                "max_new_tokens_limit": settings.max_new_tokens_limit,
                "model_context_limit": settings.model_context_limit,
                "max_tokens_policy": settings.max_tokens_policy,
                "default_max_new_tokens": settings.default_max_new_tokens,
                "prompt_warning_tokens": settings.prompt_warning_tokens,
                "prompt_reject_tokens": settings.prompt_reject_tokens,
                "min_free_gib": settings.min_free_gib,
            },
        }

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "yuzhu-lab",
                }
            ],
        }

    # ------------------------------------------------------------ OpenAI compat

    @app.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
    async def chat_completions(
        request: ChatCompletionRequest, http_request: Request
    ) -> Any:
        if not engine.loaded:
            raise HTTPException(status_code=503, detail="model is not loaded")
        try:
            messages = openai_messages_to_internal(request.messages)
        except UnsupportedContentError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        requested, effective, limited = _resolve_output_budget(
            request.max_tokens or request.max_completion_tokens, settings
        )
        params = GenerationParams(
            max_new_tokens=effective,
            temperature=request.temperature if request.temperature is not None else 0.0,
            top_p=request.top_p if request.top_p is not None else 1.0,
            stop=stop_list(request.stop),
            tools=openai_tools_to_template(request.tools),
        )
        headers = _budget_headers(
            requested=requested, effective=effective, limited=limited, settings=settings
        )
        prompt_tokens = engine.count_prompt_tokens(messages, params)
        if prompt_tokens + effective > settings.model_context_limit:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "model_context_limit_exceeded",
                    "prompt_tokens": prompt_tokens,
                    "effective_max_tokens": effective,
                    "model_context_limit": settings.model_context_limit,
                },
            )
        headers["X-Yuzhu-Prompt-Tokens"] = str(prompt_tokens)
        enforce_resource_guard(prompt_tokens, headers)
        logger.info(
            "openai chat: model=%s stream=%s messages=%d tools=%s "
            "requested_max_tokens=%d effective_max_tokens=%d limited=%s",
            request.model,
            request.stream,
            len(messages),
            [tool.get("name") for tool in (params.tools or [])],
            requested,
            effective,
            limited,
        )

        if request.stream:
            cancel = CancellationToken()
            generator = _openai_stream(engine, messages, params, settings, cancel)
            return _CancellableStreamingResponse(
                generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    **headers,
                },
                cancel=cancel,
            )

        cancel = CancellationToken()
        watchdog = asyncio.create_task(_watch_disconnect(http_request, cancel))
        try:
            result = await asyncio.to_thread(engine.generate, messages, params, cancel)
        finally:
            watchdog.cancel()
        headers["X-Yuzhu-Truncated"] = (
            "true" if result.finish_reason == "length" else "false"
        )
        headers["X-Yuzhu-Finish-Reason"] = result.finish_reason
        if result.cancel_reason:
            headers["X-Yuzhu-Cancel-Reason"] = result.cancel_reason
        return JSONResponse(
            openai_completion(
                model=settings.model_name,
                text=result.text,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                finish_reason=result.finish_reason,
            ),
            headers=headers,
        )

    def _openai_stream(
        engine_: ModelEngine,
        messages: list[dict[str, Any]],
        params: GenerationParams,
        settings_: RuntimeSettings,
        cancel: CancellationToken,
    ) -> Iterator[str]:
        response_id = new_openai_id()
        created = int(time.time())
        try:
            for delta in engine_.stream(messages, params, cancel):
                if delta.finish_reason is None:
                    if delta.text:
                        yield f"data: {json.dumps(openai_chunk(response_id=response_id, created=created, model=settings_.model_name, text=delta.text), ensure_ascii=False)}\n\n"
                else:
                    usage = delta.usage
                    yield f"data: {json.dumps(openai_chunk(response_id=response_id, created=created, model=settings_.model_name, text='', finish_reason=('stop' if delta.finish_reason == 'cancelled' else delta.finish_reason), prompt_tokens=usage.prompt_tokens if usage else None, completion_tokens=usage.completion_tokens if usage else None), ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001 - report over SSE
            logger.exception("openai stream failed")
            yield f"data: {json.dumps({'error': {'message': str(exc), 'type': 'runtime_error'}}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            cancel.cancel("client")

    # --------------------------------------------------------- Anthropic compat

    @app.post("/v1/messages", dependencies=[Depends(require_auth)])
    async def anthropic_messages(
        request: AnthropicMessagesRequest, http_request: Request
    ) -> Any:
        if not engine.loaded:
            raise HTTPException(status_code=503, detail="model is not loaded")
        try:
            messages = anthropic_messages_to_internal(request)
        except UnsupportedContentError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        requested, effective, limited = _resolve_output_budget(
            request.max_tokens, settings
        )
        params = GenerationParams(
            max_new_tokens=effective,
            temperature=request.temperature if request.temperature is not None else 0.0,
            top_p=request.top_p if request.top_p is not None else 1.0,
            stop=stop_list(request.stop_sequences),
            tools=anthropic_tools_to_template(request.tools),
        )
        headers = _budget_headers(
            requested=requested, effective=effective, limited=limited, settings=settings
        )
        prompt_tokens = engine.count_prompt_tokens(messages, params)
        if prompt_tokens + effective > settings.model_context_limit:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "model_context_limit_exceeded",
                    "prompt_tokens": prompt_tokens,
                    "effective_max_tokens": effective,
                    "model_context_limit": settings.model_context_limit,
                },
            )
        headers["X-Yuzhu-Prompt-Tokens"] = str(prompt_tokens)
        enforce_resource_guard(prompt_tokens, headers)
        logger.info(
            "anthropic messages: model=%s stream=%s messages=%d tools=%s "
            "requested_max_tokens=%d effective_max_tokens=%d limited=%s",
            request.model,
            request.stream,
            len(messages),
            [tool.get("name") for tool in (request.tools or [])],
            requested,
            effective,
            limited,
        )

        if request.stream:
            cancel = CancellationToken()
            generator = _anthropic_stream(engine, messages, params, settings, cancel)
            return _CancellableStreamingResponse(
                generator,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    **headers,
                },
                cancel=cancel,
            )

        cancel = CancellationToken()
        watchdog = asyncio.create_task(_watch_disconnect(http_request, cancel))
        try:
            result = await asyncio.to_thread(engine.generate, messages, params, cancel)
        finally:
            watchdog.cancel()
        text, calls = parse_tool_calls(result.text) if params.tools else (result.text, [])
        headers["X-Yuzhu-Truncated"] = (
            "true" if result.finish_reason == "length" else "false"
        )
        headers["X-Yuzhu-Finish-Reason"] = result.finish_reason
        if result.cancel_reason:
            headers["X-Yuzhu-Cancel-Reason"] = result.cancel_reason
        return JSONResponse(
            anthropic_message(
                model=settings.model_name,
                text=text,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                finish_reason=result.finish_reason,
                stop_sequence=result.matched_stop,
                tool_calls=calls,
            ),
            headers=headers,
        )

    def _anthropic_stream(
        engine_: ModelEngine,
        messages: list[dict[str, Any]],
        params: GenerationParams,
        settings_: RuntimeSettings,
        cancel: CancellationToken,
    ) -> Iterator[str]:
        message_id = new_anthropic_id()
        try:
            prompt_tokens = engine_.count_prompt_tokens(messages, params)
            yield _sse(
                "message_start",
                anthropic_message_start(
                    message_id=message_id, model=settings_.model_name, prompt_tokens=prompt_tokens
                ),
            )
            if params.tools:
                # Tool calls need the complete output before they can be parsed
                # deterministically.  Consume the stream directly and yield SSE
                # comments between tokens: the threadpool iterator then returns
                # to the event loop regularly, so a client disconnect can set
                # the cancel token instead of waiting for the full generation.
                final = None
                for delta in engine_.stream(messages, params, cancel):
                    if delta.finish_reason is not None:
                        final = delta
                    else:
                        yield ": keepalive\n\n"
                if final is None or final.result is None:
                    raise RuntimeError("generation finished without a result")
                result = final.result
                text, calls = parse_tool_calls(result.text)
                index = 0
                if text:
                    yield _sse("content_block_start", anthropic_content_block_start(index))
                    yield _sse("content_block_delta", anthropic_text_delta(text, index))
                    yield _sse("content_block_stop", anthropic_content_block_stop(index))
                    index += 1
                for call in calls:
                    tool_id = new_tool_use_id()
                    yield _sse(
                        "content_block_start",
                        anthropic_tool_use_block_start(index, tool_id, call.name),
                    )
                    yield _sse(
                        "content_block_delta",
                        anthropic_input_json_delta(
                            index, json.dumps(call.arguments, ensure_ascii=False)
                        ),
                    )
                    yield _sse("content_block_stop", anthropic_content_block_stop(index))
                    index += 1
                yield _sse(
                    "message_delta",
                    anthropic_message_delta(
                        finish_reason=result.finish_reason,
                        stop_sequence=result.matched_stop,
                        output_tokens=result.usage.completion_tokens,
                        has_tool_calls=bool(calls),
                    ),
                )
                yield _sse("message_stop", anthropic_message_stop())
                return
            yield _sse("content_block_start", anthropic_content_block_start())
            final = None
            for delta in engine_.stream(messages, params, cancel):
                if delta.finish_reason is None:
                    if delta.text:
                        yield _sse("content_block_delta", anthropic_text_delta(delta.text))
                else:
                    final = delta
            if final is None or final.usage is None:
                raise RuntimeError("generation finished without a result")
            yield _sse("content_block_stop", anthropic_content_block_stop())
            yield _sse(
                "message_delta",
                anthropic_message_delta(
                    finish_reason=final.finish_reason,
                    stop_sequence=None,
                    output_tokens=final.usage.completion_tokens,
                ),
            )
            yield _sse("message_stop", anthropic_message_stop())
        except Exception as exc:  # noqa: BLE001 - report over SSE
            logger.exception("anthropic stream failed")
            yield _sse(
                "error",
                {"type": "error", "error": {"type": "api_error", "message": str(exc)}},
            )
        finally:
            cancel.cancel("client")

    @app.post("/v1/messages/count_tokens", dependencies=[Depends(require_auth)])
    def count_tokens(request: AnthropicCountTokensRequest) -> dict[str, int]:
        if not engine.loaded:
            raise HTTPException(status_code=503, detail="model is not loaded")
        try:
            messages = anthropic_messages_to_internal(
                AnthropicMessagesRequest(
                    model=request.model,
                    system=request.system,
                    messages=request.messages,
                    max_tokens=1,
                    tools=request.tools,
                )
            )
        except UnsupportedContentError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        params = GenerationParams(tools=anthropic_tools_to_template(request.tools))
        return {"input_tokens": engine.count_prompt_tokens(messages, params)}

    return app