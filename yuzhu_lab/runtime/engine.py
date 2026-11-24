"""Single-GPU generation engine with streaming, cancellation, stop sequences and usage."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from .model import LoadedModel, load_model
from .settings import RuntimeSettings

logger = logging.getLogger("yuzhu.runtime.engine")


class CancellationToken:
    """Cooperative cancellation shared by the HTTP layer and the generation thread."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason: str | None = None
        # First cancellation request; used only for observability.
        self.requested_at: float | None = None

    def cancel(self, reason: str = "client") -> None:
        if self.reason is None:
            self.reason = reason
        if self.requested_at is None:
            self.requested_at = time.perf_counter()
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass
class GenerationParams:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    stop: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] | None = None


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class GenerationResult:
    text: str
    finish_reason: str
    usage: Usage
    generated_token_ids: list[int]
    elapsed_seconds: float
    cancelled: bool = False
    matched_stop: str | None = None
    cancel_reason: str | None = None


@dataclass
class StreamDelta:
    text: str
    finish_reason: str | None = None
    usage: Usage | None = None
    result: "GenerationResult | None" = None


class _StopBuffer:
    """Streaming buffer that never emits a configured stop sequence."""

    def __init__(self, stops: list[str]) -> None:
        self._stops = [stop for stop in stops if stop]
        self._buffer = ""
        self.matched: str | None = None

    def feed(self, text: str) -> str:
        if not self._stops:
            return text
        self._buffer += text
        for stop in self._stops:
            index = self._buffer.find(stop)
            if index != -1:
                emit = self._buffer[:index]
                self._buffer = ""
                self.matched = stop
                return emit
        keep = max(len(stop) for stop in self._stops) - 1
        if keep > 0 and len(self._buffer) > keep:
            emit = self._buffer[:-keep]
            self._buffer = self._buffer[-keep:]
            return emit
        return ""

    def flush(self) -> str:
        if self.matched is not None:
            self._buffer = ""
            return ""
        emit = self._buffer
        self._buffer = ""
        return emit


# MiniCPM5 marks the tool syntax (<function>, <param>, ...) as special tokens.
# They must be preserved when tools are enabled, while chat control tokens are
# stripped from visible output.
_CONTROL_TOKENS = (
    "<s>",
    "</s>",
    "<unk>",
    "<|im_start|>",
    "<|im_end|>",
    "<|im_sep|>",
    "<|thought_begin|>",
    "<|thought_end|>",
    "<|tool_call|>",
    "<|execute_start|>",
    "<|execute_end|>",
    "<tool_call>",
    "</tool_call>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "/think",
    "/no_think",
)


def strip_control_tokens(text: str) -> str:
    for token in _CONTROL_TOKENS:
        text = text.replace(token, "")
    return text


def _truncate_at_stop(text: str, stops: list[str]) -> tuple[str, str | None]:
    best: tuple[int, str] | None = None
    for stop in stops:
        if not stop:
            continue
        index = text.find(stop)
        if index != -1 and (best is None or index < best[0]):
            best = (index, stop)
    if best is None:
        return text, None
    return text[: best[0]], best[1]


class ModelEngine:
    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self._loaded: LoadedModel | None = None
        self._lock = threading.Lock()
        self._last_cancel: dict[str, Any] | None = None
        self._active_cancel: dict[str, Any] | None = None
        self._last_cancel_requested_at = 0.0

    # ------------------------------------------------------------------ loading

    def load(self) -> dict[str, Any]:
        if self._loaded is None:
            self._loaded = load_model(self.settings)
        return self._loaded.metadata

    def unload(self) -> None:
        if self._loaded is None:
            return
        import torch

        self._loaded = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def loaded(self) -> bool:
        return self._loaded is not None

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._loaded.metadata if self._loaded else {}

    @property
    def last_cancel(self) -> dict[str, Any]:
        """Timing of the most recent cancelled generation (observability only)."""
        return dict(self._last_cancel or {})

    def token_count(self, text: str) -> int:
        self._require_loaded()
        return len(self._loaded.tokenizer.encode(text, add_special_tokens=False))

    # ------------------------------------------------------------------ helpers

    def _require_loaded(self) -> LoadedModel:
        if self._loaded is None:
            raise RuntimeError("model is not loaded")
        return self._loaded

    def _render_inputs(self, messages: list[dict[str, Any]], params: GenerationParams) -> Any:
        loaded = self._require_loaded()
        rendered = loaded.tokenizer.apply_chat_template(
            messages,
            tools=params.tools or None,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.settings.enable_thinking,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = rendered["input_ids"].to(self.settings.device)
        attention_mask = rendered.get("attention_mask")
        if attention_mask is not None:
            # H5.2: a batch-of-one all-ones mask disables the memory-efficient
            # SDPA kernel and materializes an O(seq^2) attention matrix (about
            # 1.3 GiB at 4.5k tokens on this model).  It carries no information
            # when there is no padding, so pass None instead.
            if bool(attention_mask.all()):
                attention_mask = None
            else:
                attention_mask = attention_mask.to(self.settings.device)
        return input_ids, attention_mask

    def _effective_max_tokens(self, requested: int | None) -> int:
        value = requested if requested and requested > 0 else self.settings.default_max_new_tokens
        return max(1, min(int(value), self.settings.max_new_tokens_limit))

    def count_prompt_tokens(
        self, messages: list[dict[str, Any]], params: GenerationParams | None = None
    ) -> int:
        """Render the chat template and count prompt tokens without generating."""
        with self._lock:
            input_ids, _ = self._render_inputs(messages, params or GenerationParams())
            return int(input_ids.shape[1])

    # ------------------------------------------------------------------ generate

    def _record_cancel(
        self,
        cancel: CancellationToken,
        *,
        observed_at: float,
        generated_tokens: int,
    ) -> None:
        now = time.perf_counter()
        requested = cancel.requested_at or observed_at
        self._last_cancel_requested_at = requested
        record = {
            "reason": cancel.reason or "client",
            "generated_tokens": generated_tokens,
            "detect_to_stop_seconds": round(
                max(0.0, observed_at - requested), 4
            ),
            "generation_end_seconds": round(max(0.0, now - requested), 4),
        }
        self._last_cancel = record
        self._active_cancel = record
        logger.info(
            "generation cancelled: reason=%s tokens=%d detect_to_stop=%.3fs "
            "generation_end=%.3fs",
            cancel.reason,
            generated_tokens,
            observed_at - requested,
            now - requested,
        )

    def generate(
        self,
        messages: list[dict[str, Any]],
        params: GenerationParams,
        cancel: CancellationToken | None = None,
    ) -> GenerationResult:
        if cancel is None:
            cancel = CancellationToken()
        final: StreamDelta | None = None
        for delta in self.stream(messages, params, cancel):
            if delta.finish_reason is not None:
                final = delta
        # H5.2: variable-size generations fragment the CUDA caching allocator.
        # On an 8 GiB device the reserved memory otherwise creeps toward the
        # limit across turns and decoding slows to a crawl.  Release cached
        # blocks once the generation (and its KV cache) is finished.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must not fail a request
            pass
        if final is None or final.usage is None:
            raise RuntimeError("generation finished without a result")
        if final.result is not None:
            return final.result
        return GenerationResult(
            text=getattr(self, "_last_text", ""),
            finish_reason=final.finish_reason or "stop",
            usage=final.usage,
            generated_token_ids=list(getattr(self, "_last_token_ids", [])),
            elapsed_seconds=float(getattr(self, "_last_elapsed", 0.0)),
            cancelled=bool(getattr(self, "_last_cancelled", False)),
            matched_stop=getattr(self, "_last_matched_stop", None),
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        params: GenerationParams,
        cancel: CancellationToken | None = None,
        *,
        _locked: bool = False,
    ) -> Iterator[StreamDelta]:
        """Yield deltas while a producer thread owns the GPU lock.

        The producer, not the caller's generator, holds ``self._lock``.  If the
        HTTP client disconnects and stops consuming this generator, the producer
        still observes cancellation and releases the lock.  The previous design
        held the lock inside the generator and could deadlock forever when a
        client vanished mid-stream.
        """
        if cancel is None:
            cancel = CancellationToken()
        deltas: queue.Queue[Any] = queue.Queue()
        done = object()

        def producer() -> None:
            try:
                with self._lock:
                    for delta in self._stream_locked(messages, params, cancel):
                        deltas.put(delta)
            except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
                deltas.put(exc)
            finally:
                if self._active_cancel is not None:
                    self._active_cancel["lock_released_seconds"] = round(
                        time.perf_counter() - self._last_cancel_requested_at, 4
                    )
                    self._active_cancel = None
                deltas.put(done)

        thread = threading.Thread(target=producer, name="yuzhu-stream-producer", daemon=True)
        thread.start()
        try:
            while True:
                item = deltas.get()
                if item is done:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # Consumer closed early (client disconnect): stop generation so the
            # producer releases the GPU lock at the next token.
            if not cancel.cancelled:
                cancel.cancel("consumer_closed")

    def _stream_locked(
        self,
        messages: list[dict[str, Any]],
        params: GenerationParams,
        cancel: CancellationToken,
    ) -> Iterator[StreamDelta]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        from transformers import TextIteratorStreamer

        self._active_cancel = None
        if cancel.cancelled:
            requested = cancel.requested_at or time.perf_counter()
            self._last_cancel_requested_at = requested
            record = {
                "reason": cancel.reason or "client",
                "generated_tokens": 0,
                "detect_to_stop_seconds": 0.0,
                "generation_end_seconds": round(
                    max(0.0, time.perf_counter() - requested), 4
                ),
            }
            self._last_cancel = record
            self._active_cancel = record
            usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            result = GenerationResult(
                text="",
                finish_reason="cancelled",
                usage=usage,
                generated_token_ids=[],
                elapsed_seconds=0.0,
                cancelled=True,
                cancel_reason=cancel.reason or "client",
            )
            yield StreamDelta(
                text="", finish_reason="cancelled", usage=usage, result=result
            )
            return

        loaded = self._require_loaded()
        tokenizer = loaded.tokenizer
        model = loaded.model

        max_new_tokens = self._effective_max_tokens(params.max_new_tokens)
        params = GenerationParams(
            max_new_tokens=max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            stop=list(params.stop),
            tools=params.tools,
        )
        input_ids, attention_mask = self._render_inputs(messages, params)
        prompt_length = int(input_ids.shape[1])

        class _Criteria(StoppingCriteria):
            def __init__(self) -> None:
                self.reason: str | None = None
                self.observed_at: float | None = None

            def __call__(self, current_ids, scores, **kwargs):  # type: ignore[no-untyped-def]
                if cancel.cancelled:
                    if self.observed_at is None:
                        self.observed_at = time.perf_counter()
                    self.reason = cancel.reason or "client"
                    return torch.ones(current_ids.shape[0], dtype=torch.bool, device=current_ids.device)
                if not params.stop:
                    return torch.zeros(current_ids.shape[0], dtype=torch.bool, device=current_ids.device)
                generated = current_ids[0, prompt_length:]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                for stop in params.stop:
                    if stop and stop in text:
                        self.reason = "stop"
                        return torch.ones(
                            current_ids.shape[0], dtype=torch.bool, device=current_ids.device
                        )
                return torch.zeros(current_ids.shape[0], dtype=torch.bool, device=current_ids.device)

        criteria = _Criteria()
        preserve_tool_syntax = bool(params.tools)
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=not preserve_tool_syntax,
            timeout=None,
        )
        holder: dict[str, Any] = {}
        started = time.perf_counter()

        def _worker() -> None:
            try:
                generate_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=params.temperature > 0,
                    temperature=params.temperature if params.temperature > 0 else None,
                    top_p=params.top_p if params.temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=model.config.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([criteria]),
                    streamer=streamer,
                )
                # H5.2: PEFT wrappers hide ``logits_to_keep`` from generate's
                # support check, so prefill computes logits for every prompt
                # token (130k vocab -> 1.5-3 GiB on a 5k prompt).  Ask for the
                # last token only; fall back if the wrapped model rejects it.
                with torch.inference_mode():
                    try:
                        output = model.generate(**generate_kwargs, logits_to_keep=1)
                    except TypeError:
                        output = model.generate(**generate_kwargs)
                holder["output"] = output
            except BaseException as exc:  # noqa: BLE001 - surfaced to the request
                holder["error"] = exc
                end = getattr(streamer, "end", None)
                if callable(end):
                    end()

        thread = threading.Thread(target=_worker, name="yuzhu-generate", daemon=True)
        thread.start()

        watchdog: threading.Timer | None = None
        if self.settings.request_timeout_seconds > 0:
            watchdog = threading.Timer(
                self.settings.request_timeout_seconds, cancel.cancel, kwargs={"reason": "timeout"}
            )
            watchdog.daemon = True
            watchdog.start()

        stop_buffer = _StopBuffer(params.stop)
        collected: list[str] = []
        try:
            for text in streamer:
                if not text:
                    continue
                emit = stop_buffer.feed(text)
                if emit:
                    collected.append(emit)
                    yield StreamDelta(text=emit)
                if stop_buffer.matched is not None:
                    cancel.cancel(reason="stop")
        finally:
            if watchdog is not None:
                watchdog.cancel()
            # A generator may be closed from the worker thread when the HTTP
            # client disconnects; joining the current thread raises RuntimeError.
            if thread.is_alive() and thread is not threading.current_thread():
                cancel.cancel(reason=cancel.reason or "client")
                thread.join(timeout=30)
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - cleanup must not mask the result
                pass
            if "error" in holder:
                raise RuntimeError(f"generation failed: {holder['error']}") from holder["error"]

        output = holder.get("output")
        if output is None:
            raise RuntimeError("generation produced no output")

        tail = stop_buffer.flush()
        if tail:
            collected.append(tail)
            yield StreamDelta(text=tail)

        generated_ids = output[0, prompt_length:].tolist()
        if criteria.reason in {"client", "timeout"}:
            self._record_cancel(
                cancel,
                observed_at=criteria.observed_at or time.perf_counter(),
                generated_tokens=len(generated_ids),
            )
        raw_text = tokenizer.decode(
            generated_ids, skip_special_tokens=not preserve_tool_syntax
        )
        if preserve_tool_syntax:
            raw_text = strip_control_tokens(raw_text)
        text_value, matched_stop = _truncate_at_stop(raw_text, params.stop)

        usage = Usage(
            prompt_tokens=prompt_length,
            completion_tokens=len(generated_ids),
            total_tokens=prompt_length + len(generated_ids),
        )
        elapsed = time.perf_counter() - started

        if criteria.reason in {"client", "timeout"}:
            finish_reason = "cancelled"
        elif matched_stop is not None or criteria.reason == "stop":
            finish_reason = "stop"
        elif len(generated_ids) >= max_new_tokens:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        self._last_text = text_value
        self._last_token_ids = generated_ids
        self._last_elapsed = round(elapsed, 4)
        self._last_cancelled = criteria.reason in {"client", "timeout"}
        self._last_matched_stop = matched_stop

        final_result = GenerationResult(
            text=text_value,
            finish_reason=finish_reason,
            usage=usage,
            generated_token_ids=list(generated_ids),
            elapsed_seconds=round(elapsed, 4),
            cancelled=criteria.reason in {"client", "timeout"},
            matched_stop=matched_stop,
            cancel_reason=criteria.reason,
        )
        yield StreamDelta(
            text="",
            finish_reason=finish_reason,
            usage=usage,
            result=final_result,
        )