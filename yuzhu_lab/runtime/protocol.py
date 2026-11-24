"""Protocol models and conversions for OpenAI and Anthropic compatible endpoints."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel


class UnsupportedContentError(ValueError):
    """Raised when a request contains content the text-only runtime cannot handle."""


# --------------------------------------------------------------------- OpenAI


class OpenAIChatMessage(BaseModel):
    role: str
    content: str | list[Any] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[OpenAIChatMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None


# -------------------------------------------------------------------- Anthropic


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]


class AnthropicMessagesRequest(BaseModel):
    model: str | None = None
    system: str | list[dict[str, Any]] | None = None
    messages: list[AnthropicMessage]
    max_tokens: int = 512
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    metadata: dict[str, Any] | None = None


class AnthropicCountTokensRequest(BaseModel):
    model: str | None = None
    system: str | list[dict[str, Any]] | None = None
    messages: list[AnthropicMessage]
    tools: list[dict[str, Any]] | None = None


# ------------------------------------------------------------------ conversion


def _text_from_blocks(blocks: list[Any], *, kind: str) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            raise UnsupportedContentError(f"unsupported {kind} content block: {type(block).__name__}")
        block_type = block.get("type")
        if block_type == "text":
            parts.append(str(block.get("text", "")))
        else:
            raise UnsupportedContentError(f"unsupported {kind} content block type: {block_type}")
    return "\n".join(part for part in parts if part)


def _content_to_text(content: str | list[Any] | None, *, kind: str) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return _text_from_blocks(content, kind=kind)


def openai_messages_to_internal(
    messages: list[OpenAIChatMessage | dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, dict):
            message = OpenAIChatMessage.model_validate(message)
        role = message.role
        if role == "tool":
            result.append({"role": "tool", "content": _content_to_text(message.content, kind="tool")})
            continue
        if role not in {"system", "user", "assistant"}:
            raise UnsupportedContentError(f"unsupported OpenAI role: {role}")
        item: dict[str, Any] = {
            "role": role,
            "content": _content_to_text(message.content, kind=role),
        }
        if role == "assistant" and message.tool_calls:
            calls: list[dict[str, Any]] = []
            for call in message.tool_calls:
                function = call.get("function", call)
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"_raw": arguments}
                calls.append({"name": function.get("name", ""), "arguments": arguments})
            item["tool_calls"] = calls
        result.append(item)
    return result


def anthropic_messages_to_internal(
    request: AnthropicMessagesRequest,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if request.system:
        if isinstance(request.system, str):
            system_text = request.system
        else:
            system_text = _text_from_blocks(request.system, kind="system")
        result.append({"role": "system", "content": system_text})

    for message in request.messages:
        if isinstance(message.content, str):
            result.append({"role": message.role, "content": message.content})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[str] = []
        for block in message.content:
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text", "")))
            elif block_type == "tool_result":
                tool_results.append(_tool_result_text(block))
            elif block_type == "tool_use" and message.role == "assistant":
                tool_calls.append(
                    {
                        "name": block.get("name", ""),
                        "arguments": block.get("input", {}),
                    }
                )
            else:
                raise UnsupportedContentError(
                    f"unsupported Anthropic content block: {block_type} in {message.role}"
                )

        if message.role == "user" and tool_results:
            for tool_text in tool_results:
                result.append({"role": "tool", "content": tool_text})
            if text_parts:
                result.append({"role": "user", "content": "\n".join(text_parts)})
            continue

        item: dict[str, Any] = {"role": message.role, "content": "\n".join(text_parts)}
        if tool_calls:
            item["tool_calls"] = tool_calls
        result.append(item)
    return result


def _tool_result_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _text_from_blocks(content, kind="tool_result")
    return json.dumps(content, ensure_ascii=False)


def anthropic_tools_to_template(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """MiniCPM5's template serialises tool definitions as JSON; pass them through."""
    if not tools:
        return None
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        normalized.append(tool)
    return normalized or None


def openai_tools_to_template(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            normalized.append(
                {
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "input_schema": function.get("parameters", {"type": "object"}),
                }
            )
        elif "name" in tool:
            normalized.append(tool)
    return normalized or None


def stop_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return [item for item in value if item]


# -------------------------------------------------------------------- responses


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def openai_completion(
    *,
    model: str,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str,
) -> dict[str, Any]:
    return {
        "id": _new_id("chatcmpl"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop" if finish_reason == "cancelled" else finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def openai_chunk(
    *,
    response_id: str,
    created: int,
    model: str,
    text: str,
    finish_reason: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": ({"content": text} if text else {}),
                "finish_reason": finish_reason,
            }
        ],
    }
    if prompt_tokens is not None and completion_tokens is not None:
        payload["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    return payload


def anthropic_message(
    *,
    model: str,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str,
    stop_sequence: str | None,
    tool_calls: list[ParsedToolCall] | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for call in tool_calls or []:
        content.append(
            {
                "type": "tool_use",
                "id": new_tool_use_id(),
                "name": call.name,
                "input": call.arguments,
            }
        )
    return {
        "id": _new_id("msg"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": anthropic_stop_reason(
            finish_reason, stop_sequence, has_tool_calls=bool(tool_calls)
        ),
        "stop_sequence": stop_sequence,
        "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
    }


def anthropic_stop_reason(
    finish_reason: str, stop_sequence: str | None, *, has_tool_calls: bool = False
) -> str:
    if has_tool_calls:
        return "tool_use"
    if stop_sequence is not None:
        return "stop_sequence"
    if finish_reason == "length":
        return "max_tokens"
    return "end_turn"


def anthropic_message_start(
    *, message_id: str, model: str, prompt_tokens: int
) -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
        },
    }


def anthropic_content_block_start(index: int = 0) -> dict[str, Any]:
    return {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "text", "text": ""},
    }


def anthropic_text_delta(text: str, index: int = 0) -> dict[str, Any]:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }


def anthropic_content_block_stop(index: int = 0) -> dict[str, Any]:
    return {"type": "content_block_stop", "index": index}


def anthropic_tool_use_block_start(index: int, tool_id: str, name: str) -> dict[str, Any]:
    return {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}},
    }


def anthropic_input_json_delta(index: int, partial_json: str) -> dict[str, Any]:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    }


def anthropic_message_delta(
    *,
    finish_reason: str,
    stop_sequence: str | None,
    output_tokens: int,
    has_tool_calls: bool = False,
) -> dict[str, Any]:
    return {
        "type": "message_delta",
        "delta": {
            "stop_reason": anthropic_stop_reason(
                finish_reason, stop_sequence, has_tool_calls=has_tool_calls
            ),
            "stop_sequence": stop_sequence,
        },
        "usage": {"output_tokens": output_tokens},
    }


def anthropic_message_stop() -> dict[str, Any]:
    return {"type": "message_stop"}


def new_anthropic_id() -> str:
    return _new_id("msg")


def new_openai_id() -> str:
    return _new_id("chatcmpl")


def new_tool_use_id() -> str:
    return _new_id("toolu")


# ------------------------------------------------------------------ tool parse

_FUNCTION_RE = re.compile(
    r"<function\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</function>", re.DOTALL
)
_PARAM_RE = re.compile(
    r"<param\s+name=[\"']([^\"']+)[\"'][^>]*>(.*?)</param>", re.DOTALL
)
_CDATA_RE = re.compile(r"^<!\[CDATA\[(.*)\]\]>$", re.DOTALL)


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]


def _decode_param(raw: str) -> Any:
    value = raw.strip()
    cdata = _CDATA_RE.match(value)
    if cdata:
        value = cdata.group(1)
    else:
        if value.startswith("{") or value.startswith("["):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        if value in {"true", "false"}:
            return value == "true"
        if value == "null":
            return None
        if re.fullmatch(r"-?\d+", value):
            return int(value)
        if re.fullmatch(r"-?\d+\.\d+", value):
            return float(value)
    return value


def parse_tool_calls(text: str) -> tuple[str, list[ParsedToolCall]]:
    """Parse MiniCPM5's XML function syntax and remove calls from the visible text.

    The model is not trusted: only complete, well-formed `<function>` blocks become
    tool calls. Anything else stays in the text channel.
    """
    calls: list[ParsedToolCall] = []
    spans: list[tuple[int, int]] = []
    for match in _FUNCTION_RE.finditer(text):
        name = match.group(1).strip()
        if not name:
            continue
        arguments: dict[str, Any] = {}
        for param in _PARAM_RE.finditer(match.group(2)):
            arguments[param.group(1).strip()] = _decode_param(param.group(2))
        calls.append(ParsedToolCall(name=name, arguments=arguments))
        spans.append(match.span())
    if not calls:
        return text, []

    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    cleaned = "".join(pieces).strip()
    return cleaned, calls


__all__ = [
    "AnthropicCountTokensRequest",
    "AnthropicMessage",
    "AnthropicMessagesRequest",
    "ChatCompletionRequest",
    "OpenAIChatMessage",
    "ParsedToolCall",
    "UnsupportedContentError",
    "anthropic_content_block_start",
    "anthropic_content_block_stop",
    "anthropic_input_json_delta",
    "anthropic_message",
    "anthropic_message_delta",
    "anthropic_message_start",
    "anthropic_message_stop",
    "anthropic_messages_to_internal",
    "anthropic_text_delta",
    "anthropic_tool_use_block_start",
    "anthropic_tools_to_template",
    "new_anthropic_id",
    "new_openai_id",
    "new_tool_use_id",
    "openai_chunk",
    "openai_completion",
    "openai_messages_to_internal",
    "openai_tools_to_template",
    "parse_tool_calls",
    "stop_list",
]