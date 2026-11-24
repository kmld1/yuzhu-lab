"""Strict structured-action fallback.

MiniCPM5's native tool syntax is the primary channel.  When native parsing
produces nothing, the harness accepts exactly one fenced JSON block labelled
``yuzhu-action`` and nothing else.  Prose is never interpreted as a tool call.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_ACTION_BLOCK = re.compile(
    r"```yuzhu-action[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE
)


class ActionParseError(ValueError):
    pass


@dataclass(frozen=True)
class StructuredAction:
    tool: str
    arguments: dict[str, Any]
    text_without_block: str


def extract_action(text: str) -> tuple[str, StructuredAction | None]:
    """Return the remaining text and at most one validated action."""
    matches = list(_ACTION_BLOCK.finditer(text))
    if not matches:
        return text, None
    if len(matches) > 1:
        raise ActionParseError("at most one yuzhu-action block is allowed per reply")
    match = matches[0]
    payload_text = match.group(1).strip()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"yuzhu-action block is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ActionParseError("yuzhu-action payload must be a JSON object")
    tool = payload.get("tool")
    arguments = payload.get("arguments", {})
    if not isinstance(tool, str) or not tool.strip():
        raise ActionParseError("yuzhu-action payload requires a non-empty 'tool'")
    if not isinstance(arguments, dict):
        raise ActionParseError("yuzhu-action 'arguments' must be an object")
    remaining = (text[: match.start()] + text[match.end():]).strip()
    return remaining, StructuredAction(tool=tool.strip(), arguments=arguments, text_without_block=remaining)


def action_instructions() -> str:
    return (
        "If you cannot use the native function syntax, emit exactly one fenced block:\n"
        '```yuzhu-action\n{"tool": "<tool_name>", "arguments": {...}}\n```\n'
        "Do not put any other instructions inside that block. Plain prose is treated "
        "as a final answer, never as a tool call."
    )