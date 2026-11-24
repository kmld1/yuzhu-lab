"""Conservative token budgeting for mixed Chinese/English context.

The context builder can be given an exact counter backed by the serving
runtime (``ContextBuilderConfig.token_counter``); the estimate below is only a
fallback for callers that do not have a tokenizer at the context boundary.
The fallback deliberately over-estimates CJK text: character counts alone must
never be the final authority for a request that a 6144-token guard rejects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

TokenCounter = Callable[[str], int]

# CJK ideographs, kana, Hangul, CJK punctuation and full-width forms all tend to
# cost roughly one token per two to three characters in current tokenizers.
_CJK_RANGES = (
    (0x3000, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF),
)


def _is_cjk(character: str) -> bool:
    code = ord(character)
    return any(start <= code <= end for start, end in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """Conservative fallback when no serving tokenizer is available.

    Measured on the frozen MiniCPM5 prompt: 6748 characters cost 2822 tokens,
    i.e. about 0.42 tokens per character.  Using 0.6 for CJK and 0.3 for the
    rest keeps a safety margin without the threefold under-count of ``len/3``.
    """
    if not text:
        return 0
    cjk = 0
    for character in text:
        if _is_cjk(character):
            cjk += 1
    other = len(text) - cjk
    return max(1, int(cjk * 0.6 + other * 0.3) + 1)


@dataclass(frozen=True)
class BudgetPlan:
    total: int
    system: int
    summary: int
    memory: int
    rag: int


def plan_budget(total: int, *, summary_tokens: int = 0) -> BudgetPlan:
    if total < 256:
        raise ValueError("model_context_budget must be at least 256 tokens")
    system = min(160, max(48, total // 32))
    summary = min(summary_tokens, max(0, total // 5))
    memory = min(max(128, total // 4), 1800)
    rag = total - system - summary - memory - 64
    if rag < 128:
        # Tight budgets favour current evidence over historical memory.
        memory = max(64, memory // 2)
        rag = total - system - summary - memory - 32
    return BudgetPlan(
        total=total,
        system=system,
        summary=summary,
        memory=max(0, memory),
        rag=max(0, rag),
    )


def truncate_text(
    text: str, max_tokens: int, counter: TokenCounter | None = None
) -> tuple[str, bool]:
    """Trim so that the same counter used for planning stays within the budget."""
    if max_tokens <= 0:
        return "", True
    count = counter or estimate_tokens
    if count(text) <= max_tokens:
        return text, False
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if count(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    candidate = text[:low]
    newline = candidate.rfind("\n")
    if newline >= len(candidate) // 2:
        candidate = candidate[:newline]
    return candidate.rstrip() + "\n[...truncated]", True