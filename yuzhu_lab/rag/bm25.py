"""Chinese-aware tokenization and reciprocal rank fusion."""

from __future__ import annotations

import re
from typing import Iterable

import jieba

CHINESE_RE = re.compile(r"^[一-鿿]+$")
TOKEN_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_./:\\-]*"
    r"|[0-9]+(?:\.[0-9]+)*"
    r"|[一-鿿]+"
)


def smart_tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for part in TOKEN_RE.findall(text):
        if CHINESE_RE.fullmatch(part):
            tokens.extend(token.strip() for token in jieba.lcut(part) if token.strip())
        else:
            tokens.append(part.lower())
    return tokens


def prepare_for_bm25(text: str) -> str:
    return " ".join(smart_tokenize(text))


def reciprocal_rank_fusion(
    rankings: Iterable[Iterable[str]], rrf_k: int = 60
) -> list[tuple[str, float]]:
    """Fuse rankings of chunk ids. Scores from different retrievers are never added."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def search_text(chunk: dict) -> str:
    parts = [chunk.get("source", ""), chunk.get("symbol") or "", chunk.get("heading") or ""]
    parts.append(chunk.get("content", ""))
    return "\n".join(part for part in parts if part)