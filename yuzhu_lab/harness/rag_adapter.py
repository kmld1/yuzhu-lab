"""Adapter over the existing ``yuzhu_lab.rag.RagIndex``.

The index is loaded once per process and reused.  The adapter never rebuilds
the index, never re-embeds at query time beyond the query itself, and degrades
to an empty evidence list when the index or reranker is unavailable.

Every result is admitted through the shared :mod:`yuzhu_lab.source_policy`.
Filtering is applied before fusion/reranking by the index and again after
retrieval here, so an old or poisoned index cannot expose protected evaluation
assets.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..rag.search import RagIndex, RetrievalError, SearchResult
from ..source_policy import SourcePolicy
from .context_models import RagEvidenceItem
from .context_policy import sanitize_untrusted

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_INDEX_CACHE: dict[tuple[str, str, str], RagIndex] = {}
_INDEX_LOCK = threading.Lock()


@dataclass
class RagSearchOutcome:
    items: list[RagEvidenceItem] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    index_loaded: bool = False
    denied_sources: int = 0


class RagSearchPort(Protocol):
    def search(
        self,
        *,
        project_id: str,
        query: str,
        token_budget: int,
        final_k: int = 5,
    ) -> RagSearchOutcome: ...


class RagAdapter:
    def __init__(
        self,
        *,
        store_dir: Path,
        embedding_model_path: Path | str,
        device: str = "cpu",
        reranker_model_path: Path | str | None = None,
        use_reranker: bool = False,
        workspace: Path | None = None,
        allowed_prefixes: tuple[str, ...] = (),
        source_policy: SourcePolicy | None = None,
        denied_prefixes: tuple[str, ...] = (),
        denied_sources: tuple[str, ...] = (),
        index: RagIndex | None = None,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.embedding_model_path = str(embedding_model_path)
        self.device = device
        self.reranker_model_path = reranker_model_path
        self.use_reranker = use_reranker
        self.workspace = Path(workspace) if workspace else None
        self.allowed_prefixes = tuple(
            prefix.replace("\\", "/").strip("/")
            for prefix in allowed_prefixes
            if prefix
        )
        self.source_policy = source_policy or SourcePolicy(
            self.workspace or PROJECT_ROOT,
            extra_protected_prefixes=tuple(denied_prefixes),
            extra_protected_files=tuple(denied_sources),
        )
        self._index = index
        self._loaded_here = False

    def _ensure_index(self) -> RagIndex:
        if self._index is not None:
            return self._index
        key = (str(self.store_dir), self.embedding_model_path, self.device)
        with _INDEX_LOCK:
            cached = _INDEX_CACHE.get(key)
            if cached is None:
                cached = RagIndex(
                    self.store_dir,
                    embedding_model_path=self.embedding_model_path,
                    device=self.device,
                    reranker_model_path=self.reranker_model_path,
                ).load()
                _INDEX_CACHE[key] = cached
            self._index = cached
            self._loaded_here = True
            return cached

    def _source_allowed(self, source: str, citation: str | None = None) -> bool:
        decision = self.source_policy.check_relative(source, require_exists=True)
        if not decision.allowed:
            return False
        normalized = decision.normalized or ""
        if self.allowed_prefixes and not any(
            normalized.casefold() == prefix.casefold()
            or normalized.casefold().startswith(prefix.casefold() + "/")
            for prefix in self.allowed_prefixes
        ):
            return False
        if citation is not None:
            claimed = self.source_policy.check_citation(citation)
            if not claimed.allowed:
                return False
            if (claimed.normalized or "").casefold() != normalized.casefold():
                return False
        return True

    def search(
        self,
        *,
        project_id: str,
        query: str,
        token_budget: int,
        final_k: int = 5,
    ) -> RagSearchOutcome:
        started = time.perf_counter()
        index_loaded = self._index is not None
        try:
            index = self._ensure_index()
        except Exception as exc:  # noqa: BLE001 - RAG is optional context
            return RagSearchOutcome(
                degraded=[f"rag index load: {type(exc).__name__}: {exc}"],
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        try:
            results = index.search(
                query,
                retrieve_k=max(20, final_k * 4),
                rerank_k=max(10, final_k * 2),
                final_k=final_k,
                token_budget=token_budget,
                use_reranker=self.use_reranker,
                admit_source=self._source_allowed,
            )
        except RetrievalError as exc:
            return RagSearchOutcome(
                degraded=[f"rag retrieval: {exc}"],
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                index_loaded=index_loaded or self._loaded_here,
            )
        except Exception as exc:  # noqa: BLE001 - degrade, never fabricate evidence
            return RagSearchOutcome(
                degraded=[f"rag: {type(exc).__name__}: {exc}"],
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                index_loaded=index_loaded or self._loaded_here,
            )

        items: list[RagEvidenceItem] = []
        denied = 0
        for result in results:
            if not self._source_allowed(result.source, result.citation):
                denied += 1
                continue
            items.append(self._to_item(result, project_id))
        degraded = list(index.last_degraded)
        return RagSearchOutcome(
            items=items,
            degraded=degraded,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            index_loaded=index_loaded or self._loaded_here,
            denied_sources=denied,
        )

    @staticmethod
    def _to_item(result: SearchResult, project_id: str) -> RagEvidenceItem:
        content, flags = sanitize_untrusted(result.content)
        return RagEvidenceItem(
            chunk_id=result.chunk_id,
            citation=result.citation,
            source=result.source,
            symbol=result.symbol,
            start_line=result.start_line,
            end_line=result.end_line,
            content=content,
            score=result.score,
            rrf_score=result.rrf_score,
            injection_flags=flags,
            project_id=project_id,
        )


class NullRagAdapter:
    def search(
        self,
        *,
        project_id: str,
        query: str,
        token_budget: int,
        final_k: int = 5,
    ) -> RagSearchOutcome:
        return RagSearchOutcome(degraded=["rag: adapter disabled"])