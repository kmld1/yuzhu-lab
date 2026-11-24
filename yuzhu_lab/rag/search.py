"""Hybrid retrieval with RRF, optional reranking, deduplication and token budget."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .bm25 import prepare_for_bm25, reciprocal_rank_fusion, search_text
from .store import SentenceTransformerEmbedder, load_manifest


class RetrievalError(RuntimeError):
    """Raised when every retrieval channel failed."""


@dataclass
class SearchResult:
    chunk_id: str
    source: str
    symbol: str | None
    heading: str | None
    start_line: int | None
    end_line: int | None
    content: str
    citation: str
    score: float
    rrf_score: float
    faiss_rank: int | None = None
    bm25_rank: int | None = None
    reranker_score: float | None = None


def _citation(chunk: dict[str, Any]) -> str:
    location = chunk["source"]
    if chunk.get("start_line") is not None:
        location += f":{chunk['start_line']}"
        if chunk.get("end_line") not in (None, chunk.get("start_line")):
            location += f"-{chunk['end_line']}"
    if chunk.get("symbol"):
        location += f" ({chunk['symbol']})"
    return location


def _estimate_tokens(text: str) -> int:
    # Conservative mixed Chinese/English estimate; callers can inject a real
    # tokenizer via token_counter for exact budgets.
    return max(1, len(text) // 3)


class RagIndex:
    def __init__(
        self,
        store_dir: Path,
        *,
        embedding_model_path: Path | str | None = None,
        device: str = "cpu",
        reranker_model_path: Path | str | None = None,
        token_counter: Callable[[str], int] | None = None,
        embedder: Any = None,
        reranker: Any = None,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.embedding_model_path = embedding_model_path
        self.device = device
        self.reranker_model_path = reranker_model_path
        self.token_counter = token_counter or _estimate_tokens

        self.manifest: dict[str, Any] = {}
        self.chunks: list[dict[str, Any]] = []
        self.chunk_map: dict[str, dict[str, Any]] = {}
        self.index: Any = None
        self.bm25: Any = None
        self.bm25_tokenizer: Any = None
        self._embedder: Any = embedder
        self._reranker: Any = reranker
        self.last_degraded: list[str] = []

    # ------------------------------------------------------------------ loading

    def load(self) -> "RagIndex":
        import bm25s
        import faiss
        from bm25s.tokenization import Tokenizer

        self.manifest = load_manifest(self.store_dir)
        chunks_path = self.store_dir / "chunks.jsonl"
        self.chunks = []
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))
        self.chunk_map = {chunk["chunk_id"]: chunk for chunk in self.chunks}

        index_path = self.store_dir / "index.faiss"
        self.index = faiss.read_index(str(index_path)) if index_path.is_file() else None

        bm25_dir = self.store_dir / "bm25"
        if self.chunks and (bm25_dir / "params.index.json").is_file():
            self.bm25 = bm25s.BM25.load(bm25_dir, load_corpus=True)
            self.bm25_tokenizer = Tokenizer(splitter=lambda text: text.split())
            self.bm25_tokenizer.load_vocab(bm25_dir)
        return self

    def _get_embedder(self):
        if self._embedder is None:
            if self.embedding_model_path is None:
                raise RetrievalError("embedding_model_path is required for dense retrieval")
            self._embedder = SentenceTransformerEmbedder(
                self.embedding_model_path, device=self.device
            )
        return self._embedder

    def _get_reranker(self):
        if self._reranker is None:
            if self.reranker_model_path is None:
                raise RetrievalError("reranker_model_path is required for reranking")
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(str(self.reranker_model_path), device=self.device)
        return self._reranker

    # ----------------------------------------------------------------- searching

    def search(
        self,
        query: str,
        *,
        retrieve_k: int = 20,
        rerank_k: int = 10,
        final_k: int = 5,
        token_budget: int = 2000,
        use_reranker: bool = False,
        admit_source: Callable[[str], bool] | None = None,
    ) -> list[SearchResult]:
        self.last_degraded = []
        if not self.chunks:
            raise RetrievalError("index is empty")

        faiss_ranking = self._dense_search(query, retrieve_k)
        bm25_ranking = self._bm25_search(query, retrieve_k)
        if admit_source is not None:
            # Filter before fusion/reranking so protected chunks cannot consume
            # candidate slots or reach the reranker model.
            faiss_ranking = [
                chunk_id
                for chunk_id in faiss_ranking
                if chunk_id in self.chunk_map
                and admit_source(str(self.chunk_map[chunk_id].get("source") or ""))
            ]
            bm25_ranking = [
                chunk_id
                for chunk_id in bm25_ranking
                if chunk_id in self.chunk_map
                and admit_source(str(self.chunk_map[chunk_id].get("source") or ""))
            ]
        if not faiss_ranking and not bm25_ranking:
            raise RetrievalError(
                "all retrieval channels failed: " + "; ".join(self.last_degraded or ["no results"])
            )

        fused = reciprocal_rank_fusion([faiss_ranking, bm25_ranking])
        faiss_rank = {chunk_id: rank for rank, chunk_id in enumerate(faiss_ranking, start=1)}
        bm25_rank = {chunk_id: rank for rank, chunk_id in enumerate(bm25_ranking, start=1)}

        candidates = fused[: max(rerank_k, final_k)]
        reranked = self._rerank(query, candidates) if use_reranker else None

        results: list[SearchResult] = []
        used_tokens = 0
        for chunk_id, rrf_score in candidates:
            chunk = self.chunk_map[chunk_id]
            if admit_source is not None and not admit_source(
                str(chunk.get("source") or "")
            ):
                continue
            if reranked is not None and reranked:
                score = reranked.get(chunk_id, 0.0)
            else:
                score = rrf_score
            cost = self.token_counter(chunk["content"])
            if results and used_tokens + cost > token_budget:
                continue
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    source=chunk["source"],
                    symbol=chunk.get("symbol"),
                    heading=chunk.get("heading"),
                    start_line=chunk.get("start_line"),
                    end_line=chunk.get("end_line"),
                    content=chunk["content"],
                    citation=_citation(chunk),
                    score=float(score),
                    rrf_score=float(rrf_score),
                    faiss_rank=faiss_rank.get(chunk_id),
                    bm25_rank=bm25_rank.get(chunk_id),
                    reranker_score=(
                        float(reranked[chunk_id])
                        if reranked is not None and chunk_id in reranked
                        else None
                    ),
                )
            )
            used_tokens += cost
            if len(results) >= final_k:
                break

        if reranked is not None:
            results.sort(key=lambda item: item.score, reverse=True)
        else:
            results.sort(key=lambda item: item.rrf_score, reverse=True)
        return results

    def _dense_search(self, query: str, retrieve_k: int) -> list[str]:
        if self.index is None or self.index.ntotal == 0:
            return []
        try:
            vector = self._get_embedder().encode_query(query).reshape(1, -1).astype("float32")
            k = min(retrieve_k, self.index.ntotal)
            _, indices = self.index.search(vector, k)
            return [
                self.chunks[int(index)]["chunk_id"]
                for index in indices[0]
                if 0 <= int(index) < len(self.chunks)
            ]
        except Exception as exc:  # noqa: BLE001 - degraded to BM25
            self.last_degraded.append(f"dense: {type(exc).__name__}: {exc}")
            return []

    def _bm25_search(self, query: str, retrieve_k: int) -> list[str]:
        if self.bm25 is None:
            return []
        try:
            tokens = self.bm25_tokenizer.tokenize(
                [prepare_for_bm25(query)], update_vocab=False
            )
            results, _ = self.bm25.retrieve(tokens, k=min(retrieve_k, len(self.chunks)))
            return [
                item["chunk_id"]
                for item in results[0]
                if isinstance(item, dict) and "chunk_id" in item
            ]
        except Exception as exc:  # noqa: BLE001 - degraded to dense
            self.last_degraded.append(f"bm25: {type(exc).__name__}: {exc}")
            return []

    def _rerank(self, query: str, candidates: list[tuple[str, float]]) -> dict[str, float]:
        try:
            reranker = self._get_reranker()
            pairs = [
                (query, search_text(self.chunk_map[chunk_id])) for chunk_id, _ in candidates
            ]
            scores = reranker.predict(pairs, show_progress_bar=False)
            return {
                chunk_id: float(score)
                for (chunk_id, _), score in zip(candidates, scores)
            }
        except Exception as exc:  # noqa: BLE001 - fall back to RRF order
            self.last_degraded.append(f"reranker: {type(exc).__name__}: {exc}")
            return {}


def format_context(results: list[SearchResult]) -> str:
    blocks: list[str] = []
    for number, result in enumerate(results, start=1):
        blocks.append(f"[{number}] {result.citation}\n{result.content}")
    return "\n\n".join(blocks)