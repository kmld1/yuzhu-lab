"""Build and query the project hybrid index.

    python -m yuzhu_lab.rag.cli build --root . --store rag/project_index
    python -m yuzhu_lab.rag.cli search "hybrid retrieval" --store rag/project_index
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from ..source_policy import SourcePolicy
from .search import RagIndex, format_context
from .store import SentenceTransformerEmbedder, build_index

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = PROJECT_ROOT / "rag" / "project_index"
DEFAULT_EMBEDDING = PROJECT_ROOT / "models" / "embedding" / "Qwen3-Embedding-0.6B"
DEFAULT_RERANKER = PROJECT_ROOT / "models" / "reranker" / "Qwen3-Reranker-0.6B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YUZHU LAB hybrid RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build or incrementally update the index")
    build.add_argument("--root", default=str(PROJECT_ROOT))
    build.add_argument("--store", default=str(DEFAULT_STORE))
    build.add_argument("--embedding", default=str(DEFAULT_EMBEDDING))
    build.add_argument("--device", default="cpu")
    build.add_argument("--force", action="store_true")

    search = subparsers.add_parser("search", help="query the index")
    search.add_argument("query")
    search.add_argument("--store", default=str(DEFAULT_STORE))
    search.add_argument("--embedding", default=str(DEFAULT_EMBEDDING))
    search.add_argument("--reranker", default=str(DEFAULT_RERANKER))
    search.add_argument("--device", default="cpu")
    search.add_argument("--k", type=int, default=5)
    search.add_argument("--retrieve-k", type=int, default=20)
    search.add_argument("--rerank-k", type=int, default=10)
    search.add_argument("--token-budget", type=int, default=2000)
    search.add_argument("--rerank", action="store_true")
    search.add_argument("--json", action="store_true")
    search.add_argument("--context", action="store_true", help="print citation context")
    search.add_argument("--output", help="write the result as UTF-8 JSON to this file")
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()
    if args.command == "build":
        root = Path(args.root).resolve()
        embedder = SentenceTransformerEmbedder(args.embedding, device=args.device)
        stats = build_index(
            root,
            Path(args.store),
            embedder,
            force=args.force,
            source_policy=SourcePolicy(root),
        )
        print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
        return 0

    index = RagIndex(
        Path(args.store),
        embedding_model_path=args.embedding,
        device=args.device,
        reranker_model_path=args.reranker,
    )
    index.load()
    index_root = Path(index.manifest.get("root") or PROJECT_ROOT).resolve()
    policy = SourcePolicy(index_root)
    results = index.search(
        args.query,
        retrieve_k=args.retrieve_k,
        rerank_k=args.rerank_k,
        final_k=args.k,
        token_budget=args.token_budget,
        use_reranker=args.rerank,
        admit_source=lambda source: policy.check_relative(
            source, require_exists=True
        ).allowed,
    )
    payload = [
        {
            "citation": item.citation,
            "score": item.score,
            "rrf_score": item.rrf_score,
            "faiss_rank": item.faiss_rank,
            "bm25_rank": item.bm25_rank,
            "reranker_score": item.reranker_score,
            "content": item.content,
        }
        for item in results
    ]
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {len(payload)} results to {args.output}")
        if index.last_degraded:
            print("degraded:", "; ".join(index.last_degraded))
        return 0
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.context:
        print(format_context(results))
        return 0

    for number, item in enumerate(results, start=1):
        print(
            f"[{number}] {item.citation} score={item.score:.6f} "
            f"rrf={item.rrf_score:.6f} faiss={item.faiss_rank} bm25={item.bm25_rank}"
        )
        print(item.content[:500])
        print()
    if index.last_degraded:
        print("degraded:", "; ".join(index.last_degraded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())