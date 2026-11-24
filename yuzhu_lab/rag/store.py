"""Build and persist the hybrid index (FAISS + BM25 + chunk metadata)."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..source_policy import SourcePolicy
from .bm25 import prepare_for_bm25, search_text
from .chunking import Chunk, chunk_file, deduplicate, discover_files

CHUNKER_VERSION = "1"
INDEX_VERSION = 1


class Embedder(Protocol):
    dimension: int

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, query: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(str(model_path), device=device)
        self.model.max_seq_length = 1024
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("embedding model returned no dimension")
        self.dimension = int(dimension)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=4,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 64,
            ),
            dtype="float32",
        )

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                [query],
                batch_size=1,
                prompt_name="query",
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype="float32",
        )[0]


@dataclass
class BuildStats:
    files_scanned: int
    files_reused: int
    files_changed: int
    chunks_total: int
    chunks_reused_vectors: int
    chunks_encoded: int
    chunks_deduplicated: int
    elapsed_seconds: float
    files_excluded: int = 0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_from_dict(data: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=data["chunk_id"],
        source=data["source"],
        kind=data["kind"],
        content=data["content"],
        symbol=data.get("symbol"),
        start_line=data.get("start_line"),
        end_line=data.get("end_line"),
        heading=data.get("heading"),
        metadata=data.get("metadata") or {},
    )


def _read_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _load_previous(store_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray | None]:
    manifest_path = store_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}, [], None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, [], None
    chunks = _read_chunks(store_dir / "chunks.jsonl")
    vectors_path = store_dir / "vectors.npy"
    vectors = np.load(vectors_path) if vectors_path.is_file() else None
    return manifest, chunks, vectors


def build_index(
    root: Path,
    store_dir: Path,
    embedder: Embedder,
    *,
    force: bool = False,
    include_suffixes: set[str] | None = None,
    exclude_parts: set[str] | None = None,
    source_policy: SourcePolicy | None = None,
    exclude_dirs: tuple[Path, ...] = (),
) -> BuildStats:
    import bm25s
    import faiss
    from bm25s.tokenization import Tokenizer

    started = time.perf_counter()
    root = root.resolve()
    store_dir = store_dir.resolve()
    store_dir.mkdir(parents=True, exist_ok=True)

    previous_manifest, previous_chunks, previous_vectors = _load_previous(store_dir)
    chunker_changed = previous_manifest.get("chunker_version") != CHUNKER_VERSION
    if force or chunker_changed:
        previous_manifest = {}
        previous_chunks = []
        previous_vectors = None

    previous_by_source: dict[str, list[dict[str, Any]]] = {}
    for chunk in previous_chunks:
        previous_by_source.setdefault(chunk["source"], []).append(chunk)

    previous_vectors_by_hash: dict[str, np.ndarray] = {}
    if (
        previous_vectors is not None
        and len(previous_vectors) == len(previous_chunks)
        and previous_vectors.ndim == 2
        and previous_vectors.shape[1] == embedder.dimension
    ):
        for chunk, vector in zip(previous_chunks, previous_vectors):
            previous_vectors_by_hash[chunk["content_hash"]] = vector

    excluded_dirs = (store_dir, *[Path(item).resolve() for item in exclude_dirs])
    files = [
        path
        for path in discover_files(root, include_suffixes, exclude_parts)
        if not any(_is_inside(path, directory) for directory in excluded_dirs)
    ]
    files_excluded = 0
    if source_policy is not None:
        admitted: list[Path] = []
        for path in files:
            decision = source_policy.check_path(path, require_exists=True)
            if decision.allowed:
                admitted.append(path)
            else:
                files_excluded += 1
        files = admitted

    files_manifest: dict[str, Any] = {}
    chunks: list[Chunk] = []
    files_reused = 0
    files_changed = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        digest = sha256_bytes(data)
        files_manifest[relative] = {"sha256": digest, "size": len(data)}
        unchanged = previous_manifest.get("files", {}).get(relative, {}).get("sha256") == digest
        if unchanged and previous_by_source.get(relative):
            files_reused += 1
            chunks.extend(_chunk_from_dict(item) for item in previous_by_source[relative])
        else:
            files_changed += 1
            chunks.extend(chunk_file(path, root))

    total_before_dedup = len(chunks)
    chunks = deduplicate(chunks)
    chunks_deduplicated = total_before_dedup - len(chunks)

    dimension = embedder.dimension
    vectors = np.zeros((len(chunks), dimension), dtype="float32")
    missing: list[int] = []
    for index, chunk in enumerate(chunks):
        cached = previous_vectors_by_hash.get(chunk.content_hash)
        if cached is not None and cached.shape == (dimension,):
            vectors[index] = cached
        else:
            missing.append(index)

    if missing:
        encoded = np.asarray(
            embedder.encode_documents([chunks[index].content for index in missing]),
            dtype="float32",
        )
        if encoded.shape != (len(missing), dimension):
            raise RuntimeError(
                f"embedding shape mismatch: {encoded.shape} != {(len(missing), dimension)}"
            )
        vectors[missing] = encoded

    index = faiss.IndexFlatIP(dimension)
    if len(chunks):
        index.add(vectors)
    faiss.write_index(index, str(store_dir / "index.faiss"))
    np.save(store_dir / "vectors.npy", vectors)

    chunk_dicts = [chunk.to_dict() for chunk in chunks]
    chunks_path = store_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in chunk_dicts:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    bm25_dir = store_dir / "bm25"
    if bm25_dir.exists():
        shutil.rmtree(bm25_dir)
    bm25_dir.mkdir(parents=True, exist_ok=True)
    if chunk_dicts:
        prepared = [prepare_for_bm25(search_text(item)) for item in chunk_dicts]
        tokenizer = Tokenizer(splitter=lambda text: text.split())
        corpus_tokens = tokenizer.tokenize(prepared, return_as="tuple")
        retriever = bm25s.BM25(corpus=chunk_dicts)
        retriever.index(corpus_tokens, show_progress=False)
        retriever.save(bm25_dir)
        tokenizer.save_vocab(save_dir=bm25_dir)

    manifest = {
        "version": INDEX_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "embedding_dimension": dimension,
        "chunk_count": len(chunk_dicts),
        "files": files_manifest,
        "chunks_sha256": sha256_file(chunks_path),
        "vectors_sha256": sha256_file(store_dir / "vectors.npy"),
        "index_sha256": sha256_file(store_dir / "index.faiss"),
    }
    if source_policy is not None:
        manifest["source_policy"] = {
            "version": 1,
            "root": str(source_policy.root),
            "protected_prefixes": list(source_policy.protected_prefixes),
            "protected_files": sorted(source_policy.protected_files),
            "excluded_files": files_excluded,
            "excluded_digest": hashlib.sha256(
                "\n".join(sorted(source_policy.protected_files)).encode("utf-8")
            ).hexdigest(),
        }
    (store_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return BuildStats(
        files_scanned=len(files),
        files_reused=files_reused,
        files_changed=files_changed,
        chunks_total=len(chunk_dicts),
        chunks_reused_vectors=len(chunks) - len(missing),
        chunks_encoded=len(missing),
        chunks_deduplicated=chunks_deduplicated,
        elapsed_seconds=round(time.perf_counter() - started, 3),
        files_excluded=files_excluded,
    )


def _is_inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory)
        return True
    except ValueError:
        return False


def load_manifest(store_dir: Path) -> dict[str, Any]:
    return json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))


def chunk_as_dict(chunk: Chunk) -> dict[str, Any]:
    return asdict(chunk)