"""CPU tests for chunking, incremental indexing and hybrid search."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from yuzhu_lab.rag.bm25 import reciprocal_rank_fusion
from yuzhu_lab.rag.chunking import chunk_markdown, chunk_python, deduplicate
from yuzhu_lab.rag.search import RagIndex
from yuzhu_lab.rag.store import build_index


class FakeEmbedder:
    dimension = 8

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = np.frombuffer(digest[: self.dimension], dtype=np.uint8).astype("float32")
        norm = float(np.linalg.norm(vector)) or 1.0
        return vector / norm

    def encode_documents(self, texts):
        return np.stack([self._vector(text) for text in texts]).astype("float32")

    def encode_query(self, query):
        return self._vector(query)


APP_V1 = '''"""Account helpers."""

import sqlite3


def add(a, b):
    """Return the sum of two numbers."""
    return a + b


class Account:
    """A bank account."""

    def deposit(self, amount):
        return amount
'''

APP_V2 = APP_V1 + '''

def withdraw(balance, amount):
    """Withdraw money with a balance check."""
    if amount > balance:
        raise ValueError("insufficient funds")
    return balance - amount
'''

README = """# Project

This project implements a safe SQLite transfer function.

## Verification

Run the unit tests with pytest before merging.
"""


class ChunkingTest(unittest.TestCase):
    def test_python_symbols(self):
        chunks = chunk_python(APP_V1, "app.py")
        symbols = {chunk.symbol for chunk in chunks}
        self.assertIn("add", symbols)
        self.assertIn("Account", symbols)
        self.assertIn("Account.deposit", symbols)
        add_chunk = next(chunk for chunk in chunks if chunk.symbol == "add")
        self.assertEqual(add_chunk.start_line, 6)
        self.assertIsNotNone(add_chunk.end_line)

    def test_markdown_heading(self):
        chunks = chunk_markdown(README, "README.md")
        headings = " ".join(chunk.heading or "" for chunk in chunks)
        self.assertIn("Verification", headings)

    def test_deduplicate(self):
        chunks = chunk_python(APP_V1, "a.py") + chunk_python(APP_V1, "a.py")
        unique = deduplicate(chunks)
        self.assertEqual(len({chunk.content_hash for chunk in unique}), len(unique))

    def test_rrf_prefers_shared_items(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]])
        self.assertEqual(fused[0][0], "b")
        self.assertGreater(fused[0][1], fused[-1][1])


class IncrementalIndexTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "src"
        self.root.mkdir()
        self.store = Path(self.temp.name) / "index"
        (self.root / "app.py").write_text(APP_V1, encoding="utf-8")
        (self.root / "README.md").write_text(README, encoding="utf-8")
        self.embedder = FakeEmbedder()

    def tearDown(self):
        self.temp.cleanup()

    def _build(self, force=False):
        return build_index(self.root, self.store, self.embedder, force=force)

    def test_build_search_and_citation(self):
        stats = self._build()
        self.assertEqual(stats.files_scanned, 2)
        self.assertEqual(stats.chunks_deduplicated, 0)
        self.assertGreater(stats.chunks_total, 2)

        manifest = json.loads((self.store / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["chunk_count"], stats.chunks_total)
        self.assertTrue((self.store / "index.faiss").is_file())

        index = RagIndex(self.store, embedder=self.embedder).load()
        results = index.search("add function sum", final_k=5)
        self.assertTrue(results)
        self.assertTrue(any(result.source == "app.py" for result in results))
        self.assertIn("app.py", results[0].citation)

    def test_incremental_update_and_delete(self):
        self._build()
        (self.root / "app.py").write_text(APP_V2, encoding="utf-8")
        stats = self._build()
        self.assertEqual(stats.files_changed, 1)
        self.assertEqual(stats.files_reused, 1)
        self.assertGreater(stats.chunks_reused_vectors, 0)

        index = RagIndex(self.store, embedder=self.embedder).load()
        results = index.search("withdraw insufficient funds", final_k=5)
        self.assertTrue(any("withdraw" in (result.symbol or "") for result in results))

        (self.root / "README.md").unlink()
        stats = self._build()
        self.assertEqual(stats.files_scanned, 1)
        sources = {
            json.loads(line)["source"]
            for line in (self.store / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        self.assertEqual(sources, {"app.py"})

    def test_token_budget_limits_results(self):
        self._build()
        index = RagIndex(self.store, embedder=self.embedder).load()
        results = index.search("account", final_k=10, token_budget=1)
        self.assertEqual(len(results), 1)

    def test_dense_failure_degrades_to_bm25(self):
        self._build()
        index = RagIndex(self.store, embedder=None, embedding_model_path=None).load()
        results = index.search("SQLite transfer", final_k=3)
        self.assertTrue(results)
        self.assertTrue(any(item.startswith("dense:") for item in index.last_degraded))


if __name__ == "__main__":
    unittest.main()