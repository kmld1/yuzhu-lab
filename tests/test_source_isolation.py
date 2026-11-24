"""Adversarial tests for evaluation-source isolation.

All protected fixture content is synthetic canary text.  No frozen benchmark
question, criterion or answer is copied into this file or into test output.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from yuzhu_lab.harness.agent.audit import AgentAuditLog
from yuzhu_lab.harness.agent.checkpoint import CheckpointStore
from yuzhu_lab.harness.agent.loop import AgentLoop, AgentLoopConfig
from yuzhu_lab.harness.agent.model_client import ModelResponse, ModelToolCall
from yuzhu_lab.harness.agent.permissions import PermissionPolicy
from yuzhu_lab.harness.agent.registry import ToolRegistry, ToolRequest, ToolStatus
from yuzhu_lab.harness.agent.tools import build_default_registry, default_tool_context
from yuzhu_lab.harness.context_builder import ContextBuilder, ContextBuilderConfig
from yuzhu_lab.harness.context_models import (
    ContextRequest,
    MemoryContextItem,
    RagEvidenceItem,
)
from yuzhu_lab.harness.memory_adapter import (
    HindsightMemoryAdapter,
    MemorySearchOutcome,
)
from yuzhu_lab.harness.rag_adapter import RagAdapter, RagSearchOutcome
from yuzhu_lab.hindsight.models import RecallMemory
from yuzhu_lab.hindsight.service import RecallOutcome
from yuzhu_lab.rag.search import RagIndex, SearchResult
from yuzhu_lab.rag.store import build_index
from yuzhu_lab.source_policy import (
    MISSING_PROVENANCE,
    PROTECTED_EVAL_SOURCE,
    RETRIEVAL_SOURCE_DENIED,
    SourcePolicy,
)

CANARY = "PROTECTED_EVAL_CANARY_9F3A72D1_SYNTHETIC_ONLY"
MARKDOWN_MARKER = "NORMAL_MARKDOWN_MARKER_4C1B"
TEST_MARKER = "NORMAL_TEST_FILE_MARKER_7D2E"
PREFIX_MARKER = "PREFIX_COLLISION_ALLOWED_MARKER_1A9F"


class FakeEmbedder:
    dimension = 8

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = np.frombuffer(digest[: self.dimension], dtype=np.uint8).astype("float32")
        return vector / (float(np.linalg.norm(vector)) or 1.0)

    def encode_documents(self, texts):
        return np.stack([self._vector(text) for text in texts]).astype("float32")

    def encode_query(self, query):
        return self._vector(query)


class Fixture:
    """Synthetic workspace with protected eval assets and normal project files."""

    def __init__(self) -> None:
        self._temp = tempfile.TemporaryDirectory(prefix="yuzhu-source-policy-")
        self.base = Path(self._temp.name)
        self.root = self.base / "workspace"
        self.outside = self.base / "outside"
        self.root.mkdir()
        self.outside.mkdir()
        (self.root / "eval" / "blind_v2").mkdir(parents=True)
        (self.root / "src").mkdir()
        (self.root / "docs").mkdir()
        (self.root / "tests").mkdir()

        (self.root / "eval" / "sft_eval_v2.jsonl").write_text(
            json.dumps(
                {
                    "id": "synthetic_eval",
                    "category": "behavior",
                    "skill": "synthetic",
                    "prompt": f"{CANARY} prompt marker",
                    "criteria": [f"{CANARY} criterion marker"],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.root / "eval" / "blind_v2" / "score.md").write_text(
            f"# synthetic scoring\n{CANARY} scoring detail\n", encoding="utf-8"
        )
        (self.root / "src" / "make_eval_v2.py").write_text(
            f'"""eval generator fixture."""\nEVAL_TEXT = "{CANARY}"\n',
            encoding="utf-8",
        )
        (self.root / "src" / "make_eval_v2_notes.py").write_text(
            f'"""normal helper whose name starts like an eval generator."""\nNOTE = "{PREFIX_MARKER}"\n',
            encoding="utf-8",
        )
        (self.root / "src" / "app.py").write_text(
            "def add(a, b):\n    return a + b\n", encoding="utf-8"
        )
        (self.root / "docs" / "guide.md").write_text(
            f"# Guide\n\n{MARKDOWN_MARKER}\n", encoding="utf-8"
        )
        (self.root / "tests" / "test_app.py").write_text(
            f"# project engineering test\n# {TEST_MARKER}\n"
            "def test_add():\n    assert True\n",
            encoding="utf-8",
        )
        (self.root / "evaluation_notes.md").write_text(
            f"# evaluation notes\n\n{PREFIX_MARKER}\n", encoding="utf-8"
        )
        (self.outside / "secret.py").write_text(
            "OUTSIDE_SECRET = 'outside-canary'\n", encoding="utf-8"
        )

    @property
    def policy(self) -> SourcePolicy:
        return SourcePolicy(self.root)

    def close(self) -> None:
        self._temp.cleanup()


def search_result(
    source: str,
    *,
    citation: str | None = None,
    content: str = "content",
    chunk_id: str = "c1",
    symbol: str | None = "add",
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        source=source,
        symbol=symbol,
        heading=None,
        start_line=1,
        end_line=2,
        content=content,
        citation=citation if citation is not None else f"{source}:1-2 ({symbol})",
        score=0.5,
        rrf_score=0.5,
    )


class FakeIndex:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = list(results)
        self.last_degraded: list[str] = []

    def search(self, query: str, *, admit_source=None, **kwargs):
        return [
            result
            for result in self.results
            if admit_source is None or admit_source(result.source)
        ]


class SourcePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = Fixture()
        self.addCleanup(self.fx.close)
        self.policy = self.fx.policy

    def test_protected_eval_paths_are_denied_with_normalized_spellings(self) -> None:
        candidates = [
            "eval/sft_eval_v2.jsonl",
            "EVAL\\SFT_EVAL_V2.JSONL",
            "./eval/./blind_v2/score.md",
            str(self.fx.root / "eval" / "sft_eval_v2.jsonl"),
        ]
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                decision = self.policy.check_relative(candidate, require_exists=True)
                self.assertFalse(decision.allowed)
                self.assertEqual(decision.code, PROTECTED_EVAL_SOURCE)

    def test_protected_generator_files_are_denied(self) -> None:
        decision = self.policy.check_relative("src/make_eval_v2.py", require_exists=True)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, PROTECTED_EVAL_SOURCE)

    def test_prefix_collisions_and_normal_tests_are_allowed(self) -> None:
        for candidate in (
            "evaluation_notes.md",
            "src/make_eval_v2_notes.py",
            "tests/test_app.py",
            "docs/guide.md",
            "src/app.py",
        ):
            with self.subTest(candidate=candidate):
                decision = self.policy.check_relative(candidate, require_exists=True)
                self.assertTrue(decision.allowed, decision)

    def test_traversal_and_outside_paths_are_denied(self) -> None:
        for candidate in (
            "../outside/secret.py",
            "src/../../outside/secret.py",
            str(self.fx.outside / "secret.py"),
        ):
            with self.subTest(candidate=candidate):
                decision = self.policy.check_relative(candidate, require_exists=True)
                self.assertFalse(decision.allowed)

    def test_missing_or_ambiguous_provenance_is_denied(self) -> None:
        for candidate in ("", "   ", None, "\\\\server\\share\\x.py", "C:relative.py"):
            with self.subTest(candidate=candidate):
                decision = self.policy.check_relative(candidate)
                self.assertFalse(decision.allowed)
                self.assertIn(
                    decision.code,
                    {MISSING_PROVENANCE, "PATH_INVALID", "PATH_OUTSIDE_WORKSPACE"},
                )

    def test_missing_file_is_denied_when_existence_is_required(self) -> None:
        decision = self.policy.check_relative("src/does_not_exist.py", require_exists=True)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "PATH_NOT_FOUND")

    def test_forged_citation_claiming_protected_path_is_denied(self) -> None:
        forged = self.policy.check_citation("eval/sft_eval_v2.jsonl:1-2 (synthetic)")
        self.assertFalse(forged.allowed)
        self.assertEqual(forged.code, PROTECTED_EVAL_SOURCE)
        normal = self.policy.check_citation("src/app.py:1-2 (add)")
        self.assertTrue(normal.allowed, normal)
        missing = self.policy.check_citation(None)
        self.assertFalse(missing.allowed)
        self.assertEqual(missing.code, MISSING_PROVENANCE)

    def test_junction_into_eval_and_outside_are_denied(self) -> None:
        link_eval = self.fx.root / "link_eval"
        link_out = self.fx.root / "link_out"
        for link, target in (
            (link_eval, self.fx.root / "eval"),
            (link_out, self.fx.outside),
        ):
            creation = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True,
                text=True,
            )
            if creation.returncode != 0:
                self.skipTest(f"junction creation unavailable: {creation.stderr.strip()}")
        for candidate in ("link_eval/sft_eval_v2.jsonl", "link_out/secret.py"):
            with self.subTest(candidate=candidate):
                decision = self.policy.check_relative(candidate, require_exists=True)
                self.assertFalse(decision.allowed)


class RagIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = Fixture()
        self.addCleanup(self.fx.close)

    def adapter(self, results: list[SearchResult]) -> RagAdapter:
        return RagAdapter(
            store_dir=self.fx.root / "unused-index",
            embedding_model_path="unused",
            workspace=self.fx.root,
            index=FakeIndex(results),
        )

    def test_eval_source_is_removed_from_rag_results(self) -> None:
        adapter = self.adapter(
            [
                search_result(
                    "eval/blind_v2/score.md",
                    citation="eval/blind_v2/score.md:1-2 (scoring)",
                    content=CANARY,
                    chunk_id="bad",
                ),
                search_result("src/app.py", content="def add(a, b): ...", chunk_id="good"),
            ]
        )
        outcome = adapter.search(
            project_id="qianyuqwen", query="scoring", token_budget=1000, final_k=5
        )
        self.assertEqual([item.source for item in outcome.items], ["src/app.py"])
        self.assertNotIn(CANARY, "\n".join(item.content for item in outcome.items))

    def test_forged_citation_and_missing_source_are_removed(self) -> None:
        adapter = self.adapter(
            [
                search_result(
                    "src/app.py",
                    citation="eval/sft_eval_v2.jsonl:1-2 (forged)",
                    content=CANARY,
                    chunk_id="forged",
                ),
                search_result("", citation="", content=CANARY, chunk_id="missing"),
                search_result("src/app.py", content="normal", chunk_id="good"),
            ]
        )
        outcome = adapter.search(
            project_id="qianyuqwen", query="anything", token_budget=1000, final_k=5
        )
        self.assertEqual([item.chunk_id for item in outcome.items], ["good"])

    def test_clean_index_build_excludes_protected_sources(self) -> None:
        store = self.fx.base / "clean-index"
        policy = self.fx.policy
        stats = build_index(
            self.fx.root, store, FakeEmbedder(), source_policy=policy
        )
        manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
        indexed = set(manifest["files"])
        self.assertNotIn("eval/sft_eval_v2.jsonl", indexed)
        self.assertNotIn("eval/blind_v2/score.md", indexed)
        self.assertNotIn("src/make_eval_v2.py", indexed)
        self.assertIn("src/app.py", indexed)
        self.assertIn("docs/guide.md", indexed)
        # The .jsonl frozen file is outside the RAG suffix set, so only the
        # Markdown result and the generator script reach the policy filter.
        self.assertEqual(manifest["source_policy"]["excluded_files"], 2)
        self.assertGreater(stats.files_scanned, 0)

        chunks = [
            json.loads(line)
            for line in (store / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertFalse(
            any(
                chunk["source"].startswith("eval/")
                or chunk["source"] == "src/make_eval_v2.py"
                for chunk in chunks
            )
        )
        self.assertNotIn(CANARY, "\n".join(chunk["content"] for chunk in chunks))

        index = RagIndex(store, embedder=FakeEmbedder()).load()
        outcome = RagAdapter(
            store_dir=store,
            embedding_model_path="unused",
            workspace=self.fx.root,
            index=index,
        ).search(
            project_id="qianyuqwen",
            query=CANARY,
            token_budget=2000,
            final_k=5,
        )
        self.assertFalse(
            any(
                item.source.startswith("eval/")
                or item.source == "src/make_eval_v2.py"
                or CANARY in item.content
                or CANARY in item.citation
                for item in outcome.items
            )
        )
        self.assertTrue(
            any(
                item.source == "docs/guide.md"
                for item in RagAdapter(
                    store_dir=store,
                    embedding_model_path="unused",
                    workspace=self.fx.root,
                    index=index,
                ).search(
                    project_id="qianyuqwen",
                    query=MARKDOWN_MARKER,
                    token_budget=2000,
                    final_k=5,
                ).items
            )
        )
        self.assertTrue(
            any(
                item.source == "tests/test_app.py"
                for item in RagAdapter(
                    store_dir=store,
                    embedding_model_path="unused",
                    workspace=self.fx.root,
                    index=index,
                ).search(
                    project_id="qianyuqwen",
                    query=TEST_MARKER,
                    token_budget=2000,
                    final_k=5,
                ).items
            )
        )

    def test_old_polluted_index_is_filtered_at_retrieval(self) -> None:
        polluted = FakeIndex(
            [
                search_result(
                    "eval/sft_eval_v2.jsonl",
                    citation="eval/sft_eval_v2.jsonl:1 (criteria)",
                    content=CANARY,
                    chunk_id="polluted",
                )
            ]
        )
        outcome = RagAdapter(
            store_dir=self.fx.root / "unused",
            embedding_model_path="unused",
            workspace=self.fx.root,
            index=polluted,
        ).search(
            project_id="qianyuqwen", query="criteria", token_budget=1000, final_k=5
        )
        self.assertEqual(outcome.items, [])


class ContextBuilderDefenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = Fixture()
        self.addCleanup(self.fx.close)

    def test_builder_drops_protected_rag_item_even_if_adapter_misbehaves(self) -> None:
        class BadRag:
            def search(self, *, project_id, query, token_budget, final_k=5):
                return RagSearchOutcome(
                    items=[
                        RagEvidenceItem(
                            chunk_id="bad",
                            citation="eval/blind_v2/score.md:1-2 (scoring)",
                            source="eval/blind_v2/score.md",
                            symbol="scoring",
                            start_line=1,
                            end_line=2,
                            content=CANARY,
                            score=0.9,
                            rrf_score=0.9,
                            project_id=project_id,
                        ),
                        RagEvidenceItem(
                            chunk_id="good",
                            citation="src/app.py:1-2 (add)",
                            source="src/app.py",
                            symbol="add",
                            start_line=1,
                            end_line=2,
                            content="def add(a, b): ...",
                            score=0.8,
                            rrf_score=0.8,
                            project_id=project_id,
                        ),
                    ],
                    latency_ms=1.0,
                )

        builder = ContextBuilder(
            memory=None,
            rag=BadRag(),
            config=ContextBuilderConfig(
                workspace=self.fx.root, source_policy=self.fx.policy
            ),
        )
        bundle = builder.build(
            ContextRequest(
                user_message="explain the scoring",
                session_id="s",
                project_id="qianyuqwen",
                model_context_budget=4096,
                workspace=self.fx.root,
            )
        )
        self.assertEqual([item.source for item in bundle.rag_evidence], ["src/app.py"])
        self.assertNotIn(CANARY, bundle.system_context)
        self.assertTrue(
            any(w.code == RETRIEVAL_SOURCE_DENIED for w in bundle.warnings)
        )

    def test_builder_drops_memory_that_carries_protected_content(self) -> None:
        item = MemoryContextItem(
            memory_id="m-bad",
            text=CANARY,
            category="session_handoff",
            verification="verified",
            confidence=0.9,
            source="session",
            occurred_at=None,
            bank_id="yuzhu-project-qianyuqwen",
            trust="MEMORY_VERIFIED",
            metadata={},
        )
        builder = ContextBuilder(
            memory=None,
            rag=None,
            config=ContextBuilderConfig(
                workspace=self.fx.root,
                source_policy=self.fx.policy,
                memory_content_guard=True,
            ),
        )
        bundle = builder.build(
            ContextRequest(
                user_message="recall",
                session_id="s",
                project_id="qianyuqwen",
                model_context_budget=4096,
                workspace=self.fx.root,
            )
        )
        self.assertEqual(bundle.recalled_memories, ())
        self.assertNotIn(CANARY, bundle.system_context)


class HindsightIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = Fixture()
        self.addCleanup(self.fx.close)

    def test_recall_filters_protected_path_and_content(self) -> None:
        class FakeService:
            def recall(self, **kwargs):
                return RecallOutcome(
                    memories=[
                        RecallMemory(
                            memory_id="m-eval-path",
                            bank_id="bank",
                            text="safe looking text",
                            memory_type="fact",
                            category="session_handoff",
                            verification="verified",
                            confidence=0.9,
                            source="eval/sft_eval_v2.jsonl",
                            occurred_at=None,
                            mentioned_at=None,
                            document_id="d1",
                            metadata={},
                            tags=(),
                        ),
                        RecallMemory(
                            memory_id="m-eval-content",
                            bank_id="bank",
                            text=f"earlier note: {CANARY}",
                            memory_type="fact",
                            category="session_handoff",
                            verification="verified",
                            confidence=0.9,
                            source="session",
                            occurred_at=None,
                            mentioned_at=None,
                            document_id="d2",
                            metadata={"source_path": "src/make_eval_v2.py"},
                            tags=(),
                        ),
                        RecallMemory(
                            memory_id="m-normal",
                            bank_id="bank",
                            text="normal verified project memory",
                            memory_type="fact",
                            category="session_handoff",
                            verification="verified",
                            confidence=0.8,
                            source="session",
                            occurred_at=None,
                            mentioned_at=None,
                            document_id="d3",
                            metadata={},
                            tags=(),
                        ),
                    ]
                )

        adapter = HindsightMemoryAdapter(
            FakeService(),
            source_policy=self.fx.policy,
            content_guard=True,
        )
        outcome = adapter.recall(
            project_id="qianyuqwen", query="anything", max_tokens=1000
        )
        self.assertEqual([item.memory_id for item in outcome.items], ["m-normal"])


class ToolAdmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = Fixture()
        self.addCleanup(self.fx.close)
        self.policy = PermissionPolicy.create(self.fx.root)
        self.context = default_tool_context(self.policy)
        self.registry = build_default_registry(self.policy, context=self.context)

    def call(self, name: str, arguments: dict):
        return self.registry.execute(
            ToolRequest(call_id="t1", name=name, arguments=arguments)
        )

    def test_read_file_protected_path_is_denied(self) -> None:
        result = self.call("read_file", {"path": "eval/sft_eval_v2.jsonl"})
        self.assertEqual(result.status, ToolStatus.DENIED)
        self.assertIn(RETRIEVAL_SOURCE_DENIED, result.content)
        self.assertNotIn(CANARY, result.content)

    def test_search_text_does_not_return_protected_lines(self) -> None:
        result = self.call("search_text", {"query": "PROTECTED_EVAL_CANARY"})
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertNotIn(CANARY, result.content)
        self.assertEqual(result.structured["matches"], 0)

    def test_list_files_hides_protected_directory(self) -> None:
        result = self.call("list_files", {"path": ".", "recursive": True})
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertNotIn("eval/", result.content.lower())
        self.assertIn("tests/test_app.py", result.content)

    def test_normal_project_test_file_is_readable(self) -> None:
        result = self.call("read_file", {"path": "tests/test_app.py"})
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertIn(TEST_MARKER, result.content)

    def test_normal_markdown_is_readable(self) -> None:
        result = self.call("read_file", {"path": "docs/guide.md"})
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertIn(MARKDOWN_MARKER, result.content)

    def test_search_text_on_protected_base_is_denied(self) -> None:
        result = self.call("search_text", {"query": "anything", "path": "eval"})
        self.assertEqual(result.status, ToolStatus.DENIED)

    def test_run_python_on_protected_script_is_denied(self) -> None:
        policy = PermissionPolicy.create(self.fx.root, allow_execution=True)
        context = default_tool_context(policy)
        registry = build_default_registry(policy, context=context)
        result = registry.execute(
            ToolRequest(
                call_id="t2",
                name="run_python",
                arguments={"script": "src/make_eval_v2.py"},
            )
        )
        self.assertIn(result.status, {ToolStatus.DENIED, ToolStatus.ERROR})
        self.assertIn(RETRIEVAL_SOURCE_DENIED, result.content)


class _ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, *, system, messages, tools, max_tokens, temperature=0.0):
        self.calls.append(
            {
                "system": system,
                "messages": [dict(message) for message in messages],
                "tools": tools,
            }
        )
        return self.responses.pop(0)


def _tool_response(calls):
    return ModelResponse(
        text="",
        tool_calls=[ModelToolCall(name=name, arguments=args) for name, args in calls],
        request_id="req-tool",
        stop_reason="tool_use",
        usage={"input_tokens": 10, "output_tokens": 5},
    )


def _text_response(text):
    return ModelResponse(
        text=text,
        tool_calls=[],
        request_id="req-text",
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5},
    )


class HarnessLoopIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = Fixture()
        self.addCleanup(self.fx.close)

    def test_model_read_of_protected_path_is_denied_and_content_never_enters_prompt(
        self,
    ) -> None:
        model = _ScriptedModel(
            [
                _tool_response([("read_file", {"path": "eval/sft_eval_v2.jsonl"})]),
                _text_response("I cannot read protected evaluation assets."),
            ]
        )
        state = self.fx.root / ".agent"
        state.mkdir()
        policy = PermissionPolicy.create(self.fx.root)
        context = default_tool_context(policy)
        registry = ToolRegistry(policy, context=context)
        for definition in build_default_registry(policy, context=context).definitions():
            registry.register(definition)
        store = CheckpointStore(state / "state.db")
        audit = AgentAuditLog(state / "audit.jsonl")
        self.addCleanup(store.close)
        loop = AgentLoop(
            config=AgentLoopConfig(
                project_id="qianyuqwen",
                session_id="s",
                workspace=self.fx.root,
                checkpoint_db=state / "state.db",
                audit_log=state / "audit.jsonl",
                task_id="task-source-isolation",
            ),
            model=model,
            registry=registry,
            checkpoints=store,
            audit=audit,
        )
        result = loop.run("read the frozen evaluation file")
        self.assertEqual(result.status, "COMPLETED")
        calls = store.tool_calls(loop.task_id)
        self.assertEqual(len(calls), 1)
        self.assertIn(calls[0]["status"], {"DENIED", "ERROR"})
        self.assertNotIn(CANARY, model.calls[1]["messages"][-1]["content"][0]["content"])
        self.assertIn("RETRIEVAL_SOURCE_DENIED", model.calls[1]["messages"][-1]["content"][0]["content"])


if __name__ == "__main__":
    unittest.main()