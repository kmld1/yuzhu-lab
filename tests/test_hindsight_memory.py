"""Unit tests for memory admission policy and Hindsight orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from yuzhu_lab.hindsight.audit import text_digest
from yuzhu_lab.hindsight.client import HindsightAPIError
from yuzhu_lab.hindsight.config import HindsightConfig, bank_id_for
from yuzhu_lab.hindsight.models import MemoryRecord
from yuzhu_lab.hindsight.policy import MemoryAdmissionPolicy
from yuzhu_lab.hindsight.service import MemoryRejected, MemoryService


def record(**overrides):
    values = {
        "content": "YUZHU LAB 架构决策：Hindsight 使用独立 pg0 实例。",
        "project_id": "qianyuqwen",
        "category": "architecture_decision",
        "verification": "verified",
        "evidence": ("docs/decision.md",),
        "source": "unit-test",
        "memory_key": "hindsight.deployment",
    }
    values.update(overrides)
    return MemoryRecord(**values)


class FakeClient:
    def __init__(self, config: HindsightConfig) -> None:
        self.config = config
        self.retains: list[dict] = []
        self.recall_results: dict[str, list[dict]] = {}
        self.reflect_response = {
            "text": "synthesized",
            "based_on": {"memories": []},
            "usage": {"total_tokens": 1},
        }
        self.recall_error: Exception | None = None

    def retain(self, **kwargs):
        self.retains.append(kwargs)
        return {"success": True, "usage": {"total_tokens": 10}}

    def recall(self, *, bank_id, **_kwargs):
        if self.recall_error is not None:
            raise self.recall_error
        return {"results": list(self.recall_results.get(bank_id, []))}

    def reflect(self, **_kwargs):
        return dict(self.reflect_response)


class AdmissionPolicyTest(unittest.TestCase):
    def test_rejects_secret(self):
        policy = MemoryAdmissionPolicy()
        decision = policy.evaluate(record(content="api_key: placeholder-value-for-test"))
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "SECRET_DETECTED")

    def test_rejects_pii(self):
        policy = MemoryAdmissionPolicy()
        decision = policy.evaluate(record(content="联系邮箱 alice@example.com 记录在这里"))
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "PII_DETECTED")

    def test_rejects_unverified_project_claim(self):
        policy = MemoryAdmissionPolicy()
        decision = policy.evaluate(record(verification="unverified", category="hypothesis"))
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "HYPOTHESIS_REQUIRES_EXPERIMENT")

    def test_allows_unverified_hypothesis_in_experiments(self):
        policy = MemoryAdmissionPolicy()
        decision = policy.evaluate(
            record(
                scope="experiment",
                category="hypothesis",
                verification="unverified",
                evidence=(),
                memory_key=None,
            )
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.bank_id, "yuzhu-experiments")
        self.assertIn("verification:unverified", decision.tags)

    def test_verified_fix_requires_evidence(self):
        policy = MemoryAdmissionPolicy()
        decision = policy.evaluate(record(category="verified_fix", evidence=()))
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "EVIDENCE_REQUIRED")

    def test_rejects_raw_code_payload(self):
        policy = MemoryAdmissionPolicy()
        code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(6)) + "\n```"
        decision = policy.evaluate(record(content="修复如下：" + code))
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "RAW_CODE_PAYLOAD")

    def test_bank_isolation(self):
        self.assertEqual(bank_id_for("A", "project"), "yuzhu-project-a")
        self.assertEqual(bank_id_for("B", "project"), "yuzhu-project-b")
        self.assertNotEqual(bank_id_for("A", "project"), bank_id_for("B", "project"))
        self.assertEqual(bank_id_for("A", "global"), "yuzhu-global")
        self.assertEqual(bank_id_for("A", "experiment"), "yuzhu-experiments")

    def test_metadata_and_tags(self):
        policy = MemoryAdmissionPolicy()
        decision = policy.evaluate(record())
        self.assertTrue(decision.allowed)
        self.assertIn("project:qianyuqwen", decision.tags)
        self.assertIn("category:architecture_decision", decision.tags)
        self.assertIn("memory_key:hindsight.deployment", decision.tags)
        self.assertEqual(decision.metadata["verification"], "verified")


class MemoryServiceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = HindsightConfig(audit_log_path=Path(self.tmp.name) / "audit.jsonl")
        self.client = FakeClient(self.config)
        self.service = MemoryService(self.client, config=self.config)

    def tearDown(self):
        self.tmp.cleanup()

    def test_retain_records_digest_and_never_raw_content(self):
        self.service.retain(record())
        self.assertEqual(len(self.client.retains), 1)
        audit = (Path(self.tmp.name) / "audit.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("Hindsight 使用独立", audit)
        self.assertIn(text_digest("YUZHU LAB 架构决策：Hindsight 使用独立 pg0 实例。"), audit)

    def test_duplicate_is_not_retained_twice(self):
        digest = text_digest("YUZHU LAB 架构决策：Hindsight 使用独立 pg0 实例。")
        self.client.recall_results["yuzhu-project-qianyuqwen"] = [
            {
                "id": "mem-1",
                "text": "existing",
                "type": "world",
                "metadata": {
                    "memory_key": "hindsight.deployment",
                    "content_sha256": digest,
                    "verification": "verified",
                },
                "tags": [],
                "scores": {"final": 1.0},
            }
        ]
        result = self.service.retain(record())
        self.assertTrue(result.duplicate)
        self.assertEqual(self.client.retains, [])

    def test_conflict_is_preserved_not_overwritten(self):
        self.client.recall_results["yuzhu-project-qianyuqwen"] = [
            {
                "id": "mem-old",
                "text": "old decision",
                "type": "world",
                "metadata": {
                    "memory_key": "hindsight.deployment",
                    "content_sha256": "different",
                    "verification": "verified",
                },
                "tags": [],
                "scores": {"final": 1.0},
            }
        ]
        result = self.service.retain(record())
        self.assertTrue(result.conflict)
        self.assertEqual(result.code, "CONFLICT_STORED")
        self.assertIn("conflict", result.warnings[0])
        self.assertEqual(len(self.client.retains), 1)

    def test_recall_deduplicates_and_marks_cross_document_conflict(self):
        self.client.recall_results["yuzhu-project-qianyuqwen"] = [
            {
                "id": "a",
                "text": "same fact",
                "type": "world",
                "document_id": "doc-a",
                "metadata": {
                    "memory_key": "k",
                    "content_sha256": "hash-a",
                    "verification": "verified",
                },
                "tags": [],
                "scores": {"final": 1.0},
            },
            {
                "id": "b",
                "text": "same fact",
                "type": "world",
                "document_id": "doc-b",
                "metadata": {
                    "memory_key": "k",
                    "content_sha256": "hash-b",
                    "verification": "verified",
                },
                "tags": [],
                "scores": {"final": 0.5},
            },
        ]
        outcome = self.service.recall(project_id="qianyuqwen", query="fact")
        self.assertEqual(len(outcome.memories), 1)
        self.assertTrue(outcome.memories[0].conflict)

    def test_missing_bank_is_empty_not_failure(self):
        self.client.recall_error = HindsightAPIError(404, "not found")
        outcome = self.service.recall(project_id="fresh", query="anything")
        self.assertEqual(outcome.memories, [])
        self.assertIn("yuzhu-project-fresh", outcome.missing_banks)
        self.assertEqual(outcome.degraded, [])

    def test_rejected_record_raises_and_is_audited(self):
        with self.assertRaises(MemoryRejected):
            self.service.retain(record(content="password: hunter2"))
        audit = (Path(self.tmp.name) / "audit.jsonl").read_text(encoding="utf-8")
        entry = json.loads(audit.strip().splitlines()[-1])
        self.assertFalse(entry["accepted"])
        self.assertEqual(entry["code"], "SECRET_DETECTED")


if __name__ == "__main__":
    unittest.main()