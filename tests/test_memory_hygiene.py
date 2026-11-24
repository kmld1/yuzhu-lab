"""Hindsight bank classification and non-destructive cleanup manifest tests."""

from __future__ import annotations

import unittest

from yuzhu_lab.harness.agent.memory_hygiene import (
    build_cleanup_manifest,
    classify_bank,
)


class MemoryHygieneTest(unittest.TestCase):
    def test_classification(self) -> None:
        self.assertEqual(classify_bank("yuzhu-project-qianyuqwen"), "formal_project")
        self.assertEqual(classify_bank("yuzhu-global"), "global")
        self.assertEqual(classify_bank("yuzhu-experiments"), "experiment")
        self.assertEqual(classify_bank("yuzhu-project-h2-e2e-real"), "test")
        self.assertEqual(classify_bank("yuzhu-project-h1probe"), "test")
        self.assertEqual(classify_bank("yuzhu-project-h3-sqlite-e2e-x"), "test")
        self.assertEqual(classify_bank("yuzhu-project-customer-prod"), "unknown")

    def test_manifest_keeps_formal_banks_and_never_deletes(self) -> None:
        manifest = build_cleanup_manifest(
            [
                {"bank_id": "yuzhu-project-qianyuqwen", "fact_count": 4},
                {"bank_id": "yuzhu-global", "fact_count": 1},
                {"bank_id": "yuzhu-experiments", "fact_count": 5},
                {"bank_id": "yuzhu-project-h2-e2e-real", "fact_count": 7},
                {"bank_id": "yuzhu-project-h3-sqlite-e2e-abc", "fact_count": 2},
                {"bank_id": "yuzhu-project-customer-prod", "fact_count": 3},
            ]
        ).to_dict()
        self.assertFalse(manifest["policy"]["automatic_delete"])
        self.assertFalse(manifest["policy"]["delete_supported_by_harness"])
        self.assertTrue(manifest["policy"]["requires_explicit_confirmation"])
        actions = {bank["bank_id"]: bank["action"] for bank in manifest["banks"]}
        self.assertEqual(actions["yuzhu-project-qianyuqwen"], "KEEP")
        self.assertEqual(actions["yuzhu-global"], "KEEP")
        self.assertEqual(actions["yuzhu-experiments"], "KEEP")
        self.assertEqual(actions["yuzhu-project-h2-e2e-real"], "CLEANUP_CANDIDATE")
        self.assertEqual(actions["yuzhu-project-h3-sqlite-e2e-abc"], "CLEANUP_CANDIDATE")
        self.assertEqual(actions["yuzhu-project-customer-prod"], "REVIEW")
        self.assertEqual(manifest["summary"]["cleanup_candidates"], 2)
        self.assertEqual(manifest["summary"]["cleanup_fact_count"], 9)


if __name__ == "__main__":
    unittest.main()