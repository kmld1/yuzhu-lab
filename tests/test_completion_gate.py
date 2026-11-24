"""Regression tests for the H4 completion gate.

The H3 audit log contains a real ``TASK_COMPLETED`` event with
``stop_reason=max_tokens``.  These tests make that class of false completion
impossible to reintroduce silently.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.agent_test_utils import (
    ScriptedModel,
    TempWorkspace,
    make_loop,
    text_response,
    tool_response,
)
from yuzhu_lab.harness.agent.completion import (
    CompletionLevel,
    assess_completion,
)
from yuzhu_lab.harness.agent.model_client import ModelResponse


def _response(text: str, *, stop_reason: str = "end_turn") -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=[],
        request_id="req-test",
        stop_reason=stop_reason,
        usage={"input_tokens": 1, "output_tokens": 1},
    )


def _record(
    name: str,
    *,
    status: str = "COMPLETED",
    result_status: str = "OK",
    path: str | None = None,
    exit_code: int | None = None,
    error_code: str | None = None,
) -> dict:
    structured = {}
    if path is not None:
        structured.update({"path": path, "readback_verified": True})
    result = {"status": result_status, "structured": structured}
    if exit_code is not None:
        result["exit_code"] = exit_code
    if error_code:
        result["error_code"] = error_code
    return {"name": name, "status": status, "result": result}


class CompletionAssessmentTest(unittest.TestCase):
    def test_truncated_response_is_not_completed(self) -> None:
        assessment = assess_completion(
            _response("partial", stop_reason="max_tokens"),
            [],
            final_text="partial",
        )
        self.assertEqual(assessment.level, CompletionLevel.MODEL_TRUNCATED)
        self.assertFalse(assessment.task_completed)
        self.assertIn("MODEL_TRUNCATED", assessment.blockers)

    def test_tool_executed_is_distinct_from_verified(self) -> None:
        assessment = assess_completion(
            _response("done"),
            [_record("read_file")],
            final_text="done",
        )
        self.assertTrue(assessment.tool_executed)
        self.assertFalse(assessment.task_verified)
        self.assertTrue(assessment.task_completed)
        self.assertEqual(assessment.level, CompletionLevel.TASK_COMPLETED)

    def test_verified_but_empty_answer_is_task_verified(self) -> None:
        assessment = assess_completion(
            _response(""),
            [_trusted_record(_evidence())],
            final_text="",
        )
        self.assertTrue(assessment.task_verified)
        self.assertFalse(assessment.task_completed)
        self.assertEqual(assessment.level, CompletionLevel.TASK_VERIFIED)
        self.assertIn("EMPTY_FINAL_ANSWER", assessment.blockers)

    def test_code_change_without_tests_is_blocked(self) -> None:
        assessment = assess_completion(
            _response("done"),
            [_record("write_file", path="calc.py")],
            final_text="done",
            require_verification_for_code_change=True,
        )
        self.assertFalse(assessment.task_completed)
        self.assertEqual(assessment.level, CompletionLevel.UNVERIFIED_CODE)
        self.assertIn("UNVERIFIED_CODE", assessment.blockers)

    def test_failed_last_test_blocks_completion(self) -> None:
        assessment = assess_completion(
            _response("done"),
            [
                _record("write_file", path="calc.py"),
                _trusted_record(_evidence(verdict="TESTS_FAILED", failed=1), exit_code=1),
            ],
            final_text="done",
        )
        self.assertFalse(assessment.task_completed)
        self.assertIn("TESTS_FAILED", assessment.blockers)

    def test_stale_test_evidence_blocks_completion(self) -> None:
        assessment = assess_completion(
            _response("done"),
            [
                _record("write_file", path="calc.py"),
                _trusted_record(_evidence()),
                _record("edit_file", path="calc.py"),
            ],
            final_text="done",
        )
        self.assertFalse(assessment.task_completed)
        self.assertIn("TEST_EVIDENCE_STALE", assessment.blockers)

    def test_fresh_passing_test_verifies(self) -> None:
        assessment = assess_completion(
            _response("done"),
            [
                _record("write_file", path="calc.py"),
                _trusted_record(_evidence()),
            ],
            final_text="done",
        )
        self.assertTrue(assessment.task_verified)
        self.assertTrue(assessment.task_completed)
        self.assertEqual(assessment.level, CompletionLevel.TASK_COMPLETED)

    def test_no_tests_run_is_not_a_pass(self) -> None:
        assessment = assess_completion(
            _response("done"),
            [_record("run_tests", status="ERROR", result_status="ERROR", error_code="NO_TESTS_RUN")],
            final_text="done",
        )
        self.assertFalse(assessment.task_verified)
        self.assertIn("NO_TESTS_RUN", assessment.blockers)

    def test_incomplete_tool_use_is_blocked(self) -> None:
        assessment = assess_completion(
            _response("I will call a tool", stop_reason="tool_use"),
            [],
            final_text="I will call a tool",
        )
        self.assertFalse(assessment.task_completed)
        self.assertIn("INCOMPLETE_TOOL_USE", assessment.blockers)

    def test_non_code_write_does_not_require_tests(self) -> None:
        assessment = assess_completion(
            _response("documented"),
            [_record("write_file", path="notes.md")],
            final_text="documented",
            require_verification_for_code_change=True,
        )
        self.assertTrue(assessment.task_completed)
        self.assertEqual(assessment.code_files_changed, ())


class AgentLoopCompletionGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.root = self.workspace.path

    def run_task(self, model, task="do the task", **kwargs):
        loop, store, audit = make_loop(self.root, model, **kwargs)
        self.addCleanup(store.close)
        return loop.run(task), loop, store, audit

    def test_truncated_twice_fails_with_model_truncated(self) -> None:
        model = ScriptedModel(
            [
                text_response("partial answer", stop_reason="max_tokens"),
                text_response("still partial", stop_reason="max_tokens"),
            ]
        )
        result, loop, store, _ = self.run_task(model)
        self.assertEqual(result.status, "FAILED")
        self.assertIn("completion gate", result.error or "")
        self.assertEqual(result.final_text, "still partial")
        self.assertIsNotNone(result.completion)
        self.assertEqual(result.completion["level"], "MODEL_TRUNCATED")
        self.assertFalse(result.completion["task_completed"])
        task = store.get_task(loop.task_id)
        self.assertEqual(task["status"], "FAILED")
        self.assertEqual(task["state"], "FAILED")

    def test_truncated_then_complete_passes(self) -> None:
        model = ScriptedModel(
            [
                text_response("partial answer", stop_reason="max_tokens"),
                text_response("complete answer"),
            ]
        )
        result, _, _, _ = self.run_task(model)
        self.assertEqual(result.status, "COMPLETED")
        self.assertTrue(result.completion["task_completed"])
        self.assertEqual(result.completion["level"], "TASK_COMPLETED")

    def test_incomplete_tool_use_stop_reason_fails(self) -> None:
        response = ModelResponse(
            text="I will read the file",
            tool_calls=[],
            request_id="req-incomplete",
            stop_reason="tool_use",
            usage={},
        )
        result, _, _, _ = self.run_task(ScriptedModel([response]))
        self.assertEqual(result.status, "FAILED")
        self.assertIn("INCOMPLETE_TOOL_USE", result.completion["blockers"])

    def test_code_change_without_tests_fails_in_strict_mode(self) -> None:
        model = ScriptedModel(
            [
                tool_response([("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"})]),
                text_response("Implemented add()."),
            ]
        )
        result, _, _, _ = self.run_task(
            model, require_test_evidence_for_code_change=True
        )
        self.assertEqual(result.status, "FAILED")
        self.assertIn("UNVERIFIED_CODE", result.completion["blockers"])

    def test_code_change_without_tests_passes_when_not_strict(self) -> None:
        model = ScriptedModel(
            [
                tool_response([("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"})]),
                text_response("Implemented add()."),
            ]
        )
        result, _, _, _ = self.run_task(
            model, require_test_evidence_for_code_change=False
        )
        self.assertEqual(result.status, "COMPLETED")
        self.assertTrue(result.completion["task_completed"])
        self.assertFalse(result.completion["task_verified"])

    def test_failing_test_then_fix_passes(self) -> None:
        model = ScriptedModel(
            [
                tool_response(
                    [
                        ("write_file", {"path": "calc.py", "content": "def add(a, b):\n    return a - b\n"}),
                        (
                            "write_file",
                            {
                                "path": "test_calc.py",
                                "content": (
                                    "import unittest\nfrom calc import add\n\n"
                                    "class T(unittest.TestCase):\n"
                                    "    def test_add(self):\n"
                                    "        self.assertEqual(add(2, 3), 5)\n\n"
                                    "if __name__ == '__main__':\n    unittest.main()\n"
                                ),
                            },
                        ),
                    ]
                ),
                tool_response([("run_tests", {"path": ".", "framework": "unittest"})]),
                tool_response(
                    [
                        (
                            "edit_file",
                            {
                                "path": "calc.py",
                                "old_text": "return a - b",
                                "new_text": "return a + b",
                            },
                        )
                    ]
                ),
                tool_response([("run_tests", {"path": ".", "framework": "unittest"})]),
                text_response("Fixed; tests pass."),
            ]
        )
        result, _, store, _ = self.run_task(
            model,
            allow_execution=True,
            require_test_evidence_for_code_change=True,
        )
        self.assertEqual(result.status, "COMPLETED", result.error)
        self.assertTrue(result.completion["task_verified"])
        self.assertEqual(result.completion["tests_run"], 2)
        self.assertEqual(result.completion["tests_passed"], 1)

    def test_blocked_gate_does_not_retain_memory(self) -> None:
        class FakeWriter:
            def __init__(self) -> None:
                self.calls = 0

            def retain_handoff(self, **kwargs):
                self.calls += 1
                raise AssertionError("memory must not be written for a blocked completion")

        writer = FakeWriter()
        model = ScriptedModel(
            [
                text_response("partial", stop_reason="max_tokens"),
                text_response("still partial", stop_reason="max_tokens"),
            ]
        )
        result, _, _, _ = self.run_task(model, memory_writer=writer)
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(writer.calls, 0)


def _evidence(
    *,
    verdict: str = "PASS",
    executed: int = 2,
    failed: int = 0,
    skipped: int = 0,
    assertions: int = 3,
    mapped_files: tuple[str, ...] = ("calc.py",),
    test_files: tuple[dict, ...] = (),
    source_hashes: dict | None = None,
    reasons: tuple[str, ...] = (),
) -> dict:
    return {
        "schema": "yuzhu-test-evidence-1",
        "verdict": verdict,
        "framework": "unittest",
        "target": "test_calc.py",
        "exit_code": 0 if verdict == "PASS" else 1,
        "tests_discovered": executed,
        "tests_executed": executed,
        "tests_passed": executed - failed,
        "tests_failed": failed,
        "tests_skipped": skipped,
        "assertions": assertions,
        "imports": ["calc"],
        "mapped_files": list(mapped_files),
        "test_files": list(test_files),
        "source_hashes": source_hashes or {},
        "reasons": list(reasons),
    }


def _trusted_record(evidence: dict | None = None, **overrides) -> dict:
    structured = {"verification": evidence or _evidence()}
    structured.update(overrides.pop("structured", {}))
    record = {
        "name": "run_tests",
        "status": overrides.pop("status", "COMPLETED"),
        "result": {
            "status": overrides.pop("result_status", "OK"),
            "structured": structured,
            "exit_code": overrides.pop("exit_code", 0),
        },
    }
    record["result"].update(overrides)
    return record


class TrustedCompletionGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.root = self.workspace.path
        (self.root / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        (self.root / "test_calc.py").write_text(
            "import unittest\nfrom calc import add\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n",
            encoding="utf-8",
        )
        from yuzhu_lab.harness.agent.verification import sha256_file

        self.calc_hash = sha256_file(self.root / "calc.py")
        self.test_hash = sha256_file(self.root / "test_calc.py")

    def assess(self, records, **kwargs):
        return assess_completion(
            _response("done"),
            records,
            final_text="done",
            workspace=self.root,
            **kwargs,
        )

    def test_vacuous_tests_are_not_verified(self) -> None:
        record = _trusted_record(
            _evidence(verdict="TESTS_INCOMPLETE", assertions=0, reasons=("NO_ASSERTIONS",))
        )
        assessment = self.assess([_record("write_file", path="calc.py"), record])
        self.assertFalse(assessment.task_completed)
        self.assertFalse(assessment.task_verified)
        self.assertEqual(assessment.level, CompletionLevel.TESTS_INCOMPLETE)
        self.assertIn("TESTS_INCOMPLETE", assessment.blockers)

    def test_no_tests_executed_is_distinct(self) -> None:
        record = _trusted_record(
            _evidence(verdict="NO_TESTS_EXECUTED", executed=0, assertions=0)
        )
        assessment = self.assess([_record("write_file", path="calc.py"), record])
        self.assertEqual(assessment.level, CompletionLevel.NO_TESTS_EXECUTED)
        self.assertFalse(assessment.task_completed)

    def test_no_tests_discovered_is_distinct(self) -> None:
        record = _trusted_record(
            _evidence(verdict="NO_TESTS_DISCOVERED", executed=0, assertions=0)
        )
        assessment = self.assess([_record("write_file", path="calc.py"), record])
        self.assertEqual(assessment.level, CompletionLevel.NO_TESTS_DISCOVERED)

    def test_failed_tests_are_distinct(self) -> None:
        record = _trusted_record(
            _evidence(verdict="TESTS_FAILED", failed=1), exit_code=1
        )
        assessment = self.assess([_record("write_file", path="calc.py"), record])
        self.assertEqual(assessment.level, CompletionLevel.TESTS_FAILED)
        self.assertFalse(assessment.task_completed)

    def test_legacy_exit_zero_without_oracle_is_not_trusted(self) -> None:
        assessment = self.assess(
            [
                _record("write_file", path="calc.py"),
                _record("run_tests", exit_code=0),
            ]
        )
        self.assertFalse(assessment.task_verified)
        self.assertFalse(assessment.task_completed)
        self.assertEqual(assessment.level, CompletionLevel.UNVERIFIED_CODE)
        self.assertIn("VERIFICATION_UNTRUSTED", assessment.blockers)

    def test_valid_tests_complete_without_configured_acceptance(self) -> None:
        record = _trusted_record(
            _evidence(
                test_files=(
                    {"path": "test_calc.py", "sha256": self.test_hash, "parse_ok": True},
                ),
                source_hashes={"calc.py": self.calc_hash},
            )
        )
        assessment = self.assess([_record("write_file", path="calc.py"), record])
        self.assertTrue(assessment.task_verified, assessment.blockers)
        self.assertEqual(assessment.level, CompletionLevel.TASK_COMPLETED)

    def test_acceptance_failure_blocks_completion(self) -> None:
        record = _trusted_record(
            _evidence(
                test_files=(
                    {"path": "test_calc.py", "sha256": self.test_hash, "parse_ok": True},
                ),
                source_hashes={"calc.py": self.calc_hash},
            )
        )
        assessment = self.assess(
            [_record("write_file", path="calc.py"), record],
            acceptance={"passed": False, "reasons": ["ORACLE_REPORTED_FAILURE"]},
            acceptance_configured=True,
        )
        self.assertFalse(assessment.task_completed)
        self.assertEqual(assessment.level, CompletionLevel.ORACLE_FAILED)
        self.assertIn("ORACLE_FAILED", assessment.blockers)

    def test_acceptance_pass_completes(self) -> None:
        record = _trusted_record(
            _evidence(
                test_files=(
                    {"path": "test_calc.py", "sha256": self.test_hash, "parse_ok": True},
                ),
                source_hashes={"calc.py": self.calc_hash},
            )
        )
        assessment = self.assess(
            [_record("write_file", path="calc.py"), record],
            acceptance={"passed": True, "reasons": []},
            acceptance_configured=True,
        )
        self.assertTrue(assessment.task_completed)
        self.assertEqual(assessment.level, CompletionLevel.TASK_COMPLETED)

    def test_source_changed_after_test_is_stale(self) -> None:
        record = _trusted_record(
            _evidence(
                test_files=(
                    {"path": "test_calc.py", "sha256": self.test_hash, "parse_ok": True},
                ),
                source_hashes={"calc.py": "0" * 64},
            )
        )
        assessment = self.assess([_record("write_file", path="calc.py"), record])
        self.assertFalse(assessment.task_completed)
        self.assertEqual(assessment.level, CompletionLevel.TESTS_INCOMPLETE)
        self.assertIn("SOURCE_CHANGED_AFTER_TEST", assessment.blockers)

    def test_test_file_changed_after_run_is_stale(self) -> None:
        record = _trusted_record(
            _evidence(
                test_files=(
                    {"path": "test_calc.py", "sha256": "0" * 64, "parse_ok": True},
                ),
                source_hashes={"calc.py": self.calc_hash},
            )
        )
        assessment = self.assess([_record("write_file", path="calc.py"), record])
        self.assertFalse(assessment.task_completed)
        self.assertEqual(assessment.level, CompletionLevel.TESTS_INCOMPLETE)
        self.assertIn("TEST_FILE_CHANGED_AFTER_RUN", assessment.blockers)

    def test_unrelated_tests_are_incomplete(self) -> None:
        record = _trusted_record(_evidence(mapped_files=("other.py",)))
        assessment = self.assess([_record("write_file", path="calc.py"), record])
        self.assertFalse(assessment.task_completed)
        self.assertEqual(assessment.level, CompletionLevel.TESTS_INCOMPLETE)
        self.assertIn("UNRELATED_TESTS", assessment.blockers)

    def test_all_tools_failed_is_distinct(self) -> None:
        assessment = assess_completion(
            _response(""),
            [_record("read_file", status="FAILED", result_status="ERROR")],
            final_text="",
            workspace=self.root,
        )
        self.assertTrue(assessment.tool_call_failed)
        self.assertEqual(assessment.level, CompletionLevel.TOOL_CALL_FAILED)
        self.assertIn("EMPTY_FINAL_ANSWER", assessment.blockers)


class AgentLoopAcceptanceOracleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.root = self.workspace.path
        (self.root / "test_calc.py").write_text(
            "import unittest\nfrom calc import add\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n",
            encoding="utf-8",
        )
        self.oracle_dir = tempfile.TemporaryDirectory(prefix="yuzhu-loop-oracle-")
        self.addCleanup(self.oracle_dir.cleanup)
        self.oracle_path = Path(self.oracle_dir.name) / "accept.py"
        self.marker = "ORACLE_INTERNAL_MARKER_42"
        self.oracle_path.write_text(
            "import argparse, json\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--workspace')\n"
            "args = parser.parse_args()\n"
            "from pathlib import Path\n"
            "source = Path(args.workspace, 'calc.py').read_text(encoding='utf-8')\n"
            f"marker = {self.marker!r}\n"
            "print(json.dumps({'passed': 'return a + b' in source, "
            "'core_total': 1, 'marker': marker}))\n",
            encoding="utf-8",
        )
        from yuzhu_lab.harness.agent.verification import sha256_file

        self.oracle_sha = sha256_file(self.oracle_path)

    def scripted(self):
        return ScriptedModel(
            [
                tool_response(
                    [
                        (
                            "write_file",
                            {
                                "path": "calc.py",
                                "content": "def add(a, b):\n    return a + b\n",
                            },
                        )
                    ]
                ),
                tool_response([("run_tests", {"path": ".", "framework": "unittest"})]),
                text_response("Implemented add() and ran the tests."),
            ]
        )

    def test_loop_promotes_completion_with_passing_oracle(self) -> None:
        loop, store, _ = make_loop(
            self.root,
            self.scripted(),
            allow_execution=True,
            acceptance_script=self.oracle_path,
            acceptance_sha256=self.oracle_sha,
            acceptance_expected_checks=1,
            require_external_acceptance=True,
        )
        self.addCleanup(store.close)
        result = loop.run("implement add")
        self.assertEqual(result.status, "COMPLETED", result.error)
        self.assertTrue(result.completion["acceptance_passed"])
        self.assertEqual(result.completion["level"], "TASK_COMPLETED")

    def test_oracle_failure_blocks_completion_without_leaking_payload(self) -> None:
        self.oracle_path.write_text(
            "import json\n"
            "print(json.dumps({'passed': False, 'core_total': 1}))\n",
            encoding="utf-8",
        )
        from yuzhu_lab.harness.agent.verification import sha256_file

        loop, store, _ = make_loop(
            self.root,
            self.scripted(),
            allow_execution=True,
            acceptance_script=self.oracle_path,
            acceptance_sha256=sha256_file(self.oracle_path),
            acceptance_expected_checks=1,
            require_external_acceptance=True,
        )
        self.addCleanup(store.close)
        result = loop.run("implement add")
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.completion["level"], "ORACLE_FAILED")
        messages = json.dumps(store.get_task(loop.task_id)["messages"], ensure_ascii=False)
        self.assertNotIn(self.marker, messages)
        self.assertNotIn("core_total", messages)
        # The oracle runs after the final model response; the model is never
        # given acceptance details to game.
        self.assertEqual(len(loop.model.calls), 3)


if __name__ == "__main__":
    unittest.main()