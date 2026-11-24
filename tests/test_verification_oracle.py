"""H5.3 trusted verification oracle tests.

The H5.2 D3 run proved that ``exit_code == 0`` is not evidence that a test
suite is meaningful.  These tests exercise the oracle against real files and
real subprocesses: empty test files, print-only tests, all-skipped suites,
import errors, unrelated tests, stale code and a tampered acceptance script.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.agent_test_utils import TempWorkspace, make_registry
from yuzhu_lab.harness.agent.registry import ToolRequest, ToolStatus
from yuzhu_lab.harness.agent.verification import (
    VERDICT_NO_TESTS_DISCOVERED,
    VERDICT_NO_TESTS_EXECUTED,
    VERDICT_PASS,
    VERDICT_TESTS_FAILED,
    VERDICT_TESTS_INCOMPLETE,
    build_test_evidence,
    run_acceptance_oracle,
    scan_test_file,
    sha256_file,
)

CALC_SOURCE = "def add(a, b):\n    return a + b\n"
CALC_TEST = (
    "import unittest\n"
    "from calc import add\n\n"
    "class CalcTest(unittest.TestCase):\n"
    "    def test_add(self):\n"
    "        self.assertEqual(add(2, 3), 5)\n\n"
    "if __name__ == '__main__':\n"
    "    unittest.main()\n"
)

PRINT_ONLY_TEST = (
    "import unittest\n\n"
    "class VacuousTest(unittest.TestCase):\n"
    "    def test_print(self):\n"
    "        print('looks fine')\n"
)


class RunTestsToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.root = self.workspace.path
        self.registry, _ = make_registry(self.root, allow_execution=True)

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def run_tests(self, path: str, framework: str = "unittest"):
        return self.registry.execute(
            ToolRequest(
                call_id="call-test",
                name="run_tests",
                arguments={"path": path, "framework": framework},
            )
        )

    def test_empty_test_file_is_rejected(self) -> None:
        self.write("test_empty.py", "")
        result = self.run_tests("test_empty.py")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn(
            result.error_code,
            {"NO_TESTS_DISCOVERED", "NO_TESTS_EXECUTED"},
        )
        self.assertFalse(result.structured["passed"])
        self.assertNotEqual(result.structured["verification"]["verdict"], VERDICT_PASS)

    def test_print_only_test_is_rejected(self) -> None:
        self.write("test_print_only.py", PRINT_ONLY_TEST)
        result = self.run_tests("test_print_only.py")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "TESTS_INCOMPLETE")
        evidence = result.structured["verification"]
        self.assertEqual(evidence["assertions"], 0)
        self.assertIn("NO_ASSERTIONS", evidence["reasons"])

    def test_all_skipped_is_rejected(self) -> None:
        self.write(
            "test_skip.py",
            "import unittest\n\n"
            "class SkipTest(unittest.TestCase):\n"
            "    @unittest.skip('not today')\n"
            "    def test_never(self):\n"
            "        self.assertEqual(1, 1)\n",
        )
        result = self.run_tests("test_skip.py")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "TESTS_INCOMPLETE")
        evidence = result.structured["verification"]
        self.assertGreater(evidence["tests_executed"], 0)
        self.assertEqual(evidence["tests_skipped"], evidence["tests_executed"])
        self.assertIn("ALL_TESTS_SKIPPED", evidence["reasons"])

    def test_import_error_is_failed_not_passed(self) -> None:
        self.write(
            "test_broken_import.py",
            "import unittest\nimport module_that_does_not_exist\n",
        )
        result = self.run_tests("test_broken_import.py")
        self.assertNotEqual(result.status, ToolStatus.OK)
        self.assertFalse(result.structured["passed"])

    def test_real_passing_test_is_accepted(self) -> None:
        self.write("calc.py", CALC_SOURCE)
        self.write("test_calc.py", CALC_TEST)
        result = self.run_tests("test_calc.py")
        self.assertEqual(result.status, ToolStatus.OK, result.content)
        self.assertTrue(result.structured["passed"])
        evidence = result.structured["verification"]
        self.assertEqual(evidence["verdict"], VERDICT_PASS)
        self.assertGreaterEqual(evidence["tests_executed"], 1)
        self.assertGreaterEqual(evidence["assertions"], 1)
        self.assertIn("calc", evidence["imports"])
        self.assertIn("calc.py", evidence["mapped_files"])

    def test_vacuous_database_test_is_rejected(self) -> None:
        self.write("transfer.py", "def transfer(conn, a, b, amount):\n    return None\n")
        self.write(
            "test_transfer.py",
            "import sqlite3\nimport unittest\n"
            "from transfer import transfer\n\n"
            "class TestTransfer(unittest.TestCase):\n"
            "    def setUp(self):\n"
            "        self.conn = sqlite3.connect(':memory:')\n"
            "        self.conn.execute('CREATE TABLE accounts (id INTEGER, balance INTEGER)')\n\n"
            "    def tearDown(self):\n"
            "        self.conn.close()\n\n"
            "    def test_transfer(self):\n"
            "        conn = sqlite3.connect(':memory:')\n"
            "        conn.close()\n",
        )
        result = self.run_tests("test_transfer.py")
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "TESTS_INCOMPLETE")
        self.assertEqual(result.structured["verification"]["assertions"], 0)

    def test_missing_main_guard_still_discovers_tests(self) -> None:
        # A unittest file without ``unittest.main()`` must still be discovered
        # through ``python -m unittest`` rather than executed as a script.
        self.write("calc.py", CALC_SOURCE)
        self.write(
            "test_noguard.py",
            "import unittest\nfrom calc import add\n\n"
            "class NoGuardTest(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 1), 2)\n",
        )
        result = self.run_tests("test_noguard.py")
        self.assertEqual(result.status, ToolStatus.OK, result.content)
        self.assertGreaterEqual(result.structured["tests_run"], 1)


class ScanTestFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.root = self.workspace.path

    def test_scan_counts_tests_and_assertions(self) -> None:
        path = self.root / "test_calc.py"
        path.write_text(CALC_TEST, encoding="utf-8")
        scan = scan_test_file(path)
        self.assertTrue(scan["parse_ok"])
        self.assertEqual(scan["test_cases"], 1)
        self.assertGreaterEqual(scan["assertions"], 1)
        self.assertIn("calc", scan["imports"])
        self.assertEqual(scan["sha256"], sha256_file(path))

    def test_scan_flags_syntax_error(self) -> None:
        path = self.root / "test_broken.py"
        path.write_text("def broken(:\n", encoding="utf-8")
        scan = scan_test_file(path)
        self.assertFalse(scan["parse_ok"])
        self.assertIn("parse_error", scan)


class BuildTestEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.root = self.workspace.path
        (self.root / "calc.py").write_text(CALC_SOURCE, encoding="utf-8")
        (self.root / "test_calc.py").write_text(CALC_TEST, encoding="utf-8")

    def test_unrelated_test_is_incomplete(self) -> None:
        (self.root / "other.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.root / "test_other.py").write_text(
            "import unittest\nimport other\n\n"
            "class OtherTest(unittest.TestCase):\n"
            "    def test_value(self):\n"
            "        self.assertEqual(other.VALUE, 1)\n",
            encoding="utf-8",
        )
        evidence = build_test_evidence(
            self.root,
            target="test_other.py",
            framework="unittest",
            output="Ran 1 test in 0.001s\n\nOK\n",
            exit_code=0,
            changed_code_files=("calc.py",),
        )
        self.assertEqual(evidence["verdict"], VERDICT_TESTS_INCOMPLETE)
        self.assertIn("UNRELATED_TESTS", evidence["reasons"])

    def test_empty_output_is_no_tests_executed(self) -> None:
        evidence = build_test_evidence(
            self.root,
            target="test_calc.py",
            framework="unittest",
            output="",
            exit_code=0,
        )
        self.assertEqual(evidence["verdict"], VERDICT_NO_TESTS_EXECUTED)

    def test_zero_discovered_is_reported(self) -> None:
        (self.root / "test_calc.py").unlink()
        evidence = build_test_evidence(
            self.root,
            target=".",
            framework="unittest",
            output="Ran 0 tests in 0.000s\n\nNO TESTS RAN\n",
            exit_code=0,
        )
        self.assertEqual(evidence["verdict"], VERDICT_NO_TESTS_DISCOVERED)

    def test_failed_exit_is_tests_failed(self) -> None:
        evidence = build_test_evidence(
            self.root,
            target="test_calc.py",
            framework="unittest",
            output="Ran 1 test in 0.001s\n\nFAILED (failures=1)\n",
            exit_code=1,
        )
        self.assertEqual(evidence["verdict"], VERDICT_TESTS_FAILED)


class AcceptanceOracleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.root = self.workspace.path
        self.outside = tempfile.TemporaryDirectory(prefix="yuzhu-oracle-")
        self.addCleanup(self.outside.cleanup)
        self.oracle_dir = Path(self.outside.name)

    def write_oracle(self, text: str, name: str = "accept.py") -> Path:
        path = self.oracle_dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_acceptance_inside_workspace_is_rejected(self) -> None:
        script = self.root / "accept.py"
        script.write_text("print('{}')\n", encoding="utf-8")
        result = run_acceptance_oracle(
            self.root,
            script=script,
            expected_sha256=sha256_file(script),
            timeout_seconds=30,
        )
        self.assertFalse(result["passed"])
        self.assertIn("ORACLE_INSIDE_WORKSPACE", result["reasons"])

    def test_acceptance_hash_mismatch_is_rejected(self) -> None:
        script = self.write_oracle("print('{\"passed\": true}')\n")
        result = run_acceptance_oracle(
            self.root,
            script=script,
            expected_sha256="0" * 64,
            timeout_seconds=30,
        )
        self.assertFalse(result["passed"])
        self.assertIn("ORACLE_HASH_MISMATCH", result["reasons"])

    def test_acceptance_pass_is_parsed(self) -> None:
        script = self.write_oracle(
            "import json, sys\n"
            "print(json.dumps({'passed': True, 'core_total': 10, 'core_passed': True}))\n"
        )
        result = run_acceptance_oracle(
            self.root,
            script=script,
            expected_sha256=sha256_file(script),
            timeout_seconds=30,
        )
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["payload"]["core_total"], 10)

    def test_acceptance_failure_is_reported(self) -> None:
        script = self.write_oracle(
            "import json\n"
            "print(json.dumps({'passed': False, 'core_total': 10, 'core_passed': False}))\n"
        )
        result = run_acceptance_oracle(
            self.root,
            script=script,
            expected_sha256=sha256_file(script),
            timeout_seconds=30,
        )
        self.assertFalse(result["passed"])
        self.assertIn("ORACLE_REPORTED_FAILURE", result["reasons"])

    def test_acceptance_timeout_is_reported(self) -> None:
        script = self.write_oracle("import time\ntime.sleep(30)\n")
        result = run_acceptance_oracle(
            self.root,
            script=script,
            expected_sha256=sha256_file(script),
            timeout_seconds=1,
        )
        self.assertFalse(result["passed"])
        self.assertIn("ORACLE_TIMEOUT", result["reasons"])

    def test_acceptance_invalid_json_is_reported(self) -> None:
        script = self.write_oracle("print('not json')\n")
        result = run_acceptance_oracle(
            self.root,
            script=script,
            expected_sha256=sha256_file(script),
            timeout_seconds=30,
        )
        self.assertFalse(result["passed"])
        self.assertIn("ORACLE_INVALID_OUTPUT", result["reasons"])


if __name__ == "__main__":
    unittest.main()