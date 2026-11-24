"""Bounded subprocess output, timeout, cancellation and failure classification."""

from __future__ import annotations

import sys
import threading
import time
import unittest

from tests.agent_test_utils import TempWorkspace, make_registry
from yuzhu_lab.harness.agent.registry import ToolRequest, ToolStatus
from yuzhu_lab.harness.agent.tools.common import run_process
from yuzhu_lab.harness.agent.tools.exec import classify_failure


class RunProcessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.root = self.workspace.path

    def run_py(self, code: str, **kwargs):
        return run_process(
            [sys.executable, "-c", code],
            cwd=self.root,
            timeout=kwargs.pop("timeout", 30),
            **kwargs,
        )

    def test_normal_output(self) -> None:
        result = self.run_py("print('hello')")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("hello", result.stdout)
        self.assertFalse(result.output_truncated)
        self.assertFalse(result.read_incomplete)

    def test_large_stdout_is_bounded(self) -> None:
        result = self.run_py(
            "import sys; sys.stdout.write('a' * 2000000)",
            max_output_chars=5000,
            max_stdout_bytes=10000,
            max_stderr_bytes=10000,
            max_total_bytes=20000,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.stdout_truncated)
        self.assertGreaterEqual(result.stdout_bytes, 2_000_000)
        self.assertLessEqual(len(result.stdout), 5100)
        self.assertIn("truncated", result.stdout)

    def test_large_stderr_is_bounded(self) -> None:
        result = self.run_py(
            "import sys; sys.stderr.write('e' * 1000000)",
            max_output_chars=4000,
            max_stdout_bytes=10000,
            max_stderr_bytes=8000,
            max_total_bytes=16000,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.stderr_truncated)
        self.assertGreaterEqual(result.stderr_bytes, 1_000_000)
        self.assertLessEqual(len(result.stderr), 4100)

    def test_stdout_and_stderr_together(self) -> None:
        code = (
            "import sys\n"
            "sys.stdout.write('o' * 800000)\n"
            "sys.stderr.write('e' * 800000)\n"
        )
        result = self.run_py(
            code,
            max_output_chars=3000,
            max_stdout_bytes=6000,
            max_stderr_bytes=6000,
            max_total_bytes=10000,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.stdout_truncated)
        self.assertTrue(result.stderr_truncated)
        self.assertLessEqual(len(result.stdout), 3100)
        self.assertLessEqual(len(result.stderr), 3100)

    def test_timeout(self) -> None:
        started = time.perf_counter()
        result = self.run_py("import time; time.sleep(30)", timeout=1)
        elapsed = time.perf_counter() - started
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.exit_code)
        self.assertLess(elapsed, 10)

    def test_cancellation(self) -> None:
        cancel = threading.Event()

        def mark() -> None:
            time.sleep(0.5)
            cancel.set()

        threading.Thread(target=mark, daemon=True).start()
        result = self.run_py(
            "import time; time.sleep(30)",
            timeout=30,
            cancel=cancel.is_set,
        )
        self.assertTrue(result.cancelled)
        self.assertIsNone(result.exit_code)

    def test_abnormal_exit_code(self) -> None:
        result = self.run_py("import sys; sys.exit(7)")
        self.assertEqual(result.exit_code, 7)

    def test_cancel_race_is_consistent(self) -> None:
        cancel = threading.Event()
        cancel.set()
        result = self.run_py("print('fast')", cancel=cancel.is_set)
        self.assertFalse(result.cancelled and result.exit_code is not None)
        self.assertFalse(result.read_incomplete)

    def test_pipe_deadlock_does_not_occur(self) -> None:
        code = (
            "import sys\n"
            "sys.stdout.write('o' * 5000000)\n"
            "sys.stderr.write('e' * 5000000)\n"
        )
        started = time.perf_counter()
        result = self.run_py(
            code,
            timeout=30,
            max_stdout_bytes=20000,
            max_stderr_bytes=20000,
            max_total_bytes=40000,
            max_output_chars=2000,
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(result.exit_code, 0)
        self.assertLess(elapsed, 20)
        self.assertGreaterEqual(result.stdout_bytes, 5_000_000)
        self.assertGreaterEqual(result.stderr_bytes, 5_000_000)


class ClassificationTest(unittest.TestCase):
    def test_powershell_command_not_found(self) -> None:
        code, suggestion = classify_failure(
            "run_powershell",
            "powershell_probe.ps1 : The term 'powershell_probe.ps1' is not recognized "
            "as the name of a cmdlet, function, script file, or operable program.",
            1,
        )
        self.assertEqual(code, "COMMAND_NOT_FOUND")
        self.assertIn(".\\", suggestion or "")

    def test_powershell_path_not_found(self) -> None:
        code, _ = classify_failure(
            "run_powershell", "Get-Content : Cannot find path 'C:\\x\\y'", 1
        )
        self.assertEqual(code, "PATH_NOT_FOUND")

    def test_powershell_access_denied(self) -> None:
        code, _ = classify_failure(
            "run_powershell", "Access is denied. (UnauthorizedAccessException)", 1
        )
        self.assertEqual(code, "ACCESS_DENIED")

    def test_powershell_parameter_binding(self) -> None:
        code, _ = classify_failure(
            "run_powershell",
            "A parameter cannot be found that matches parameter name 'NoSuchParameter'.",
            1,
        )
        self.assertEqual(code, "INVALID_PARAMETERS")

    def test_python_missing_module(self) -> None:
        code, _ = classify_failure(
            "run_python", "ModuleNotFoundError: No module named 'requests'", 1
        )
        self.assertEqual(code, "MISSING_MODULE")

    def test_success_has_no_classification(self) -> None:
        self.assertEqual(classify_failure("run_python", "ok", 0), (None, None))


class ToolOutputTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.root = self.workspace.path

    def test_run_python_reports_truncation_without_faking_success(self) -> None:
        (self.root / "big.py").write_text(
            "import sys\nsys.stdout.write('x' * 1000000)\nsys.exit(0)\n",
            encoding="utf-8",
        )
        registry, _ = make_registry(self.root, allow_execution=True)
        result = registry.execute(
            ToolRequest(call_id="c", name="run_python", arguments={"script": "big.py"})
        )
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.structured["output_truncated"])
        self.assertIn("output truncated", result.content)

    def test_run_powershell_missing_script_has_suggestion(self) -> None:
        registry, _ = make_registry(self.root, allow_execution=True)
        result = registry.execute(
            ToolRequest(
                call_id="c",
                name="run_powershell",
                arguments={"script": "missing.ps1"},
            )
        )
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "SCRIPT_NOT_FOUND")
        self.assertIn(".\\", result.content)


if __name__ == "__main__":
    unittest.main()