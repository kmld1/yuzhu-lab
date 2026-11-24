"""Tool failure classification and repeated-failure recovery."""

from __future__ import annotations

import shutil
import unittest

from tests.agent_test_utils import (
    ScriptedModel,
    TempWorkspace,
    make_loop,
    make_registry,
    text_response,
    tool_response,
)
from yuzhu_lab.harness.agent.registry import ToolRequest, ToolStatus


class ToolRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.root = self.workspace.path
        self.has_powershell = bool(
            shutil.which("pwsh") or shutil.which("powershell")
        )

    def test_missing_powershell_script_is_structured(self) -> None:
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
        self.assertIn("suggestion", result.structured)
        self.assertIn(".\\", result.structured["suggestion"])

    def test_bad_parameter_is_classified(self) -> None:
        if not self.has_powershell:
            self.skipTest("PowerShell is not available")
        (self.root / "badparam.ps1").write_text(
            "$ErrorActionPreference = 'Stop'\n"
            "try { Get-ChildItem -NoSuchParameter } catch { Write-Error $_; exit 1 }\n",
            encoding="utf-8",
        )
        registry, _ = make_registry(self.root, allow_execution=True)
        result = registry.execute(
            ToolRequest(
                call_id="c",
                name="run_powershell",
                arguments={"script": "badparam.ps1"},
            )
        )
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("suggestion", result.structured)
        self.assertTrue(result.structured["suggestion"])

    def test_repeated_failing_call_is_blocked(self) -> None:
        if not self.has_powershell:
            self.skipTest("PowerShell is not available")
        (self.root / "good.ps1").write_text("Write-Output 'OK'\n", encoding="utf-8")
        model = ScriptedModel(
            [
                tool_response([("run_powershell", {"script": "missing.ps1"})]),
                tool_response([("run_powershell", {"script": "missing.ps1"})]),
                tool_response([("run_powershell", {"script": "good.ps1"})]),
                text_response("done"),
            ]
        )
        loop, store, _ = make_loop(self.root, model, allow_execution=True)
        self.addCleanup(store.close)
        result = loop.run("run the script")
        self.assertEqual(result.status, "COMPLETED", result.error)
        calls = store.tool_calls(loop.task_id)
        self.assertEqual(calls[0]["result"]["error_code"], "SCRIPT_NOT_FOUND")
        self.assertEqual(calls[1]["result"]["error_code"], "REPEATED_FAILED_CALL")
        self.assertEqual(calls[1]["result"]["structured"]["failure_count"], 1)
        self.assertEqual(calls[2]["status"], "COMPLETED")
        self.assertEqual(calls[2]["result"]["exit_code"], 0)

    def test_repeatable_test_runner_is_not_blocked(self) -> None:
        (self.root / "test_always_fail.py").write_text(
            "import unittest\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_fail(self):\n"
            "        self.assertEqual(1, 2)\n\n"
            "if __name__ == '__main__':\n    unittest.main()\n",
            encoding="utf-8",
        )
        model = ScriptedModel(
            [
                tool_response(
                    [("run_tests", {"path": "test_always_fail.py", "framework": "unittest"})]
                ),
                tool_response(
                    [("run_tests", {"path": "test_always_fail.py", "framework": "unittest"})]
                ),
                text_response("tests still fail"),
            ]
        )
        loop, store, _ = make_loop(self.root, model, allow_execution=True)
        self.addCleanup(store.close)
        result = loop.run("run tests")
        # The repeatable runner is not blocked, but a failing last test must not
        # be reported as a completed task (H4 completion gate).
        self.assertEqual(result.status, "FAILED", result.error)
        self.assertIn("LAST_TEST_FAILED", result.completion["blockers"])
        calls = store.tool_calls(loop.task_id)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["result"]["exit_code"], 1)
        self.assertEqual(calls[1]["result"]["exit_code"], 1)
        self.assertNotEqual(calls[1]["result"].get("error_code"), "REPEATED_FAILED_CALL")

    def test_repeated_successful_test_runs_are_capped(self) -> None:
        (self.root / "test_pass.py").write_text(
            "import unittest\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_ok(self):\n"
            "        self.assertTrue(True)\n\n"
            "if __name__ == '__main__':\n    unittest.main()\n",
            encoding="utf-8",
        )
        responses = [
            tool_response([("run_tests", {"path": "test_pass.py", "framework": "unittest"})])
            for _ in range(3)
        ] + [text_response("done")]
        model = ScriptedModel(responses)
        loop, store, _ = make_loop(self.root, model, allow_execution=True)
        self.addCleanup(store.close)
        result = loop.run("run tests")
        self.assertEqual(result.status, "COMPLETED", result.error)
        calls = store.tool_calls(loop.task_id)
        self.assertEqual(calls[0]["result"]["exit_code"], 0)
        self.assertEqual(calls[1]["result"]["exit_code"], 0)
        self.assertEqual(calls[2]["result"]["error_code"], "REPEATED_SUCCESS_CALL")

    def test_retry_after_file_fix_is_allowed(self) -> None:
        (self.root / "broken.py").write_text(
            "import sys\nsys.exit(1)\n", encoding="utf-8"
        )
        model = ScriptedModel(
            [
                tool_response([("run_python", {"script": "broken.py"})]),
                tool_response(
                    [
                        (
                            "edit_file",
                            {
                                "path": "broken.py",
                                "old_text": "sys.exit(1)",
                                "new_text": "sys.exit(0)",
                            },
                        )
                    ]
                ),
                tool_response([("run_python", {"script": "broken.py"})]),
                text_response("fixed and reran"),
            ]
        )
        loop, store, _ = make_loop(self.root, model, allow_execution=True)
        self.addCleanup(store.close)
        result = loop.run("fix the script and rerun")
        self.assertEqual(result.status, "COMPLETED", result.error)
        calls = store.tool_calls(loop.task_id)
        self.assertEqual(calls[0]["result"]["exit_code"], 1)
        self.assertEqual(calls[1]["status"], "COMPLETED")
        self.assertEqual(calls[1]["result"]["structured"]["readback_verified"], True)
        self.assertEqual(calls[2]["result"]["exit_code"], 0)
        self.assertNotEqual(calls[2]["result"].get("error_code"), "REPEATED_FAILED_CALL")

    def test_consecutive_failure_budget_stops_task(self) -> None:
        model = ScriptedModel(
            [
                tool_response([("run_powershell", {"script": f"missing{i}.ps1"})])
                for i in range(4)
            ]
        )
        loop, store, _ = make_loop(
            self.root,
            model,
            allow_execution=True,
            max_consecutive_failures=3,
        )
        self.addCleanup(store.close)
        result = loop.run("run missing scripts")
        self.assertEqual(result.status, "FAILED")
        self.assertIn("tool failure budget exhausted", result.error or "")
        self.assertEqual(store.get_task(loop.task_id)["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()