"""Real execution tests for the local coding tools (no model, no GPU)."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from tests.agent_test_utils import TempWorkspace, make_registry
from yuzhu_lab.harness.agent.registry import ToolRequest, ToolResult, ToolStatus


class LocalCodingToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.root = self.workspace.path

    def call(
        self,
        name: str,
        arguments: dict,
        *,
        allow_execution: bool = False,
        allow_destructive: bool = False,
        registry=None,
    ) -> ToolResult:
        if registry is None:
            registry, _ = make_registry(
                self.root,
                allow_execution=allow_execution,
                allow_destructive=allow_destructive,
            )
        return registry.execute(
            ToolRequest(call_id=f"call-{name}", name=name, arguments=arguments)
        )

    # -------------------------------------------------------------- file tools

    def test_write_read_edit_read_roundtrip(self) -> None:
        registry, _ = make_registry(self.root)
        write = self.call(
            "write_file",
            {"path": "src/app.py", "content": "def add(a, b):\n    return a + b\n", "create_parents": True},
            registry=registry,
        )
        self.assertEqual(write.status, ToolStatus.OK, write.content)
        self.assertTrue(write.structured["readback_verified"])

        read = self.call("read_file", {"path": "src/app.py"}, registry=registry)
        self.assertIn("return a + b", read.content)
        self.assertEqual(read.structured["total_lines"], 2)

        edit = self.call(
            "edit_file",
            {
                "path": "src/app.py",
                "old_text": "return a + b",
                "new_text": "return a + b + 0",
            },
            registry=registry,
        )
        self.assertEqual(edit.status, ToolStatus.OK, edit.content)
        self.assertTrue(edit.structured["readback_verified"])
        final = self.call("read_file", {"path": "src/app.py"}, registry=registry)
        self.assertIn("return a + b + 0", final.content)

    def test_write_existing_fails_without_overwrite(self) -> None:
        (self.root / "a.txt").write_text("old", encoding="utf-8")
        result = self.call("write_file", {"path": "a.txt", "content": "new"})
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "FILE_EXISTS")
        self.assertEqual((self.root / "a.txt").read_text(encoding="utf-8"), "old")

    def test_overwrite_with_approval(self) -> None:
        (self.root / "a.txt").write_text("old", encoding="utf-8")
        result = self.call(
            "write_file",
            {"path": "a.txt", "content": "new", "overwrite": True},
            allow_destructive=True,
        )
        self.assertEqual(result.status, ToolStatus.OK, result.content)
        self.assertEqual((self.root / "a.txt").read_text(encoding="utf-8"), "new")

    def test_edit_ambiguous_match_fails(self) -> None:
        (self.root / "a.txt").write_text("x\nx\n", encoding="utf-8")
        result = self.call("edit_file", {"path": "a.txt", "old_text": "x", "new_text": "y"})
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "AMBIGUOUS_MATCH")

    def test_edit_non_utf8_rejected(self) -> None:
        (self.root / "latin.txt").write_bytes(b"caf\xe9\n")
        result = self.call(
            "edit_file", {"path": "latin.txt", "old_text": "caf", "new_text": "cafe"}
        )
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "ENCODING_UNSUPPORTED")
        self.assertEqual((self.root / "latin.txt").read_bytes(), b"caf\xe9\n")

    def test_edit_missing_text_fails(self) -> None:
        (self.root / "a.txt").write_text("hello\n", encoding="utf-8")
        result = self.call("edit_file", {"path": "a.txt", "old_text": "absent", "new_text": "x"})
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "OLD_TEXT_NOT_FOUND")

    def test_apply_patch(self) -> None:
        (self.root / "app.py").write_text(
            "def add(a, b):\n    return a + b\n", encoding="utf-8"
        )
        patch = (
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a + b\n"
            "+    return a + b + 1\n"
        )
        result = self.call("apply_patch", {"path": "app.py", "patch": patch})
        self.assertEqual(result.status, ToolStatus.OK, result.content)
        self.assertIn("return a + b + 1", (self.root / "app.py").read_text(encoding="utf-8"))

    def test_apply_patch_conflict(self) -> None:
        (self.root / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        patch = (
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a - b\n"
            "+    return a + b\n"
        )
        result = self.call("apply_patch", {"path": "app.py", "patch": patch})
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "PATCH_CONFLICT")

    def test_read_binary_rejected(self) -> None:
        (self.root / "blob.bin").write_bytes(b"\x00\x01\x02binary")
        result = self.call("read_file", {"path": "blob.bin"})
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "BINARY_FILE")

    def test_read_line_range(self) -> None:
        (self.root / "lines.txt").write_text("\n".join(str(i) for i in range(1, 11)), encoding="utf-8")
        result = self.call("read_file", {"path": "lines.txt", "start_line": 3, "max_lines": 2})
        self.assertEqual(result.content, "3\n4")
        self.assertTrue(result.structured["truncated"])

    def test_list_files_recursive(self) -> None:
        (self.root / "src").mkdir()
        (self.root / "src" / "a.py").write_text("x", encoding="utf-8")
        (self.root / "top.txt").write_text("y", encoding="utf-8")
        result = self.call("list_files", {"path": ".", "recursive": True})
        self.assertIn("src/a.py", result.content)
        self.assertIn("top.txt", result.content)

    def test_search_text(self) -> None:
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text("alpha\nbeta_target\ngamma\n", encoding="utf-8")
        result = self.call("search_text", {"query": "beta_target"})
        self.assertIn("src/app.py:2", result.content)
        self.assertEqual(result.structured["matches"], 1)

    def test_get_file_info_hash(self) -> None:
        (self.root / "a.txt").write_text("hello", encoding="utf-8")
        result = self.call("get_file_info", {"path": "a.txt"})
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(len(result.structured["sha256"]), 64)
        self.assertEqual(result.structured["lines"], 1)

    # ------------------------------------------------------------ exec tools

    def test_run_tests_real_pass_and_fail(self) -> None:
        registry, _ = make_registry(self.root, allow_execution=True)
        (self.root / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        (self.root / "test_calc.py").write_text(
            "import unittest\n"
            "from calc import add\n\n"
            "class CalcTest(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(2, 3), 5)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        passed = self.call("run_tests", {"path": ".", "framework": "unittest"}, registry=registry)
        self.assertEqual(passed.status, ToolStatus.OK, passed.content)
        self.assertEqual(passed.exit_code, 0)
        self.assertTrue(passed.structured["passed"])
        self.assertIn("TESTS_PASSED", passed.content)

        (self.root / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        failed = self.call("run_tests", {"path": ".", "framework": "unittest"}, registry=registry)
        self.assertEqual(failed.exit_code, 1)
        self.assertFalse(failed.structured["passed"])
        self.assertIn("TESTS_FAILED", failed.content)

    def test_run_tests_with_no_tests_is_not_a_pass(self) -> None:
        registry, _ = make_registry(self.root, allow_execution=True)
        (self.root / "empty").mkdir()
        result = self.call(
            "run_tests", {"path": "empty", "framework": "unittest"}, registry=registry
        )
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "NO_TESTS_DISCOVERED")
        self.assertEqual(result.structured["tests_run"], 0)
        self.assertFalse(result.structured["passed"])

    def test_run_python_exit_code(self) -> None:
        registry, _ = make_registry(self.root, allow_execution=True)
        (self.root / "ok.py").write_text("import sys\nprint('value=7')\nsys.exit(0)\n", encoding="utf-8")
        result = self.call("run_python", {"script": "ok.py"}, registry=registry)
        self.assertEqual(result.status, ToolStatus.OK, result.content)
        self.assertEqual(result.exit_code, 0)
        self.assertIn("value=7", result.content)

        (self.root / "bad.py").write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
        result = self.call("run_python", {"script": "bad.py"}, registry=registry)
        self.assertEqual(result.exit_code, 3)

    def test_run_python_timeout(self) -> None:
        registry, _ = make_registry(self.root, allow_execution=True)
        (self.root / "sleep.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
        started = time.perf_counter()
        result = self.call(
            "run_python", {"script": "sleep.py", "timeout_seconds": 1}, registry=registry
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(result.status, ToolStatus.TIMEOUT)
        self.assertLess(elapsed, 15)

    def test_run_python_cancellation(self) -> None:
        registry, context = make_registry(self.root, allow_execution=True)
        (self.root / "sleep.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
        cancel = threading.Event()

        def mark() -> None:
            time.sleep(0.5)
            cancel.set()

        threading.Thread(target=mark, daemon=True).start()
        registry.cancel_check = cancel.is_set
        started = time.perf_counter()
        result = self.call("run_python", {"script": "sleep.py"}, registry=registry)
        elapsed = time.perf_counter() - started
        self.assertEqual(result.status, ToolStatus.CANCELLED)
        self.assertLess(elapsed, 15)

    def test_run_powershell_script(self) -> None:
        import shutil

        if shutil.which("pwsh") is None and shutil.which("powershell") is None:
            self.skipTest("PowerShell is not available")
        registry, _ = make_registry(self.root, allow_execution=True)
        (self.root / "hello.ps1").write_text("Write-Output 'ps-ok'\n", encoding="utf-8")
        result = self.call("run_powershell", {"script": "hello.ps1"}, registry=registry)
        self.assertEqual(result.status, ToolStatus.OK, result.content)
        self.assertIn("ps-ok", result.content)

    def test_inline_command_argument_rejected(self) -> None:
        result = self.call("run_powershell", {"script": "x.ps1", "command": "whoami"})
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "INVALID_ARGUMENTS")

    # -------------------------------------------------------------- git tools

    def test_git_status_not_a_repository(self) -> None:
        result = self.call("git_status", {})
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertIn(result.error_code, {"GIT_NOT_REPOSITORY", "GIT_ERROR"})

    def test_git_status_in_repository(self) -> None:
        import shutil
        import subprocess

        if shutil.which("git") is None:
            self.skipTest("git is not available")
        subprocess.run(["git", "init"], cwd=self.root, capture_output=True, check=False)
        (self.root / "a.txt").write_text("x", encoding="utf-8")
        result = self.call("git_status", {})
        self.assertEqual(result.status, ToolStatus.OK, result.content)
        self.assertIn("a.txt", result.content)


if __name__ == "__main__":
    unittest.main()