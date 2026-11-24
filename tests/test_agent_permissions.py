"""Workspace boundary and permission policy tests."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

from tests.agent_test_utils import TempWorkspace, make_policy
from yuzhu_lab.harness.agent.errors import PathViolation
from yuzhu_lab.harness.agent.permissions import PermissionLevel, WorkspaceGuard


class WorkspaceGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.root = self.workspace.path
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text("print(1)\n", encoding="utf-8")
        self.guard = WorkspaceGuard(self.root)

    def test_relative_path_resolves_inside(self) -> None:
        resolved = self.guard.resolve("src/app.py", must_exist=True)
        self.assertEqual(resolved, (self.root / "src" / "app.py").resolve())

    def test_absolute_path_inside_allowed(self) -> None:
        absolute = str((self.root / "src" / "app.py").resolve())
        self.assertTrue(self.guard.resolve(absolute, must_exist=True).is_file())

    def test_parent_traversal_denied(self) -> None:
        with self.assertRaises(PathViolation) as caught:
            self.guard.resolve("../outside.txt")
        self.assertEqual(caught.exception.code, "PATH_OUTSIDE_WORKSPACE")

    def test_absolute_outside_denied(self) -> None:
        with self.assertRaises(PathViolation) as caught:
            self.guard.resolve("C:/Windows/win.ini", must_exist=False)
        self.assertEqual(caught.exception.code, "PATH_OUTSIDE_WORKSPACE")

    def test_unc_path_denied(self) -> None:
        with self.assertRaises(PathViolation) as caught:
            self.guard.resolve("\\\\server\\share\\file.txt")
        self.assertEqual(caught.exception.code, "PATH_UNC")

    def test_alternate_data_stream_denied(self) -> None:
        with self.assertRaises(PathViolation) as caught:
            self.guard.resolve("src/app.py:hidden")
        self.assertEqual(caught.exception.code, "PATH_ADS")

    def test_drive_relative_denied(self) -> None:
        with self.assertRaises(PathViolation) as caught:
            self.guard.resolve("C:app.py")
        self.assertEqual(caught.exception.code, "PATH_DRIVE_RELATIVE")

    def test_protected_prefix(self) -> None:
        (self.root / "eval").mkdir()
        protected = (self.root / "eval" / "sft_eval_v2.jsonl").resolve()
        self.assertTrue(self.guard.is_protected(protected))
        with self.assertRaises(PathViolation) as caught:
            self.guard.ensure_write_allowed(protected)
        self.assertEqual(caught.exception.code, "PATH_PROTECTED")

    def test_junction_outside_denied(self) -> None:
        if os.name != "nt":
            self.skipTest("junction test requires Windows")
        outside = TempWorkspace()
        self.addCleanup(outside.close)
        (outside.path / "secret.txt").write_text("secret", encoding="utf-8")
        link = self.root / "link"
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside.path)],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            self.skipTest(f"cannot create junction: {completed.stderr.strip()[:120]}")
        with self.assertRaises(PathViolation) as caught:
            self.guard.resolve("link/secret.txt", must_exist=True)
        self.assertIn(
            caught.exception.code,
            {"PATH_REPARSE_POINT", "PATH_OUTSIDE_WORKSPACE"},
        )

    def test_nonexistent_write_path_inside_allowed(self) -> None:
        resolved = self.guard.resolve("src/new.py", for_write=True)
        self.assertEqual(resolved.parent, (self.root / "src").resolve())


class PermissionPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.root = self.workspace.path

    def test_read_allowed(self) -> None:
        policy = make_policy(self.root)
        decision = policy.check(name="read_file", arguments={}, level=PermissionLevel.READ)
        self.assertTrue(decision.allowed)

    def test_write_outside_denied(self) -> None:
        policy = make_policy(self.root)
        decision = policy.check(
            name="write_file",
            arguments={"path": "../escape.txt"},
            level=PermissionLevel.WRITE_WORKSPACE,
        )
        self.assertFalse(decision.allowed)

    def test_protected_write_denied(self) -> None:
        (self.root / "eval").mkdir()
        policy = make_policy(self.root)
        decision = policy.check(
            name="write_file",
            arguments={"path": "eval/sft_eval_v2.jsonl"},
            level=PermissionLevel.WRITE_WORKSPACE,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("PATH_PROTECTED", decision.reason)

    def test_execution_requires_enable_flag(self) -> None:
        policy = make_policy(self.root, allow_execution=False)
        decision = policy.check(
            name="run_tests", arguments={}, level=PermissionLevel.EXECUTE_APPROVED
        )
        self.assertFalse(decision.allowed)
        enabled = make_policy(self.root, allow_execution=True)
        decision = enabled.check(
            name="run_tests", arguments={}, level=PermissionLevel.EXECUTE_APPROVED
        )
        self.assertTrue(decision.allowed)

    def test_destructive_requires_approval_callback(self) -> None:
        policy = make_policy(self.root)
        denied = policy.check(
            name="write_file", arguments={}, level=PermissionLevel.DESTRUCTIVE
        )
        self.assertFalse(denied.allowed)

        requested = []

        def approve(name: str, arguments: dict) -> bool:
            requested.append(name)
            return True

        approving = make_policy(self.root, approval=approve)
        allowed = approving.check(
            name="write_file", arguments={}, level=PermissionLevel.DESTRUCTIVE
        )
        self.assertTrue(allowed.allowed)
        self.assertEqual(requested, ["write_file"])

    def test_preapproved_destructive(self) -> None:
        policy = make_policy(self.root, allow_destructive=True)
        decision = policy.check(
            name="write_file", arguments={}, level=PermissionLevel.DESTRUCTIVE
        )
        self.assertTrue(decision.allowed)


if __name__ == "__main__":
    unittest.main()