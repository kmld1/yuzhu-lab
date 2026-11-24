"""Unit tests for tool registration, schema validation and permissions."""

from __future__ import annotations

import unittest

from tests.agent_test_utils import TempWorkspace, make_policy, make_registry
from yuzhu_lab.harness.agent.permissions import PermissionLevel
from yuzhu_lab.harness.agent.registry import (
    ToolContext,
    ToolDefinition,
    ToolOrigin,
    ToolRegistry,
    ToolRequest,
    ToolResult,
    ToolStatus,
)


class ToolRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.registry, self.context = make_registry(self.workspace.path)

    def call(self, name: str, arguments: dict) -> ToolResult:
        return self.registry.execute(
            ToolRequest(call_id="call-1", name=name, arguments=arguments)
        )

    def test_expected_tools_registered(self) -> None:
        names = set(self.registry.names())
        expected = {
            "read_file",
            "list_files",
            "search_text",
            "get_file_info",
            "write_file",
            "edit_file",
            "apply_patch",
            "run_python",
            "run_powershell",
            "run_tests",
            "git_status",
            "git_diff",
            "git_log",
        }
        self.assertEqual(names, expected)

    def test_anthropic_tool_payload_shape(self) -> None:
        tools = self.registry.anthropic_tools()
        read = next(tool for tool in tools if tool["name"] == "read_file")
        self.assertEqual(read["input_schema"]["type"], "object")
        self.assertIn("path", read["input_schema"]["properties"])

    def test_unknown_tool_returns_structured_error(self) -> None:
        result = self.call("delete_everything", {})
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "UNKNOWN_TOOL")
        self.assertIn("read_file", result.content)

    def test_missing_required_argument(self) -> None:
        result = self.call("read_file", {})
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "INVALID_ARGUMENTS")
        self.assertIn("missing required property 'path'", result.content)

    def test_wrong_argument_type(self) -> None:
        result = self.call("read_file", {"path": 42})
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "INVALID_ARGUMENTS")

    def test_unexpected_argument_rejected(self) -> None:
        result = self.call("read_file", {"path": "a.py", "shell": "rm -rf /"})
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "INVALID_ARGUMENTS")
        self.assertIn("unexpected property", result.content)

    def test_execution_disabled_by_default(self) -> None:
        script = self.workspace.path / "hello.py"
        script.write_text("print('hi')\n", encoding="utf-8")
        result = self.call("run_python", {"script": "hello.py"})
        self.assertEqual(result.status, ToolStatus.DENIED)
        self.assertEqual(result.error_code, "PERMISSION_DENIED")

    def test_write_permission_level_reported(self) -> None:
        result = self.call("write_file", {"path": "a.txt", "content": "x"})
        self.assertEqual(result.status, ToolStatus.OK)
        self.assertEqual(result.permission, PermissionLevel.WRITE_WORKSPACE.value)

    def test_overwrite_requires_approval(self) -> None:
        path = self.workspace.path / "a.txt"
        path.write_text("old", encoding="utf-8")
        result = self.call(
            "write_file", {"path": "a.txt", "content": "new", "overwrite": True}
        )
        self.assertEqual(result.status, ToolStatus.DENIED)
        self.assertEqual(result.permission, PermissionLevel.DESTRUCTIVE.value)
        self.assertEqual(path.read_text(encoding="utf-8"), "old")

    def test_tool_crash_is_contained(self) -> None:
        policy = make_policy(self.workspace.path)
        registry = ToolRegistry(policy, context=self.context)

        def explode(ctx: ToolContext, arguments: dict) -> ToolResult:
            raise RuntimeError("boom")

        registry.register(
            ToolDefinition(
                name="explode",
                description="test",
                input_schema={"type": "object", "properties": {}},
                execute=explode,
            )
        )
        result = registry.execute(ToolRequest(call_id="c", name="explode", arguments={}))
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.error_code, "TOOL_CRASH")

    def test_audit_log_receives_digest_not_content(self) -> None:
        records = []
        workspace = TempWorkspace()
        self.addCleanup(workspace.close)
        registry, _ = make_registry(workspace.path, audit=records.append)
        payload_text = "content-that-must-not-be-audited"
        registry.execute(
            ToolRequest(
                call_id="a",
                name="write_file",
                arguments={"path": "x.txt", "content": payload_text},
                origin=ToolOrigin.DETERMINISTIC_FALLBACK,
            )
        )
        self.assertEqual(records[0]["tool"], "write_file")
        self.assertEqual(records[0]["origin"], "DETERMINISTIC_FALLBACK")
        self.assertIn("arguments_digest", records[0])
        self.assertNotIn(payload_text, str(records[0]))

    def test_declared_timeout_is_passed_to_tools(self) -> None:
        policy = make_policy(self.workspace.path)
        registry = ToolRegistry(policy, context=self.context)
        seen = {}

        def slow(ctx: ToolContext, arguments: dict) -> ToolResult:
            seen["deadline"] = ctx.deadline
            ctx.check_deadline()
            return ToolResult(call_id="", name="slow", status=ToolStatus.OK, content="ok")

        registry.register(
            ToolDefinition(
                name="slow",
                description="test",
                input_schema={"type": "object", "properties": {}},
                execute=slow,
                timeout_seconds=5,
            )
        )
        registry.execute(ToolRequest(call_id="c", name="slow", arguments={}))
        self.assertIsNotNone(seen["deadline"])


if __name__ == "__main__":
    unittest.main()