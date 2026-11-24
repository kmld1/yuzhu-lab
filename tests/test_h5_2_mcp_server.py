"""Unit tests for the H5.2 MCP stdio bridge (no model, no network)."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from yuzhu_lab.harness.mcp_server import (
    McpServer,
    McpTool,
    ToolCancelled,
    ToolError,
)


class RecordingWriter:
    def __init__(self) -> None:
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.buffer = b""
        self.lock = threading.Lock()

    def write(self, data: bytes) -> int:
        with self.lock:
            self.buffer += data
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                if line.strip():
                    self.queue.put(json.loads(line.decode("utf-8")))
        return len(data)

    def flush(self) -> None:
        return

    def response(self, request_id: Any, timeout: float = 5.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                message = self.queue.get(timeout=max(0.01, deadline - time.time()))
            except queue.Empty:
                break
            if message.get("id") == request_id:
                return message
        raise AssertionError(f"no response for {request_id}")


class SilentLog:
    def write(self, data: bytes) -> int:
        return len(data)

    def flush(self) -> None:
        return


def _echo(arguments: dict[str, Any], cancel: threading.Event) -> str:
    return json.dumps({"echo": arguments}, ensure_ascii=False)


def _fail(arguments: dict[str, Any], cancel: threading.Event) -> str:
    raise ToolError("expected failure")


def _slow(arguments: dict[str, Any], cancel: threading.Event) -> str:
    if cancel.wait(10.0):
        raise ToolCancelled("cancelled")
    return "slow finished"


class McpServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.read_fd, self.write_fd = os.pipe()
        self.reader = os.fdopen(self.read_fd, "rb", buffering=0)
        self.writer = RecordingWriter()
        self.tools = [
            McpTool(
                name="echo",
                description="echo arguments",
                input_schema={"type": "object", "properties": {}, "additionalProperties": True},
                handler=_echo,
                timeout_seconds=5.0,
            ),
            McpTool(
                name="fail",
                description="always fails",
                input_schema={"type": "object", "properties": {}},
                handler=_fail,
                timeout_seconds=5.0,
            ),
            McpTool(
                name="slow",
                description="waits for cancellation",
                input_schema={"type": "object", "properties": {}},
                handler=_slow,
                timeout_seconds=5.0,
            ),
            McpTool(
                name="timeout",
                description="times out quickly",
                input_schema={"type": "object", "properties": {}},
                handler=_slow,
                timeout_seconds=0.2,
            ),
        ]
        self.server = McpServer(
            self.tools,
            reader=self.reader,
            writer=self.writer,  # type: ignore[arg-type]
            log=SilentLog(),  # type: ignore[arg-type]
        )
        self.thread = threading.Thread(target=self.server.serve, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        try:
            os.close(self.write_fd)
        except OSError:
            pass
        self.thread.join(timeout=6)
        try:
            self.reader.close()
        except Exception:
            pass

    def send(self, message: dict[str, Any]) -> None:
        os.write(self.write_fd, (json.dumps(message) + "\n").encode("utf-8"))

    def request(self, request_id: Any, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            message["params"] = params
        self.send(message)
        return self.writer.response(request_id)

    def test_initialize_handshake(self) -> None:
        response = self.request(
            "init",
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        )
        result = response["result"]
        self.assertEqual(result["protocolVersion"], "2025-06-18")
        self.assertEqual(result["serverInfo"]["name"], "yuzhu-harness")
        self.assertIn("tools", result["capabilities"])

    def test_unknown_protocol_falls_back(self) -> None:
        response = self.request("init2", "initialize", {"protocolVersion": "1999-01-01"})
        self.assertIn(response["result"]["protocolVersion"], ("2024-11-05", "2025-03-26", "2025-06-18"))

    def test_tools_list(self) -> None:
        response = self.request("list", "tools/list", {})
        names = [tool["name"] for tool in response["result"]["tools"]]
        self.assertEqual(names, ["echo", "fail", "slow", "timeout"])
        self.assertTrue(all("inputSchema" in tool for tool in response["result"]["tools"]))

    def test_ping(self) -> None:
        response = self.request("ping1", "ping", {})
        self.assertEqual(response["result"], {})

    def test_unknown_method(self) -> None:
        response = self.request("m1", "no/such/method", {})
        self.assertEqual(response["error"]["code"], -32601)

    def test_unknown_tool(self) -> None:
        response = self.request("t1", "tools/call", {"name": "nope", "arguments": {}})
        self.assertEqual(response["error"]["code"], -32602)

    def test_tool_error_is_reported(self) -> None:
        response = self.request("t2", "tools/call", {"name": "fail", "arguments": {}})
        self.assertTrue(response["result"]["isError"])
        self.assertIn("expected failure", response["result"]["content"][0]["text"])

    def test_tool_success(self) -> None:
        response = self.request(
            "t3", "tools/call", {"name": "echo", "arguments": {"x": 1}}
        )
        self.assertFalse(response["result"]["isError"])
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertEqual(payload["echo"], {"x": 1})

    def test_cancellation(self) -> None:
        request_id = "cancel-me"
        self.send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": "slow", "arguments": {}},
            }
        )
        time.sleep(0.2)
        self.send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": {"requestId": request_id, "reason": "test"},
            }
        )
        response = self.writer.response(request_id, timeout=5.0)
        self.assertEqual(response["error"]["code"], -32800)

    def test_timeout(self) -> None:
        response = self.request("timeout-me", "tools/call", {"name": "timeout", "arguments": {}})
        self.assertEqual(response["error"]["code"], -32603)
        self.assertIn("timed out", response["error"]["message"])

    def test_parse_error_does_not_kill_server(self) -> None:
        os.write(self.write_fd, b"{not json}\n")
        ping = self.request("ping2", "ping", {})
        self.assertEqual(ping["result"], {})

    def test_notification_without_id_gets_no_response(self) -> None:
        self.send({"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "echo", "arguments": {}}})
        with self.assertRaises(AssertionError):
            self.writer.response(None, timeout=0.3)


if __name__ == "__main__":
    unittest.main()