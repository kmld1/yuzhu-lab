"""Minimal MCP stdio server exposing the YUZHU agent harness.

This is the H5.2 bridge between Claude Code and the standalone harness.  It
implements the MCP stdio transport directly (JSON-RPC 2.0 over newline-delimited
UTF-8 JSON) so no new dependency is introduced.

Security model:
    * every tool is workspace-confined and uses the existing harness permission
      policy; there is no shell and no arbitrary command endpoint;
    * execution tools are disabled unless the server process was started with
      ``--allow-execute``; a client cannot elevate that by request argument;
    * retention into Hindsight is disabled unless ``--allow-retain`` is passed;
    * stdout carries protocol messages only; logs go to stderr.

The server never accepts a credential from the client and never prints one.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable

from .agent.factory import (
    DEFAULT_AUDIT_LOG,
    DEFAULT_CHECKPOINT_DB,
    DEFAULT_EMBEDDING,
    DEFAULT_RAG_STORE,
    PROJECT_ROOT,
    AgentFactoryConfig,
    build_agent_runtime,
)
from .agent.loop import CancelToken
from .agent.permissions import PermissionPolicy
from .memory_adapter import HindsightMemoryAdapter
from .rag_adapter import RagAdapter
from .agent.resources import check_services
from ..source_policy import SourcePolicy, protected_content_guard

SERVER_NAME = "yuzhu-harness"
SERVER_VERSION = "0.2.0-h5.2"
SUPPORTED_PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
DEFAULT_STATE_DIR = PROJECT_ROOT / "data" / "h5_2_mcp"

JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603
JSONRPC_REQUEST_CANCELLED = -32800


class ToolError(RuntimeError):
    """A tool failed in a way the client should see as a tool error."""


class ToolCancelled(RuntimeError):
    """The request was cancelled by the client or by the server deadline."""


class _Tee:
    """Duplicate log writes to stderr and an evidence file."""

    def __init__(self, *streams: BinaryIO) -> None:
        self.streams = streams
        self._lock = threading.Lock()

    def write(self, data: bytes) -> int:
        with self._lock:
            for stream in self.streams:
                try:
                    stream.write(data)
                    stream.flush()
                except Exception:  # noqa: BLE001 - logging must not break the server
                    pass
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:  # noqa: BLE001
                pass


@dataclass(frozen=True)
class McpServerSettings:
    workspace: Path = PROJECT_ROOT
    project_id: str = "qianyuqwen"
    runtime_url: str = "http://127.0.0.1:8710"
    hindsight_url: str = "http://127.0.0.1:8890"
    rag_store: Path = DEFAULT_RAG_STORE
    embedding_model: Path = DEFAULT_EMBEDDING
    state_dir: Path = DEFAULT_STATE_DIR
    allow_execute: bool = False
    allow_retain: bool = False
    max_task_seconds: float = 600.0
    max_turns_limit: int = 20


@dataclass(frozen=True)
class McpTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any], threading.Event], str]
    timeout_seconds: float = 60.0


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _snippet(text: str, limit: int = 700) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


class HarnessTools:
    """Builds MCP tools backed by the real RAG, Hindsight and agent harness."""

    def __init__(self, settings: McpServerSettings) -> None:
        self.settings = settings
        self._rag: RagAdapter | None = None
        self._rag_lock = threading.Lock()
        self._memory: HindsightMemoryAdapter | None = None
        self._memory_client: Any = None
        self._memory_lock = threading.Lock()
        self._agent_lock = threading.Lock()
        settings.state_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- services

    def _rag_adapter(self) -> RagAdapter:
        with self._rag_lock:
            if self._rag is None:
                self._rag = RagAdapter(
                    store_dir=self.settings.rag_store,
                    embedding_model_path=self.settings.embedding_model,
                    device="cpu",
                    workspace=self.settings.workspace,
                    allowed_prefixes=(),
                    source_policy=SourcePolicy(self.settings.workspace),
                )
            return self._rag

    def _memory_adapter(self) -> HindsightMemoryAdapter:
        with self._memory_lock:
            if self._memory is None:
                from ..hindsight.client import HindsightHttpClient
                from ..hindsight.config import HindsightConfig
                from ..hindsight.service import MemoryService

                self._memory_client = HindsightHttpClient(
                    HindsightConfig(base_url=self.settings.hindsight_url)
                )
                source_policy = SourcePolicy(self.settings.workspace)
                self._memory = HindsightMemoryAdapter(
                    MemoryService(self._memory_client),
                    source_policy=source_policy,
                    content_guard=protected_content_guard(source_policy),
                )
            return self._memory

    # ----------------------------------------------------------------- tools

    def tool_definitions(self) -> list[McpTool]:
        return [
            McpTool(
                name="yuzhu_service_status",
                description=(
                    "Read-only health and resource snapshot of the YUZHU system: "
                    "MiniCPM5 runtime, Hindsight, RAG index, GPU and RAM."
                ),
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                handler=self.handle_status,
                timeout_seconds=30.0,
            ),
            McpTool(
                name="yuzhu_rag_search",
                description=(
                    "Search the project Hybrid RAG index (BM25 + FAISS + RRF) and "
                    "return cited code/document evidence with paths and line ranges."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 2000},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=self.handle_rag_search,
                timeout_seconds=90.0,
            ),
            McpTool(
                name="yuzhu_memory_recall",
                description=(
                    "Recall verified long-term project memories from Hindsight. "
                    "Read-only; memories are historical context, not current facts."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1, "maxLength": 2000},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=self.handle_memory_recall,
                timeout_seconds=60.0,
            ),
            McpTool(
                name="yuzhu_agent_task",
                description=(
                    "Run one task through the YUZHU single-agent harness: Hindsight "
                    "recall + Hybrid RAG + MiniCPM5 + local coding tools + checkpoint. "
                    "Returns the completion gate assessment and tool evidence. "
                    "Execution is available only when the server was started with "
                    "--allow-execute; retention only with --allow-retain."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "minLength": 1, "maxLength": 8000},
                        "max_turns": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
                        "max_tokens": {"type": "integer", "minimum": 64, "maximum": 2048, "default": 768},
                        "timeout_seconds": {"type": "integer", "minimum": 30, "maximum": 900, "default": 600},
                        "allow_execution": {"type": "boolean", "default": False},
                        "session_id": {"type": "string", "maxLength": 200},
                    },
                    "required": ["task"],
                    "additionalProperties": False,
                },
                handler=self.handle_agent_task,
                timeout_seconds=float(self.settings.max_task_seconds) + 30.0,
            ),
        ]

    # -------------------------------------------------------------- handlers

    def handle_status(self, args: dict[str, Any], cancel: threading.Event) -> str:
        report = check_services(
            runtime_url=self.settings.runtime_url,
            hindsight_url=self.settings.hindsight_url,
            rag_store=self.settings.rag_store,
        )
        data = report.to_dict()
        data["workspace"] = str(self.settings.workspace)
        data["project_id"] = self.settings.project_id
        data["bridge"] = {
            "server": SERVER_NAME,
            "version": SERVER_VERSION,
            "execution_enabled": self.settings.allow_execute,
            "retention_enabled": self.settings.allow_retain,
        }
        if cancel.is_set():
            raise ToolCancelled("cancelled")
        return _json_text(data)

    def handle_rag_search(self, args: dict[str, Any], cancel: threading.Event) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ToolError("query must not be empty")
        top_k = int(args.get("top_k") or 5)
        if not 1 <= top_k <= 10:
            raise ToolError("top_k must be between 1 and 10")
        outcome = self._rag_adapter().search(
            project_id=self.settings.project_id,
            query=query,
            token_budget=1600,
            final_k=top_k,
        )
        if cancel.is_set():
            raise ToolCancelled("cancelled")
        payload = {
            "query": query,
            "degraded": list(outcome.degraded),
            "index_loaded": outcome.index_loaded,
            "denied_sources": outcome.denied_sources,
            "latency_ms": outcome.latency_ms,
            "results": [
                {
                    "citation": item.citation,
                    "source": item.source,
                    "symbol": item.symbol,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "score": item.score,
                    "rrf_score": item.rrf_score,
                    "snippet": _snippet(item.content),
                }
                for item in outcome.items
            ],
        }
        if not outcome.items and outcome.degraded:
            raise ToolError(_json_text(payload))
        return _json_text(payload)

    def handle_memory_recall(self, args: dict[str, Any], cancel: threading.Event) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ToolError("query must not be empty")
        limit = int(args.get("limit") or 5)
        if not 1 <= limit <= 10:
            raise ToolError("limit must be between 1 and 10")
        outcome = self._memory_adapter().recall(
            project_id=self.settings.project_id,
            query=query,
            max_tokens=1200,
            include_unverified=False,
        )
        if cancel.is_set():
            raise ToolCancelled("cancelled")
        items = outcome.items[:limit]
        payload = {
            "query": query,
            "degraded": list(outcome.degraded),
            "banks_queried": list(outcome.banks_queried),
            "missing_banks": list(outcome.missing_banks),
            "excluded_unverified": outcome.excluded_unverified,
            "latency_ms": outcome.latency_ms,
            "memories": [
                {
                    "memory_id": item.memory_id,
                    "citation": f"hindsight://{item.bank_id}/{item.memory_id}",
                    "text": _snippet(item.text),
                    "category": item.category,
                    "verification": item.verification,
                    "trust": item.trust,
                    "confidence": item.confidence,
                    "occurred_at": item.occurred_at,
                }
                for item in items
            ],
        }
        if not items and outcome.degraded:
            raise ToolError(_json_text(payload))
        return _json_text(payload)

    def handle_agent_task(self, args: dict[str, Any], cancel: threading.Event) -> str:
        task = str(args.get("task") or "").strip()
        if not task:
            raise ToolError("task must not be empty")
        if len(task) > 8000:
            raise ToolError("task is limited to 8000 characters")
        max_turns = int(args.get("max_turns") or 8)
        max_tokens = int(args.get("max_tokens") or 768)
        timeout_seconds = float(args.get("timeout_seconds") or 600)
        requested_execution = bool(args.get("allow_execution", False))
        if requested_execution and not self.settings.allow_execute:
            raise ToolError(
                "execution is disabled for this MCP server; start it with "
                "--allow-execute only inside an approved workspace"
            )
        effective_execution = requested_execution and self.settings.allow_execute
        max_turns = max(1, min(max_turns, self.settings.max_turns_limit))
        max_tokens = max(64, min(max_tokens, 2048))
        timeout_seconds = max(30.0, min(timeout_seconds, self.settings.max_task_seconds))
        session_id = str(args.get("session_id") or f"mcp-{uuid.uuid4().hex[:10]}")

        if not self._agent_lock.acquire(blocking=False):
            raise ToolError("another agent task is already running; one task at a time")
        cancel_token = CancelToken()
        watcher_done = threading.Event()

        def watch() -> None:
            while not watcher_done.is_set():
                if cancel.wait(0.2):
                    cancel_token.cancel("mcp-cancel")
                    return

        watcher = threading.Thread(target=watch, name="mcp-cancel-watch", daemon=True)
        watcher.start()
        started = time.perf_counter()
        runtime = None
        try:
            config = AgentFactoryConfig(
                workspace=self.settings.workspace,
                project_id=self.settings.project_id,
                session_id=session_id,
                task_id=f"mcp-task-{uuid.uuid4().hex[:10]}",
                runtime_url=self.settings.runtime_url,
                hindsight_url=self.settings.hindsight_url,
                rag_store=self.settings.rag_store,
                embedding_model=self.settings.embedding_model,
                checkpoint_db=self.settings.state_dir / "agent_state.db",
                audit_log=self.settings.state_dir / "agent_audit.jsonl",
                allow_execution=effective_execution,
                allow_destructive=False,
                use_memory=True,
                use_rag=True,
                retain_memory=self.settings.allow_retain,
                max_turns=max_turns,
                max_tool_calls=max_turns * 3,
                max_tools_per_turn=3,
                task_timeout_seconds=max(30.0, timeout_seconds - 5.0),
                max_tokens=max_tokens,
                model_timeout_seconds=300.0,
                model_context_budget=8192,
                context_char_limit=24000,
            )
            runtime = build_agent_runtime(config, cancel_token=cancel_token)
            if not runtime.services.get("runtime", {}).get("ok"):
                raise ToolError("MiniCPM5 runtime is not healthy")
            result = runtime.loop.run(task)
            store = runtime.checkpoints
            task_id = runtime.loop.task_id
            record = store.get_task(task_id)
            calls = store.tool_calls(task_id)
            context_meta = (record.get("context") or {}).get("context_metadata", {})
            payload = {
                "status": result.status,
                "state": result.state,
                "final_text": result.final_text[:8000],
                "turns": result.turns,
                "tool_calls": result.tool_calls,
                "duration_seconds": round(time.perf_counter() - started, 2),
                "task_id": task_id,
                "session_id": session_id,
                "completion": result.completion,
                "error": result.error,
                "warnings": result.warnings,
                "memory_refs": record.get("memory_refs", []),
                "rag_refs": record.get("rag_refs", []),
                "context_metadata": {
                    "memory_count": context_meta.get("memory_count"),
                    "rag_count": context_meta.get("rag_count"),
                    "memory_degraded": context_meta.get("memory_degraded"),
                    "rag_degraded": context_meta.get("rag_degraded"),
                    "duplicates_removed": context_meta.get("duplicates_removed"),
                    "conflicts": context_meta.get("conflicts"),
                    "token_estimate": (
                        context_meta.get("budget_plan")
                        if isinstance(context_meta, dict)
                        else None
                    ),
                },
                "tool_evidence": [
                    {
                        "name": call.get("name"),
                        "status": call.get("status"),
                        "result_status": (call.get("result") or {}).get("status"),
                        "exit_code": (call.get("result") or {}).get("exit_code"),
                    }
                    for call in calls
                ],
                "execution_enabled": effective_execution,
            }
            return _json_text(payload)
        except ToolCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - report as tool error
            if cancel_token.cancelled:
                raise ToolCancelled("cancelled") from exc
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            watcher_done.set()
            if runtime is not None:
                try:
                    runtime.close()
                except Exception:  # noqa: BLE001 - shutdown is best effort
                    pass
            self._agent_lock.release()


@dataclass
class _Inflight:
    request_id: Any
    event: threading.Event = field(default_factory=threading.Event)
    responded: bool = False
    cancelled: bool = False
    timed_out: bool = False


class McpServer:
    def __init__(
        self,
        tools: list[McpTool],
        *,
        reader: BinaryIO,
        writer: BinaryIO,
        log: BinaryIO | None = None,
        protocol_version: str = "2024-11-05",
    ) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.reader = reader
        self.writer = writer
        self.log = log or sys.stderr.buffer
        self.protocol_version = protocol_version
        self._write_lock = threading.Lock()
        self._inflight: dict[Any, _Inflight] = {}
        self._inflight_lock = threading.Lock()
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._stopping = threading.Event()

    # ------------------------------------------------------------ transport

    def _log(self, message: str) -> None:
        try:
            self.log.write((message + "\n").encode("utf-8", errors="replace"))
            self.log.flush()
        except Exception:  # noqa: BLE001 - logging must not break the server
            pass

    def _send(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        with self._write_lock:
            if self._stopping.is_set():
                return
            try:
                self.writer.write(data + b"\n")
                self.writer.flush()
            except Exception as exc:  # noqa: BLE001 - client may be gone
                self._log(f"write failed: {type(exc).__name__}")

    def _error(self, request_id: Any, code: int, message: str, data: Any = None) -> None:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        self._send({"jsonrpc": "2.0", "id": request_id, "error": error})

    def _result(self, request_id: Any, result: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _reader_loop(self) -> None:
        try:
            while True:
                line = self.reader.readline()
                if not line:
                    break
                self._queue.put(line)
        finally:
            self._queue.put(None)

    # ------------------------------------------------------------- lifecycle

    def serve(self) -> int:
        reader_thread = threading.Thread(target=self._reader_loop, name="mcp-stdin", daemon=True)
        reader_thread.start()
        self._log(f"{SERVER_NAME} {SERVER_VERSION} ready")
        try:
            while True:
                line = self._queue.get()
                if line is None:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line.decode("utf-8", errors="replace"))
                except ValueError as exc:
                    self._error(None, JSONRPC_PARSE_ERROR, f"parse error: {exc}")
                    continue
                if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                    self._error(None, JSONRPC_INVALID_REQUEST, "invalid JSON-RPC 2.0 message")
                    continue
                self._dispatch(message)
        finally:
            self._stopping.set()
            with self._inflight_lock:
                inflight = list(self._inflight.values())
            for item in inflight:
                item.cancelled = True
                item.event.set()
            # Give workers a moment to unwind before the process exits.
            deadline = time.time() + 3.0
            while time.time() < deadline:
                with self._inflight_lock:
                    if not self._inflight:
                        break
                time.sleep(0.05)
        return 0

    def _dispatch(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") or {}
        is_notification = "id" not in message

        if method == "notifications/initialized":
            return
        if method == "notifications/cancelled":
            if isinstance(params, dict):
                self._cancel_request(params.get("requestId"), str(params.get("reason") or "client"))
            return
        if method == "initialize":
            requested = ""
            if isinstance(params, dict):
                requested = str(params.get("protocolVersion") or "")
            version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else self.protocol_version
            self._result(
                request_id,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            )
            return
        if method == "ping":
            self._result(request_id, {})
            return
        if method == "tools/list":
            self._result(
                request_id,
                {
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.input_schema,
                        }
                        for tool in self.tools.values()
                    ]
                },
            )
            return
        if method == "tools/call":
            if is_notification:
                return
            if not isinstance(params, dict):
                self._error(request_id, JSONRPC_INVALID_PARAMS, "params must be an object")
                return
            name = params.get("name")
            arguments = params.get("arguments") or {}
            tool = self.tools.get(str(name))
            if tool is None:
                self._error(request_id, JSONRPC_INVALID_PARAMS, f"unknown tool: {name}")
                return
            if not isinstance(arguments, dict):
                self._error(request_id, JSONRPC_INVALID_PARAMS, "arguments must be an object")
                return
            self._start_tool(request_id, tool, arguments)
            return
        if is_notification:
            return
        self._error(request_id, JSONRPC_METHOD_NOT_FOUND, f"method not found: {method}")

    # ---------------------------------------------------------------- worker

    def _start_tool(self, request_id: Any, tool: McpTool, arguments: dict[str, Any]) -> None:
        inflight = _Inflight(request_id=request_id)
        with self._inflight_lock:
            self._inflight[request_id] = inflight

        def worker() -> None:
            started = time.perf_counter()
            try:
                text = tool.handler(arguments, inflight.event)
                with self._inflight_lock:
                    if inflight.responded:
                        return
                    inflight.responded = True
                    self._inflight.pop(request_id, None)
                self._result(
                    request_id,
                    {"content": [{"type": "text", "text": text}], "isError": False},
                )
                self._log(
                    f"tool {tool.name} ok in {round(time.perf_counter() - started, 2)}s"
                )
            except ToolCancelled:
                with self._inflight_lock:
                    if inflight.responded:
                        return
                    inflight.responded = True
                    self._inflight.pop(request_id, None)
                if inflight.cancelled:
                    self._error(request_id, JSONRPC_REQUEST_CANCELLED, "request cancelled")
                else:
                    self._error(request_id, JSONRPC_INTERNAL_ERROR, "tool timed out")
                self._log(f"tool {tool.name} cancelled/timed out")
            except ToolError as exc:
                with self._inflight_lock:
                    if inflight.responded:
                        return
                    inflight.responded = True
                    self._inflight.pop(request_id, None)
                self._result(
                    request_id,
                    {"content": [{"type": "text", "text": str(exc)}], "isError": True},
                )
                self._log(f"tool {tool.name} tool-error in {round(time.perf_counter() - started, 2)}s")
            except Exception as exc:  # noqa: BLE001 - never kill the server
                with self._inflight_lock:
                    if inflight.responded:
                        return
                    inflight.responded = True
                    self._inflight.pop(request_id, None)
                self._log(traceback.format_exc(limit=6))
                self._result(
                    request_id,
                    {
                        "content": [
                            {"type": "text", "text": f"{type(exc).__name__}: {exc}"}
                        ],
                        "isError": True,
                    },
                )

        threading.Thread(target=worker, name=f"mcp-tool-{tool.name}", daemon=True).start()
        self._log(f"tool {tool.name} started")

        deadline = time.time() + tool.timeout_seconds

        def watchdog() -> None:
            if inflight.event.wait(max(0.0, deadline - time.time())):
                return
            with self._inflight_lock:
                if inflight.responded:
                    return
                inflight.timed_out = True
            inflight.event.set()

        threading.Thread(target=watchdog, name=f"mcp-watch-{tool.name}", daemon=True).start()

    def _cancel_request(self, request_id: Any, reason: str) -> None:
        with self._inflight_lock:
            inflight = self._inflight.get(request_id)
        if inflight is None:
            return
        inflight.cancelled = True
        inflight.event.set()
        self._log(f"cancel request {request_id}: {reason}")


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YUZHU LAB MCP stdio server")
    parser.add_argument("--workspace", default=str(PROJECT_ROOT))
    parser.add_argument("--project-id", default="qianyuqwen")
    parser.add_argument("--runtime-url", default="http://127.0.0.1:8710")
    parser.add_argument("--hindsight-url", default="http://127.0.0.1:8890")
    parser.add_argument("--rag-store", default=str(DEFAULT_RAG_STORE))
    parser.add_argument("--embedding-model", default=str(DEFAULT_EMBEDDING))
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--allow-execute", action="store_true")
    parser.add_argument("--allow-retain", action="store_true")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--max-task-seconds", type=_positive_float, default=600.0)
    return parser


def _preload_rag_imports() -> None:
    """Import the RAG stack on the main thread.

    Windows + faiss 1.15 can deadlock when `faiss` is first imported from a
    non-main thread while the main thread waits on stdin/queue.  Importing the
    modules here makes later worker-thread use safe without loading the index
    or the embedding model eagerly.
    """
    import importlib

    for name in ("faiss", "bm25s", "bm25s.tokenization", "sentence_transformers"):
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - RAG will degrade later
            print(f"preload {name} failed: {type(exc).__name__}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _preload_rag_imports()
    if os.environ.get("YUZHU_MCP_DEBUG_HANG") == "1":
        # Diagnostic only: dump every thread stack if a tool call wedges.
        import faulthandler

        faulthandler.dump_traceback_later(30, repeat=True)
    settings = McpServerSettings(
        workspace=Path(args.workspace).resolve(),
        project_id=args.project_id,
        runtime_url=args.runtime_url,
        hindsight_url=args.hindsight_url,
        rag_store=Path(args.rag_store),
        embedding_model=Path(args.embedding_model),
        state_dir=Path(args.state_dir),
        allow_execute=bool(args.allow_execute),
        allow_retain=bool(args.allow_retain),
        max_task_seconds=args.max_task_seconds,
    )
    tools = HarnessTools(settings).tool_definitions()
    log_stream: BinaryIO = sys.stderr.buffer
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_stream = _Tee(sys.stderr.buffer, open(log_path, "ab", buffering=0))
    server = McpServer(
        tools,
        reader=sys.stdin.buffer,
        writer=sys.stdout.buffer,
        log=log_stream,
    )
    return server.serve()


if __name__ == "__main__":
    raise SystemExit(main())