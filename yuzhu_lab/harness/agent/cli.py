"""Command line entry point for the H2 agent harness.

    python -m yuzhu_lab.harness.agent.cli check
    python -m yuzhu_lab.harness.agent.cli run --task "..." --workspace ...
    python -m yuzhu_lab.harness.agent.cli resume --task-id ...
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

from .checkpoint import CheckpointStore
from .errors import CheckpointError
from .factory import (
    DEFAULT_AUDIT_LOG,
    DEFAULT_CHECKPOINT_DB,
    PROJECT_ROOT,
    AgentFactoryConfig,
    build_agent_runtime,
)
from .loop import CancelToken
from .permissions import PermissionPolicy
from .resources import check_services
from .tools import build_default_registry, default_tool_context


def _new_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"


def _base_config(args: argparse.Namespace) -> AgentFactoryConfig:
    return AgentFactoryConfig(
        workspace=Path(args.workspace).resolve() if args.workspace else PROJECT_ROOT,
        project_id=args.project_id,
        session_id=args.session_id or _new_id("session"),
        task_id=args.task_id,
        runtime_url=args.runtime_url,
        hindsight_url=args.hindsight_url,
        checkpoint_db=Path(args.checkpoint_db),
        audit_log=Path(args.audit_log),
        allow_execution=args.allow_execute,
        allow_destructive=args.approve_destructive,
        use_memory=not args.no_memory,
        use_rag=not args.no_rag,
        retain_memory=not args.no_memory_write,
        max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls,
        max_tools_per_turn=args.max_tools_per_turn,
        task_timeout_seconds=args.timeout,
        max_tokens=args.max_tokens,
        model_context_budget=args.model_context_budget,
        allowed_tools=(
            tuple(tool.strip() for tool in args.tools.split(",") if tool.strip())
            if args.tools
            else None
        ),
        require_test_evidence_for_code_change=not args.allow_unverified_code,
    )


def _install_signal_handler(token: CancelToken) -> None:
    def handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        token.cancel("SIGINT")

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:
        pass


def _dump(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _transcript(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _add_common(
    parser: argparse.ArgumentParser, *, task_id_required: bool = False
) -> None:
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--project-id", default="qianyuqwen")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--task-id", default=None, required=task_id_required)
    parser.add_argument("--runtime-url", default="http://127.0.0.1:8710")
    parser.add_argument("--hindsight-url", default="http://127.0.0.1:8890")
    parser.add_argument("--checkpoint-db", default=str(DEFAULT_CHECKPOINT_DB))
    parser.add_argument("--audit-log", default=str(DEFAULT_AUDIT_LOG))
    parser.add_argument("--allow-execute", action="store_true")
    parser.add_argument("--approve-destructive", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--no-memory-write", action="store_true")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-tool-calls", type=int, default=24)
    parser.add_argument("--max-tools-per-turn", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--model-context-budget", type=int, default=8192)
    parser.add_argument(
        "--tools",
        default=None,
        help="comma-separated tool allow-list (default: all registered tools)",
    )
    parser.add_argument("--allow-unverified-code", action="store_true")
    parser.add_argument("--transcript", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YUZHU LAB H2 agent harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a new agent task")
    _add_common(run)
    run.add_argument("--task", required=True)

    resume = subparsers.add_parser("resume", help="resume an interrupted task")
    _add_common(resume, task_id_required=True)

    check = subparsers.add_parser("check", help="report service and resource health")
    _add_common(check)

    list_cmd = subparsers.add_parser("list", help="list recent tasks")
    list_cmd.add_argument("--checkpoint-db", default=str(DEFAULT_CHECKPOINT_DB))
    list_cmd.add_argument("--session-id", default=None)
    list_cmd.add_argument("--limit", type=int, default=20)

    show = subparsers.add_parser("show", help="show a task, its events and tool calls")
    show.add_argument("--checkpoint-db", default=str(DEFAULT_CHECKPOINT_DB))
    show.add_argument("--task-id", required=True)

    tools = subparsers.add_parser("tools", help="list registered tools")
    tools.add_argument("--workspace", default=str(PROJECT_ROOT))
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)

    if args.command == "check":
        report = check_services(
            runtime_url=args.runtime_url,
            hindsight_url=args.hindsight_url,
            rag_store=PROJECT_ROOT / "rag" / "project_index",
        )
        _dump(report.to_dict())
        return 0

    if args.command == "list":
        try:
            store = CheckpointStore(Path(args.checkpoint_db))
        except CheckpointError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
        try:
            _dump(store.list_tasks(args.session_id, args.limit))
        finally:
            store.close()
        return 0

    if args.command == "show":
        try:
            store = CheckpointStore(Path(args.checkpoint_db))
        except CheckpointError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
        try:
            task = store.get_task(args.task_id)
            task["messages"] = f"{len(task['messages'])} messages"
            task.pop("messages_json", None)
            task.pop("context_json", None)
            task["context"] = {
                key: ("<omitted>" if key == "system_prompt" else value)
                for key, value in task["context"].items()
            }
            _dump(
                {
                    "task": task,
                    "tool_calls": store.tool_calls(args.task_id),
                    "events": store.events(args.task_id),
                }
            )
        finally:
            store.close()
        return 0

    if args.command == "tools":
        workspace = Path(args.workspace)
        policy = PermissionPolicy.create(workspace, allow_execution=False)
        registry = build_default_registry(
            policy, context=default_tool_context(policy, interpreter=sys.executable)
        )
        _dump(
            [
                {
                    "name": definition.name,
                    "permission": definition.permission_level.value,
                    "idempotent": definition.idempotent,
                    "timeout_seconds": definition.timeout_seconds,
                    "required": definition.input_schema.get("required", []),
                }
                for definition in registry.definitions()
            ]
        )
        return 0

    if args.command == "resume":
        try:
            store = CheckpointStore(Path(args.checkpoint_db))
        except CheckpointError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 2
        try:
            record = store.get_task(args.task_id)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"error": f"cannot load task: {exc}"}), file=sys.stderr)
            return 2
        finally:
            store.close()
        args.workspace = record["workspace"]
        args.project_id = record["project_id"]
        args.session_id = record["session_id"]

    config = _base_config(args)
    if args.command == "resume":
        config.task_id = args.task_id

    token = CancelToken()
    _install_signal_handler(token)
    try:
        runtime = build_agent_runtime(config, cancel_token=token)
    except CheckpointError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    try:
        if not runtime.services.get("runtime", {}).get("ok"):
            print(
                json.dumps(
                    {
                        "error": "MiniCPM5 runtime is not healthy; start it with "
                        "scripts/start_runtime.ps1",
                        "services": runtime.services,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 2
        if args.command == "run":
            result = runtime.loop.run(args.task)
        else:
            result = runtime.loop.run("", resume=True)
        payload = {
            "result": result.to_dict(),
            "services": runtime.services,
            "task": {
                "task_id": config.task_id or runtime.loop.task_id,
                "project_id": config.project_id,
                "session_id": config.session_id,
                "workspace": str(config.workspace),
            },
        }
        _dump(payload)
        if args.transcript:
            store = runtime.checkpoints
            _transcript(
                Path(args.transcript),
                {
                    **payload,
                    "checkpoint": store.get_task(runtime.loop.task_id),
                    "tool_calls": store.tool_calls(runtime.loop.task_id),
                    "events": store.events(runtime.loop.task_id),
                },
            )
        return 0 if result.status in {"COMPLETED", "CANCELLED"} else 1
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())