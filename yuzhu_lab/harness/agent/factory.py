"""Factory that wires the H2 agent from existing H1 components."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...source_policy import SourcePolicy, protected_content_guard
from ..context_builder import ContextBuilder, ContextBuilderConfig
from ...hindsight.client import HindsightHttpClient
from ...hindsight.config import HindsightConfig
from ...hindsight.service import MemoryService
from ..memory_adapter import HindsightMemoryAdapter, NullMemoryAdapter
from ..rag_adapter import NullRagAdapter, RagAdapter
from .audit import AgentAuditLog
from .checkpoint import CheckpointStore
from .loop import AgentLoop, AgentLoopConfig
from .memory_writer import DEFAULT_PENDING_PATH, AgentMemoryWriter
from .model_client import ModelPort, RuntimeModelClient
from .permissions import PermissionPolicy
from .registry import ToolRegistry
from .resources import check_services
from .tools import build_default_registry, default_tool_context

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAG_STORE = PROJECT_ROOT / "rag" / "project_index"
DEFAULT_EMBEDDING = PROJECT_ROOT / "models" / "embedding" / "Qwen3-Embedding-0.6B"
DEFAULT_CHECKPOINT_DB = PROJECT_ROOT / "data" / "h2_agent" / "agent_state.db"
DEFAULT_AUDIT_LOG = PROJECT_ROOT / "logs" / "h2_agent" / "agent_audit.jsonl"


@dataclass
class AgentFactoryConfig:
    workspace: Path = field(default_factory=lambda: PROJECT_ROOT)
    project_id: str = "qianyuqwen"
    session_id: str = "session-local"
    task_id: str | None = None
    runtime_url: str = "http://127.0.0.1:8710"
    hindsight_url: str = "http://127.0.0.1:8890"
    rag_store: Path = DEFAULT_RAG_STORE
    embedding_model: Path = DEFAULT_EMBEDDING
    checkpoint_db: Path = DEFAULT_CHECKPOINT_DB
    audit_log: Path = DEFAULT_AUDIT_LOG
    allow_execution: bool = False
    allow_destructive: bool = False
    use_memory: bool = True
    use_rag: bool = True
    retain_memory: bool = True
    max_turns: int = 12
    max_tool_calls: int = 24
    max_tools_per_turn: int = 4
    task_timeout_seconds: float = 900.0
    max_tokens: int = 1024
    model_timeout_seconds: float = 300.0
    model_context_budget: int = 8192
    context_char_limit: int = 60000
    allowed_tools: tuple[str, ...] | None = None
    interpreter: Path | str | None = None
    rag_min_ram_gib: float = 1.5
    rag_allowed_prefixes: tuple[str, ...] = ()
    require_test_evidence_for_code_change: bool = True
    acceptance_script: Path | None = None
    acceptance_sha256: str | None = None
    acceptance_expected_checks: int | None = None
    acceptance_timeout_seconds: float = 300.0
    require_external_acceptance: bool = False
    retain_pending_path: Path = DEFAULT_PENDING_PATH


@dataclass
class AgentRuntime:
    loop: AgentLoop
    registry: ToolRegistry
    checkpoints: CheckpointStore
    audit: AgentAuditLog
    services: dict[str, Any]
    _closers: list[Any] = field(default_factory=list)

    def close(self) -> None:
        for closer in self._closers:
            try:
                closer()
            except Exception:  # noqa: BLE001 - shutdown is best effort
                pass
        self.checkpoints.close()


def build_agent_runtime(
    config: AgentFactoryConfig,
    *,
    model: ModelPort | None = None,
    context_builder: ContextBuilder | None = None,
    memory_service: MemoryService | None = None,
    cancel_token: Any = None,
) -> AgentRuntime:
    closers: list[Any] = []
    services = check_services(
        runtime_url=config.runtime_url,
        hindsight_url=config.hindsight_url,
        rag_store=config.rag_store,
        min_ram_gib=config.rag_min_ram_gib,
    )

    if model is None:
        client = RuntimeModelClient(
            config.runtime_url, timeout_seconds=config.model_timeout_seconds
        )
        closers.append(client.close)
        model = client

    hindsight_client = None
    if memory_service is None and config.use_memory and services.memory_enabled:
        hindsight_client = HindsightHttpClient(HindsightConfig(base_url=config.hindsight_url))
        closers.append(hindsight_client.close)
        memory_service = MemoryService(hindsight_client)

    source_policy = SourcePolicy(config.workspace)
    content_guard = protected_content_guard(source_policy)
    memory_port = (
        HindsightMemoryAdapter(
            memory_service,
            source_policy=source_policy,
            content_guard=content_guard,
        )
        if memory_service is not None and config.use_memory
        else NullMemoryAdapter()
    )

    rag_port: Any = NullRagAdapter()
    if config.use_rag and services.rag_enabled and services.rag_index_present:
        rag_port = RagAdapter(
            store_dir=config.rag_store,
            embedding_model_path=config.embedding_model,
            device="cpu",
            workspace=config.workspace,
            allowed_prefixes=config.rag_allowed_prefixes,
            source_policy=source_policy,
        )

    if context_builder is None:
        context_builder = ContextBuilder(
            memory=memory_port,
            rag=rag_port,
            config=ContextBuilderConfig(
                workspace=config.workspace,
                default_budget=config.model_context_budget,
                token_counter=getattr(model, "count_text_tokens", None),
                source_policy=source_policy,
                memory_content_guard=content_guard,
            ),
        )

    policy = PermissionPolicy.create(
        config.workspace,
        allow_execution=config.allow_execution,
        allow_destructive=config.allow_destructive,
    )
    audit = AgentAuditLog(config.audit_log)
    tool_context = default_tool_context(
        policy,
        interpreter=config.interpreter or sys.executable,
    )
    registry = build_default_registry(policy, context=tool_context, audit=audit)
    if config.allowed_tools:
        allowed = set(config.allowed_tools)
        available = {definition.name for definition in registry.definitions()}
        unknown = sorted(allowed - available)
        if unknown:
            raise ValueError(f"unknown tools requested: {', '.join(unknown)}")
        filtered = ToolRegistry(policy, context=tool_context, audit=audit)
        for definition in registry.definitions():
            if definition.name in allowed:
                filtered.register(definition)
        registry = filtered
    checkpoints = CheckpointStore(config.checkpoint_db)

    import uuid

    task_id = config.task_id or f"task-{uuid.uuid4().hex[:12]}"
    loop_config = AgentLoopConfig(
        project_id=config.project_id,
        session_id=config.session_id,
        workspace=config.workspace,
        checkpoint_db=config.checkpoint_db,
        audit_log=config.audit_log,
        task_id=task_id,
        max_turns=config.max_turns,
        max_tool_calls=config.max_tool_calls,
        max_tools_per_turn=config.max_tools_per_turn,
        task_timeout_seconds=config.task_timeout_seconds,
        max_tokens=config.max_tokens,
        model_context_budget=config.model_context_budget,
        context_char_limit=config.context_char_limit,
        allow_execution=config.allow_execution,
        allow_destructive=config.allow_destructive,
        retain_memory=config.retain_memory,
        require_test_evidence_for_code_change=config.require_test_evidence_for_code_change,
        acceptance_script=config.acceptance_script,
        acceptance_sha256=config.acceptance_sha256,
        acceptance_expected_checks=config.acceptance_expected_checks,
        acceptance_timeout_seconds=config.acceptance_timeout_seconds,
        require_external_acceptance=config.require_external_acceptance,
    )
    writer = AgentMemoryWriter(
        memory_service,
        enabled=config.retain_memory,
        pending_path=config.retain_pending_path,
        source_policy=source_policy,
        content_guard=content_guard,
    )
    loop = AgentLoop(
        config=loop_config,
        model=model,
        registry=registry,
        checkpoints=checkpoints,
        audit=audit,
        context_builder=context_builder,
        memory_writer=writer,
        cancel_token=cancel_token,
    )
    services_dict = services.to_dict()
    services_dict["memory_adapter"] = type(memory_port).__name__
    services_dict["rag_adapter"] = type(rag_port).__name__
    services_dict["interpreter"] = str(config.interpreter or sys.executable)
    if not config.use_memory:
        services_dict["notes"].append("memory disabled by configuration")
    if not config.use_rag:
        services_dict["notes"].append("rag disabled by configuration")
    return AgentRuntime(
        loop=loop,
        registry=registry,
        checkpoints=checkpoints,
        audit=audit,
        services=services_dict,
        _closers=closers,
    )