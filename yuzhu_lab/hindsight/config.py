"""Configuration for the local Hindsight deployment.

The service is the official ``hindsight-api-slim`` 1.2.1 running against an
embedded pg0 PostgreSQL instance.  Bank names are stable project identifiers,
not per-session values, so memory survives process restarts.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GLOBAL_BANK = "yuzhu-global"
PROJECT_BANK_PREFIX = "yuzhu-project-"
EXPERIMENTS_BANK = "yuzhu-experiments"

DEFAULT_BASE_URL = "http://127.0.0.1:8890"
DEFAULT_AUDIT_LOG = PROJECT_ROOT / "logs" / "hindsight" / "memory_audit.jsonl"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _slug(value: str, *, max_length: int = 48) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    if not value:
        raise ValueError("project_id must contain at least one alphanumeric character")
    return value[:max_length].rstrip("-") or "project"


def bank_id_for(project_id: str, scope: str = "project") -> str:
    """Return the bank that owns ``project_id`` for the requested scope.

    A bank per project is the isolation boundary.  Global preferences and
    explicitly experimental hypotheses never share a project bank.
    """

    if scope == "global":
        return GLOBAL_BANK
    if scope == "experiment":
        return EXPERIMENTS_BANK
    if scope != "project":
        raise ValueError(f"unknown memory scope: {scope!r}")
    return PROJECT_BANK_PREFIX + _slug(project_id)


def project_tag(project_id: str) -> str:
    return "project:" + _slug(project_id)


@dataclass(frozen=True)
class HindsightConfig:
    base_url: str = field(
        default_factory=lambda: os.environ.get("YUZHU_HINDSIGHT_URL", DEFAULT_BASE_URL)
    )
    connect_timeout_seconds: float = 5.0
    request_timeout_seconds: float = field(
        default_factory=lambda: _env_float("YUZHU_HINDSIGHT_TIMEOUT", 300.0)
    )
    recall_timeout_seconds: float = field(
        default_factory=lambda: _env_float("YUZHU_HINDSIGHT_RECALL_TIMEOUT", 30.0)
    )
    reflect_timeout_seconds: float = field(
        default_factory=lambda: _env_float("YUZHU_HINDSIGHT_REFLECT_TIMEOUT", 180.0)
    )
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("YUZHU_HINDSIGHT_MAX_RETRIES", "2"))
    )
    audit_log_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("YUZHU_HINDSIGHT_AUDIT_LOG", str(DEFAULT_AUDIT_LOG))
        )
    )
    max_memory_chars: int = 4000
    min_memory_chars: int = 8