"""Runtime paths and limits. All values can be overridden by environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "base" / "MiniCPM5-2B-BF16"
DEFAULT_ADAPTER_DIR = (
    PROJECT_ROOT
    / "data"
    / "yuzhu_python_coding_v1"
    / "experiments"
    / "P04.0"
    / "p04_0_minicpm5_python_sft_20261112_213322"
    / "adapter"
)
DEFAULT_MODEL_MANIFEST = PROJECT_ROOT / "eval" / "model_assets" / "minicpm5_bf16_sha256.json"

# Frozen by P04.0 evidence; a mismatch must fail closed instead of silently serving
# a different adapter.
P04_0_ADAPTER_SHA256 = "3693a0b903746a1f9a5a497d5e623209c94e6cf298977b372c1f87c4330574fc"


def _path_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class RuntimeSettings:
    model_dir: Path = field(default_factory=lambda: _path_env("YUZHU_MODEL_DIR", DEFAULT_MODEL_DIR))
    adapter_dir: Path = field(
        default_factory=lambda: _path_env("YUZHU_ADAPTER_DIR", DEFAULT_ADAPTER_DIR)
    )
    model_manifest: Path = field(
        default_factory=lambda: _path_env("YUZHU_MODEL_MANIFEST", DEFAULT_MODEL_MANIFEST)
    )
    adapter_sha256: str = field(
        default_factory=lambda: os.environ.get("YUZHU_ADAPTER_SHA256", P04_0_ADAPTER_SHA256)
    )
    model_name: str = field(
        default_factory=lambda: os.environ.get("YUZHU_MODEL_NAME", "yuzhu-minicpm5-2b-p04.0")
    )
    device: str = field(default_factory=lambda: os.environ.get("YUZHU_DEVICE", "cuda:0"))
    attn_implementation: str = "sdpa"
    enable_thinking: bool = False
    max_new_tokens_limit: int = field(
        default_factory=lambda: int(os.environ.get("YUZHU_MAX_NEW_TOKENS", "2048"))
    )
    # The model's usable context window.  Distinct from the output budget: a
    # request may ask for 32000 output tokens, but the model can only generate
    # max_new_tokens_limit, and prompt + output must fit this window.
    model_context_limit: int = field(
        default_factory=lambda: int(os.environ.get("YUZHU_MODEL_CONTEXT_LIMIT", "131072"))
    )
    # "clamp": generate the supported maximum and advertise the clamp via
    # headers.  "reject": return HTTP 400 when the request exceeds the limit.
    max_tokens_policy: str = field(
        default_factory=lambda: os.environ.get("YUZHU_MAX_TOKENS_POLICY", "clamp").strip().lower()
    )
    default_max_new_tokens: int = field(
        default_factory=lambda: int(os.environ.get("YUZHU_DEFAULT_MAX_NEW_TOKENS", "512"))
    )
    # H5.3 resource guard: a 4.5k-token prompt peaked near 7.9 GiB on the 8 GiB
    # device.  Warn early and refuse prompts that cannot be served safely.
    prompt_warning_tokens: int = field(
        default_factory=lambda: int(os.environ.get("YUZHU_PROMPT_WARNING_TOKENS", "3500"))
    )
    prompt_reject_tokens: int = field(
        default_factory=lambda: int(os.environ.get("YUZHU_PROMPT_REJECT_TOKENS", "6144"))
    )
    min_free_gib: float = field(
        default_factory=lambda: float(os.environ.get("YUZHU_MIN_FREE_GIB", "0.35"))
    )
    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("YUZHU_REQUEST_TIMEOUT", "600"))
    )
    host: str = field(default_factory=lambda: os.environ.get("YUZHU_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("YUZHU_PORT", "8710")))
    api_key: str | None = field(default_factory=lambda: os.environ.get("YUZHU_API_KEY"))
    verify_model_files: bool = field(
        default_factory=lambda: os.environ.get("YUZHU_VERIFY_MODEL_FILES", "0") == "1"
    )
    verify_adapter: bool = field(
        default_factory=lambda: os.environ.get("YUZHU_VERIFY_ADAPTER", "1") == "1"
    )

    def __post_init__(self) -> None:
        if self.max_new_tokens_limit < 1:
            raise ValueError("max_new_tokens_limit must be >= 1")
        if self.model_context_limit < self.max_new_tokens_limit:
            raise ValueError("model_context_limit must be >= max_new_tokens_limit")
        if self.max_tokens_policy not in {"clamp", "reject"}:
            raise ValueError("max_tokens_policy must be 'clamp' or 'reject'")
        if self.prompt_warning_tokens < 1:
            raise ValueError("prompt_warning_tokens must be >= 1")
        if self.prompt_reject_tokens < self.prompt_warning_tokens:
            raise ValueError(
                "prompt_reject_tokens must be >= prompt_warning_tokens"
            )
        if self.min_free_gib < 0:
            raise ValueError("min_free_gib must be >= 0")