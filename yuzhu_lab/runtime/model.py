"""Load the frozen MiniCPM5-2B base model with the P04.0 LoRA adapter, read-only."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .settings import RuntimeSettings


class ModelLoadError(RuntimeError):
    """Raised when the model or adapter does not match the frozen assets."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_adapter_integrity(adapter_dir: Path, expected_sha256: str) -> dict[str, Any]:
    weights = adapter_dir / "adapter_model.safetensors"
    config = adapter_dir / "adapter_config.json"
    for path in (weights, config):
        if not path.is_file():
            raise ModelLoadError(f"adapter file is missing: {path}")
    actual = sha256_file(weights)
    if actual.lower() != expected_sha256.lower():
        raise ModelLoadError(
            f"adapter SHA256 mismatch: {actual} != expected {expected_sha256}"
        )
    return {
        "adapter_dir": str(adapter_dir),
        "adapter_weights_sha256": actual,
        "adapter_config_sha256": sha256_file(config),
        "adapter_size_bytes": weights.stat().st_size,
    }


def verify_model_files(model_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Full 11-file SHA256 verification. Optional because it costs several seconds."""
    if not manifest_path.is_file():
        raise ModelLoadError(f"model manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified = 0
    for entry in manifest.get("files", []):
        filename = entry.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ModelLoadError("model manifest contains an invalid filename entry")
        path = model_dir / filename
        if not path.is_file():
            raise ModelLoadError(f"model file is missing: {path}")
        if path.stat().st_size != entry.get("size_bytes"):
            raise ModelLoadError(f"model file size mismatch: {filename}")
        if sha256_file(path).lower() != str(entry.get("sha256", "")).lower():
            raise ModelLoadError(f"model file SHA256 mismatch: {filename}")
        verified += 1
    return {
        "files_verified": verified,
        "model_id": manifest.get("model_id"),
        "revision": manifest.get("revision"),
    }


@dataclass
class LoadedModel:
    tokenizer: Any
    model: Any
    metadata: dict[str, Any]


def load_model(settings: RuntimeSettings) -> LoadedModel:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not settings.model_dir.is_dir():
        raise ModelLoadError(f"base model directory is missing: {settings.model_dir}")
    if not settings.adapter_dir.is_dir():
        raise ModelLoadError(f"adapter directory is missing: {settings.adapter_dir}")

    started = time.perf_counter()
    metadata: dict[str, Any] = {
        "model_dir": str(settings.model_dir),
        "adapter_dir": str(settings.adapter_dir),
        "device": settings.device,
        "enable_thinking": settings.enable_thinking,
    }
    if settings.verify_model_files:
        metadata["model_integrity"] = verify_model_files(
            settings.model_dir, settings.model_manifest
        )
    if settings.verify_adapter:
        metadata["adapter_integrity"] = verify_adapter_integrity(
            settings.adapter_dir, settings.adapter_sha256
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(settings.model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(settings.model_dir),
        dtype=torch.bfloat16,
        device_map={"": settings.device},
        low_cpu_mem_usage=True,
        attn_implementation=settings.attn_implementation,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )
    model = PeftModel.from_pretrained(model, str(settings.adapter_dir), is_trainable=False)
    model.eval()
    model.config.use_cache = True

    metadata["model_class"] = type(model).__name__
    metadata["load_seconds"] = round(time.perf_counter() - started, 3)
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
        metadata["peak_allocated_gib"] = round(
            torch.cuda.max_memory_allocated() / (1024**3), 4
        )
    return LoadedModel(tokenizer=tokenizer, model=model, metadata=metadata)