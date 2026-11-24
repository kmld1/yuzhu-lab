"""Trust policy: retrieved text is data, never instructions.

Everything returned by Hindsight or the RAG index is untrusted.  The builder
neutralizes instruction-like spans, keeps the original text for inspection, and
never promotes retrieved text into the system instruction section.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .context_models import ContextWarning, MemoryContextItem, RagEvidenceItem

_INSTRUCTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "OVERRIDE_INSTRUCTIONS",
        re.compile(
            r"(?i)\b(?:ignore|disregard|forget|override)\b.{0,40}"
            r"\b(?:previous|prior|above|system)\b.{0,24}"
            r"\b(?:instruction|prompt|direction)s?\b"
        ),
    ),
    (
        "OVERRIDE_INSTRUCTIONS_ZH",
        re.compile(r"(?:忽略|无视|忘记|覆盖|绕过).{0,20}(?:之前|以上|前面|系统).{0,12}(?:指令|提示|要求|设定)"),
    ),
    (
        "ROLE_IMPERSONATION",
        re.compile(r"(?i)\b(?:system|assistant|developer|tool)\s*[:：]"),
    ),
    (
        "CHAT_TEMPLATE_TOKEN",
        re.compile(r"<\|(?:im_start|im_end|system|user|assistant)\|>|\[/?INST\]|<<SYS>>"),
    ),
    (
        "TOOL_ESCALATION",
        re.compile(
            r"(?i)\b(?:run|execute|invoke)\s+(?:the\s+)?(?:shell|command|powershell|bash)\b"
            r"|执行.{0,8}(?:命令|脚本)"
            r"|\brm\s+-rf\b"
            r"|\bcurl\b[^\n|]{0,200}\|\s*(?:sh|bash)\b"
            r"|\bpowershell\s+(?:-|/)(?:enc|encodedcommand)\b"
        ),
    ),
    (
        "PERSONA_REPLACEMENT",
        re.compile(
            r"(?i)\bfrom now on,?\s+you are\b|\byou are now\b|从现在开始[,，]?\s*你是"
        ),
    ),
)

_TAG_BREAKOUT = re.compile(
    r"</?(?:yuzhu_context|untrusted_evidence|trusted_memory|memory_item|memory_context|"
    r"rag_evidence|warnings?|warning|system|instructions?)>",
    re.IGNORECASE,
)


def sanitize_untrusted(text: str) -> tuple[str, tuple[str, ...]]:
    flags: list[str] = []
    sanitized = text
    for code, pattern in _INSTRUCTION_PATTERNS:
        if pattern.search(sanitized):
            flags.append(code)
            sanitized = pattern.sub("[NEUTRALIZED_INSTRUCTION]", sanitized)
    if _TAG_BREAKOUT.search(sanitized):
        flags.append("TAG_BREAKOUT")
        sanitized = _TAG_BREAKOUT.sub("[NEUTRALIZED_TAG]", sanitized)
    return sanitized, tuple(flags)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _within(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def detect_conflicts(
    memories: list[MemoryContextItem],
    evidence: list[RagEvidenceItem],
    workspace: Path | None,
) -> list[ContextWarning]:
    warnings: list[ContextWarning] = []

    for memory in memories:
        if memory.conflict:
            warnings.append(
                ContextWarning(
                    "MEMORY_CONFLICT",
                    f"memory {memory.memory_id} shares a memory_key with a different "
                    "verified record; both are kept for review",
                    severity="error",
                )
            )
        refs = [
            item.strip()
            for item in re.split(r"[|,;]", memory.metadata.get("code_refs", ""))
            if item.strip()
        ]
        expected_hash = memory.metadata.get("code_hash", "").strip()
        if not refs:
            continue
        if workspace is None:
            warnings.append(
                ContextWarning(
                    "MEMORY_CODE_UNCHECKED",
                    f"memory {memory.memory_id} references code but no workspace was supplied",
                )
            )
            continue
        for ref in refs:
            candidate = workspace / ref
            if not _within(workspace, candidate) or not candidate.is_file():
                warnings.append(
                    ContextWarning(
                        "MEMORY_CODE_REF_MISSING",
                        f"memory {memory.memory_id} references missing file {ref}",
                    )
                )
                continue
            actual = _sha256_file(candidate)
            if expected_hash and actual != expected_hash:
                warnings.append(
                    ContextWarning(
                        "MEMORY_CODE_CONFLICT",
                        f"memory {memory.memory_id} describes {ref} at hash "
                        f"{expected_hash[:12]}, current file is {actual[:12]}; "
                        "current code wins",
                        severity="error",
                    )
                )

    memory_texts = {item.text.strip() for item in memories}
    for item in evidence:
        if item.content.strip() and item.content.strip() in memory_texts:
            warnings.append(
                ContextWarning(
                    "MEMORY_RAG_DUPLICATE",
                    f"memory and RAG evidence contain the same text; evidence {item.citation} "
                    "remains authoritative for current code",
                )
            )
    return warnings


def conflict_count(warnings: list[ContextWarning]) -> int:
    return sum(1 for item in warnings if item.code in {"MEMORY_CONFLICT", "MEMORY_CODE_CONFLICT"})