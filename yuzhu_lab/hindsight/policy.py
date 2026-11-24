"""Memory admission policy.

Hindsight extracts facts from whatever it is given.  That makes the write path
the security and quality boundary: secrets, personal data, raw transcripts and
unverified model guesses must be rejected before an LLM ever sees them.
"""

from __future__ import annotations

import re

from .config import HindsightConfig, _slug, bank_id_for
from .models import AdmissionDecision, MemoryRecord

ALLOWED_CATEGORIES = frozenset(
    {
        "user_preference",
        "architecture_decision",
        "verified_fix",
        "engineering_experience",
        "phase_status",
        "open_item",
        "failure_cause",
        "session_handoff",
        "hypothesis",
    }
)

GLOBAL_CATEGORIES = frozenset(
    {"user_preference", "engineering_experience", "architecture_decision"}
)

EVIDENCE_REQUIRED_CATEGORIES = frozenset(
    {"architecture_decision", "verified_fix", "failure_cause"}
)

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "API_KEY",
        re.compile(
            r"\b(?:sk|pk|rk|ghp|gho|ghu|ghs|github_pat|xox[baprs])[-_][A-Za-z0-9_\-]{12,}\b"
        ),
    ),
    ("AWS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("BEARER_TOKEN", re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}", re.IGNORECASE)),
    (
        "PASSWORD",
        re.compile(r"(?:password|passwd|pwd|密码|口令)\s*[:=：]\s*\S{4,}", re.IGNORECASE),
    ),
    (
        "TOKEN_ASSIGNMENT",
        re.compile(
            r"(?:api[_\- ]?key|access[_\- ]?token|refresh[_\- ]?token|secret|令牌|密钥)"
            r"\s*[:=：]\s*\S{6,}",
            re.IGNORECASE,
        ),
    ),
)

PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE_CN", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("ID_CN", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
)

RAW_TRANSCRIPT = re.compile(
    r"(?:^|\n)\s*(?:user|assistant|system|human|ai)\s*[:：]", re.IGNORECASE
)

FENCED_CODE = re.compile(r"```(?:[A-Za-z0-9_+-]+)?\s*\n(.*?)```", re.DOTALL)


def _normalize(content: str) -> str:
    return re.sub(r"\s+", " ", content).strip()


def _metadata_value(value: object) -> str:
    text = str(value).strip()
    return text[:500]


class MemoryAdmissionPolicy:
    """Validate and normalize a memory before any Hindsight write."""

    def __init__(self, config: HindsightConfig | None = None) -> None:
        self.config = config or HindsightConfig()

    def evaluate(self, record: MemoryRecord) -> AdmissionDecision:
        content = _normalize(record.content)
        bank_id = bank_id_for(record.project_id, record.scope)

        def refuse(code: str, reason: str) -> AdmissionDecision:
            return AdmissionDecision(
                allowed=False,
                code=code,
                reason=reason,
                bank_id=bank_id,
                normalized_content=content,
                tags=(),
                metadata={},
            )

        if len(content) < self.config.min_memory_chars:
            return refuse("TOO_SHORT", "memory content is too short to be durable")
        if len(content) > self.config.max_memory_chars:
            return refuse(
                "TOO_LONG",
                "memory content exceeds the durable-memory limit; store a conclusion, not a transcript",
            )
        if record.category not in ALLOWED_CATEGORIES:
            return refuse("CATEGORY_NOT_ALLOWED", f"category not allowed: {record.category!r}")
        if record.scope not in {"project", "global", "experiment"}:
            return refuse("SCOPE_NOT_ALLOWED", f"scope not allowed: {record.scope!r}")
        if record.verification not in {"verified", "unverified"}:
            return refuse(
                "VERIFICATION_INVALID",
                "verification must be 'verified' or 'unverified'",
            )

        if record.scope == "global" and record.category not in GLOBAL_CATEGORIES:
            return refuse(
                "GLOBAL_CATEGORY_NOT_ALLOWED",
                "global bank only accepts preferences and cross-project engineering knowledge",
            )
        if record.category == "hypothesis" and record.scope != "experiment":
            return refuse(
                "HYPOTHESIS_REQUIRES_EXPERIMENT",
                "hypotheses belong in yuzhu-experiments and must not enter project/global memory",
            )
        if record.scope == "experiment" and record.category != "hypothesis":
            return refuse(
                "EXPERIMENT_CATEGORY_NOT_ALLOWED",
                "the experiments bank only accepts hypothesis records",
            )
        if record.verification != "verified" and record.scope != "experiment":
            return refuse(
                "UNVERIFIED_REQUIRES_EXPERIMENT",
                "unverified guesses must not enter project/global memory",
            )
        if record.verification == "verified" and not record.source.strip():
            return refuse("SOURCE_REQUIRED", "verified memory requires a source")
        if (
            record.verification == "verified"
            and record.category in EVIDENCE_REQUIRED_CATEGORIES
            and not any(item.strip() for item in record.evidence)
        ):
            return refuse(
                "EVIDENCE_REQUIRED",
                f"{record.category} requires at least one evidence reference",
            )

        for code, pattern in SECRET_PATTERNS:
            if pattern.search(content):
                return refuse(
                    "SECRET_DETECTED",
                    f"secret-like content detected ({code}); refusing to store it",
                )
        for code, pattern in PII_PATTERNS:
            if pattern.search(content):
                return refuse(
                    "PII_DETECTED",
                    f"personal data detected ({code}); refusing to store it",
                )

        if len(RAW_TRANSCRIPT.findall(content)) >= 3:
            return refuse(
                "RAW_TRANSCRIPT",
                "raw conversation transcripts must be summarized before retention",
            )
        for block in FENCED_CODE.findall(record.content):
            if len([line for line in block.splitlines() if line.strip()]) >= 4:
                return refuse(
                    "RAW_CODE_PAYLOAD",
                    "code belongs in the repository/RAG, not in long-term memory",
                )

        tags = {
            f"scope:{record.scope}",
            f"category:{record.category}",
            f"verification:{record.verification}",
            f"source:{_slug(record.source, max_length=32)}",
        }
        if record.scope == "project":
            tags.add("project:" + _slug(record.project_id))
        if record.memory_key:
            tags.add("memory_key:" + _slug(record.memory_key, max_length=64))
        for tag in record.tags:
            cleaned = tag.strip()
            if cleaned and " " not in cleaned:
                tags.add(cleaned)

        metadata = {key: _metadata_value(value) for key, value in record.metadata.items()}
        metadata.update(
            {
                "project_id": record.project_id,
                "category": record.category,
                "verification": record.verification,
                "source": record.source,
                "scope": record.scope,
            }
        )
        if record.memory_key:
            metadata["memory_key"] = record.memory_key
        if record.session_id:
            metadata["session_id"] = record.session_id
        if record.occurred_at:
            metadata["occurred_at"] = record.occurred_at
        if record.evidence:
            metadata["evidence"] = " | ".join(item.strip() for item in record.evidence if item.strip())

        return AdmissionDecision(
            allowed=True,
            code="OK",
            reason="accepted",
            bank_id=bank_id,
            normalized_content=content,
            tags=tuple(sorted(tags)),
            metadata=metadata,
        )