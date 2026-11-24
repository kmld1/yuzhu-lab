"""Classify Hindsight banks for cleanup review without deleting anything.

Formal project/global/experiment banks are never cleanup candidates.  Test
banks are listed with their fact counts and require an explicit, targeted
operation outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

FORMAL_PROJECT_IDS = frozenset({"qianyuqwen"})
TEST_PROJECT_PREFIXES = ("h1", "h2", "h3", "test", "tmp", "e2e")


def project_from_bank(bank_id: str) -> str | None:
    prefix = "yuzhu-project-"
    if not bank_id.startswith(prefix):
        return None
    return bank_id[len(prefix):]


def classify_bank(bank_id: str) -> str:
    if bank_id == "yuzhu-global":
        return "global"
    if bank_id == "yuzhu-experiments":
        return "experiment"
    project = project_from_bank(bank_id)
    if project is None:
        return "unknown"
    if project in FORMAL_PROJECT_IDS:
        return "formal_project"
    if project.startswith(TEST_PROJECT_PREFIXES):
        return "test"
    return "unknown"


@dataclass(frozen=True)
class BankEntry:
    bank_id: str
    classification: str
    fact_count: int
    action: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank_id": self.bank_id,
            "classification": self.classification,
            "fact_count": self.fact_count,
            "action": self.action,
            "reason": self.reason,
        }


@dataclass
class CleanupManifest:
    entries: list[BankEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        cleanup = [entry for entry in self.entries if entry.action == "CLEANUP_CANDIDATE"]
        keep = [entry for entry in self.entries if entry.action == "KEEP"]
        review = [entry for entry in self.entries if entry.action == "REVIEW"]
        return {
            "schema": "yuzhu-memory-bank-cleanup-1",
            "policy": {
                "automatic_delete": False,
                "delete_supported_by_harness": False,
                "requires_explicit_confirmation": True,
                "formal_banks_never_candidates": sorted(FORMAL_PROJECT_IDS),
            },
            "summary": {
                "total": len(self.entries),
                "keep": len(keep),
                "cleanup_candidates": len(cleanup),
                "review": len(review),
                "cleanup_fact_count": sum(entry.fact_count for entry in cleanup),
            },
            "banks": [entry.to_dict() for entry in self.entries],
        }


def build_cleanup_manifest(banks: Iterable[dict[str, Any]]) -> CleanupManifest:
    entries: list[BankEntry] = []
    for bank in banks:
        bank_id = str(bank.get("bank_id") or "")
        if not bank_id:
            continue
        fact_count = int(bank.get("fact_count") or 0)
        classification = classify_bank(bank_id)
        if classification in {"formal_project", "global", "experiment"}:
            action = "KEEP"
            reason = f"{classification} bank is part of the formal memory scope"
        elif classification == "test":
            action = "CLEANUP_CANDIDATE"
            reason = (
                "test bank identified by project prefix; deleting requires an "
                "explicit targeted request"
            )
        else:
            action = "REVIEW"
            reason = "unknown bank; inspect before any action"
        entries.append(
            BankEntry(
                bank_id=bank_id,
                classification=classification,
                fact_count=fact_count,
                action=action,
                reason=reason,
            )
        )
    entries.sort(key=lambda entry: (entry.action, entry.bank_id))
    return CleanupManifest(entries=entries)


def fetch_banks(base_url: str, timeout: float = 10.0) -> list[dict[str, Any]]:
    import httpx

    response = httpx.get(
        base_url.rstrip("/") + "/v1/default/banks", timeout=timeout, trust_env=False
    )
    response.raise_for_status()
    payload = response.json()
    banks = payload.get("banks")
    if not isinstance(banks, list):
        raise ValueError("unexpected Hindsight bank listing response")
    return banks