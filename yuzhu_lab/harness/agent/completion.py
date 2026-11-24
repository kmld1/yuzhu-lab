"""Unified completion gate for agent runs.

The H3 evidence showed a real defect: an answer returned with
``stop_reason=max_tokens`` was persisted as ``TASK_COMPLETED``.  H5.2 D3
showed a second defect: a model-written test file exited 0 without executing
any real test, and the gate treated that as verification.

This module separates the facts the loop can actually prove:

    MODEL_STOPPED          the model returned a non-truncated response
    MODEL_TRUNCATED        the model hit its output limit
    TOOL_EXECUTED          at least one tool call ran and returned success
    TOOL_CALL_FAILED       every attempted tool call failed
    NO_TESTS_DISCOVERED    no test file/case exists
    NO_TESTS_EXECUTED      tests exist but the runner executed none
    TESTS_FAILED           the runner reported a failure/error
    TESTS_INCOMPLETE       tests ran but are skipped, vacuous, stale or unrelated
    UNVERIFIED_CODE        code changed without trusted verification
    ORACLE_FAILED          the independent acceptance oracle did not pass
    TASK_VERIFIED          trusted tests passed on the current code
    TASK_COMPLETED         the final answer is complete and every applicable check passed

The gate never trusts a model's prose or a bare exit code.  It reads the
runner evidence produced by :mod:`verification`, checks file hashes so later
edits invalidate old evidence, and consumes the independent acceptance oracle
when one is configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from .model_client import ModelResponse
from .verification import (
    VERDICT_NO_TESTS_DISCOVERED,
    VERDICT_NO_TESTS_EXECUTED,
    VERDICT_PASS,
    VERDICT_TESTS_FAILED,
    VERDICT_TESTS_INCOMPLETE,
    current_hashes,
)


class CompletionLevel(str, Enum):
    MODEL_STOPPED = "MODEL_STOPPED"
    MODEL_TRUNCATED = "MODEL_TRUNCATED"
    TOOL_EXECUTED = "TOOL_EXECUTED"
    TOOL_CALL_FAILED = "TOOL_CALL_FAILED"
    NO_TESTS_DISCOVERED = "NO_TESTS_DISCOVERED"
    NO_TESTS_EXECUTED = "NO_TESTS_EXECUTED"
    TESTS_FAILED = "TESTS_FAILED"
    TESTS_INCOMPLETE = "TESTS_INCOMPLETE"
    UNVERIFIED_CODE = "UNVERIFIED_CODE"
    ORACLE_FAILED = "ORACLE_FAILED"
    TASK_VERIFIED = "TASK_VERIFIED"
    TASK_COMPLETED = "TASK_COMPLETED"


_MODIFYING_TOOLS = {"write_file", "edit_file", "apply_patch"}
_CODE_SUFFIXES = (".py", ".pyi")
_NO_TEST_CODES = {
    "NO_TESTS_RUN",
    "NO_TESTS_DISCOVERED",
    "NO_TESTS_EXECUTED",
}


@dataclass(frozen=True)
class CompletionAssessment:
    level: CompletionLevel
    model_stopped: bool
    model_truncated: bool
    final_answer_complete: bool
    tool_executed: bool
    tool_call_failed: bool
    task_verified: bool
    task_completed: bool
    modified_files: tuple[str, ...] = ()
    code_files_changed: tuple[str, ...] = ()
    tests_run: int = 0
    tests_passed: int = 0
    last_test_passed: bool | None = None
    verification_fresh: bool | None = None
    evidence_verdict: str | None = None
    acceptance_passed: bool | None = None
    blockers: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "model_stopped": self.model_stopped,
            "model_truncated": self.model_truncated,
            "final_answer_complete": self.final_answer_complete,
            "tool_executed": self.tool_executed,
            "tool_call_failed": self.tool_call_failed,
            "task_verified": self.task_verified,
            "task_completed": self.task_completed,
            "modified_files": list(self.modified_files),
            "code_files_changed": list(self.code_files_changed),
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "last_test_passed": self.last_test_passed,
            "verification_fresh": self.verification_fresh,
            "evidence_verdict": self.evidence_verdict,
            "acceptance_passed": self.acceptance_passed,
            "blockers": list(self.blockers),
            "reasons": list(self.reasons),
        }

    @property
    def summary(self) -> str:
        if self.task_completed:
            return "task completed with verified evidence"
        return "task not completed: " + ", ".join(self.blockers or ("UNKNOWN",))


def _record_status(record: dict[str, Any]) -> str:
    return str(record.get("status") or "")


def _record_result(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result")
    return result if isinstance(result, dict) else {}


def _tool_ok(record: dict[str, Any]) -> bool:
    if _record_status(record) not in {"COMPLETED", "DUPLICATE"}:
        return False
    return str(_record_result(record).get("status") or "") in {"OK", "DUPLICATE"}


def _modified_path(record: dict[str, Any]) -> str | None:
    if record.get("name") not in _MODIFYING_TOOLS:
        return None
    result = _record_result(record)
    if not _tool_ok(record):
        return None
    structured = result.get("structured") or {}
    if not structured.get("readback_verified"):
        return None
    path = structured.get("path")
    return str(path) if path else None


def _is_code_path(path: str) -> bool:
    return path.lower().endswith(_CODE_SUFFIXES)


def _has_unclosed_tool_syntax(text: str) -> bool:
    lowered = text.lower()
    if lowered.count("<function") > lowered.count("</function>"):
        return True
    if lowered.count("<param") > lowered.count("</param>"):
        return True
    if lowered.lstrip().startswith("```yuzhu-action") and "```" not in lowered[3:]:
        return True
    return False


def _evidence_from(record: dict[str, Any]) -> dict[str, Any] | None:
    structured = _record_result(record).get("structured") or {}
    evidence = structured.get("verification")
    return evidence if isinstance(evidence, dict) else None


def _hash_mismatches(
    workspace: Path | None,
    expected: dict[str, str] | None,
) -> list[str]:
    if workspace is None or not expected:
        return []
    observed = current_hashes(workspace, expected)
    return sorted(path for path, digest in expected.items() if observed.get(path) != digest)


def assess_completion(
    response: ModelResponse,
    tool_calls: Sequence[dict[str, Any]],
    *,
    final_text: str,
    require_verification_for_code_change: bool = True,
    workspace: Path | None = None,
    acceptance: dict[str, Any] | None = None,
    acceptance_configured: bool = False,
) -> CompletionAssessment:
    """Assess whether a final answer may be recorded as TASK_COMPLETED.

    ``tool_calls`` is the checkpointed call list for the task.  The function is
    deterministic for a given evidence set and filesystem state, so the same
    evidence can be re-evaluated after a resume or an audit.
    """

    reasons: list[str] = []
    blockers: list[str] = []

    model_truncated = response.stop_reason == "max_tokens" or response.truncated
    model_stopped = not model_truncated
    if model_truncated:
        blockers.append("MODEL_TRUNCATED")
        reasons.append("model stop_reason was max_tokens")
    elif response.stop_reason == "tool_use":
        blockers.append("INCOMPLETE_TOOL_USE")
        reasons.append("model stopped for tool_use but no tool call was parsed")

    text = (final_text or "").strip()
    final_answer_complete = (
        bool(text) and not model_truncated and response.stop_reason != "tool_use"
    )
    if not text:
        blockers.append("EMPTY_FINAL_ANSWER")
    if text and _has_unclosed_tool_syntax(text):
        final_answer_complete = False
        blockers.append("UNCLOSED_TOOL_SYNTAX")
        reasons.append("final answer contains an incomplete tool syntax block")

    executed = [record for record in tool_calls if _tool_ok(record)]
    tool_executed = bool(executed)
    tool_call_failed = bool(tool_calls) and not executed
    if tool_call_failed:
        reasons.append("every attempted tool call failed")

    modified: list[str] = []
    last_code_modification_index = -1
    for index, record in enumerate(tool_calls):
        path = _modified_path(record)
        if path is None:
            continue
        modified.append(path)
        if _is_code_path(path):
            last_code_modification_index = index

    code_files_changed = tuple(sorted({path for path in modified if _is_code_path(path)}))
    changed_set = set(code_files_changed)

    test_records: list[tuple[int, dict[str, Any], dict[str, Any] | None]] = []
    no_tests_observed = False
    for index, record in enumerate(tool_calls):
        if record.get("name") != "run_tests":
            continue
        result = _record_result(record)
        if result.get("error_code") == "NO_TESTS_RUN":
            no_tests_observed = True
        evidence = _evidence_from(record)
        if evidence is None and result.get("exit_code") is None:
            # A harness-blocked repeat/budget call did not execute tests; it
            # must not overwrite the last real test evidence.
            continue
        test_records.append((index, record, evidence))

    tests_run = 0
    tests_passed = 0
    evidence_verdict: str | None = None
    last_test_passed: bool | None = None
    verification_fresh: bool | None = None
    task_verified = False

    if test_records:
        last_index, last_record, last_evidence = test_records[-1]
        verification_fresh = last_index > last_code_modification_index
        for _, _, evidence in test_records:
            if evidence:
                tests_run += int(evidence.get("tests_executed") or 0)
                tests_passed += int(evidence.get("tests_passed") or 0)

        if last_evidence is None:
            blockers.append("VERIFICATION_UNTRUSTED")
            reasons.append(
                "the last run_tests result has no trusted verification evidence"
            )
            evidence_verdict = "UNTRUSTED"
        else:
            evidence_verdict = str(last_evidence.get("verdict") or "")
            last_test_passed = evidence_verdict == VERDICT_PASS
            if evidence_verdict == VERDICT_TESTS_FAILED:
                blockers.append("TESTS_FAILED")
                blockers.append("LAST_TEST_FAILED")
                reasons.append("the last trusted test run reported a failure")
            elif evidence_verdict == VERDICT_NO_TESTS_DISCOVERED:
                blockers.append("NO_TESTS_DISCOVERED")
                reasons.append("no test case was discovered")
            elif evidence_verdict == VERDICT_NO_TESTS_EXECUTED:
                blockers.append("NO_TESTS_EXECUTED")
                reasons.append("tests exist but none were executed")
            elif evidence_verdict == VERDICT_TESTS_INCOMPLETE:
                blockers.append("TESTS_INCOMPLETE")
                reasons.extend(str(item) for item in last_evidence.get("reasons") or [])
            elif not verification_fresh:
                blockers.append("TEST_EVIDENCE_STALE")
                reasons.append("a code file changed after the last successful test run")
            else:
                source_changed = _hash_mismatches(
                    workspace, last_evidence.get("source_hashes") or {}
                )
                if source_changed:
                    blockers.append("SOURCE_CHANGED_AFTER_TEST")
                    reasons.append(
                        "source files changed after the test run: "
                        + ", ".join(source_changed)
                    )
                test_hashes = {
                    str(item.get("path")): str(item.get("sha256"))
                    for item in last_evidence.get("test_files") or []
                    if item.get("path") and item.get("sha256")
                }
                test_changed = _hash_mismatches(workspace, test_hashes)
                if test_changed:
                    blockers.append("TEST_FILE_CHANGED_AFTER_RUN")
                    reasons.append(
                        "test files changed after the test run: "
                        + ", ".join(test_changed)
                    )
                mapped = {str(item) for item in last_evidence.get("mapped_files") or []}
                if changed_set and not mapped.intersection(changed_set):
                    blockers.append("UNRELATED_TESTS")
                    reasons.append(
                        "the test files do not import any changed code file"
                    )
                if any(
                    blocker in blockers
                    for blocker in (
                        "SOURCE_CHANGED_AFTER_TEST",
                        "TEST_FILE_CHANGED_AFTER_RUN",
                        "UNRELATED_TESTS",
                    )
                ):
                    blockers.append("TESTS_INCOMPLETE")
                    evidence_verdict = VERDICT_TESTS_INCOMPLETE
                else:
                    task_verified = True

    if code_files_changed and not task_verified:
        if require_verification_for_code_change:
            blockers.append("UNVERIFIED_CODE")
            reasons.append("code files changed without trusted verification")
        else:
            reasons.append(
                "code files changed and verification is not required by configuration"
            )

    acceptance_passed: bool | None = None
    if acceptance_configured:
        if acceptance is None:
            blockers.append("ORACLE_FAILED")
            reasons.append("ORACLE_NOT_RUN")
            acceptance_passed = False
        else:
            acceptance_passed = bool(acceptance.get("passed"))
            if not acceptance_passed:
                blockers.append("ORACLE_FAILED")
                reasons.extend(str(item) for item in acceptance.get("reasons") or [])
                reasons.append("the independent acceptance oracle did not pass")

    if no_tests_observed and not task_verified:
        blockers.append("NO_TESTS_RUN")
        reasons.append("run_tests executed but discovered zero tests")

    # Deduplicate while preserving order.
    blockers = list(dict.fromkeys(blockers))
    reasons = list(dict.fromkeys(reasons))
    task_completed = final_answer_complete and not blockers

    if model_truncated:
        level = CompletionLevel.MODEL_TRUNCATED
    elif task_completed:
        level = CompletionLevel.TASK_COMPLETED
    elif "TESTS_FAILED" in blockers:
        level = CompletionLevel.TESTS_FAILED
    elif "ORACLE_FAILED" in blockers:
        level = CompletionLevel.ORACLE_FAILED
    elif "NO_TESTS_DISCOVERED" in blockers:
        level = CompletionLevel.NO_TESTS_DISCOVERED
    elif "NO_TESTS_EXECUTED" in blockers:
        level = CompletionLevel.NO_TESTS_EXECUTED
    elif "TESTS_INCOMPLETE" in blockers:
        level = CompletionLevel.TESTS_INCOMPLETE
    elif "UNVERIFIED_CODE" in blockers or "VERIFICATION_UNTRUSTED" in blockers:
        level = CompletionLevel.UNVERIFIED_CODE
    elif tool_call_failed:
        level = CompletionLevel.TOOL_CALL_FAILED
    elif task_verified:
        level = CompletionLevel.TASK_VERIFIED
    elif tool_executed:
        level = CompletionLevel.TOOL_EXECUTED
    else:
        level = CompletionLevel.MODEL_STOPPED

    return CompletionAssessment(
        level=level,
        model_stopped=model_stopped,
        model_truncated=model_truncated,
        final_answer_complete=final_answer_complete,
        tool_executed=tool_executed,
        tool_call_failed=tool_call_failed,
        task_verified=task_verified,
        task_completed=task_completed,
        modified_files=tuple(dict.fromkeys(modified)),
        code_files_changed=code_files_changed,
        tests_run=tests_run,
        tests_passed=tests_passed,
        last_test_passed=last_test_passed,
        verification_fresh=verification_fresh,
        evidence_verdict=evidence_verdict,
        acceptance_passed=acceptance_passed,
        blockers=tuple(blockers),
        reasons=tuple(reasons),
    )


__all__ = [
    "CompletionAssessment",
    "CompletionLevel",
    "assess_completion",
]