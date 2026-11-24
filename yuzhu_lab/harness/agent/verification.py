"""Trusted verification oracle for agent test evidence.

H5.2 D3 showed that a model-written test file can exit 0 without executing any
real test: the file defined a ``unittest.TestCase`` but had no
``unittest.main()`` guard, and the harness ran it as a plain script.  The
completion gate then treated ``exit_code == 0`` as verification.

This module builds evidence that does not depend on the model's prose:

* tests are discovered from the filesystem and parsed with ``ast``;
* the test runner is invoked through ``python -m unittest`` / ``pytest`` so a
  missing ``if __name__ == '__main__'`` guard cannot silently skip execution;
* the runner output is parsed for executed / passed / failed / skipped counts;
* test and source file hashes are recorded so a later edit invalidates the
  evidence;
* an independent acceptance script outside the workspace can judge semantics.

Static assertion counts are an auxiliary signal, not a semantic judgement.
Only the external acceptance oracle can prove task-specific behaviour.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

VERDICT_PASS = "PASS"
VERDICT_NO_TESTS_DISCOVERED = "NO_TESTS_DISCOVERED"
VERDICT_NO_TESTS_EXECUTED = "NO_TESTS_EXECUTED"
VERDICT_TESTS_FAILED = "TESTS_FAILED"
VERDICT_TESTS_INCOMPLETE = "TESTS_INCOMPLETE"

TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")

_IGNORED_DIRS = {
    ".agent",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "node_modules",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def discover_test_files(workspace: Path, target: str | Path) -> list[Path]:
    """Return the test files the runner is expected to load."""
    workspace = workspace.resolve()
    candidate = (workspace / target).resolve() if not Path(target).is_absolute() else Path(target)
    if candidate.is_file():
        return [candidate]
    if not candidate.exists():
        return []
    files: list[Path] = []
    for pattern in TEST_FILE_PATTERNS:
        files.extend(candidate.rglob(pattern))
    return sorted(
        {
            path
            for path in files
            if path.is_file()
            and not any(part in _IGNORED_DIRS for part in path.relative_to(workspace).parts)
        }
    )


def scan_test_file(path: Path) -> dict[str, Any]:
    """Parse one test file and report structural evidence."""
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    result: dict[str, Any] = {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "parse_ok": True,
        "test_cases": 0,
        "assertions": 0,
        "skipped_cases": 0,
        "has_main_guard": False,
        "imports": [],
        "parse_error": None,
    }
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        result["parse_ok"] = False
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
        return result

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
        elif isinstance(node, ast.Assert):
            result["assertions"] += 1
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr.startswith("assert"):
                result["assertions"] += 1
            elif isinstance(func, ast.Name) and func.id.startswith("assert_"):
                result["assertions"] += 1
            elif isinstance(func, ast.Attribute) and func.attr in {"raises", "warns"}:
                # pytest.raises / pytest.warns
                owner = func.value
                if isinstance(owner, ast.Name) and owner.id == "pytest":
                    result["assertions"] += 1
        elif isinstance(node, ast.If):
            # if __name__ == '__main__':
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                result["has_main_guard"] = True

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not node.name.startswith("Test"):
                continue
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith(
                    "test"
                ):
                    result["test_cases"] += 1
                    if _is_skipped(item):
                        result["skipped_cases"] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                result["test_cases"] += 1
                if _is_skipped(node):
                    result["skipped_cases"] += 1

    result["imports"] = sorted(imports)
    return result


def _is_skipped(node: ast.AST) -> bool:
    for decorator in getattr(node, "decorator_list", []):
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = ""
        if isinstance(target, ast.Attribute):
            name = target.attr
        elif isinstance(target, ast.Name):
            name = target.id
        if name in {"skip", "skipIf", "skipUnless", "skipTest", "xfail"}:
            return True
    return False


def _imports_to_files(workspace: Path, imports: Iterable[str]) -> list[str]:
    mapped: set[str] = set()
    for module in imports:
        parts = module.split(".")
        for end in range(len(parts), 0, -1):
            stem = parts[:end]
            candidate = workspace.joinpath(*stem).with_suffix(".py")
            package = workspace.joinpath(*stem) / "__init__.py"
            if candidate.is_file():
                mapped.add(candidate.relative_to(workspace).as_posix())
                break
            if package.is_file():
                mapped.add(package.relative_to(workspace).as_posix())
                break
    return sorted(mapped)


def _parse_unittest(output: str) -> dict[str, int]:
    executed = 0
    match = re.search(r"Ran (\d+) tests?", output)
    if match:
        executed = int(match.group(1))
    failed = 0
    match = re.search(r"FAILED \((.*?)\)", output)
    if match:
        details = match.group(1)
        for label in ("failures", "errors"):
            found = re.search(rf"{label}=(\d+)", details)
            if found:
                failed += int(found.group(1))
    skipped = len(re.findall(r"\.\.\. skipped", output))
    match = re.search(r"OK \(skipped=(\d+)\)", output)
    if match:
        skipped = max(skipped, int(match.group(1)))
    return {
        "tests_executed": executed,
        "tests_failed": failed,
        "tests_skipped": skipped,
    }


def _parse_pytest(output: str) -> dict[str, int]:
    lowered = output.lower()
    if "no tests ran" in lowered or "collected 0 items" in lowered:
        return {"tests_executed": 0, "tests_failed": 0, "tests_skipped": 0}

    def count(label: str) -> int:
        match = re.search(rf"(\d+) {label}", lowered)
        return int(match.group(1)) if match else 0

    passed = count("passed")
    failed = count("failed")
    errors = count("errors?")
    skipped = count("skipped")
    return {
        "tests_executed": passed + failed + errors + skipped,
        "tests_failed": failed + errors,
        "tests_skipped": skipped,
    }


def parse_runner_output(framework: str, output: str) -> dict[str, int]:
    if framework == "pytest":
        return _parse_pytest(output)
    return _parse_unittest(output)


def build_test_evidence(
    workspace: Path,
    *,
    target: str,
    framework: str,
    output: str,
    exit_code: int | None,
    changed_code_files: Sequence[str] = (),
) -> dict[str, Any]:
    """Build trusted evidence for one ``run_tests`` execution.

    ``output`` must be the combined stdout/stderr captured by the harness, not
    a model summary.  ``changed_code_files`` is the set of code files modified
    by the agent before this run; it is used for the test/target mapping check.
    """
    workspace = workspace.resolve()
    scans = [scan_test_file(path) for path in discover_test_files(workspace, target)]
    for scan in scans:
        scan["path"] = Path(scan["path"]).resolve().relative_to(workspace).as_posix()

    counts = parse_runner_output(framework, output)
    tests_discovered = sum(int(scan["test_cases"]) for scan in scans)
    assertions = sum(int(scan["assertions"]) for scan in scans)
    skipped_static = sum(int(scan["skipped_cases"]) for scan in scans)
    imports: set[str] = set()
    for scan in scans:
        imports.update(scan["imports"])
    mapped_files = _imports_to_files(workspace, imports)

    reasons: list[str] = []
    verdict = VERDICT_PASS

    no_tests_output = "NO TESTS RAN" in output or "Ran 0 tests" in output
    if counts["tests_executed"] == 0 and (tests_discovered == 0 or no_tests_output):
        if tests_discovered == 0:
            verdict = VERDICT_NO_TESTS_DISCOVERED
            reasons.append("NO_TESTS_DISCOVERED")
        else:
            verdict = VERDICT_NO_TESTS_EXECUTED
            reasons.append("NO_TESTS_EXECUTED")
    elif exit_code not in (0, None) or counts["tests_failed"] > 0:
        verdict = VERDICT_TESTS_FAILED
        reasons.append("TESTS_FAILED")
    elif counts["tests_executed"] == 0:
        if tests_discovered == 0:
            verdict = VERDICT_NO_TESTS_DISCOVERED
            reasons.append("NO_TESTS_DISCOVERED")
        else:
            verdict = VERDICT_NO_TESTS_EXECUTED
            reasons.append("NO_TESTS_EXECUTED")
    elif any(not scan["parse_ok"] for scan in scans):
        verdict = VERDICT_TESTS_INCOMPLETE
        reasons.append("INVALID_TEST_FILE")
    elif counts["tests_skipped"] >= counts["tests_executed"] or (
        skipped_static >= tests_discovered and tests_discovered > 0
    ):
        verdict = VERDICT_TESTS_INCOMPLETE
        reasons.append("ALL_TESTS_SKIPPED")
    elif tests_discovered == 0:
        verdict = VERDICT_NO_TESTS_DISCOVERED
        reasons.append("NO_TESTS_DISCOVERED")
    elif assertions == 0:
        verdict = VERDICT_TESTS_INCOMPLETE
        reasons.append("NO_ASSERTIONS")
    elif changed_code_files:
        changed = {Path(path).as_posix() for path in changed_code_files}
        if not changed.intersection(mapped_files):
            verdict = VERDICT_TESTS_INCOMPLETE
            reasons.append("UNRELATED_TESTS")

    if counts["tests_skipped"] and verdict == VERDICT_PASS:
        reasons.append("SOME_TESTS_SKIPPED")

    source_hashes = _source_hashes(workspace, mapped_files, changed_code_files)
    return {
        "schema": "yuzhu-test-evidence-1",
        "verdict": verdict,
        "framework": framework,
        "target": target,
        "exit_code": exit_code,
        "tests_discovered": tests_discovered,
        "tests_executed": counts["tests_executed"],
        "tests_passed": max(0, counts["tests_executed"] - counts["tests_failed"]),
        "tests_failed": counts["tests_failed"],
        "tests_skipped": counts["tests_skipped"],
        "assertions": assertions,
        "imports": sorted(imports),
        "mapped_files": mapped_files,
        "test_files": scans,
        "source_hashes": source_hashes,
        "reasons": reasons,
        "output_sha256": _sha256_text(output),
    }


def _source_hashes(
    workspace: Path,
    mapped_files: Sequence[str],
    changed_code_files: Sequence[str],
) -> dict[str, str]:
    """Snapshot source files used to judge whether tests ran on stale code."""
    paths: set[Path] = set()
    for path in changed_code_files:
        candidate = (workspace / path).resolve()
        if candidate.is_file():
            paths.add(candidate)
    for mapped in mapped_files:
        candidate = (workspace / mapped).resolve()
        if candidate.is_file():
            paths.add(candidate)
    hashes: dict[str, str] = {}
    for path in paths:
        try:
            hashes[path.relative_to(workspace).as_posix()] = sha256_file(path)
        except OSError:
            continue
    return hashes


def current_hashes(workspace: Path, relative_paths: Iterable[str]) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for relative in relative_paths:
        candidate = (workspace / relative).resolve()
        try:
            candidate.relative_to(workspace.resolve())
        except ValueError:
            hashes[Path(relative).as_posix()] = None
            continue
        hashes[Path(relative).as_posix()] = sha256_file(candidate) if candidate.is_file() else None
    return hashes


def run_acceptance_oracle(
    workspace: Path,
    *,
    script: Path,
    expected_sha256: str | None = None,
    timeout_seconds: float = 300.0,
    expected_checks: int | None = None,
) -> dict[str, Any]:
    """Run an independent acceptance script outside the model workspace.

    The script is trusted only when it lives outside the workspace and its
    configured SHA256 matches.  Its JSON payload is never added to the model's
    conversation; only the completion gate consumes it.
    """
    workspace = workspace.resolve()
    script = script.resolve()
    reasons: list[str] = []
    result: dict[str, Any] = {
        "schema": "yuzhu-acceptance-oracle-1",
        "script": str(script),
        "expected_sha256": expected_sha256,
        "sha256": None,
        "exit_code": None,
        "duration_ms": 0.0,
        "payload": None,
        "passed": False,
        "reasons": reasons,
        "stdout_tail": "",
        "stderr_tail": "",
    }

    try:
        script.relative_to(workspace)
        inside = True
    except ValueError:
        inside = False
    if inside:
        reasons.append("ORACLE_INSIDE_WORKSPACE")
        return result
    if not script.is_file():
        reasons.append("ORACLE_SCRIPT_NOT_FOUND")
        return result

    actual = sha256_file(script)
    result["sha256"] = actual
    if expected_sha256 and actual != expected_sha256:
        reasons.append("ORACLE_HASH_MISMATCH")
        return result

    from .tools.common import run_process

    output_limit = 1024 * 1024
    completed = run_process(
        [sys.executable, str(script), "--workspace", str(workspace)],
        cwd=workspace,
        timeout=timeout_seconds,
        max_output_chars=output_limit,
        max_stdout_bytes=output_limit,
        max_stderr_bytes=output_limit,
        max_total_bytes=output_limit * 2,
    )
    result["duration_ms"] = completed.duration_ms
    if completed.timed_out:
        reasons.append("ORACLE_TIMEOUT")
        return result
    result["exit_code"] = completed.exit_code
    result["stdout_tail"] = completed.stdout[-4000:]
    result["stderr_tail"] = completed.stderr[-4000:]
    if completed.output_truncated or completed.read_incomplete:
        reasons.append("ORACLE_INCOMPLETE_OUTPUT")
        return result

    try:
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise ValueError("oracle payload must be a JSON object")
    except Exception as exc:  # noqa: BLE001 - report, never accept
        reasons.append("ORACLE_INVALID_OUTPUT")
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["payload"] = payload
    if expected_checks is not None:
        observed = payload.get("core_total", payload.get("checks_total"))
        if observed is not None and int(observed) != int(expected_checks):
            reasons.append("ORACLE_CHECK_COUNT_MISMATCH")
    if "passed" in payload:
        passed = bool(payload.get("passed"))
    elif "core_passed" in payload:
        passed = bool(payload.get("core_passed")) and bool(
            payload.get("extension_passed", True)
        )
    else:
        passed = False
    result["passed"] = passed and not reasons
    if not passed:
        reasons.append("ORACLE_REPORTED_FAILURE")
    return result


__all__ = [
    "VERDICT_NO_TESTS_DISCOVERED",
    "VERDICT_NO_TESTS_EXECUTED",
    "VERDICT_PASS",
    "VERDICT_TESTS_FAILED",
    "VERDICT_TESTS_INCOMPLETE",
    "build_test_evidence",
    "current_hashes",
    "discover_test_files",
    "parse_runner_output",
    "run_acceptance_oracle",
    "scan_test_file",
    "sha256_file",
]