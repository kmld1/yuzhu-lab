"""Resource and service health checks for graceful degradation.

The harness never starts or stops user services.  It only reports health and
decides whether optional context sources (RAG, memory) can be loaded within the
current RAM budget.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

GIB = 1024**3


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def ram_available_bytes() -> int | None:
    if hasattr(ctypes, "windll"):
        try:
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:  # noqa: BLE001 - health must not fail
            return None
    return None


def gpu_snapshot() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}
    if completed.returncode != 0:
        return {"available": False, "reason": (completed.stderr or "").strip()[:200]}
    lines = (completed.stdout or "").strip().splitlines()
    if not lines:
        return {"available": False, "reason": "unexpected nvidia-smi output"}
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) < 4:
        return {"available": False, "reason": "unexpected nvidia-smi output"}
    name, total, used, free = parts[:4]
    return {
        "available": True,
        "name": name,
        "total_mib": int(float(total)),
        "used_mib": int(float(used)),
        "free_mib": int(float(free)),
    }


def windows_memory() -> dict[str, Any]:
    """Physical RAM and commit charge; availability is not the same as commit."""
    if not hasattr(ctypes, "windll"):
        return {"available": False}
    try:
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {"available": False}
    except Exception as exc:  # noqa: BLE001 - diagnostics must not fail
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "total_phys_gib": round(status.ullTotalPhys / GIB, 2),
        "avail_phys_gib": round(status.ullAvailPhys / GIB, 2),
        "total_commit_gib": round(status.ullTotalPageFile / GIB, 2),
        "avail_commit_gib": round(status.ullAvailPageFile / GIB, 2),
        "memory_load_percent": int(status.dwMemoryLoad),
    }


def process_memory(names: tuple[str, ...] = ("python", "hindsight-api", "postgres", "node")) -> list[dict[str, Any]]:
    """Working-set snapshot per matching process (psutil preferred, tasklist fallback)."""
    try:
        import psutil  # type: ignore
    except Exception:  # noqa: BLE001
        return _process_memory_tasklist(names)
    items: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            name = process.info.get("name") or ""
            if not any(token.lower() in name.lower() for token in names):
                continue
            info = process.info.get("memory_info")
            if info is None:
                continue
            items.append(
                {
                    "pid": process.info["pid"],
                    "name": name,
                    "rss_mib": round(info.rss / (1024**2), 1),
                    "vms_mib": round(info.vms / (1024**2), 1),
                }
            )
        except Exception:  # noqa: BLE001 - process may exit during iteration
            continue
    return sorted(items, key=lambda item: item["rss_mib"], reverse=True)


def _process_memory_tasklist(names: tuple[str, ...]) -> list[dict[str, Any]]:
    executable = shutil.which("tasklist")
    if executable is None:
        return []
    try:
        completed = subprocess.run(
            [executable, "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return []
    items = []
    for line in completed.stdout.splitlines():
        parts = [part.strip('"') for part in line.split('","')]
        if len(parts) < 5:
            continue
        name = parts[0]
        if not any(token.lower() in name.lower() for token in names):
            continue
        try:
            rss_kib = int(parts[4].replace(",", "").replace(" K", "").strip())
        except ValueError:
            continue
        items.append(
            {
                "pid": int(parts[1]),
                "name": name,
                "rss_mib": round(rss_kib / 1024, 1),
                "vms_mib": None,
            }
        )
    return sorted(items, key=lambda item: item["rss_mib"], reverse=True)


def gpu_processes() -> list[dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except Exception:  # noqa: BLE001
        return []
    if completed.returncode != 0:
        return []
    items = []
    for line in (completed.stdout or "").strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            items.append(
                {
                    "pid": int(parts[0]),
                    "name": parts[1],
                    "used_mib": int(float(parts[2])),
                }
            )
        except ValueError:
            continue
    return items


def http_health(url: str, timeout: float = 3.0) -> dict[str, Any]:
    try:
        response = httpx.get(url.rstrip("/") + "/health", timeout=timeout, trust_env=False)
        return {
            "ok": response.status_code == 200,
            "status_code": response.status_code,
            "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else {},
        }
    except Exception as exc:  # noqa: BLE001 - health is diagnostic
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


@dataclass
class ServiceReport:
    runtime_url: str
    hindsight_url: str
    rag_store: Path
    ram_available_gib: float | None = None
    gpu: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    hindsight: dict[str, Any] = field(default_factory=dict)
    rag_index_present: bool = False
    rag_enabled: bool = False
    memory_enabled: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_url": self.runtime_url,
            "hindsight_url": self.hindsight_url,
            "ram_available_gib": self.ram_available_gib,
            "gpu": self.gpu,
            "runtime": self.runtime,
            "hindsight": self.hindsight,
            "rag_store": str(self.rag_store),
            "rag_index_present": self.rag_index_present,
            "rag_enabled": self.rag_enabled,
            "memory_enabled": self.memory_enabled,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ResourceThresholds:
    min_runtime_gpu_free_mib: int = 5000
    min_rag_ram_gib: float = 1.5
    min_hindsight_ram_gib: float = 0.5
    min_agent_ram_gib: float = 0.75


@dataclass(frozen=True)
class ResourceDecision:
    service: str
    allowed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "allowed": self.allowed,
            "reasons": list(self.reasons),
        }


def preflight(
    service: str,
    *,
    snapshot: dict[str, Any],
    thresholds: ResourceThresholds | None = None,
) -> ResourceDecision:
    """Decide whether a service may be loaded, without starting or stopping it."""
    limits = thresholds or ResourceThresholds()
    ram = snapshot.get("ram_available_gib")
    gpu = snapshot.get("gpu") or {}
    reasons: list[str] = []

    if service == "runtime":
        if snapshot.get("runtime", {}).get("ok"):
            return ResourceDecision(service, True, ("runtime is already healthy",))
        if not gpu.get("available"):
            reasons.append("GPU telemetry unavailable")
        elif int(gpu.get("free_mib", 0)) < limits.min_runtime_gpu_free_mib:
            reasons.append(
                f"GPU free {gpu.get('free_mib')} MiB < {limits.min_runtime_gpu_free_mib} MiB"
            )
        if ram is not None and ram < limits.min_agent_ram_gib:
            reasons.append(f"RAM available {ram:.2f} GiB < {limits.min_agent_ram_gib:.2f} GiB")
        return ResourceDecision(service, not reasons, tuple(reasons))

    if service == "rag":
        if not snapshot.get("rag_index_present", False):
            reasons.append("RAG index files are missing")
        if ram is None:
            reasons.append("RAM availability unknown")
        elif ram < limits.min_rag_ram_gib:
            reasons.append(f"RAM available {ram:.2f} GiB < {limits.min_rag_ram_gib:.2f} GiB")
        return ResourceDecision(service, not reasons, tuple(reasons))

    if service == "hindsight":
        if ram is not None and ram < limits.min_hindsight_ram_gib:
            reasons.append(
                f"RAM available {ram:.2f} GiB < {limits.min_hindsight_ram_gib:.2f} GiB"
            )
        return ResourceDecision(service, not reasons, tuple(reasons))

    if service == "agent":
        if not snapshot.get("runtime", {}).get("ok"):
            reasons.append("MiniCPM5 runtime is not healthy")
        if ram is not None and ram < limits.min_agent_ram_gib:
            reasons.append(f"RAM available {ram:.2f} GiB < {limits.min_agent_ram_gib:.2f} GiB")
        return ResourceDecision(service, not reasons, tuple(reasons))

    return ResourceDecision(service, False, (f"unknown service: {service}",))


def full_snapshot() -> dict[str, Any]:
    memory = windows_memory()
    return {
        "memory": memory,
        "ram_available_gib": memory.get("avail_phys_gib") if memory.get("available") else None,
        "gpu": gpu_snapshot(),
        "gpu_processes": gpu_processes(),
        "processes": process_memory(),
    }


def check_services(
    *,
    runtime_url: str,
    hindsight_url: str,
    rag_store: Path,
    min_ram_gib: float = 1.5,
) -> ServiceReport:
    report = ServiceReport(
        runtime_url=runtime_url,
        hindsight_url=hindsight_url,
        rag_store=Path(rag_store),
    )
    available = ram_available_bytes()
    report.ram_available_gib = round(available / GIB, 2) if available else None
    report.gpu = gpu_snapshot()
    report.runtime = http_health(runtime_url)
    report.hindsight = http_health(hindsight_url)
    report.rag_index_present = (Path(rag_store) / "manifest.json").is_file() and (
        Path(rag_store) / "index.faiss"
    ).is_file()

    if report.ram_available_gib is None:
        report.rag_enabled = report.rag_index_present
        report.notes.append("RAM availability unknown; RAG left enabled but unverified")
    elif report.ram_available_gib < min_ram_gib:
        report.rag_enabled = False
        report.notes.append(
            f"available RAM {report.ram_available_gib:.2f} GiB < {min_ram_gib:.2f} GiB; "
            "RAG index not loaded"
        )
    else:
        report.rag_enabled = report.rag_index_present
        if not report.rag_index_present:
            report.notes.append("RAG index files are missing")
    report.memory_enabled = bool(report.hindsight.get("ok"))
    if not report.memory_enabled:
        report.notes.append("Hindsight is not healthy; memory context disabled")
    if not report.runtime.get("ok"):
        report.notes.append("MiniCPM5 runtime is not healthy")
    return report