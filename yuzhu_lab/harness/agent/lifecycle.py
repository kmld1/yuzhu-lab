"""Service lifecycle and resource-budget governance.

The manager never starts or stops a user process.  It records what is loaded,
refuses a second high-resource load when the budget is insufficient, and
produces explicit plans that a human can approve.  The actual GPU/RAM sampling
stays in :mod:`resources`; this module only makes the policy explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable

from .resources import ResourceThresholds, preflight


class ServiceState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    STARTING = "STARTING"
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STOPPED = "STOPPED"
    UNKNOWN = "UNKNOWN"


STARTUP_ORDER = ("runtime", "hindsight", "rag", "agent")
_DEPENDENCIES = {
    "runtime": (),
    "hindsight": (),
    "rag": (),
    "agent": ("runtime",),
}


@dataclass(frozen=True)
class ResourceBudget:
    min_ram_gib: float = 1.5
    min_commit_headroom_gib: float = 2.0
    min_gpu_free_mib: int = 1500
    min_runtime_gpu_free_mib: int = 5000
    min_training_ram_gib: float = 4.0
    min_training_gpu_free_mib: int = 7000
    min_training_commit_headroom_gib: float = 8.0

    def evaluate(self, snapshot: dict[str, Any]) -> "BudgetReport":
        memory = snapshot.get("memory") or {}
        gpu = snapshot.get("gpu") or {}
        ram = snapshot.get("ram_available_gib")
        if ram is None and memory.get("available"):
            ram = memory.get("avail_phys_gib")
        gpu_free = int(gpu.get("free_mib") or 0) if gpu.get("available") else None
        commit_headroom = memory.get("avail_commit_gib")
        total_commit = memory.get("total_commit_gib")
        commit_used_percent = None
        if commit_headroom is not None and total_commit:
            commit_used_percent = round(
                max(0.0, (total_commit - commit_headroom) / total_commit * 100.0), 2
            )

        checks = {
            "ram_ok": ram is not None and ram >= self.min_ram_gib,
            "commit_ok": commit_headroom is not None
            and commit_headroom >= self.min_commit_headroom_gib,
            "gpu_ok": gpu_free is not None and gpu_free >= self.min_gpu_free_mib,
            "training_ram_ok": ram is not None and ram >= self.min_training_ram_gib,
            "training_gpu_ok": gpu_free is not None
            and gpu_free >= self.min_training_gpu_free_mib,
            "training_commit_ok": commit_headroom is not None
            and commit_headroom >= self.min_training_commit_headroom_gib,
        }
        reasons: list[str] = []
        if not checks["ram_ok"]:
            reasons.append(f"RAM available {ram} GiB < {self.min_ram_gib} GiB")
        if not checks["commit_ok"]:
            reasons.append(
                f"commit headroom {commit_headroom} GiB < {self.min_commit_headroom_gib} GiB"
            )
        if not checks["gpu_ok"]:
            reasons.append(f"GPU free {gpu_free} MiB < {self.min_gpu_free_mib} MiB")
        if not checks["training_ram_ok"]:
            reasons.append(
                f"training RAM {ram} GiB < {self.min_training_ram_gib} GiB"
            )
        if not checks["training_gpu_ok"]:
            reasons.append(
                f"training GPU free {gpu_free} MiB < {self.min_training_gpu_free_mib} MiB"
            )
        if not checks["training_commit_ok"]:
            reasons.append(
                f"training commit headroom {commit_headroom} GiB "
                f"< {self.min_training_commit_headroom_gib} GiB"
            )
        return BudgetReport(
            ram_available_gib=ram,
            gpu_free_mib=gpu_free,
            commit_headroom_gib=commit_headroom,
            commit_used_percent=commit_used_percent,
            checks=checks,
            reasons=tuple(reasons),
        )


@dataclass(frozen=True)
class BudgetReport:
    ram_available_gib: float | None
    gpu_free_mib: int | None
    commit_headroom_gib: float | None
    commit_used_percent: float | None
    checks: dict[str, bool]
    reasons: tuple[str, ...]

    @property
    def can_start_normal_service(self) -> bool:
        return self.checks["ram_ok"] and self.checks["commit_ok"] and self.checks["gpu_ok"]

    @property
    def can_start_training(self) -> bool:
        return all(
            self.checks[key]
            for key in ("training_ram_ok", "training_gpu_ok", "training_commit_ok")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ram_available_gib": self.ram_available_gib,
            "gpu_free_mib": self.gpu_free_mib,
            "commit_headroom_gib": self.commit_headroom_gib,
            "commit_used_percent": self.commit_used_percent,
            "checks": dict(self.checks),
            "can_start_normal_service": self.can_start_normal_service,
            "can_start_training": self.can_start_training,
            "reasons": list(self.reasons),
        }


@dataclass
class ServiceRecord:
    name: str
    url: str | None = None
    owned: bool = False
    state: ServiceState = ServiceState.UNKNOWN
    pid: int | None = None
    model_id: str | None = None
    adapter_sha256: str | None = None
    last_health: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "owned": self.owned,
            "state": self.state.value,
            "pid": self.pid,
            "model_id": self.model_id,
            "adapter_sha256": self.adapter_sha256,
            "last_health": dict(self.last_health),
            "note": self.note,
        }


@dataclass(frozen=True)
class LifecycleDecision:
    service: str
    allowed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "allowed": self.allowed,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class DegradationDecision:
    enabled: tuple[str, ...]
    disabled: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": list(self.enabled), "disabled": list(self.disabled)}


class ServiceLifecycleManager:
    def __init__(
        self,
        budget: ResourceBudget | None = None,
        *,
        thresholds: ResourceThresholds | None = None,
        snapshot_provider: Callable[[], dict[str, Any]] | None = None,
        health_provider: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.budget = budget or ResourceBudget()
        self.thresholds = thresholds or ResourceThresholds()
        self._snapshot = snapshot_provider
        self._health = health_provider
        self._services: dict[str, ServiceRecord] = {}

    # ------------------------------------------------------------ registration

    def register(
        self,
        name: str,
        *,
        url: str | None = None,
        owned: bool = False,
        pid: int | None = None,
        model_id: str | None = None,
        adapter_sha256: str | None = None,
        state: ServiceState | None = None,
        note: str = "",
    ) -> ServiceRecord:
        record = ServiceRecord(
            name=name,
            url=url,
            owned=owned,
            pid=pid,
            model_id=model_id,
            adapter_sha256=adapter_sha256,
            state=state or ServiceState.UNKNOWN,
            note=note,
        )
        self._services[name] = record
        return record

    def get(self, name: str) -> ServiceRecord | None:
        return self._services.get(name)

    def snapshot(self) -> dict[str, Any]:
        if self._snapshot is None:
            return {}
        return self._snapshot()

    # ----------------------------------------------------------------- health

    def health_check(self, name: str) -> dict[str, Any]:
        record = self._services.get(name)
        if record is None:
            return {"service": name, "ok": False, "state": ServiceState.UNKNOWN.value,
                    "error": "service is not registered"}
        if self._health is None or not record.url:
            result = {"service": name, "ok": False, "state": ServiceState.UNKNOWN.value,
                      "error": "no health provider or URL"}
            record.last_health = result
            record.state = ServiceState.UNKNOWN
            return result
        payload = self._health(record.url)
        ok = bool(payload.get("ok"))
        record.last_health = payload
        record.state = (
            ServiceState.HEALTHY
            if ok
            else (ServiceState.DEGRADED if payload.get("status_code") else ServiceState.UNKNOWN)
        )
        return {"service": name, "ok": ok, "state": record.state.value, **payload}

    # ------------------------------------------------------------ start policy

    def can_start(
        self,
        name: str,
        *,
        snapshot: dict[str, Any] | None = None,
        model_id: str | None = None,
        adapter_sha256: str | None = None,
    ) -> LifecycleDecision:
        record = self._services.get(name)
        if record is not None and record.state is ServiceState.HEALTHY:
            if name == "runtime" and model_id and record.model_id and model_id != record.model_id:
                return LifecycleDecision(
                    name,
                    False,
                    (
                        f"another model is already loaded: {record.model_id}; "
                        f"refusing to load {model_id} concurrently",
                    ),
                )
            return LifecycleDecision(name, False, ("service is already healthy; do not load it twice",))

        if name == "runtime" and model_id:
            for other in self._services.values():
                if (
                    other.name != name
                    and other.state is ServiceState.HEALTHY
                    and other.model_id
                    and other.model_id != model_id
                ):
                    return LifecycleDecision(
                        name,
                        False,
                        (
                            f"model {other.model_id} is already loaded by {other.name}; "
                            f"refusing to load a second model {model_id}",
                        ),
                    )

        current = snapshot if snapshot is not None else self.snapshot()
        if name == "agent":
            runtime = self._services.get("runtime")
            if runtime is None or runtime.state is not ServiceState.HEALTHY:
                return LifecycleDecision(
                    name, False, ("agent requires a healthy runtime",)
                )

        decision = preflight(name, snapshot=current, thresholds=self.thresholds)
        reasons = list(decision.reasons)
        if name == "runtime":
            gpu = current.get("gpu") or {}
            free = int(gpu.get("free_mib") or 0) if gpu.get("available") else 0
            if free < self.thresholds.min_runtime_gpu_free_mib:
                reasons.append(
                    f"GPU free {free} MiB < {self.thresholds.min_runtime_gpu_free_mib} MiB"
                )
            if model_id and adapter_sha256:
                # Identity is recorded for the caller; a changed adapter is a
                # different load and must not reuse the current process.
                if record and record.adapter_sha256 and record.adapter_sha256 != adapter_sha256:
                    reasons.append("adapter hash changed; the loaded model is not the requested one")
        allowed = decision.allowed and not reasons
        return LifecycleDecision(name, allowed, tuple(dict.fromkeys(reasons)))

    def startup_plan(self, requested: Iterable[str]) -> list[dict[str, Any]]:
        wanted = set(requested)
        plan: list[dict[str, Any]] = []
        for name in STARTUP_ORDER:
            if name not in wanted:
                continue
            missing = [
                dependency
                for dependency in _DEPENDENCIES[name]
                if dependency in self._services
                and self._services[dependency].state is not ServiceState.HEALTHY
            ]
            plan.append(
                {
                    "service": name,
                    "depends_on": list(_DEPENDENCIES[name]),
                    "blocked_by": missing,
                }
            )
        return plan

    # --------------------------------------------------------- degradation

    def degradation_plan(
        self, *, snapshot: dict[str, Any] | None = None
    ) -> DegradationDecision:
        current = snapshot if snapshot is not None else self.snapshot()
        budget = self.budget.evaluate(current)
        enabled: list[str] = []
        disabled: list[dict[str, str]] = []

        budget_ok = budget.checks["ram_ok"] and budget.checks["commit_ok"]
        runtime_record = self._services.get("runtime")
        if runtime_record is not None and runtime_record.state is ServiceState.HEALTHY:
            # Never propose stopping a service the user is already using.
            enabled.append("runtime")
        elif budget_ok:
            enabled.append("runtime")
        else:
            disabled.append({"service": "runtime", "reason": "RAM/commit budget insufficient"})

        hindsight_record = self._services.get("hindsight")
        if hindsight_record is not None and hindsight_record.state is ServiceState.HEALTHY:
            enabled.append("hindsight")
        elif budget_ok:
            enabled.append("hindsight")
        else:
            disabled.append({"service": "hindsight", "reason": "RAM budget insufficient"})

        rag_record = self._services.get("rag")
        if rag_record is not None and rag_record.state is ServiceState.HEALTHY:
            enabled.append("rag")
        elif budget_ok and current.get("rag_index_present", False):
            enabled.append("rag")
        else:
            reason = (
                "RAG index missing"
                if not current.get("rag_index_present", False)
                else "RAM/commit budget insufficient"
            )
            disabled.append({"service": "rag", "reason": reason})

        runtime = self._services.get("runtime")
        if runtime is not None and runtime.state is ServiceState.HEALTHY:
            enabled.append("agent")
        else:
            disabled.append({"service": "agent", "reason": "runtime is not healthy"})

        return DegradationDecision(tuple(enabled), tuple(disabled))

    # ------------------------------------------------------------- training

    def training_release_plan(
        self, *, snapshot: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return a plan only; never stop a process.

        Stopping a user-owned service requires explicit confirmation.  The plan
        lists the runtime first because it holds the GPU allocation.
        """
        current = snapshot if snapshot is not None else self.snapshot()
        budget = self.budget.evaluate(current)
        stop_order: list[dict[str, Any]] = []
        for name in ("runtime", "rag", "hindsight"):
            record = self._services.get(name)
            if record is None or record.state in {ServiceState.STOPPED, ServiceState.NOT_STARTED}:
                continue
            stop_order.append(
                {
                    "service": name,
                    "owned": record.owned,
                    "pid": record.pid,
                    "requires_user_confirmation": True,
                    "reason": "free GPU/RAM before a training run",
                }
            )
        return {
            "requires_user_confirmation": True,
            "stop_order": stop_order,
            "budget": budget.to_dict(),
            "training_allowed_by_budget": budget.can_start_training,
            "blockers": list(budget.reasons) if not budget.can_start_training else [],
            "note": "This is a plan only. The manager never stops an external process.",
        }

    # --------------------------------------------------------------- marking

    def mark_started(self, name: str, *, pid: int | None = None) -> None:
        record = self._services[name]
        if not record.owned:
            raise PermissionError(
                f"service {name} is not owned by this manager; refusing to mark it started"
            )
        record.state = ServiceState.STARTING
        if pid is not None:
            record.pid = pid

    def mark_stopped(self, name: str) -> None:
        record = self._services[name]
        if not record.owned:
            raise PermissionError(
                f"service {name} is external; the manager must not mark or stop it"
            )
        record.state = ServiceState.STOPPED

    def to_dict(self) -> dict[str, Any]:
        return {
            "services": {name: record.to_dict() for name, record in self._services.items()},
            "budget": self.budget.__dict__,
        }


__all__ = [
    "BudgetReport",
    "DegradationDecision",
    "LifecycleDecision",
    "ResourceBudget",
    "ServiceLifecycleManager",
    "ServiceRecord",
    "ServiceState",
    "STARTUP_ORDER",
]