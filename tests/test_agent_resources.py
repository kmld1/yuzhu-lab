"""Service health and resource-aware degradation tests."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from tests.agent_test_utils import TempWorkspace
from yuzhu_lab.harness.agent import resources


class ResourceDegradationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = TempWorkspace()
        self.addCleanup(self.workspace.close)
        self.store = self.workspace.path / "rag_store"

    def test_unreachable_services_degrade(self) -> None:
        report = resources.check_services(
            runtime_url="http://127.0.0.1:1",
            hindsight_url="http://127.0.0.1:1",
            rag_store=self.store,
        )
        self.assertFalse(report.runtime["ok"])
        self.assertFalse(report.hindsight["ok"])
        self.assertFalse(report.memory_enabled)
        self.assertFalse(report.rag_enabled)
        self.assertTrue(report.notes)

    def test_rag_disabled_when_ram_is_low(self) -> None:
        self.store.mkdir()
        (self.store / "manifest.json").write_text("{}", encoding="utf-8")
        (self.store / "index.faiss").write_text("x", encoding="utf-8")
        with mock.patch.object(resources, "http_health", return_value={"ok": True}), mock.patch.object(
            resources, "gpu_snapshot", return_value={"available": False}
        ), mock.patch.object(
            resources, "ram_available_bytes", return_value=int(1.0 * resources.GIB)
        ):
            report = resources.check_services(
                runtime_url="http://x",
                hindsight_url="http://x",
                rag_store=self.store,
                min_ram_gib=1.5,
            )
        self.assertTrue(report.rag_index_present)
        self.assertFalse(report.rag_enabled)
        self.assertTrue(
            any("RAM" in note and "RAG" in note for note in report.notes)
        )

    def test_rag_enabled_with_enough_ram_and_index(self) -> None:
        self.store.mkdir()
        (self.store / "manifest.json").write_text("{}", encoding="utf-8")
        (self.store / "index.faiss").write_text("x", encoding="utf-8")
        with mock.patch.object(resources, "http_health", return_value={"ok": True}), mock.patch.object(
            resources, "gpu_snapshot", return_value={"available": False}
        ), mock.patch.object(
            resources, "ram_available_bytes", return_value=int(4.0 * resources.GIB)
        ):
            report = resources.check_services(
                runtime_url="http://x",
                hindsight_url="http://x",
                rag_store=self.store,
                min_ram_gib=1.5,
            )
        self.assertTrue(report.rag_enabled)
        self.assertTrue(report.memory_enabled)
        self.assertEqual(report.notes, [])

    def test_memory_disabled_when_hindsight_unhealthy(self) -> None:
        self.store.mkdir()
        (self.store / "manifest.json").write_text("{}", encoding="utf-8")
        (self.store / "index.faiss").write_text("x", encoding="utf-8")

        def fake_health(url: str, timeout: float = 3.0) -> dict:
            return {"ok": "8890" in url}

        with mock.patch.object(resources, "http_health", side_effect=fake_health), mock.patch.object(
            resources, "gpu_snapshot", return_value={"available": False}
        ), mock.patch.object(
            resources, "ram_available_bytes", return_value=int(4.0 * resources.GIB)
        ):
            report = resources.check_services(
                runtime_url="http://127.0.0.1:8710",
                hindsight_url="http://127.0.0.1:8890",
                rag_store=self.store,
                min_ram_gib=1.5,
            )
        self.assertTrue(report.memory_enabled)
        self.assertTrue(report.rag_enabled)
        self.assertFalse(report.runtime["ok"])
        self.assertIn("runtime", " ".join(report.notes).lower())


class ResourcePreflightTest(unittest.TestCase):
    def test_runtime_requires_gpu_headroom(self) -> None:
        snapshot = {
            "ram_available_gib": 4.0,
            "gpu": {"available": True, "free_mib": 100},
            "runtime": {"ok": False},
        }
        decision = resources.preflight("runtime", snapshot=snapshot)
        self.assertFalse(decision.allowed)
        self.assertTrue(any("GPU free" in reason for reason in decision.reasons))

        snapshot["gpu"]["free_mib"] = 6000
        decision = resources.preflight("runtime", snapshot=snapshot)
        self.assertTrue(decision.allowed)

    def test_runtime_already_healthy_is_allowed(self) -> None:
        decision = resources.preflight(
            "runtime",
            snapshot={"runtime": {"ok": True}, "gpu": {"available": False}},
        )
        self.assertTrue(decision.allowed)

    def test_rag_requires_ram_and_index(self) -> None:
        snapshot = {
            "ram_available_gib": 1.0,
            "rag_index_present": True,
            "runtime": {"ok": True},
        }
        decision = resources.preflight("rag", snapshot=snapshot)
        self.assertFalse(decision.allowed)
        snapshot["ram_available_gib"] = 3.0
        decision = resources.preflight("rag", snapshot=snapshot)
        self.assertTrue(decision.allowed)

    def test_agent_requires_healthy_runtime(self) -> None:
        decision = resources.preflight(
            "agent",
            snapshot={"runtime": {"ok": False}, "ram_available_gib": 3.0},
        )
        self.assertFalse(decision.allowed)
        decision = resources.preflight(
            "agent",
            snapshot={"runtime": {"ok": True}, "ram_available_gib": 3.0},
        )
        self.assertTrue(decision.allowed)

    def test_unknown_service_is_denied(self) -> None:
        decision = resources.preflight("unknown", snapshot={})
        self.assertFalse(decision.allowed)

    def test_full_snapshot_shape(self) -> None:
        snapshot = resources.full_snapshot()
        self.assertIn("memory", snapshot)
        self.assertIn("gpu", snapshot)
        self.assertIn("processes", snapshot)
        self.assertIsInstance(snapshot["processes"], list)


class GpuQueryRobustnessTest(unittest.TestCase):
    """nvidia-smi output must not crash diagnostics when decoding fails."""

    def test_gpu_snapshot_handles_missing_stdout(self) -> None:
        completed = mock.Mock(returncode=0, stdout=None, stderr=None)
        with mock.patch.object(
            resources.shutil, "which", return_value="nvidia-smi"
        ), mock.patch.object(resources.subprocess, "run", return_value=completed) as run:
            snapshot = resources.gpu_snapshot()
        self.assertFalse(snapshot["available"])
        self.assertIn("unexpected", snapshot["reason"])
        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(run.call_args.kwargs["errors"], "replace")

    def test_gpu_processes_handles_missing_stdout(self) -> None:
        completed = mock.Mock(returncode=0, stdout=None, stderr=None)
        with mock.patch.object(
            resources.shutil, "which", return_value="nvidia-smi"
        ), mock.patch.object(resources.subprocess, "run", return_value=completed) as run:
            processes = resources.gpu_processes()
        self.assertEqual(processes, [])
        self.assertEqual(run.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(run.call_args.kwargs["errors"], "replace")


if __name__ == "__main__":
    unittest.main()