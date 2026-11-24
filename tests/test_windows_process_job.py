"""Windows-native process lifecycle regression, using only owned fixtures."""

import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from yuzhu_lab.harness.agent.errors import ToolExecutionError
from yuzhu_lab.harness.agent.tools import common

if os.name == "nt":
    import ctypes
    from ctypes import wintypes
    import subprocess

    from yuzhu_lab.harness.agent.tools import windows_job

    _kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel.OpenProcess.restype = wintypes.HANDLE
    _kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4))
    _kernel.GetProcessTimes.restype = wintypes.BOOL
    _kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    _kernel.WaitForSingleObject.restype = wintypes.DWORD
    _kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel.CloseHandle.restype = wintypes.BOOL
    _kernel.GetCurrentProcess.argtypes = ()
    _kernel.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel.GetProcessHandleCount.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    _kernel.GetProcessHandleCount.restype = wintypes.BOOL

_TREE_SCRIPT = """
from pathlib import Path
import os, subprocess, sys, time
root = Path(sys.argv[1])
stage = int(sys.argv[2])
depth = int(sys.argv[3])
mode = sys.argv[4]
print(f"stdout-level-{stage}", flush=True)
print(f"stderr-level-{stage}", file=sys.stderr, flush=True)
temp = root / f"{stage}.pending"
temp.write_text(str(os.getpid()), encoding="ascii")
temp.replace(root / f"{stage}.pid")
if stage < depth:
    subprocess.Popen(
        [sys.executable, "-B", __file__, str(root), str(stage+1), str(depth), mode],
        stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr, close_fds=True,
    )
if stage == 0 and mode != "wait":
    deadline = time.monotonic() + 3
    while not (root / "observed").exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    sys.exit(7 if mode == "error" else 0)
time.sleep(8)
"""


@unittest.skipUnless(os.name == "nt", "Windows native process accounting required")
class WindowsProcessJobTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.handles = {}
        self.addCleanup(self._close_handles)

    def _close_handles(self) -> None:
        for handle, _, _ in self.handles.values():
            _kernel.CloseHandle(handle)
        self.handles.clear()

    def _observe(self, depth: int) -> None:
        for stage in range(depth + 1):
            if stage in self.handles:
                continue
            path = self.root / f"{stage}.pid"
            if not path.exists():
                continue
            pid = int(path.read_text(encoding="ascii"))
            handle = _kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
            if not handle:
                continue
            timestamps = [wintypes.FILETIME() for _ in range(4)]
            if not _kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in timestamps)):
                _kernel.CloseHandle(handle)
                continue
            birth = (timestamps[0].dwHighDateTime << 32) | timestamps[0].dwLowDateTime
            self.handles[stage] = (handle, pid, birth)

    def _fixture(self, depth: int, mode: str) -> list[str]:
        script = self.root / "fixture.py"
        script.write_text(_TREE_SCRIPT, encoding="utf-8")
        return [sys.executable, "-B", str(script), str(self.root), "0", str(depth), mode]

    def _assert_terminated(self, depth: int) -> str:
        self.assertEqual(set(self.handles), set(range(depth + 1)), self.handles.keys())
        parts = []
        for handle, pid, birth in self.handles.values():
            state = _kernel.WaitForSingleObject(handle, 500)
            self.assertEqual(state, 0, f"owned process {pid} (FILETIME {birth}) still running")
            parts.append(f"{pid}@{birth}:SIGNALED")
        return ",".join(parts)

    def _run_tree(self, depth: int, mode: str, *, cancel_when_ready: bool = False):
        timeout = 1 if mode == "wait" and not cancel_when_ready else 4
        argv = self._fixture(depth, mode)

        def check():
            self._observe(depth)
            if len(self.handles) == depth + 1:
                (self.root / "observed").touch()
                return cancel_when_ready
            return False

        result = common.run_process(argv, cwd=self.root, timeout=timeout, cancel=check)
        self.assertEqual(set(self.handles), set(range(depth + 1)), result.combined())
        identities = self._assert_terminated(depth)
        self.assertFalse(result.read_incomplete, result.reader_errors)
        self.assertIsNotNone(result.raw_returncode)
        self.assertLess(result.duration_ms, (timeout + 3) * 1000)
        for stage in range(depth + 1):
            self.assertIn(f"stdout-level-{stage}", result.stdout)
            self.assertIn(f"stderr-level-{stage}", result.stderr)
        print(
            f"NATIVE_JOB {self._testMethodName} timeout={timeout} "
            f"elapsed_ms={result.duration_ms:.1f} raw_exit={result.raw_returncode} "
            f"timed_out={result.timed_out} cancelled={result.cancelled} "
            f"pipes_complete=True identities={identities}",
            flush=True,
        )
        return result

    def test_short_task_with_isolated_stdin(self):
        result = common.run_process(
            [sys.executable, "-B", "-c", "import sys;print(len(sys.stdin.buffer.read(1)))"],
            cwd=self.root, timeout=1,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout.strip(), "0")
        self.assertFalse(result.read_incomplete)

    def test_one_second_direct_timeout_does_not_use_taskkill(self):
        with patch.object(subprocess, "run", side_effect=AssertionError("unexpected taskkill dependency")):
            result = self._run_tree(0, "wait")
        self.assertTrue(result.timed_out)
        self.assertIsNone(result.exit_code)

    def test_timeout_terminates_single_descendant_holding_pipes(self):
        self.assertTrue(self._run_tree(1, "wait").timed_out)

    def test_timeout_terminates_multiple_descendants_and_spares_unrelated_process(self):
        unrelated = subprocess.Popen(
            [sys.executable, "-B", "-c", "import time;time.sleep(8)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            self.assertTrue(self._run_tree(2, "wait").timed_out)
            self.assertIsNone(unrelated.poll())
            print(f"NATIVE_JOB unrelated_control_pid={unrelated.pid} survived=True", flush=True)
        finally:
            if unrelated.poll() is None:
                unrelated.kill()
            unrelated.wait(timeout=3)

    def test_cancel_terminates_tree_then_allows_next_command(self):
        result = self._run_tree(2, "wait", cancel_when_ready=True)
        self.assertTrue(result.cancelled)
        self.assertFalse(result.timed_out)
        self.test_short_task_with_isolated_stdin()

    def test_normal_root_exit_cleans_up_inherited_pipes(self):
        result = self._run_tree(2, "exit")
        self.assertEqual(result.exit_code, 0)

    def test_nonzero_root_exit_still_cleans_up_descendants(self):
        result = self._run_tree(2, "error")
        self.assertEqual(result.exit_code, 7)

    def test_repeated_timeouts_reap_roots_and_do_not_leak_handles(self):
        def count_handles():
            count = wintypes.DWORD()
            self.assertTrue(_kernel.GetProcessHandleCount(_kernel.GetCurrentProcess(), ctypes.byref(count)))
            return count.value

        # Warm up the interpreter's process/pipe path before measuring handles.
        self.test_short_task_with_isolated_stdin()
        before = count_handles()
        for _ in range(4):
            result = common.run_process(
                [sys.executable, "-B", "-c", "import time;time.sleep(8)"],
                cwd=self.root, timeout=0.25,
            )
            self.assertTrue(result.timed_out)
            self.assertIsNotNone(result.raw_returncode)
            self.assertFalse(result.read_incomplete)
        after = count_handles()
        print(f"NATIVE_JOB handle_count_before={before} handle_count_after={after}", flush=True)
        self.assertEqual(after, before)

    def test_assignment_failure_never_runs_unmanaged_payload(self):
        marker = self.root / "ran"
        argv = [sys.executable, "-B", "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]
        with patch.object(windows_job, "_assign_job", return_value=0):
            with self.assertRaises(ToolExecutionError) as error:
                common.run_process(argv, cwd=self.root, timeout=1)
        self.assertEqual(error.exception.code, "PROCESS_SETUP_FAILED")
        self.assertFalse(marker.exists())
        self.assertEqual(common.run_process(argv, cwd=self.root, timeout=1).exit_code, 0)
        self.assertTrue(marker.exists())

    def test_unverified_cleanup_fails_closed_and_blocks_future_execution(self):
        argv = self._fixture(0, "wait")

        def check():
            self._observe(0)
            return bool(self.handles)

        with patch.object(common, "_CLEANUP_FAILED", threading.Event()):
            # The real job close still enforces kill-on-close in this failure path.
            with patch.object(windows_job, "_terminate_job", return_value=0):
                with self.assertRaises(ToolExecutionError) as error:
                    common.run_process(argv, cwd=self.root, timeout=2, cancel=check)
            self.assertEqual(error.exception.code, "PROCESS_CLEANUP_FAILED")
            self._assert_terminated(0)
            with patch.object(subprocess, "Popen", side_effect=AssertionError("unexpected launch")):
                with self.assertRaises(ToolExecutionError) as later:
                    common.run_process(argv, cwd=self.root, timeout=1)
            self.assertEqual(later.exception.code, "PROCESS_CLEANUP_FAILED")


    def test_cleanup_failure_stops_later_agent_tools_and_tasks(self):
        from tests.agent_test_utils import ScriptedModel, make_loop, tool_response

        script = self.root / "delay.py"
        script.write_text("import time\ntime.sleep(8)\n", encoding="utf-8")
        response = tool_response([
            ("run_python", {"script": "delay.py", "timeout_seconds": 1}),
            ("write_file", {"path": "should-not-exist.txt", "content": "blocked"}),
        ])
        model = ScriptedModel([response])
        loop, store, _ = make_loop(self.root, model, allow_execution=True)
        self.addCleanup(store.close)
        with patch.object(common, "_CLEANUP_FAILED", threading.Event()):
            with patch.object(windows_job, "_terminate_job", return_value=0):
                result = loop.run("Run the owned delay fixture.")
            self.assertEqual(result.status, "FAILED")
            self.assertIn("PROCESS_CLEANUP_FAILED", result.error or "")
            self.assertFalse((self.root / "should-not-exist.txt").exists())
            self.assertEqual(len(store.tool_calls("task-test")), 1)
            next_model = ScriptedModel([])
            later, later_store, _ = make_loop(self.root, next_model, task_id="later")
            self.addCleanup(later_store.close)
            result = later.run("Start another task.")
            self.assertEqual(result.status, "FAILED")
            self.assertIn("previous process cleanup could not be verified", result.error or "")
            self.assertEqual(next_model.calls, [])

    def test_stdio_run_tests_keeps_protocol_input_and_serves_next_request(self):
        import json

        from tests.agent_test_utils import make_policy
        from tests.test_h5_2_mcp_server import RecordingWriter, SilentLog
        from yuzhu_lab.harness.agent.registry import ToolRequest
        from yuzhu_lab.harness.agent.tools import build_default_registry
        from yuzhu_lab.harness.mcp_server import McpServer, McpTool

        (self.root / "test_stdin.py").write_text(
            "import sys, unittest\n"
            "class CheckStdin(unittest.TestCase):\n"
            "    def test_isolated(self):\n"
            "        self.assertEqual(sys.stdin.buffer.read(1), b'')\n",
            encoding="utf-8",
        )
        registry = build_default_registry(make_policy(self.root, allow_execution=True))

        def handler(arguments, cancel):
            request = ToolRequest("stdin", "run_tests", {
                "path": "test_stdin.py", "framework": "unittest", "timeout_seconds": 2,
            })
            result = registry.execute(request)
            return json.dumps({"passed": result.structured.get("passed"), "state": result.status.value})

        read_fd, write_fd = os.pipe()
        writer = RecordingWriter()
        with os.fdopen(read_fd, "rb", buffering=0) as reader:
            with os.fdopen(write_fd, "wb", buffering=0) as client:
                tool = McpTool("fixture_tests", "Check test stdin",
                               {"type": "object", "properties": {}}, handler, 4)
                server = McpServer([tool], reader=reader, writer=writer, log=SilentLog())
                active = threading.Thread(target=server.serve, daemon=True)
                active.start()

                def request(request_id, method, params=None):
                    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
                    if params is not None:
                        message["params"] = params
                    client.write(json.dumps(message).encode("utf-8") + b"\n")
                    return writer.response(request_id)

                try:
                    for request_id in (1, 2):
                        response = request(request_id, "tools/call", {"name": "fixture_tests", "arguments": {}})
                        self.assertEqual(
                            json.loads(response["result"]["content"][0]["text"]),
                            {"passed": True, "state": "OK"},
                        )
                    self.assertEqual(request(3, "ping")["result"], {})
                finally:
                    client.close()
                    active.join(timeout=5)
            self.assertFalse(active.is_alive())

