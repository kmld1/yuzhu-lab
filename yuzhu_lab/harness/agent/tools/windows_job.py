"""Manage an owned Windows subprocess and its descendants via an anonymous Job."""

from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
import msvcrt
import subprocess
import time

_CREATE_SUSPENDED = 0x00000004
_KILL_ON_JOB_CLOSE = 0x00002000
_EXTENDED_LIMITS = 9
_BASIC_ACCOUNTING = 1
_SNAPTHREAD = 0x00000004
_THREAD_SUSPEND_RESUME = 0x0002
_THREAD_QUERY_LIMITED_INFORMATION = 0x0800
_INVALID_HANDLE = C.c_void_p(-1).value


class _BasicLimits(C.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", C.c_int64),
        ("PerJobUserTimeLimit", C.c_int64),
        ("LimitFlags", W.DWORD),
        ("MinimumWorkingSetSize", C.c_size_t),
        ("MaximumWorkingSetSize", C.c_size_t),
        ("ActiveProcessLimit", W.DWORD),
        ("Affinity", C.c_size_t),
        ("PriorityClass", W.DWORD),
        ("SchedulingClass", W.DWORD),
    ]


class _ExtendedLimits(C.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", C.c_uint64 * 6),
        ("ProcessMemoryLimit", C.c_size_t),
        ("JobMemoryLimit", C.c_size_t),
        ("PeakProcessMemoryUsed", C.c_size_t),
        ("PeakJobMemoryUsed", C.c_size_t),
    ]


class _Accounting(C.Structure):
    _fields_ = [
        ("Times", C.c_int64 * 4),
        ("TotalPageFaultCount", W.DWORD),
        ("TotalProcesses", W.DWORD),
        ("ActiveProcesses", W.DWORD),
        ("TotalTerminatedProcesses", W.DWORD),
    ]


class _ThreadEntry(C.Structure):
    _fields_ = [
        ("dwSize", W.DWORD),
        ("cntUsage", W.DWORD),
        ("th32ThreadID", W.DWORD),
        ("th32OwnerProcessID", W.DWORD),
        ("tpBasePri", W.LONG),
        ("tpDeltaPri", W.LONG),
        ("dwFlags", W.DWORD),
    ]


_kernel = C.WinDLL("kernel32", use_last_error=True)


def _api(name, restype, *argtypes):
    function = getattr(_kernel, name)
    function.restype = restype
    function.argtypes = argtypes
    return function


_create_job = _api("CreateJobObjectW", W.HANDLE, C.c_void_p, W.LPCWSTR)
_set_job = _api("SetInformationJobObject", W.BOOL, W.HANDLE, C.c_int, C.c_void_p, W.DWORD)
_assign_job = _api("AssignProcessToJobObject", W.BOOL, W.HANDLE, W.HANDLE)
_terminate_job = _api("TerminateJobObject", W.BOOL, W.HANDLE, W.UINT)
_query_job = _api("QueryInformationJobObject", W.BOOL, W.HANDLE, C.c_int,
                  C.c_void_p, W.DWORD, C.POINTER(W.DWORD))
_close_handle = _api("CloseHandle", W.BOOL, W.HANDLE)
_snapshot = _api("CreateToolhelp32Snapshot", W.HANDLE, W.DWORD, W.DWORD)
_first_thread = _api("Thread32First", W.BOOL, W.HANDLE, C.POINTER(_ThreadEntry))
_next_thread = _api("Thread32Next", W.BOOL, W.HANDLE, C.POINTER(_ThreadEntry))
_open_thread = _api("OpenThread", W.HANDLE, W.DWORD, W.BOOL, W.DWORD)
_thread_owner = _api("GetProcessIdOfThread", W.DWORD, W.HANDLE)
_resume_thread = _api("ResumeThread", W.DWORD, W.HANDLE)
_peek_pipe = _api("PeekNamedPipe", W.BOOL, W.HANDLE, C.c_void_p, W.DWORD,
                  C.c_void_p, C.POINTER(W.DWORD), C.c_void_p)


def _winerror(operation: str) -> OSError:
    code = C.get_last_error()
    return OSError(f"{operation} failed (WinError {code})")


def _resume_primary_thread(pid: int) -> None:
    # Popen keeps the process handle but closes the creation thread handle.
    # The suspended child has never executed user code or spawned descendants.
    snapshot = _snapshot(_SNAPTHREAD, 0)
    if snapshot in (None, _INVALID_HANDLE):
        raise _winerror("CreateToolhelp32Snapshot")
    try:
        entry = _ThreadEntry()
        entry.dwSize = C.sizeof(entry)
        found = _first_thread(snapshot, C.byref(entry))
        while found and entry.th32OwnerProcessID != pid:
            entry.dwSize = C.sizeof(entry)
            found = _next_thread(snapshot, C.byref(entry))
        if not found:
            raise _winerror("locating suspended process thread")
        thread = _open_thread(_THREAD_SUSPEND_RESUME | _THREAD_QUERY_LIMITED_INFORMATION,
                              False, entry.th32ThreadID)
        if not thread:
            raise _winerror("OpenThread")
        try:
            if _thread_owner(thread) != pid:
                raise OSError("suspended process thread no longer belongs to the child")
            if _resume_thread(thread) != 1:
                raise OSError("could not resume the suspended process exactly once")
        finally:
            _close_handle(thread)
    finally:
        _close_handle(snapshot)


class WindowsProcessJob:
    creation_flags = _CREATE_SUSPENDED | subprocess.CREATE_NEW_PROCESS_GROUP

    def __init__(self) -> None:
        self.assigned = False
        self.handle = _create_job(None, None)
        if not self.handle:
            raise _winerror("CreateJobObjectW")
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
        if not _set_job(self.handle, _EXTENDED_LIMITS, C.byref(limits), C.sizeof(limits)):
            error = _winerror("SetInformationJobObject")
            self.close()
            raise error

    def start(self, process: subprocess.Popen) -> None:
        # The CPython-owned handle identifies the child across PID reuse.
        # No handle to this job is inheritable and no breakaway flag is set.
        if not _assign_job(self.handle, int(process._handle)):
            raise _winerror("AssignProcessToJobObject")
        self.assigned = True
        _resume_primary_thread(process.pid)

    def active_count(self) -> int:
        info = _Accounting()
        if not _query_job(self.handle, _BASIC_ACCOUNTING, C.byref(info), C.sizeof(info), None):
            raise _winerror("QueryInformationJobObject")
        return int(info.ActiveProcesses)

    def finish(self, deadline: float) -> None:
        if not self.active_count():
            return
        if not _terminate_job(self.handle, 1):
            error = _winerror("TerminateJobObject")
            if self.active_count():
                raise error
            return
        while self.active_count():
            if time.monotonic() >= deadline:
                raise TimeoutError("Windows job still has active processes after termination")
            time.sleep(0.01)

    def close(self) -> None:
        if self.handle and not _close_handle(self.handle):
            raise _winerror("CloseHandle(job)")
        self.handle = None


def pipe_available(stream) -> int | None:
    """Return buffered bytes, 0 for an open empty pipe, or None at EOF."""
    available = W.DWORD()
    handle = msvcrt.get_osfhandle(stream.fileno())
    if _peek_pipe(handle, None, 0, None, C.byref(available), None):
        return int(available.value)
    if C.get_last_error() in (109, 232, 233):  # broken pipe, no data, disconnected
        return None
    raise _winerror("PeekNamedPipe")

