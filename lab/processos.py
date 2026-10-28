"""Processos no Windows sem dependências (ctypes): RAM (RSS), vivo?, matar."""
import ctypes
import os
import subprocess
from ctypes import wintypes

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_CONSULTA = 0x1000 | 0x0010          # PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ
_AINDA_ATIVO = 259


class _Memoria(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


_k32.OpenProcess.restype = wintypes.HANDLE
_k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_k32.GetCurrentProcess.restype = wintypes.HANDLE
_psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(_Memoria), wintypes.DWORD)
_k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
_k32.CloseHandle.argtypes = (wintypes.HANDLE,)


def rss_mb(pid=None, pico=False):
    """Working set (≈ RSS) em MB do processo pid (padrão: este); pico=True dá o máximo."""
    h = _k32.GetCurrentProcess() if pid is None else _k32.OpenProcess(_CONSULTA, False, pid)
    if not h:
        return None
    try:
        m = _Memoria()
        m.cb = ctypes.sizeof(m)
        if not _psapi.GetProcessMemoryInfo(h, ctypes.byref(m), m.cb):
            return None
        return round((m.PeakWorkingSetSize if pico else m.WorkingSetSize) / 2 ** 20, 1)
    finally:
        if pid is not None:
            _k32.CloseHandle(h)


def vivo(pid):
    h = _k32.OpenProcess(_CONSULTA, False, pid)
    if not h:
        return False
    try:
        cod = wintypes.DWORD()
        return bool(_k32.GetExitCodeProcess(h, ctypes.byref(cod))) and cod.value == _AINDA_ATIVO
    finally:
        _k32.CloseHandle(h)


def matar(pid, arvore=True):
    """Mata o processo à força (taskkill /F); com arvore=True, os filhos também (/T)."""
    subprocess.run(["taskkill", "/PID", str(pid), "/F"] + (["/T"] if arvore else []),
                   capture_output=True)


if __name__ == "__main__":
    print(os.getpid(), rss_mb(), rss_mb(pico=True), vivo(os.getpid()), vivo(999999))
