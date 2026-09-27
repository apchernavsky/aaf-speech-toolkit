"""Subprocess helpers that avoid flashing console windows on Windows."""

from __future__ import annotations

import os
import subprocess
from typing import Any


def _with_hidden_window(kwargs: dict[str, Any]) -> dict[str, Any]:
    if os.name != "nt":
        return kwargs

    updated = dict(kwargs)
    updated["creationflags"] = int(updated.get("creationflags") or 0) | int(
        getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )

    startupinfo = updated.get("startupinfo")
    if startupinfo is None:
        startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    updated["startupinfo"] = startupinfo
    return updated


def run_hidden(*popenargs: Any, **kwargs: Any) -> subprocess.CompletedProcess:
    """Run a child process without opening a new console window on Windows."""
    return subprocess.run(*popenargs, **_with_hidden_window(kwargs))


def popen_hidden(*popenargs: Any, **kwargs: Any) -> subprocess.Popen:
    """Start a child process without opening a new console window on Windows."""
    return subprocess.Popen(*popenargs, **_with_hidden_window(kwargs))
