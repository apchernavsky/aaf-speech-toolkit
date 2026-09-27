from __future__ import annotations

import gc
import os
import shutil
import stat
import time
from pathlib import Path



def _rmtree_onerror(func, path, _exc_info) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        return
    try:
        func(path)
    except Exception:
        return


def prune_empty_parents(start: Path, *, stop_at: Path) -> None:
    """
    Remove empty directories upward from ``start`` until ``stop_at`` exclusive.
    """
    try:
        cur = Path(start).resolve()
        stop = Path(stop_at).resolve()
    except Exception:
        return
    while True:
        if cur == stop or cur.name == "__aaf_tool_work":
            return
        try:
            if cur.is_dir() and not any(cur.iterdir()):
                cur.rmdir()
            else:
                return
        except Exception:
            return
        nxt = cur.parent
        if nxt == cur:
            return
        cur = nxt


def robust_rmtree(path: Path, *, retries: int = 80, delay_sec: float = 0.1) -> bool:
    """
    Best-effort recursive delete for Windows-heavy temp trees.
    """
    target = Path(path)
    for _ in range(max(1, int(retries))):
        try:
            if not target.exists():
                return True
            gc.collect()
            shutil.rmtree(target, onerror=_rmtree_onerror)
        except Exception:
            pass
        if not target.exists():
            return True
        time.sleep(max(0.0, float(delay_sec)))
    return not target.exists()


def robust_rmtree_with_windows_fallback(path: Path, *, retries: int = 80, delay_sec: float = 0.1) -> bool:
    """Legacy public name; cleanup never interprets paths through a shell."""
    return robust_rmtree(path, retries=retries, delay_sec=delay_sec)


def cleanup_work_dir(work_dir: Path, *, input_parent: Path) -> None:
    """
    Delete one pipeline run child, retaining its shared allocation root.
    """
    target = Path(work_dir).resolve()
    parent = Path(input_parent).resolve()
    if target in (parent, parent / "__aaf_tool_work") or not target.is_relative_to(parent):
        raise ValueError(f"Refusing cleanup outside the run parent: {target}")
    if not robust_rmtree(target):
        raise OSError(f"Could not remove owned workspace: {target}")
    # The common root has process-shared lifetime; each run owns its child only.


def cleanup_aaf_tool_work_root(input_parent: Path) -> None:
    """Keep the shared allocation root stable while other runs may start.

    Compatibility entry point: run-owned children are removed by cleanup_work_dir.
    """
    return None
