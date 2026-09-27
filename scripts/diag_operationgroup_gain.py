#!/usr/bin/env python3
"""
Inspect OperationGroup chains above SourceClips to detect gain/automation.

This is a heuristic dump: for each timeline SourceClip, if it is nested under an OperationGroup,
print the Operation name and any parameter-like fields we can read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable


def _ensure_toolkit_on_path() -> None:
    here = Path(__file__).resolve()
    toolkit = here.parents[1]
    aaf_io_root = toolkit / "aaf_io"
    aaf_speech_root = toolkit / "aaf_speech_filter"
    for p in (toolkit, aaf_io_root, aaf_speech_root):
        if p.is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)


_ensure_toolkit_on_path()

from aaf_io.converter import AAFConverter  # noqa: E402
from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient  # noqa: E402
from aaf_speech_filter.timeline_walk import (  # noqa: E402
    _is_sourceclip,
    _slot_edit_rate,
    _slot_is_soundish,
    _vectors_on_container,
)


def _iter_timeline_sourceclips_with_stack(aaf: Any):
    compositions = list(aaf.content.compositionmobs())

    def walk(container: Any, edit_rate: float, stack: list[Any]):
        if container is None:
            return
        stack2 = stack + [container]
        for vec in _vectors_on_container(container):
            i = 0
            while True:
                try:
                    n = len(vec)
                except Exception:
                    break
                if i >= n:
                    break
                try:
                    node = vec[i]
                except IndexError:
                    break
                if _is_sourceclip(node):
                    yield edit_rate, node, stack2
                else:
                    yield from walk(node, edit_rate, stack2)
                i += 1
        selected = getattr(container, "selected", None)
        if selected is not None:
            yield from walk(selected, edit_rate, stack2)

    for comp in compositions:
        for slot in getattr(comp, "slots", []) or []:
            if not _slot_is_soundish(slot):
                continue
            er = float(_slot_edit_rate(slot) or 48000.0)
            seg = getattr(slot, "segment", None)
            if seg is None:
                continue
            yield from walk(seg, er, stack=[comp, slot])


def _safe_str(v: Any, cap: int = 160) -> str:
    try:
        s = str(v)
    except Exception:
        s = repr(v)
    if len(s) > cap:
        s = s[:cap] + "…"
    return s


def _opgroup_summary(og: Any) -> str:
    op_name = ""
    try:
        op = og.operation
        op_name = _safe_str(getattr(op, "name", "") or op)
    except Exception:
        try:
            op_name = _safe_str(og["Operation"].value)
        except Exception:
            op_name = ""

    parts = [f"OperationGroup(op={op_name or '?'})"]
    # Try to dump parameters vector if present
    params = None
    for attr in ("parameters", "Parameters"):
        try:
            params = getattr(og, attr)
            if params is not None:
                break
        except Exception:
            params = None
    if params is None:
        try:
            params = og["Parameters"]
        except Exception:
            params = None

    if params is not None:
        try:
            n = len(params)
        except Exception:
            n = 0
        if n:
            parts.append(f"params={n}")
            # sample first few (prefer iteration over indexing; some vectors are not reliably indexable)
            sample: list[Any] = []
            try:
                for p in params:
                    sample.append(p)
                    if len(sample) >= 4:
                        break
            except Exception:
                sample = []
                for i in range(min(4, n)):
                    try:
                        sample.append(params[i])
                    except Exception:
                        continue

            for p in sample:
                try:
                    _ = p.__class__.__name__
                except Exception:
                    pass
                pname = ""
                try:
                    pname = _safe_str(getattr(p, "name", "") or "")
                except Exception:
                    pname = ""
                pdef = ""
                try:
                    d = getattr(p, "definition", None)
                    if d is not None:
                        pdef = _safe_str(getattr(d, "name", "") or d)
                except Exception:
                    pdef = ""

                # Try common AAF2 parameter shapes
                candidates: list[str] = []
                try:
                    candidates.append(f"type={p.__class__.__name__}")
                except Exception:
                    pass
                for expr in (
                    lambda: getattr(p, "value"),
                    lambda: getattr(p, "Value"),
                    lambda: p["Value"].value,
                    lambda: p["Value"],
                    lambda: p["Value"].data if hasattr(p["Value"], "data") else None,
                ):
                    try:
                        v = expr()
                        if v is None:
                            continue
                        candidates.append(_safe_str(v))
                    except Exception:
                        continue

                # VaryingValue may have a control_points list
                try:
                    cps = getattr(p, "control_points", None)
                    if cps is not None:
                        try:
                            cn = len(cps)
                        except Exception:
                            cn = 0
                        if cn:
                            candidates.append(f"control_points={cn}")
                            for j in range(min(3, cn)):
                                try:
                                    cp = cps[j]
                                except Exception:
                                    continue
                                try:
                                    candidates.append("  cp=" + _safe_str(cp))
                                except Exception:
                                    pass
                except Exception:
                    pass

                desc = pname or pdef or p.__class__.__name__
                if candidates:
                    parts.append(f"  - {desc}: " + " | ".join(candidates))
                else:
                    parts.append(f"  - {desc}: ?")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("aaf", type=Path)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    aaf_path = Path(args.aaf).resolve()
    if not aaf_path.is_file():
        print(f"AAF not found: {aaf_path}")
        return 2

    conv = AAFConverter(aaf_path)
    work_path = conv.output_dir / f"{aaf_path.stem}.__diag_opg_work{aaf_path.suffix}"
    try:
        work_path.unlink(missing_ok=True)
    except Exception:
        pass
    ok = conv.prepare_external_media_copy(work_path)
    if not ok or not work_path.is_file():
        print("prepare_external_media_copy failed")
        return 1

    printed = 0
    seen: set[int] = set()
    try:
        with open_aaf_lenient(work_path, "r") as aaf:
            for _er, _node, stack in _iter_timeline_sourceclips_with_stack(aaf):
                ogs = [x for x in stack if "OperationGroup" in x.__class__.__name__]
                if not ogs:
                    continue
                og = ogs[-1]
                oid = id(og)
                if oid in seen:
                    continue
                seen.add(oid)
                print(_opgroup_summary(og))
                print("---")
                printed += 1
                if printed >= int(args.limit):
                    break
    finally:
        try:
            conv._unlink_mapped_essence()
        except Exception:
            pass
        try:
            work_path.unlink(missing_ok=True)
        except Exception:
            pass

    print(f"\nPrinted unique OperationGroups: {printed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
