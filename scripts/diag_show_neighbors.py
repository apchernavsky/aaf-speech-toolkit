from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _ensure_toolkit_on_path() -> None:
    here = Path(__file__).resolve()
    toolkit = here.parents[1]
    for p in (toolkit, toolkit / "aaf_io"):
        if p.is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)


_ensure_toolkit_on_path()

from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient  # noqa: E402


def _node_len(node: Any) -> int:
    try:
        return int(getattr(node, "length", 0) or 0)
    except Exception:
        pass
    try:
        return int(node["Length"].value)
    except Exception:
        return 0


def _iter_sequences(comp: Any):
    for slot in getattr(comp, "slots", []) or []:
        seg = getattr(slot, "segment", None)
        if seg is None:
            continue
        nm = seg.__class__.__name__
        if "Sequence" in nm:
            yield seg
            continue
        if "OperationGroup" in nm:
            try:
                inner = getattr(seg, "segments", None)
                if inner and len(inner) >= 1 and "Sequence" in inner[0].__class__.__name__:
                    yield inner[0]
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aaf", type=Path, required=True)
    ap.add_argument("--lane", type=int, required=True)
    ap.add_argument("--center", type=int, required=True)
    ap.add_argument("--window", type=int, default=6)
    args = ap.parse_args()

    aaf_path = Path(args.aaf).resolve()
    with open_aaf_lenient(aaf_path, "r") as aaf:
        comps = list(aaf.content.compositionmobs())
        if not comps:
            print("no compositions")
            return 1
        comp = comps[0]
        lanes = list(_iter_sequences(comp))
        if not lanes:
            print("no sequences")
            return 1
        li = max(0, min(int(args.lane), len(lanes) - 1))
        seq = lanes[li]
        vec = seq.components
        n = len(vec)
        ci = max(0, min(int(args.center), n - 1))
        w = max(0, int(args.window))
        lo = max(0, ci - w)
        hi = min(n, ci + w + 1)

        print(f"== {aaf_path.name} lane={li} center={ci} window={w} ==")

        pos = 0
        for i in range(lo):
            pos += max(0, _node_len(vec[i]))
        for i in range(lo, hi):
            nd = vec[i]
            ln = max(0, _node_len(nd))
            print(f"idx={i:4d} T_naive={pos:10d} L={ln:8d} type={nd.__class__.__name__}")
            pos += ln

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
