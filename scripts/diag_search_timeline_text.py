from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

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


def _iter_stringish_props(obj: Any) -> Iterable[tuple[str, str]]:
    # Best-effort scan for string properties (shallow).
    try:
        props = obj.properties()
        if hasattr(props, "items"):
            for k, v in props.items():
                try:
                    vv = getattr(v, "value", v)
                except Exception:
                    vv = v
                if isinstance(vv, str):
                    yield str(k), vv
            return
    except Exception:
        pass
    for k in ("Name", "Comment", "MobName"):
        try:
            vv = obj[k].value
            if isinstance(vv, str):
                yield k, vv
        except Exception:
            pass


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
            yield slot, seg
            continue
        if "OperationGroup" in nm:
            try:
                inner = getattr(seg, "segments", None)
                if inner and len(inner) >= 1 and "Sequence" in inner[0].__class__.__name__:
                    yield slot, inner[0]
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aaf", type=Path, required=True)
    ap.add_argument("--needle", type=str, required=True)
    ap.add_argument("--max", type=int, default=50)
    args = ap.parse_args()

    needle = args.needle
    aaf_path = Path(args.aaf).resolve()
    hits = 0
    with open_aaf_lenient(aaf_path, "r") as aaf:
        comps = list(aaf.content.compositionmobs())
        if not comps:
            print("no comps")
            return 1
        comp = comps[0]
        lanes = list(_iter_sequences(comp))
        for lane_idx, (_slot, seq) in enumerate(lanes):
            vec = seq.components
            try:
                n = len(vec)
            except Exception:
                continue
            pos = 0
            for i in range(n):
                node = vec[i]
                ln = max(0, _node_len(node))
                T = pos
                pos += ln
                # only scan blocks that are not filler
                if "Filler" in node.__class__.__name__:
                    continue
                found = []
                for k, v in _iter_stringish_props(node):
                    if needle in v:
                        found.append((k, v))
                if found:
                    hits += 1
                    print(f"HIT#{hits}: lane={lane_idx} idx={i} T={T} L={ln} type={node.__class__.__name__}")
                    for k, v in found[:10]:
                        print(f"  {k}: {v}")
                    if hits >= int(args.max):
                        return 0
    print("hits", hits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
