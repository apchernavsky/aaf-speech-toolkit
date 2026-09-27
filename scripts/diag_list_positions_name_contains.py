from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

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
from aaf_io.clip_naming import resolve_clip_display_name  # noqa: E402


def _node_len(node: Any) -> int:
    try:
        return int(getattr(node, "length", 0) or 0)
    except Exception:
        pass
    try:
        return int(node["Length"].value)
    except Exception:
        return 0


def _inner_sourceclip(node: Any) -> Optional[Any]:
    stack = [node]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        i = id(cur)
        if i in seen:
            continue
        seen.add(i)
        try:
            cn = cur.__class__.__name__
        except Exception:
            cn = ""
        if "SourceClip" in cn:
            return cur
        for attr in ("components", "segments"):
            ch = getattr(cur, attr, None)
            if not ch:
                continue
            try:
                n = len(ch)
            except Exception:
                continue
            for j in range(n):
                try:
                    stack.append(ch[j])
                except Exception:
                    pass
        try:
            ins = cur["InputSegments"]
            for j in range(len(ins)):
                try:
                    stack.append(ins[j])
                except Exception:
                    pass
        except Exception:
            pass
        sel = getattr(cur, "selected", None)
        if sel is not None:
            stack.append(sel)
    return None


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
    ap.add_argument("--max", type=int, default=200)
    args = ap.parse_args()

    aaf_path = Path(args.aaf).resolve()
    needle = args.needle

    hits = 0
    with open_aaf_lenient(aaf_path, "r") as aaf:
        comps = list(aaf.content.compositionmobs())
        if not comps:
            print("no comps")
            return 1
        comp = comps[0]
        cache = {}
        for lane_idx, (_slot, seq) in enumerate(_iter_sequences(comp)):
            top = seq.components
            pos = 0
            try:
                tn = len(top)
            except Exception:
                continue
            for ti in range(tn):
                node = top[ti]
                ln = _node_len(node)
                T = pos
                pos += max(0, ln)
                if "Filler" in node.__class__.__name__:
                    continue
                nm = resolve_clip_display_name(aaf, node, cache=cache).display_name or ""
                if needle not in nm:
                    continue
                sc = _inner_sourceclip(node)
                sid = st = sl = None
                if sc is not None:
                    try:
                        sid = sc["SourceID"].value
                    except Exception:
                        sid = getattr(sc, "source_id", None)
                    try:
                        st = sc["StartTime"].value
                    except Exception:
                        st = getattr(sc, "start", None)
                    try:
                        sl = sc["Length"].value
                    except Exception:
                        sl = getattr(sc, "length", None)
                hits += 1
                print(
                    f"HIT#{hits}: lane={lane_idx} idx={ti} T={T} L={ln} "
                    f"type={node.__class__.__name__} name={nm}"
                )
                if sid is not None:
                    print(f"  SourceID: {sid}")
                if st is not None or sl is not None:
                    print(f"  StartTime: {st}  Length: {sl}")
                if hits >= int(args.max):
                    return 0
    print("hits", hits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
