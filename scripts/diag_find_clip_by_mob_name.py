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
    for p in (toolkit, toolkit / "aaf_io", toolkit / "aaf_speech_filter"):
        if p.is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)


_ensure_toolkit_on_path()

from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient  # noqa: E402


def _mob_name(mob: Any) -> str:
    try:
        v = getattr(mob, "name", None)
        if v:
            return str(v)
    except Exception:
        pass
    try:
        return str(mob["Name"].value)
    except Exception:
        return ""


def _resolve_mob(aaf: Any, mob_id: Any) -> Any:
    try:
        mobs = aaf.content.mobs
        it = mobs.values() if hasattr(mobs, "values") else mobs
        for m in it:
            if getattr(m, "mob_id", None) == mob_id:
                return m
    except Exception:
        return None
    return None


def _node_length(node: Any) -> int:
    try:
        return int(getattr(node, "length", 0) or 0)
    except Exception:
        pass
    try:
        return int(node["Length"].value)
    except Exception:
        return 0


def _iter_all_sequences(comp: Any):
    for slot in getattr(comp, "slots", []) or []:
        seg = getattr(slot, "segment", None)
        if seg is None:
            continue
        name = seg.__class__.__name__
        if "Sequence" in name:
            yield slot, seg
        elif "OperationGroup" in name:
            try:
                inner = getattr(seg, "segments", None)
                if inner is not None and len(inner) >= 1 and "Sequence" in inner[0].__class__.__name__:
                    yield slot, inner[0]
            except Exception:
                pass


def _slot_edit_rate(slot: Any) -> float:
    try:
        er = getattr(slot, "edit_rate", None)
        if er is not None:
            return float(er)
    except Exception:
        pass
    try:
        er = slot["EditRate"].value
        # AAF SDK xml style can store as "48000/1"
        if isinstance(er, str) and "/" in er:
            a, b = er.split("/", 1)
            return float(a) / max(1.0, float(b))
        return float(er)
    except Exception:
        return 0.0


def _walk_find_sourceclip(node: Any) -> list[Any]:
    out: list[Any] = []
    if node is None:
        return out
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
            out.append(cur)
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
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aaf", type=Path, required=True)
    ap.add_argument("--mob-name", type=str, required=True)
    ap.add_argument("--max", type=int, default=30)
    args = ap.parse_args()

    want = args.mob_name.strip()
    aaf_path = Path(args.aaf).resolve()

    with open_aaf_lenient(aaf_path, "r") as aaf:
        comps = list(aaf.content.compositionmobs())
        if not comps:
            print("No compositions")
            return 1
        total_hits = 0
        for comp_idx, comp in enumerate(comps):
            lanes = list(_iter_all_sequences(comp))
            if not lanes:
                continue

            hits = 0
            for lane_idx, (slot, seq) in enumerate(lanes):
                vec = seq.components
                try:
                    n = len(vec)
                except Exception:
                    continue
                er = float(_slot_edit_rate(slot) or 0.0)
                if er <= 0.0:
                    er = 48000.0
                pos = 0
                for i in range(n):
                    node = vec[i]
                    ln = max(0, _node_length(node))
                    # for top-level T, we count all components (including transitions)
                    T_top = pos
                    if ln > 0:
                        pos += ln
                    # Search inside this top-level node for any SourceClip.
                    scs = _walk_find_sourceclip(node)
                    for sc in scs:
                        try:
                            sid = sc["SourceID"].value
                        except Exception:
                            sid = getattr(sc, "source_id", None)
                        if sid is None:
                            continue
                        mob = _resolve_mob(aaf, sid)
                        if mob is None:
                            continue
                        nm = _mob_name(mob)
                        if nm != want:
                            continue
                        hits += 1
                        total_hits += 1
                        t_sec = float(T_top) / float(er) if er > 0 else 0.0
                        print(
                            f"HIT#{total_hits}: comp={comp_idx} lane={lane_idx} top_idx={i} "
                            f"top_T={T_top} ({t_sec:.3f}s) top_L={ln} node={node.__class__.__name__}"
                        )
                        print(f"  edit_rate={er}")
                        print(f"  sourceclip_len={_node_length(sc)} source_id={sid}")
                        print(f"  mob={mob.__class__.__name__} name={nm}")
                        if total_hits >= int(args.max):
                            return 0

    print("hits", total_hits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
