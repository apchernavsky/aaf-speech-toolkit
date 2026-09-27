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


def _resolve_source_mob(aaf: Any, source_id: Any) -> Optional[Any]:
    # Best-effort: scan all mobs
    try:
        mobs = aaf.content.mobs
        it = mobs.values() if hasattr(mobs, "values") else mobs
    except Exception:
        return None
    for m in it:
        try:
            if "SourceMob" not in m.__class__.__name__:
                continue
        except Exception:
            continue
        try:
            if m.mob_id == source_id:
                return m
        except Exception:
            pass
        try:
            if m["MobID"].value == source_id:
                return m
        except Exception:
            pass
    return None


def _print_descriptor(desc: Any) -> None:
    if desc is None:
        print("descriptor: <none>")
        return
    try:
        print("descriptor_type:", desc.__class__.__name__)
    except Exception:
        print("descriptor_type: <unknown>")
    # Common audio fields
    for k in ("Channels", "SampleRate", "QuantizationBits", "AudioSamplingRate", "AverageBPS"):
        try:
            v = desc[k].value
            print(f"  {k}: {v}")
        except Exception:
            pass
    # Try to show essence streams count
    for k in ("EssenceStream", "EssenceStreams", "Locator", "Locators"):
        try:
            v = desc[k]
            try:
                ln = len(v)
                print(f"  {k}: len={ln}")
            except Exception:
                print(f"  {k}: {type(v)}")
        except Exception:
            pass


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
    ap.add_argument("--name-contains", type=str, required=True)
    args = ap.parse_args()

    aaf_path = Path(args.aaf).resolve()
    needle = args.name_contains
    with open_aaf_lenient(aaf_path, "r") as aaf:
        comps = list(aaf.content.compositionmobs())
        if not comps:
            print("no comps")
            return 1
        comp = comps[0]
        cache = {}
        for seq in _iter_sequences(comp):
            vec = seq.components
            try:
                n = len(vec)
            except Exception:
                continue
            for i in range(n):
                node = vec[i]
                if "Filler" in node.__class__.__name__:
                    continue
                nm = resolve_clip_display_name(aaf, node, cache=cache).display_name or ""
                if needle not in nm:
                    continue
                sc = _inner_sourceclip(node)
                if sc is None:
                    print("found node but no inner sourceclip; name=", nm)
                    return 0
                try:
                    sid = sc["SourceID"].value
                except Exception:
                    sid = getattr(sc, "source_id", None)
                print("match_name:", nm)
                print("node_type:", node.__class__.__name__)
                print("source_id:", sid)
                sm = _resolve_source_mob(aaf, sid)
                if sm is None:
                    print("source_mob: <not found>")
                    return 0
                try:
                    print("source_mob_name:", getattr(sm, "name", None))
                except Exception:
                    pass
                desc = getattr(sm, "descriptor", None)
                _print_descriptor(desc)
                return 0
    print("no match for", needle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
