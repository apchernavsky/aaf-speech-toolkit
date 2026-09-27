from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
from aaf_io.clip_naming import resolve_clip_display_name  # noqa: E402


def _iter_sound_sequences(comp):
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
                if (
                    inner is not None
                    and len(inner) >= 1
                    and "Sequence" in inner[0].__class__.__name__
                ):
                    yield slot, inner[0]
            except Exception:
                pass


def _node_length(node) -> int:
    try:
        return int(getattr(node, "length", 0) or 0)
    except Exception:
        pass
    try:
        return int(node["Length"].value)
    except Exception:
        return 0


def collect_positions(aaf_path: Path, *, targets: set[str]) -> list[tuple[str, int, int, int]]:
    out: list[tuple[str, int, int, int]] = []
    with open_aaf_lenient(aaf_path, "r") as aaf:
        cache = {}
        comps = list(aaf.content.compositionmobs())
        if not comps:
            return out
        # This diagnostic inspects the first composition.
        comp = comps[0]
        lanes = list(_iter_sound_sequences(comp))
        for lane_idx, (_slot, seq) in enumerate(lanes):
            try:
                vec = seq.components
                n = len(vec)
            except Exception:
                continue
            pos = 0
            for i in range(n):
                try:
                    node = vec[i]
                except Exception:
                    break
                ln = max(0, _node_length(node))
                cname = node.__class__.__name__
                if "Filler" in cname:
                    pos += ln
                    continue
                if ("SourceClip" in cname) or ("OperationGroup" in cname):
                    nm = resolve_clip_display_name(aaf, node, cache=cache).display_name
                    base = nm.split("/")[-1].split("\\")[-1] if nm else ""
                    if base in targets or nm in targets:
                        out.append((base or nm, lane_idx, pos, ln))
                pos += ln
    out.sort()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument(
        "--target",
        action="append",
        default=[],
        help="Clip base name to compare (repeatable)",
    )
    args = ap.parse_args()

    targets = set(args.target or [])
    if not targets:
        targets = {"09.mp4_L", "09.mp4_R-01"}

    a = collect_positions(Path(args.src), targets=targets)
    b = collect_positions(Path(args.dst), targets=targets)

    print("src", str(Path(args.src).resolve()))
    print("dst", str(Path(args.dst).resolve()))
    print("targets", sorted(targets))
    print("src_matches", len(a))
    print("dst_matches", len(b))

    if a == b:
        print("OK: positions identical (T and L preserved)")
        return 0

    print("DIFF: positions changed")
    import itertools

    for i, (x, y) in enumerate(itertools.zip_longest(a, b)):
        if x != y:
            print("src", x)
            print("dst", y)
            if i >= 30:
                break
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
