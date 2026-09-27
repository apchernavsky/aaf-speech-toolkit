from __future__ import annotations

import argparse
import sys
from collections import Counter
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
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    aaf_path = Path(args.aaf).resolve()
    with open_aaf_lenient(aaf_path, "r") as aaf:
        comps = list(aaf.content.compositionmobs())
        if not comps:
            print("no comps")
            return 1
        comp = comps[0]
        lanes = list(_iter_sequences(comp))
        cache = {}
        c = Counter()
        blocks = 0
        for seq in lanes:
            vec = seq.components
            try:
                n = len(vec)
            except Exception:
                continue
            for i in range(n):
                node = vec[i]
                if "Filler" in node.__class__.__name__:
                    continue
                ln = _node_len(node)
                if ln <= 0:
                    continue
                nm = resolve_clip_display_name(aaf, node, cache=cache).display_name or "<без имени>"
                c[nm] += 1
                blocks += 1
        print("file", str(aaf_path))
        print("blocks_counted", blocks)
        for name, cnt in c.most_common(int(args.top)):
            print(cnt, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
