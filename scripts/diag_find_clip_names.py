from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("aaf", type=Path)
    ap.add_argument("needle", type=str)
    args = ap.parse_args()

    # Allow running as a standalone script from repo root.
    repo_root = Path(__file__).resolve().parents[1]
    # aaf_io and aaf_speech_filter are namespace-ish (aaf_io/aaf_io, aaf_speech_filter/aaf_speech_filter)
    # so we add both the repo root and their package roots.
    for p in (
        repo_root,
        repo_root / "aaf_io",
        repo_root / "aaf_speech_filter",
    ):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
    from aaf_speech_filter.aaf_yamnet_lane_layout import (
        _is_operation_group,
        _is_sourceclip,
        _pick_main_composition,
        _sound_sequence_timeline_slots,
        resolve_clip_display_name,
    )

    names: set[str] = set()
    cache: dict[object, object] = {}

    with open_aaf_lenient(Path(args.aaf), "r") as aaf:
        comps = list(aaf.content.compositionmobs())
        if not comps:
            print("No composition mobs.")
            return 1
        comp = _pick_main_composition(comps)
        lanes, _skipped = _sound_sequence_timeline_slots(comp, cancel_check=lambda: None)

        def clip_name(node) -> str:
            try:
                r = resolve_clip_display_name(aaf, node, cache=cache)
                return (r.display_name or "").strip()
            except Exception:
                return ""

        for _lane_idx, (_slot, seq) in enumerate(lanes):
            top = seq.components
            try:
                tn = len(top)
            except Exception:
                continue
            for ti in range(tn):
                try:
                    node = top[ti]
                except Exception:
                    break
                if _is_operation_group(node) or _is_sourceclip(node):
                    nm = clip_name(node)
                    if args.needle in nm:
                        names.add(nm)

    print(f"MATCHES: {len(names)}")
    for nm in sorted(names):
        print(nm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
