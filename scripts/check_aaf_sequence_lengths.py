from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _ensure_toolkit_on_path() -> Path:
    here = Path(__file__).resolve()
    toolkit = here.parents[1]
    for p in (toolkit, toolkit / "aaf_io", toolkit / "aaf_speech_filter"):
        if p.is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
    return toolkit


_ensure_toolkit_on_path()

from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient  # noqa: E402
from aaf_speech_filter.aaf_yamnet_lane_layout import (  # noqa: E402
    _is_transition_node,
    _node_length_units,
    _pick_main_composition,
    _sound_sequence_timeline_slots,
)


def _sequence_expected_declared_length(nodes: list[Any]) -> int:
    component_total = sum(max(0, int(_node_length_units(node))) for node in nodes)
    transition_total = sum(
        max(0, int(_node_length_units(node)))
        for node in nodes
        if _is_transition_node(node)
    )
    return max(0, int(component_total) - 2 * int(transition_total))


def compare_sequence_lengths(
    aaf_path: Path,
    *,
    max_report: int,
) -> tuple[int, dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    lanes_checked = 0
    with open_aaf_lenient(aaf_path, "r") as aaf:
        comp = _pick_main_composition(list(aaf.content.compositionmobs()))
        sound_lanes, skipped = _sound_sequence_timeline_slots(comp, cancel_check=lambda: None)
        for lane_idx, (_slot, seq) in enumerate(sound_lanes):
            nodes = list(getattr(seq, "components", []) or [])
            declared = int(_node_length_units(seq))
            component_total = sum(max(0, int(_node_length_units(node))) for node in nodes)
            transition_total = sum(
                max(0, int(_node_length_units(node)))
                for node in nodes
                if _is_transition_node(node)
            )
            expected = _sequence_expected_declared_length(nodes)
            lanes_checked += 1
            if declared == expected:
                continue
            mismatches.append(
                {
                    "lane": int(lane_idx),
                    "declared": int(declared),
                    "component_total": int(component_total),
                    "transition_total": int(transition_total),
                    "expected_declared": int(expected),
                    "diff": int(declared) - int(expected),
                    "components": len(nodes),
                }
            )
    report = {
        "aaf": str(aaf_path),
        "lanes_checked": int(lanes_checked),
        "skipped_slots": int(skipped),
        "mismatches": len(mismatches),
        "mismatch_sample": mismatches[: max(0, int(max_report))],
    }
    return (1 if mismatches else 0), report


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Verify that sound Sequence.ComponentLength matches the AAF transition "
            "overlap invariant: sum(children) - 2 * sum(transitions)."
        )
    )
    ap.add_argument("aaf", type=Path)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--max-report", type=int, default=40)
    args = ap.parse_args()

    code, report = compare_sequence_lengths(args.aaf, max_report=args.max_report)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if code == 0:
        print("OK: sound sequence declared lengths match transition overlap invariant.")
    else:
        print("ERROR: sound sequence declared lengths violate transition overlap invariant.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
