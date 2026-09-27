from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any


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
from aaf_io.converter import AAFConverter  # noqa: E402
from aaf_speech_filter.aaf_yamnet_lane_layout import (  # noqa: E402
    _apply_edit_timeline_unification,
    _assign_target_lanes,
    _classify_timeline_block_with_scores,
    _event_owned_transition_indices,
    _is_filler,
    _is_operation_group,
    _is_sourceclip,
    _is_transition_node,
    _lane_yamnet_cfg,
    _node_length_units,
    _og_first_sourceclip,
    _pick_main_composition,
    _slot_edit_rate,
    _sound_sequence_timeline_slots,
)
from aaf_speech_filter.config import FilterConfig  # noqa: E402
from aaf_speech_filter.timeline_walk import _aaf_int_length, _aaf_int_start  # noqa: E402


def _parse_timecode_units(value: str, *, fps: float, edit_rate: float) -> int:
    hh, mm, ss, ff = str(value).strip().split(":")
    seconds = (
        int(hh) * 3600.0
        + int(mm) * 60.0
        + int(ss)
        + float(ff) / float(fps)
    )
    return int(round(seconds * float(edit_rate)))


def _source_id(sc: Any) -> str:
    try:
        return str(sc["SourceID"].value)
    except Exception:
        return str(getattr(sc, "source_id", ""))


def _source_track(sc: Any) -> int:
    for prop in ("SourceMobSlotID", "SourceSlotID"):
        try:
            return int(sc[prop].value)
        except Exception:
            pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aaf", type=Path, required=True)
    ap.add_argument("--timecode", action="append", required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--edit-rate", type=float, default=48000.0)
    args = ap.parse_args()

    timecodes = [
        (tc, _parse_timecode_units(tc, fps=args.fps, edit_rate=args.edit_rate))
        for tc in args.timecode
    ]

    with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix="__diag_plan_") as td:
        work_dir = Path(td)
        converter = AAFConverter(args.aaf, work_dir=work_dir)
        work_path = work_dir / "work.aaf"
        ok = converter.prepare_external_media_copy(work_path)
        runtime = converter.runtime_essence_paths_for_filter()
        print(f"prepare={int(bool(ok))} runtime={0 if runtime is None else len(runtime)}")
        if not ok or runtime is None:
            return 1

        try:
            with open_aaf_lenient(work_path, "r") as aaf:
                comp = _pick_main_composition(list(aaf.content.compositionmobs()))
                lanes, skipped = _sound_sequence_timeline_slots(comp, cancel_check=lambda: None)
                print(f"lanes={len(lanes)} skipped={skipped}")
                cfg = FilterConfig(
                    remove_quiet_clips=False,
                    experimental_yamnet_lane_layout=True,
                )
                yamnet_cfg = _lane_yamnet_cfg(cfg)
                events: list[dict[str, Any]] = []
                for lane_idx, (slot, seq) in enumerate(lanes):
                    top = seq.components
                    pos = 0
                    trans_before = 0
                    try:
                        n = len(top)
                    except Exception:
                        continue
                    try:
                        owner_by_transition = _event_owned_transition_indices(
                            [top[i] for i in range(n)]
                        )
                    except Exception:
                        owner_by_transition = {}
                    er = float(_slot_edit_rate(slot) or 48000.0)
                    for top_idx in range(n):
                        node = top[top_idx]
                        length = int(_node_length_units(node))
                        if _is_filler(node):
                            pos += length
                            continue
                        if _is_transition_node(node):
                            if int(top_idx) not in owner_by_transition:
                                pass
                            trans_before += max(0, length)
                            pos += length
                            continue
                        if not (_is_operation_group(node) or _is_sourceclip(node)):
                            pos += length
                            continue
                        inner = _og_first_sourceclip(node) if _is_operation_group(node) else node
                        if inner is None:
                            pos += length
                            continue
                        kind, ys, ym, yn = _classify_timeline_block_with_scores(
                            aaf,
                            node,
                            runtime,
                            yamnet_cfg,
                            er,
                            lambda: None,
                            work_dir=work_dir,
                            log_callback=None,
                        )
                        events.append(
                            {
                                "src_lane": int(lane_idx),
                                "top_idx": int(top_idx),
                                "T": int(pos),
                                "T_nuendo": int(pos) - 2 * int(trans_before),
                                "transition_offset_before": int(trans_before),
                                "L": int(length),
                                "node": node,
                                "kind": str(kind),
                                "yamnet_s": float(ys),
                                "yamnet_m": float(ym),
                                "yamnet_n": float(yn),
                                "source_id": _source_id(inner),
                                "source_track": _source_track(inner),
                                "source_start": int(_aaf_int_start(inner)),
                                "source_length": int(_aaf_int_length(inner)),
                            }
                        )
                        pos += length

                _apply_edit_timeline_unification(events)
                _assign_target_lanes(events, len(lanes))

                for label, unit in timecodes:
                    print(f"\nTC {label} units={unit}")
                    covering = [
                        e
                        for e in events
                        if int(e.get("T_edit", e.get("T_nuendo", e["T"]))) <= unit
                        < int(e.get("T_edit", e.get("T_nuendo", e["T"]))) + int(e["L"])
                    ]
                    for e in sorted(covering, key=lambda x: (int(x["src_lane"]), int(x["top_idx"]))):
                        sid_tail = str(e["source_id"]).split(".")[-1]
                        print(
                            "lane "
                            f"{int(e['src_lane']):2d}->{int(e['target_lane']):2d} "
                            f"idx={int(e['top_idx']):3d} kind={e['kind']:<7} "
                            f"s={float(e['yamnet_s']):.3f} "
                            f"m={float(e['yamnet_m']):.3f} "
                            f"n={float(e['yamnet_n']):.3f} "
                            f"Tedit={int(e.get('T_edit', e.get('T_nuendo', e['T'])))} "
                            f"L={int(e['L'])} sid_tail={sid_tail} "
                            f"src_start={int(e['source_start'])} "
                            f"src_len={int(e['source_length'])}"
                        )
        finally:
            try:
                converter._unlink_mapped_essence()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
