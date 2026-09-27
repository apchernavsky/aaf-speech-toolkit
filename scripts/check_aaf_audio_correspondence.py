from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


def _ensure_toolkit_on_path() -> Path:
    here = Path(__file__).resolve()
    toolkit = here.parents[1]
    for p in (toolkit, toolkit / "aaf_io", toolkit / "aaf_speech_filter"):
        if p.is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
    return toolkit


REPO_ROOT = _ensure_toolkit_on_path()

from aaf_io.clip_naming import resolve_clip_display_name  # noqa: E402
from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient  # noqa: E402
from aaf_io.converter import AAFConverter  # noqa: E402
from aaf_speech_filter.aaf_yamnet_lane_layout import (  # noqa: E402
    _is_filler,
    _is_operation_group,
    _is_sourceclip,
    _is_transition_node,
    _node_length_units,
    _og_first_sourceclip,
    _pick_main_composition,
    _slot_edit_rate,
    _sound_sequence_timeline_slots,
)
from aaf_speech_filter.media_resolve import _resolve_wave_path_for_sourceclip  # noqa: E402
from aaf_speech_filter.speech_vad import read_media_segment_pcm16_mono  # noqa: E402
from aaf_speech_filter.timeline_timing import sourceclip_audio_timing  # noqa: E402


@dataclass(frozen=True)
class ClipAudioFingerprint:
    name: str
    source_id: str
    source_track: int
    source_start: int
    source_length: int
    timeline_start: int
    edit_start: int
    timeline_length: int
    lane: int
    top_index: int
    sample_rate: int
    pcm_bytes: int
    pcm_sha256: str

    @property
    def raw_key(self) -> tuple[str, int, int, int, int, int]:
        return (
            self.source_id,
            int(self.source_track),
            int(self.source_start),
            int(self.source_length),
            int(self.timeline_start),
            int(self.timeline_length),
        )

    @property
    def edit_key(self) -> tuple[str, int, int, int, int, int]:
        return (
            self.source_id,
            int(self.source_track),
            int(self.source_start),
            int(self.source_length),
            int(self.edit_start),
            int(self.timeline_length),
        )


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


def _aaf_int_prop(obj: Any, prop: str, attr: str) -> int:
    try:
        return int(obj[prop].value)
    except Exception:
        return int(getattr(obj, attr, 0) or 0)


def _parse_timecode_units(value: str, *, fps: float, edit_rate: float) -> int:
    parts = str(value).strip().split(":")
    if len(parts) != 4:
        raise ValueError("timecode must be HH:MM:SS:FF or HH:MM:SS:FF.subframes")
    hh = int(parts[0])
    mm = int(parts[1])
    ss = int(parts[2])
    frame_part = parts[3]
    frames = float(frame_part)
    seconds = (hh * 3600.0) + (mm * 60.0) + ss + (frames / float(fps))
    return int(round(seconds * float(edit_rate)))


def _collect_fingerprints(aaf_path: Path, *, name_contains: Optional[str]) -> list[ClipAudioFingerprint]:
    out: list[ClipAudioFingerprint] = []
    with tempfile.TemporaryDirectory(prefix="aaf_audio_check_") as td:
        conv = AAFConverter(aaf_path, work_dir=Path(td))
        try:
            with open_aaf_lenient(aaf_path, "r") as aaf:
                conv.essence_map = {}
                conv._extract_all(aaf)
            runtime = conv.runtime_essence_paths_for_filter()

            with open_aaf_lenient(aaf_path, "r") as aaf:
                comp = _pick_main_composition(list(aaf.content.compositionmobs()))
                sound_lanes, _skipped = _sound_sequence_timeline_slots(
                    comp, cancel_check=lambda: None
                )
                cache: dict[Any, Any] = {}
                for lane_idx, (slot, seq) in enumerate(sound_lanes):
                    try:
                        nodes = list(seq.components)
                    except Exception:
                        continue
                    pos = 0
                    transition_before = 0
                    for top_idx, node in enumerate(nodes):
                        length = int(_node_length_units(node))
                        if _is_transition_node(node):
                            transition_before += max(0, int(length))
                            pos += length
                            continue
                        if _is_filler(node) or length <= 0:
                            pos += length
                            continue
                        if not (_is_operation_group(node) or _is_sourceclip(node)):
                            pos += length
                            continue
                        inner = _og_first_sourceclip(node) if _is_operation_group(node) else node
                        if inner is None:
                            pos += length
                            continue
                        try:
                            name = (
                                resolve_clip_display_name(aaf, node, cache=cache).display_name
                                or ""
                            ).strip()
                        except Exception:
                            name = ""
                        if name_contains and name_contains not in name:
                            pos += length
                            continue
                        wav = _resolve_wave_path_for_sourceclip(aaf, inner, runtime)
                        if wav is None or not Path(wav).is_file():
                            raise RuntimeError(f"media not resolved for clip {name!r}")
                        edit_rate = float(_slot_edit_rate(slot) or 48000.0)
                        visible_duration_sec = max(0.0, float(length) / edit_rate)
                        start_sec, dur_sec = sourceclip_audio_timing(
                            aaf,
                            edit_rate,
                            inner,
                            wav,
                            runtime,
                            visible_duration_sec=visible_duration_sec,
                        )
                        pcm, sample_rate = read_media_segment_pcm16_mono(wav, start_sec, dur_sec)
                        out.append(
                            ClipAudioFingerprint(
                                name=name,
                                source_id=_source_id(inner),
                                source_track=_source_track(inner),
                                source_start=_aaf_int_prop(inner, "StartTime", "start"),
                                source_length=_aaf_int_prop(inner, "Length", "length"),
                                timeline_start=int(pos),
                                edit_start=int(pos) - (2 * int(transition_before)),
                                timeline_length=int(length),
                                lane=int(lane_idx),
                                top_index=int(top_idx),
                                sample_rate=int(sample_rate),
                                pcm_bytes=len(pcm),
                                pcm_sha256=hashlib.sha256(pcm).hexdigest(),
                            )
                        )
                        pos += length
        finally:
            try:
                conv._unlink_mapped_essence()
            except Exception:
                pass
    return out


def _sample(items: list[Any], limit: int) -> list[Any]:
    return items[: max(0, int(limit))]


def _covering(
    items: list[ClipAudioFingerprint],
    unit: Optional[int],
    *,
    visible: bool,
) -> list[ClipAudioFingerprint]:
    if unit is None:
        return []
    out: list[ClipAudioFingerprint] = []
    for item in items:
        start = int(item.edit_start if visible else item.timeline_start)
        if start <= int(unit) < start + int(item.timeline_length):
            out.append(item)
    return out


def compare_audio(
    src: Path,
    dst: Path,
    *,
    name_contains: Optional[str],
    timecode_units: Optional[int],
    max_report: int,
) -> tuple[int, dict[str, Any]]:
    src_items = _collect_fingerprints(src, name_contains=name_contains)
    dst_items = _collect_fingerprints(dst, name_contains=name_contains)
    src_by_key: dict[tuple[str, int, int, int, int, int], list[ClipAudioFingerprint]] = defaultdict(list)
    dst_by_key: dict[tuple[str, int, int, int, int, int], list[ClipAudioFingerprint]] = defaultdict(list)
    for item in src_items:
        src_by_key[item.raw_key].append(item)
    for item in dst_items:
        dst_by_key[item.raw_key].append(item)

    src_counter = Counter(item.raw_key for item in src_items)
    dst_counter = Counter(item.raw_key for item in dst_items)
    new_or_shifted_keys = list((dst_counter - src_counter).elements())
    raw_removed_or_shifted_keys = list((src_counter - dst_counter).elements())
    src_edit_counter = Counter(item.edit_key for item in src_items)
    dst_edit_counter = Counter(item.edit_key for item in dst_items)
    edit_new_or_shifted_keys = list((dst_edit_counter - src_edit_counter).elements())
    edit_removed_keys = list((src_edit_counter - dst_edit_counter).elements())

    src_by_edit_key: dict[tuple[str, int, int, int, int, int], list[ClipAudioFingerprint]] = defaultdict(list)
    dst_by_edit_key: dict[tuple[str, int, int, int, int, int], list[ClipAudioFingerprint]] = defaultdict(list)
    for item in src_items:
        src_by_edit_key[item.edit_key].append(item)
    for item in dst_items:
        dst_by_edit_key[item.edit_key].append(item)

    src_by_edit_pcm_key: dict[tuple[str, int, int, int, int, int], list[ClipAudioFingerprint]] = defaultdict(list)
    dst_by_edit_pcm_key: dict[tuple[str, int, int, int, int, int], list[ClipAudioFingerprint]] = defaultdict(list)
    for item in src_items:
        src_by_edit_pcm_key[item.edit_key].append(item)
    for item in dst_items:
        dst_by_edit_pcm_key[item.edit_key].append(item)

    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(src_by_edit_pcm_key) & set(dst_by_edit_pcm_key), key=lambda k: (k[4], k[0], k[2])):
        src_list = list(src_by_edit_pcm_key[key])
        dst_list = list(dst_by_edit_pcm_key[key])
        pair_count = min(len(src_list), len(dst_list))
        for idx in range(pair_count):
            s = src_list[idx]
            d = dst_list[idx]
            if (
                s.sample_rate == d.sample_rate
                and s.pcm_bytes == d.pcm_bytes
                and s.pcm_sha256 == d.pcm_sha256
            ):
                continue
            mismatches.append(
                {
                    "name": d.name or s.name,
                    "edit_start": key[4],
                    "timeline_length": key[5],
                    "src_lane": s.lane,
                    "dst_lane": d.lane,
                    "src_hash": s.pcm_sha256,
                    "dst_hash": d.pcm_sha256,
                    "src_pcm_bytes": s.pcm_bytes,
                    "dst_pcm_bytes": d.pcm_bytes,
                }
            )

    new_or_shifted: list[dict[str, Any]] = []
    for key in new_or_shifted_keys:
        item = dst_by_key[key].pop(0)
        new_or_shifted.append(asdict(item))

    raw_removed_or_shifted: list[dict[str, Any]] = []
    for key in raw_removed_or_shifted_keys:
        item = src_by_key[key].pop(0)
        raw_removed_or_shifted.append(asdict(item))

    edit_new_or_shifted: list[dict[str, Any]] = []
    for key in edit_new_or_shifted_keys:
        item = dst_by_edit_key[key].pop(0)
        edit_new_or_shifted.append(asdict(item))

    src_removed_lookup: dict[tuple[str, int, int, int, int, int], list[ClipAudioFingerprint]] = defaultdict(list)
    for item in src_items:
        src_removed_lookup[item.edit_key].append(item)
    removed: list[dict[str, Any]] = []
    for key in edit_removed_keys:
        item = src_removed_lookup[key].pop(0)
        removed.append(asdict(item))

    src_covering = _covering(src_items, timecode_units, visible=True)
    dst_covering = _covering(dst_items, timecode_units, visible=True)
    report = {
        "src": str(src),
        "dst": str(dst),
        "name_contains": name_contains,
        "timecode_units": timecode_units,
        "src_clips_checked": len(src_items),
        "dst_clips_checked": len(dst_items),
        "new_or_raw_shifted": len(new_or_shifted),
        "removed_or_raw_shifted": len(raw_removed_or_shifted),
        "new_or_edit_shifted": len(edit_new_or_shifted),
        "removed": max(0, len(src_items) - len(dst_items)),
        "removed_or_edit_shifted": len(removed),
        "pcm_mismatches": len(mismatches),
        "new_or_raw_shifted_sample": _sample(new_or_shifted, max_report),
        "removed_or_raw_shifted_sample": _sample(raw_removed_or_shifted, max_report),
        "new_or_edit_shifted_sample": _sample(edit_new_or_shifted, max_report),
        "removed_or_edit_shifted_sample": _sample(removed, max_report),
        "pcm_mismatch_sample": _sample(mismatches, max_report),
        "src_covering_timecode": [asdict(item) for item in src_covering],
        "dst_covering_timecode": [asdict(item) for item in dst_covering],
    }
    code = 1 if edit_new_or_shifted or mismatches else 0
    return code, report


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Verify that surviving AAF clips keep the same DAW-visible horizontal "
            "position and decoded PCM window after processing."
        )
    )
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--name-contains")
    ap.add_argument("--timecode", help="HH:MM:SS:FF or HH:MM:SS:FF.subframes")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--edit-rate", type=float, default=48000.0)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--max-report", type=int, default=40)
    args = ap.parse_args()

    tc_units = None
    if args.timecode:
        tc_units = _parse_timecode_units(args.timecode, fps=args.fps, edit_rate=args.edit_rate)
    code, report = compare_audio(
        args.src,
        args.dst,
        name_contains=args.name_contains,
        timecode_units=tc_units,
        max_report=args.max_report,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if code == 0:
        print("OK: surviving clips preserved DAW-visible position and decoded PCM window.")
    else:
        print("ERROR: surviving clips changed DAW-visible position or decoded PCM window.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
