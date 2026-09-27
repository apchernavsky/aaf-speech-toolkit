from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


def _ensure_toolkit_on_path() -> None:
    here = Path(__file__).resolve()
    toolkit = here.parents[1]
    for p in (toolkit, toolkit / "aaf_io", toolkit / "aaf_speech_filter"):
        if p.is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)


_ensure_toolkit_on_path()

from aaf_io.composition import select_aaf_composition
from aaf_io.clip_naming import resolve_clip_display_name  # noqa: E402
from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient  # noqa: E402
from aaf_speech_filter.timeline_walk import (  # noqa: E402
    _aaf_int_length,
    _aaf_int_start,
    _inner_sourceclip_for_block,
    _is_sourceclip,
    _slot_is_soundish,
)


@dataclass(frozen=True)
class ClipPosition:
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

    @property
    def identity(self) -> tuple[str, int, int, int, str]:
        return (
            self.source_id,
            int(self.source_track),
            int(self.source_start),
            int(self.source_length),
            self.name,
        )

    @property
    def position_key(self) -> tuple[tuple[str, int, int, int, str], int, int]:
        return (self.identity, int(self.timeline_start), int(self.timeline_length))

    @property
    def edit_position_key(self) -> tuple[tuple[str, int, int, int, str], int, int]:
        return (self.identity, int(self.edit_start), int(self.timeline_length))


@dataclass(frozen=True)
class ClassifiedOccurrence:
    source_id: str
    source_track: int
    source_start: int
    source_length: int
    edit_start: int
    timeline_length: int
    source_lane: int
    class_kind: str

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> "ClassifiedOccurrence":
        kind = str(event["kind"])
        if kind not in {"speech", "noise", "music", "unknown"}:
            raise ValueError(f"Unsupported classified kind: {kind}")
        return cls(str(event["source_id"]), int(event["source_track"]),
            int(event["source_start"]), int(event["source_length"]), int(event["T_edit"]),
            int(event["L"]), int(event["src_lane"]), kind)

    @property
    def position_identity(self) -> tuple[Any, ...]:
        return (self.source_id, self.source_track, self.source_start, self.source_length,
                self.edit_start, self.timeline_length)

    @property
    def source_key(self) -> tuple[Any, ...]:
        return (*self.position_identity, self.source_lane)


@dataclass(frozen=True)
class ClassifiedLayoutPlan:
    composition_id: str
    events: tuple[ClassifiedOccurrence, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClassifiedLayoutPlan":
        return cls(str(value["composition_id"]), tuple(ClassifiedOccurrence(**item) for item in value["events"]))


def _stable_position_identity(item: ClipPosition) -> tuple[Any, ...]:
    return (item.source_id, item.source_track, item.source_start, item.source_length,
            item.edit_start, item.timeline_length)


def _classified_order_inversions(src_items, dst_items, plan: ClassifiedLayoutPlan):
    """Check actual class groups; absent or ambiguous plan identities are errors."""
    sources = Counter((*_stable_position_identity(item), item.lane) for item in src_items)
    outputs: dict[tuple[Any, ...], list[ClipPosition]] = defaultdict(list)
    for item in dst_items:
        outputs[_stable_position_identity(item)].append(item)
    planned: dict[tuple[Any, ...], list[ClassifiedOccurrence]] = defaultdict(list)
    errors: list[str] = []
    for event in plan.events:
        if event.class_kind not in {"speech", "noise", "music", "unknown"}:
            errors.append(f"Unsupported classified kind: {event.class_kind}")
        planned[event.position_identity].append(event)
    requested_sources = Counter(event.source_key for event in plan.events)
    for key, count in requested_sources.items():
        if sources[key] < count:
            errors.append(f"Classified source multiplicity exceeds input: {key}; planned={count}, input={sources[key]}")
    for key in sorted(planned.keys() | outputs.keys()):
        records = planned.get(key, [])
        if len(records) != len(outputs.get(key, [])):
            errors.append(f"Classified output multiplicity mismatch: {key}; planned={len(records)}, output={len(outputs.get(key, []))}")
        if len({event.class_kind for event in records}) > 1:
            errors.append(f"Indistinguishable output occurrences have ambiguous classes: {key}")
    if errors:
        return [], errors
    groups: dict[tuple[Any, ...], list[tuple[ClassifiedOccurrence, ClipPosition]]] = defaultdict(list)
    for key, records in planned.items():
        # Equal-identity, equal-class copies are indistinguishable in the output.
        # Pair monotonically; the full identity sequence still exposes A,B,A -> A,A,B.
        source_copies = sorted(records, key=lambda event: event.source_lane)
        output_copies = sorted(outputs[key], key=lambda item: (item.lane, item.top_index))
        for event, item in zip(source_copies, output_copies):
            group_key = (event.edit_start, event.timeline_length, event.source_start,
                         event.source_length, event.class_kind)
            groups[group_key].append((event, item))
    inversions = []
    for key, group in groups.items():
        source_order = sorted(group, key=lambda pair: pair[0].source_lane)
        target_order = sorted(group, key=lambda pair: pair[1].lane)
        if [pair[0].position_identity for pair in source_order] == [pair[0].position_identity for pair in target_order]:
            continue
        inversions.append(dict(timeline_start=key[0], timeline_length=key[1],
            source_start=key[2], source_length=key[3], actual_class=key[4],
            src_order=[dict(source_id=event.source_id, lane=event.source_lane) for event, _ in source_order],
            dst_order=[dict(source_id=event.source_id, lane=item.lane) for event, item in target_order]))
    return inversions, []


def _node_length(node: Any) -> int:
    try:
        return int(_aaf_int_length(node))
    except Exception:
        return 0


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


def _pick_main_composition(comps: list[Any]) -> Optional[Any]:
    return select_aaf_composition(comps) if comps else None


def _sound_sequences(comp: Any) -> Iterable[tuple[Any, Any]]:
    for slot in getattr(comp, "slots", []) or []:
        if not _slot_is_soundish(slot):
            continue
        seg = getattr(slot, "segment", None)
        if seg is None:
            continue
        if "Sequence" in seg.__class__.__name__:
            yield slot, seg
            continue
        if "OperationGroup" in seg.__class__.__name__:
            try:
                inner = getattr(seg, "segments", None)
                if inner is not None and len(inner) >= 1 and "Sequence" in inner[0].__class__.__name__:
                    yield slot, inner[0]
            except Exception:
                pass


def collect_clip_positions(aaf_path: Path, *, composition_id: Optional[str] = None) -> list[ClipPosition]:
    out: list[ClipPosition] = []
    with open_aaf_lenient(aaf_path, "r") as aaf:
        comps = list(aaf.content.compositionmobs())
        comp = select_aaf_composition(comps, composition_id) if composition_id is not None else _pick_main_composition(comps)
        if comp is None:
            return out
        name_cache: dict[Any, Any] = {}
        for lane_idx, (_slot, seq) in enumerate(_sound_sequences(comp)):
            try:
                top = seq.components
                n = len(top)
            except Exception:
                continue
            pos = 0
            transition_before = 0
            for top_idx in range(n):
                try:
                    node = top[top_idx]
                except Exception:
                    break
                ln = max(0, _node_length(node))
                cname = node.__class__.__name__
                if "Transition" in cname:
                    transition_before += ln
                    pos += ln
                    continue
                if "Filler" in cname:
                    pos += ln
                    continue
                if "OperationGroup" not in cname and not _is_sourceclip(node):
                    pos += ln
                    continue
                sc = _inner_sourceclip_for_block(node)
                if sc is None:
                    pos += ln
                    continue
                try:
                    name = resolve_clip_display_name(aaf, node, cache=name_cache).display_name or ""
                except Exception:
                    name = ""
                out.append(
                    ClipPosition(
                        name=str(name),
                        source_id=_source_id(sc),
                        source_track=_source_track(sc),
                        source_start=int(_aaf_int_start(sc)),
                        source_length=int(_aaf_int_length(sc)),
                        timeline_start=int(pos),
                        edit_start=int(pos) - 2 * int(transition_before),
                        timeline_length=int(ln),
                        lane=int(lane_idx),
                        top_index=int(top_idx),
                    )
                )
                pos += ln
    return out


def _sample(items: list[Any], limit: int) -> list[Any]:
    return items[: max(0, int(limit))]


def _aligned_group_key(item: ClipPosition) -> tuple[int, int, int, int]:
    return (
        int(item.timeline_start),
        int(item.timeline_length),
        int(item.source_start),
        int(item.source_length),
    )


def _aligned_order_inversions(
    src_items: list[ClipPosition],
    dst_items: list[ClipPosition],
) -> list[dict[str, Any]]:
    src_groups: dict[tuple[int, int, int, int], list[ClipPosition]] = defaultdict(list)
    dst_groups: dict[tuple[int, int, int, int], list[ClipPosition]] = defaultdict(list)
    for item in src_items:
        src_groups[_aligned_group_key(item)].append(item)
    for item in dst_items:
        dst_groups[_aligned_group_key(item)].append(item)

    dst_lane_count = max((int(item.lane) for item in dst_items), default=-1) + 1

    def lane_band(lane: int) -> str:
        if dst_lane_count <= 0:
            return "unknown"
        speech_end = max(1, int(dst_lane_count) // 3)
        music_start = max(speech_end, (2 * int(dst_lane_count)) // 3)
        if int(lane) < speech_end:
            return "speech"
        if int(lane) >= music_start:
            return "music"
        return "noise"

    inversions: list[dict[str, Any]] = []
    for key, src_group in src_groups.items():
        dst_group = dst_groups.get(key, [])
        if len(src_group) < 2 or len(dst_group) < 2:
            continue
        dst_counts = Counter(item.identity for item in dst_group)
        src_ordered: list[ClipPosition] = []
        for item in sorted(src_group, key=lambda x: (x.lane, x.top_index)):
            if dst_counts[item.identity] <= 0:
                continue
            dst_counts[item.identity] -= 1
            src_ordered.append(item)
        dst_ordered = sorted(dst_group, key=lambda x: (x.lane, x.top_index))
        if len(src_ordered) < 2 or len(src_ordered) != len(dst_ordered):
            continue
        dst_by_band: dict[str, list[ClipPosition]] = defaultdict(list)
        for item in dst_ordered:
            dst_by_band[lane_band(item.lane)].append(item)
        for band, band_items in dst_by_band.items():
            if len(band_items) < 2:
                continue
            band_counts = Counter(item.identity for item in band_items)
            src_band_order: list[ClipPosition] = []
            for item in src_ordered:
                if band_counts[item.identity] <= 0:
                    continue
                band_counts[item.identity] -= 1
                src_band_order.append(item)
            if [item.identity for item in src_band_order] == [item.identity for item in band_items]:
                continue
            inversions.append(
                {
                    "timeline_start": int(key[0]),
                    "timeline_length": int(key[1]),
                    "source_start": int(key[2]),
                    "source_length": int(key[3]),
                    "target_band": band,
                    "src_order": [
                        {"name": item.name, "lane": item.lane, "top_index": item.top_index}
                        for item in src_band_order
                    ],
                    "dst_order": [
                        {"name": item.name, "lane": item.lane, "top_index": item.top_index}
                        for item in band_items
                    ],
                }
            )
    return inversions


def compare_positions(
    src: Path, dst: Path, *, max_report: int,
    classified_plan: Optional[ClassifiedLayoutPlan] = None,
) -> tuple[int, dict[str, Any]]:
    selection = {} if classified_plan is None else {"composition_id": classified_plan.composition_id}
    src_items = collect_clip_positions(src, **selection)
    dst_items = collect_clip_positions(dst, **selection)
    return compare_collected_positions(src_items, dst_items, max_report=max_report,
        classified_plan=classified_plan, src_label=src, dst_label=dst)


def compare_collected_positions(
    src_items: list[ClipPosition], dst_items: list[ClipPosition], *, max_report: int,
    classified_plan: Optional[ClassifiedLayoutPlan] = None,
    src_label: str | Path = "", dst_label: str | Path = "",
) -> tuple[int, dict[str, Any]]:
    """Compare persisted position records with the same strict checks as live AAFs."""
    src_counter = Counter(item.position_key for item in src_items)
    dst_counter = Counter(item.position_key for item in dst_items)
    src_edit_counter = Counter(item.edit_position_key for item in src_items)
    dst_edit_counter = Counter(item.edit_position_key for item in dst_items)

    src_by_identity: dict[tuple[str, int, int, int, str], list[ClipPosition]] = defaultdict(list)
    for item in src_items:
        src_by_identity[item.identity].append(item)

    extras = list((dst_counter - src_counter).elements())
    shifted: list[dict[str, Any]] = []
    edit_shifted: list[dict[str, Any]] = []
    new_items: list[dict[str, Any]] = []
    dst_lookup: dict[tuple[tuple[str, int, int, int, str], int, int], list[ClipPosition]] = defaultdict(list)
    for item in dst_items:
        dst_lookup[item.position_key].append(item)

    for key in extras:
        identity, t_dst, l_dst = key
        item = dst_lookup[key].pop()
        if identity in src_by_identity:
            shifted.append(
                {
                    "name": item.name,
                    "source_id": item.source_id,
                    "source_track": item.source_track,
                    "source_start": item.source_start,
                    "source_length": item.source_length,
                    "dst_timeline_start": int(t_dst),
                    "dst_timeline_length": int(l_dst),
                    "dst_lane": item.lane,
                    "src_positions": [
                        {
                            "timeline_start": s.timeline_start,
                            "timeline_length": s.timeline_length,
                            "lane": s.lane,
                        }
                        for s in src_by_identity[identity]
                    ],
                }
            )
        else:
            new_items.append(asdict(item))

    edit_extras = list((dst_edit_counter - src_edit_counter).elements())
    dst_edit_lookup: dict[tuple[tuple[str, int, int, int, str], int, int], list[ClipPosition]] = defaultdict(list)
    for item in dst_items:
        dst_edit_lookup[item.edit_position_key].append(item)
    for key in edit_extras:
        identity, edit_dst, l_dst = key
        item = dst_edit_lookup[key].pop()
        if identity not in src_by_identity:
            continue
        edit_shifted.append(
            {
                "name": item.name,
                "source_id": item.source_id,
                "source_track": item.source_track,
                "source_start": item.source_start,
                "source_length": item.source_length,
                "dst_timeline_start": item.timeline_start,
                "dst_edit_start": int(edit_dst),
                "dst_timeline_length": int(l_dst),
                "dst_lane": item.lane,
                "src_positions": [
                    {
                        "timeline_start": s.timeline_start,
                        "edit_start": s.edit_start,
                        "timeline_length": s.timeline_length,
                        "lane": s.lane,
                    }
                    for s in src_by_identity[identity]
                ],
            }
        )

    missing = list((src_edit_counter - dst_edit_counter).elements())
    classified_errors: list[str] = []
    if classified_plan is None:
        inversions = _aligned_order_inversions(src_items, dst_items)
    else:
        inversions, classified_errors = _classified_order_inversions(src_items, dst_items, classified_plan)
    report = {
        "src": str(src_label),
        "dst": str(dst_label),
        "src_clips": len(src_items),
        "dst_clips": len(dst_items),
        "missing_or_removed": len(missing),
        "new_at_unseen_identity": len(new_items),
        "raw_shifted_or_length_changed": len(shifted),
        "edit_shifted_or_length_changed": len(edit_shifted),
        "aligned_order_inversions": len(inversions),
        "aligned_order_basis": "actual_classified_plan" if classified_plan is not None else "inferred_lane_bands",
        "classified_plan_errors": classified_errors,
        "shifted_or_length_changed": len(shifted),
        "shifted_sample": _sample(shifted, max_report),
        "edit_shifted_sample": _sample(edit_shifted, max_report),
        "aligned_order_inversion_sample": _sample(inversions, max_report),
        "new_sample": _sample(new_items, max_report),
    }
    return (1 if edit_shifted or new_items or inversions or classified_errors else 0), report


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Verify that lane layout did not change DAW-visible horizontal positions. "
            "Removed clips are allowed; surviving clips must keep original visible/edit start, "
            "Length, source window, and aligned channel order. Raw AAF T may change when SDK "
            "transition normalization requires it to preserve the visible position."
        )
    )
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--classified-plan", type=Path, help="Actual-run classified occurrence manifest")
    ap.add_argument("--max-report", type=int, default=40)
    args = ap.parse_args()

    plan = None if args.classified_plan is None else ClassifiedLayoutPlan.from_dict(
        json.loads(args.classified_plan.read_text(encoding="utf8")))
    code, report = compare_positions(args.src, args.dst, max_report=args.max_report, classified_plan=plan)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if code == 0:
        print("OK: surviving clips preserved visible timeline position and aligned channel order.")
    else:
        print("ERROR: surviving clips changed visible timeline position or aligned channel order.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
