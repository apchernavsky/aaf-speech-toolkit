from __future__ import annotations

from .composition import select_composition
from .lane_order import class_ordered_lanes, reorder_xml_tracks
from .lane_assignment import ordered_lane_assignment

import copy
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Optional

from aaf_io.sdk_tools import find_aaffmtconv, run_aaffmtconv_to_aaf, run_aaffmtconv_to_xml
from aaf_io.sdk_xml_edit import splice_composition_package
from aaf_io.audio_silence import SilenceSignature, xml_silence_replacement_signature
from aaf_io.heal.cfb import copy_and_sync_fat_header

_XmlLanePart = tuple[
    int,
    int,
    Any,
    str,
    Optional[int],
    int,
    int,
    Optional[str],
    int,
    int,
    Optional[int],
    Optional[int],
]


def _xml_ns(tag: str) -> str:
    return "{http://www.aafassociation.org/aafx/v1.1/20090617}" + tag


def _xml_lane_render_signature(wrapper, lane):
    if wrapper is None:
        return ()
    operation = wrapper.findtext(_xml_ns("Operation"), "").strip()
    if operation not in ("OperationDef_MonoAudioPan", "9d2ea893-0968-11d3-8a38-0050040ef7d2", "urn:uuid:9d2ea893-0968-11d3-8a38-0050040ef7d2"):
        return ("unsupported", lane)
    parameters = wrapper.find(_xml_ns("Parameters"))
    constant_pan = parameters is not None and all(child.tag == _xml_ns("ConstantValue") for child in parameters)
    def semantic_tree(element):
        return (element.tag, (element.text or "").strip(),
                tuple(sorted((k, v) for k, v in element.attrib.items() if k != "uid")),
                tuple(semantic_tree(child) for child in element
                      if child.tag != _xml_ns("InputSegments")
                      and not (constant_pan and element is wrapper and child.tag == _xml_ns("ComponentLength"))))
    return semantic_tree(wrapper)


def _rebuilt_lane_total_length(original_total: int, placed_end: int) -> int:
    return max(int(original_total), int(placed_end))


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) < min(int(a1), int(b1))


def _dedupe_exact_transition_spans(
    spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for t, ln in spans:
        key = (int(t), int(ln))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _raw_from_visible_with_transition_spans(
    transition_spans: list[tuple[int, int]],
    visible_t: int,
    preferred_offset: int = 0,
) -> int:
    transition_spans = _dedupe_exact_transition_spans(list(transition_spans))
    preferred_offset = int(max(0, preferred_offset))

    def offset_before(raw_t: int) -> int:
        total = 0
        for t, l in sorted(transition_spans):
            if (
                preferred_offset > 0
                and int(l) == int(preferred_offset)
                and int(t) + int(l) == int(raw_t)
            ):
                continue
            if int(t) < int(raw_t):
                total += int(l)
        return int(total)

    raw = int(visible_t) + 2 * int(preferred_offset)
    for _ in range(64):
        nxt = int(visible_t) + 2 * (offset_before(raw) + int(preferred_offset))
        if int(nxt) == int(raw):
            return int(raw)
        raw = int(nxt)
    return int(raw)


def _spans_without_exact_matches(
    spans: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    excluded_keys = {(int(t), int(ln)) for t, ln in excluded}
    out: list[tuple[int, int]] = []
    for span in spans:
        key = (int(span[0]), int(span[1]))
        if key not in excluded_keys:
            out.append(key)
    return out


def _raw_from_visible_for_part(
    transition_spans: list[tuple[int, int]],
    own_transition_spans: list[tuple[int, int]],
    visible_t: int,
    preferred_offset: int,
    event_offset: int,
) -> int:
    external_spans = _spans_without_exact_matches(
        list(transition_spans),
        list(own_transition_spans),
    )
    raw_t = _raw_from_visible_with_transition_spans(
        external_spans,
        int(visible_t),
        int(preferred_offset),
    )
    return int(raw_t) - int(event_offset)


def _xml_lane_occupancy_intervals(parts: list[_XmlLanePart]) -> list[tuple[int, int]]:
    return [
        (int(t), int(l))
        for t, l, *_rest in parts
        if int(l) > 0
    ]


def _xml_component_length(node_xml: Any) -> int:
    ln_el = node_xml.find(_xml_ns("ComponentLength"))
    try:
        return int((ln_el.text or "0").strip()) if ln_el is not None else 0
    except Exception:
        return 0


def _xml_component_total_length(nodes: list[Any]) -> int:
    return sum(max(0, _xml_component_length(node)) for node in nodes)


def _xml_transition_overlap_length(nodes: list[Any]) -> int:
    return sum(
        max(0, _xml_component_length(node))
        for node in nodes
        if node.tag == _xml_ns("Transition")
    )


def _xml_declared_sequence_length_from_children(nodes: list[Any]) -> int:
    component_total = _xml_component_total_length(nodes)
    transition_overlap = _xml_transition_overlap_length(nodes)
    return max(0, int(component_total) - 2 * int(transition_overlap))


def _xml_part_nodes(node_xml: Any) -> list[Any]:
    return list(node_xml) if isinstance(node_xml, list) else [node_xml]


def _xml_part_transition_spans(t: int, node_xml: Any) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cur = int(t)
    for node in _xml_part_nodes(node_xml):
        ln = max(0, _xml_component_length(node))
        if node.tag == _xml_ns("Transition") and ln > 0:
            spans.append((int(cur), int(ln)))
        cur += int(ln)
    return spans


def _trim_leading_xml_transitions_to_cursor(
    t: int,
    length: int,
    node_xml: Any,
    cursor: int,
    *,
    emitted_transitions: Optional[dict[int, Any]] = None,
    transition_sources: Optional[dict[int, Any]] = None,
) -> tuple[int, int, Any]:
    """Coalesce only an original edge already emitted at this raw position."""
    out_t = int(t)
    out_l = int(length)
    if out_t >= int(cursor):
        return out_t, out_l, node_xml
    nodes = _xml_part_nodes(node_xml)
    if not nodes:
        return out_t, out_l, node_xml
    original_was_list = isinstance(node_xml, list)
    changed = False
    while nodes and out_t < int(cursor):
        node = nodes[0]
        node_len = max(0, _xml_component_length(node))
        if node.tag != _xml_ns("Transition") or node_len <= 0:
            break
        if out_t + node_len > int(cursor):
            break
        source = (transition_sources or {}).get(id(node), node)
        if (emitted_transitions or {}).get(out_t) is not source:
            break
        out_t += node_len
        out_l -= node_len
        nodes.pop(0)
        changed = True
    if not changed:
        return int(t), int(length), node_xml
    if original_was_list:
        return int(out_t), max(0, int(out_l)), nodes
    if len(nodes) == 1:
        return int(out_t), max(0, int(out_l)), nodes[0]
    return int(out_t), max(0, int(out_l)), nodes


def _event_candidate_lanes(
    *,
    current_lane: int,
    n_lanes: int,
    class_kind: Optional[str],
    candidate_lanes_by_kind: Optional[dict[str, list[int]]],
) -> list[int]:
    current = int(current_lane)
    if candidate_lanes_by_kind is None or class_kind is None:
        return [current] + [i for i in range(int(n_lanes)) if i != current]
    allowed_raw = candidate_lanes_by_kind.get(str(class_kind))
    if not allowed_raw:
        return [current]
    allowed: list[int] = []
    for lane in allowed_raw:
        lane_i = int(lane)
        if lane_i < 0 or lane_i >= int(n_lanes) or lane_i in allowed:
            continue
        allowed.append(lane_i)
    if current in allowed:
        return [current] + [lane for lane in allowed if lane != current]
    return list(allowed)


@dataclass(frozen=True)
class LaneLayoutEvent:
    src_lane: int
    top_idx: int
    T_edit: int
    L: int
    target_lane: int
    kind: str = "event"
    class_kind: Optional[str] = None
    visible_t: Optional[int] = None
    preferred_transition_offset: int = 0
    pre_top_idx: Optional[int] = None
    pre_len: int = 0
    post_top_idx: Optional[int] = None
    post_len: int = 0
    source_start: Optional[int] = None
    source_length: Optional[int] = None
    raw_t_authoritative: bool = False
    silence_signature: Optional[SilenceSignature] = None
    lane_bounds: Optional[tuple[int, int]] = None


def apply_lane_layout_via_aaf_sdk_xml(
    *,
    input_aaf: Path,
    output_aaf: Path,
    events: list[LaneLayoutEvent],
    n_lanes: int,
    candidate_lanes_by_kind: Optional[dict[str, list[int]]] = None,
    aaf_tools_dir: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    cancel_check: Optional[Callable[[], None]] = None,
    immutable_lanes: Optional[set[int]] = None,
    composition_id: Optional[str] = None,
    order_tracks_by_class: bool = False,
    result_out: Optional[dict[str, Any]] = None,
) -> None:
    """
    Apply lane layout by editing AAF SDK XML and rebuilding with aaffmtconv.

    This is the Nuendo-safe path for Premiere exports when PyAAF2 write-back is fragile.
    """
    input_aaf = Path(input_aaf).resolve()
    output_aaf = Path(output_aaf).resolve()
    immutable_lane_set = {int(lane) for lane in (immutable_lanes or set())}

    aaffmtconv = find_aaffmtconv(aaf_tools_dir)
    wd = Path(work_dir).resolve() if work_dir is not None else output_aaf.parent
    wd.mkdir(parents=True, exist_ok=True)
    xml_path = wd / f"{output_aaf.name}.__lane_layout.xml"
    streams_dir = xml_path.parent / f"{xml_path.stem}_streams"
    # If the caller didn't provide an explicit per-run work dir, some call sites pass a stable
    # default under "__aaf_tool_work/<stem>.__lane_layout". We should clean it up too to avoid
    # leaving a permanent temp folder after a successful run.
    created_default_lane_dir = False
    try:
        if work_dir is not None:
            # Recognize the default pattern: .../<input_parent>/__aaf_tool_work/<stem>.__lane_layout
            created_default_lane_dir = (
                wd.name.endswith(".__lane_layout") and wd.parent.name == "__aaf_tool_work"
            )
    except Exception:
        created_default_lane_dir = False

    try:
        # Some "bad" Nuendo/Premiere AAFs crash AAF SDK tools due to inconsistent CFB header
        # (fat_sector_count mismatch). Retry `-xml` once on a healed copy.
        xml_input = input_aaf
        try:
            run_aaffmtconv_to_xml(aaffmtconv, xml_input, xml_path, cancel_check=cancel_check)
        except RuntimeError as ex:
            msg = str(ex)
            crashish = ("aaffmtconv -xml failed" in msg) and (
                "3221225477" in msg
                or "3221226519" in msg
                or "0xC0000005" in msg.upper()
                or "C0000005" in msg.upper()
            )
            if not crashish:
                raise
            healed = wd / f"{input_aaf.name}.__sdk_healed.aaf"
            try:
                copy_and_sync_fat_header(input_aaf, healed)
                xml_input = healed
                run_aaffmtconv_to_xml(aaffmtconv, xml_input, xml_path, cancel_check=cancel_check)
            finally:
                try:
                    healed.unlink(missing_ok=True)
                except Exception:
                    pass

        xml_text = xml_path.read_text(encoding="utf-8", errors="replace")
        root = ET.fromstring(xml_text)

        comp_pkgs = root.findall(".//" + _xml_ns("CompositionPackage"))
        comp = select_composition(((package, package.findtext(_xml_ns("PackageID")),
            package.findtext(_xml_ns("PackageUsage")) == "Usage_TopLevel")
            for package in comp_pkgs), composition_id)

        tracks = comp.find(_xml_ns("PackageTracks"))
        if tracks is None:
            raise RuntimeError("SDK XML: CompositionPackage/PackageTracks not found")

        def _og_first_sequence(track_segment):
            og = track_segment.find(_xml_ns("OperationGroup"))
            if og is None:
                return None, None
            ins = og.find(_xml_ns("InputSegments"))
            if ins is None:
                return og, None
            return og, ins.find(_xml_ns("Sequence"))

        # Sound TimelineTracks that have a Sequence (direct or wrapped) with Sound DD.
        sound_tracks: list[tuple[Any, Any, Any, Any]] = []
        for tt in list(tracks):
            if tt.tag != _xml_ns("TimelineTrack"):
                continue
            seg = tt.find(_xml_ns("TrackSegment"))
            if seg is None:
                continue
            seq = seg.find(_xml_ns("Sequence"))
            seq_wrapper = None
            if seq is None:
                seq_wrapper, seq = _og_first_sequence(seg)
            if seq is None:
                continue
            comps = seq.find(_xml_ns("ComponentObjects"))
            if comps is None:
                continue
            first = None
            try:
                first = list(comps)[0]
            except Exception:
                first = None
            if first is None:
                continue
            dd = first.find(_xml_ns("ComponentDataDefinition"))
            if dd is None or (dd.text or "").strip() != "DataDef_Sound":
                continue
            sound_tracks.append((tt, seq, comps, seq_wrapper))

        if len(sound_tracks) != int(n_lanes):
            raise RuntimeError(
                f"SDK XML: sound track count mismatch: xml={len(sound_tracks)} lanes={n_lanes}"
            )

        lane_clocks = []
        for lane, (track, _seq, _comps, wrapper) in enumerate(sound_tracks):
            rate_text = track.findtext(_xml_ns("EditRate"))
            origin_text = track.findtext(_xml_ns("Origin"), "0")
            lane_clocks.append((Fraction(rate_text) if rate_text else None, int(origin_text), _xml_lane_render_signature(wrapper, lane)))

        original_declared_len_by_lane: list[int] = []
        for _tt, seq, comps, _wrapper in sound_tracks:
            total_len_el = seq.find(_xml_ns("ComponentLength"))
            try:
                declared_len = (
                    int((total_len_el.text or "0").strip())
                    if total_len_el is not None
                    else 0
                )
            except Exception:
                declared_len = 0
            if declared_len <= 0:
                declared_len = _xml_declared_sequence_length_from_children(list(comps))
            original_declared_len_by_lane.append(max(0, int(declared_len)))
        composition_declared_len = max(original_declared_len_by_lane, default=0)

        filler_template = None
        for _tt, _seq, comps, _wrapper in sound_tracks:
            for ch in list(comps):
                if ch.tag == _xml_ns("Filler"):
                    dd = ch.find(_xml_ns("ComponentDataDefinition"))
                    if dd is not None and (dd.text or "").strip() == "DataDef_Sound":
                        filler_template = ch
                        break
            if filler_template is not None:
                break
        if filler_template is None:
            filler_template = ET.Element(_xml_ns("Filler"))

        def make_filler(length: int):
            f = copy.deepcopy(filler_template)
            ln_el = f.find(_xml_ns("ComponentLength"))
            if ln_el is None:
                ln_el = ET.SubElement(f, _xml_ns("ComponentLength"))
            ln_el.text = str(int(length))
            dd_el = f.find(_xml_ns("ComponentDataDefinition"))
            if dd_el is None:
                dd_el = ET.SubElement(f, _xml_ns("ComponentDataDefinition"))
            dd_el.text = "DataDef_Sound"
            return f

        by_target: dict[int, list[_XmlLanePart]] = {
            i: [] for i in range(n_lanes)
        }
        covered_indices: dict[int, set[int]] = {i: set() for i in range(n_lanes)}
        planned_indices: set[tuple[int, int]] = set()
        event_bounds = {(int(e.src_lane), int(e.top_idx)): e.lane_bounds for e in events}

        def lane_within_bounds(src_lane, top_idx, target_lane):
            bounds = event_bounds.get((int(src_lane), int(top_idx)))
            return bounds is None or int(bounds[0]) <= int(target_lane) <= int(bounds[1])

        transition_sources: dict[int, Any] = {}

        def copy_transition(node):
            cloned = copy.deepcopy(node)
            transition_sources[id(cloned)] = node
            return cloned

        def record_emitted_transitions(start, node_xml, emitted):
            position = int(start)
            for node in _xml_part_nodes(node_xml):
                if node.tag == _xml_ns("Transition"):
                    emitted[position] = transition_sources.get(id(node), node)
                position += max(0, _xml_component_length(node))

        for e in events:
            if cancel_check is not None:
                cancel_check()
            src_lane = int(e.src_lane)
            tl = int(e.target_lane)
            ti = int(e.top_idx)
            if src_lane < 0 or src_lane >= n_lanes or tl < 0 or tl >= n_lanes:
                raise RuntimeError("SDK XML: event lane out of range")
            if not lane_within_bounds(src_lane, ti, tl):
                raise RuntimeError("SDK XML: event crossed a fixed aligned member")
            if lane_clocks[src_lane] != lane_clocks[tl]:
                raise RuntimeError("SDK XML: event cannot move between incompatible lane clocks")
            if (src_lane, ti) in planned_indices:
                raise RuntimeError("SDK XML: duplicate planned component")
            planned_indices.add((src_lane, ti))
            if src_lane in immutable_lane_set:
                if tl != src_lane:
                    raise RuntimeError(
                        "SDK XML: immutable lane event cannot move "
                        f"src_lane={src_lane} target_lane={tl} top_idx={ti}"
                    )
                continue
            if tl in immutable_lane_set:
                raise RuntimeError(
                    "SDK XML: mutable event cannot target immutable lane "
                    f"src_lane={src_lane} target_lane={tl} top_idx={ti}"
                )
            _tt, _seq, comps, _wrapper = sound_tracks[src_lane]
            children = list(comps)
            if ti < 0 or ti >= len(children):
                raise RuntimeError(
                    f"SDK XML: component index mismatch lane={src_lane} idx={ti} len={len(children)}"
                )
            if _xml_component_length(children[ti]) != int(e.L):
                raise RuntimeError("SDK XML: planned component length mismatch")
            covered_indices[src_lane].add(ti)
            part_kind = str(e.kind or "event")
            if part_kind == "silence":
                signature = xml_silence_replacement_signature(children, ti)
                if (tl != src_lane or e.silence_signature is None
                        or signature != e.silence_signature
                        or sum(_xml_component_length(ch) for ch in children[:ti]) != int(e.T_edit)):
                    raise RuntimeError("SDK XML: planned silence identity mismatch")
                # Covered silence becomes a gap; normal filler emission preserves timing.
                continue
            node_xml: Any = (
                copy_transition(children[ti])
                if children[ti].tag == _xml_ns("Transition")
                else copy.deepcopy(children[ti])
            )
            event_offset = 0
            part_len = int(e.L)
            part_t = int(e.T_edit)
            if part_kind == "event":
                nodes: list[Any] = []
                pre_len = max(0, int(e.pre_len or 0))
                if e.pre_top_idx is not None and pre_len > 0:
                    pre_idx = int(e.pre_top_idx)
                    if pre_idx < 0 or pre_idx >= len(children):
                        raise RuntimeError(
                            f"SDK XML: pre-transition index mismatch lane={src_lane} idx={pre_idx} len={len(children)}"
                        )
                    if children[pre_idx].tag != _xml_ns("Transition") or _xml_component_length(children[pre_idx]) != pre_len:
                        raise RuntimeError("SDK XML: pre-transition identity mismatch")
                    covered_indices[src_lane].add(pre_idx)
                    nodes.append(copy_transition(children[pre_idx]))
                    event_offset = int(pre_len)
                nodes.append(copy.deepcopy(children[ti]))
                post_len = max(0, int(e.post_len or 0))
                if e.post_top_idx is not None and post_len > 0:
                    post_idx = int(e.post_top_idx)
                    if post_idx < 0 or post_idx >= len(children):
                        raise RuntimeError(
                            f"SDK XML: post-transition index mismatch lane={src_lane} idx={post_idx} len={len(children)}"
                        )
                    if children[post_idx].tag != _xml_ns("Transition") or _xml_component_length(children[post_idx]) != post_len:
                        raise RuntimeError("SDK XML: post-transition identity mismatch")
                    covered_indices[src_lane].add(post_idx)
                    nodes.append(copy_transition(children[post_idx]))
                node_xml = nodes
                part_len = int(pre_len) + int(e.L) + int(post_len)
                part_t = int(e.T_edit) - int(event_offset)
            by_target[tl].append(
                (
                    int(part_t),
                    int(part_len),
                    node_xml,
                    part_kind,
                    None if e.visible_t is None or bool(e.raw_t_authoritative) else int(e.visible_t),
                    int(e.preferred_transition_offset),
                    int(event_offset),
                    None if e.class_kind is None else str(e.class_kind),
                    int(src_lane),
                    int(ti),
                    None if e.source_start is None else int(e.source_start),
                    None if e.source_length is None else int(e.source_length),
                )
            )

        # A plan may cover only selected clips. Preserve every unclaimed original
        # non-filler node at its original raw coordinate, including opaque objects.
        for lane, (_track, _seq, comps, _wrapper) in enumerate(sound_tracks):
            if lane in immutable_lane_set:
                continue
            position = 0
            for top_idx, node in enumerate(comps):
                length = _xml_component_length(node)
                if length < 0:
                    raise RuntimeError("SDK XML: negative original component length")
                if top_idx not in covered_indices[lane] and node.tag != _xml_ns("Filler"):
                    kind = "transition" if node.tag == _xml_ns("Transition") else "structural"
                    retained = copy_transition(node) if kind == "transition" else copy.deepcopy(node)
                    by_target[lane].append((position, length, retained, kind,
                        None, 0, 0, None, lane, top_idx, None, None))
                    covered_indices[lane].add(top_idx)
                position += length
            expected = {i for i, node in enumerate(comps) if node.tag != _xml_ns("Filler")}
            if not expected.issubset(covered_indices[lane]):
                raise RuntimeError("SDK XML: incomplete original component coverage")

        def transition_spans_for_lane(lane: int) -> list[tuple[int, int]]:
            spans: list[tuple[int, int]] = []
            for t, l, node, kind, *_rest in by_target.get(int(lane), []):
                if str(kind) == "transition" and int(l) > 0:
                    spans.append((int(t), int(l)))
                elif str(kind) == "event":
                    spans.extend(_xml_part_transition_spans(int(t), node))
            return sorted(spans)

        def lane_preserves_event_visible_time(
            lane: int,
            t: int,
            node_xml: Any,
            kind: str,
            visible_t: Optional[int],
            preferred_offset: int,
            event_offset: int,
        ) -> bool:
            if visible_t is None:
                return True
            out_t = _raw_from_visible_for_part(
                transition_spans_for_lane(int(lane)),
                _xml_part_transition_spans(int(t), node_xml) if str(kind) == "event" else [],
                int(visible_t),
                int(preferred_offset),
                int(event_offset),
            )
            return int(out_t) == int(t)

        def part_t_for_lane(
            lane: int,
            t: int,
            node_xml: Any,
            kind: str,
            visible_t: Optional[int],
            preferred_offset: int,
            event_offset: int,
        ) -> int:
            if visible_t is None:
                return int(t)
            return _raw_from_visible_for_part(
                transition_spans_for_lane(int(lane)),
                _xml_part_transition_spans(int(t), node_xml) if str(kind) == "event" else [],
                int(visible_t),
                int(preferred_offset),
                int(event_offset),
            )

        def output_t_for_current_lane_cursor(
            t: int,
            kind: str,
            visible_t: Optional[int],
            event_offset: int,
            *,
            node_xml: Any,
            emitted_transitions: dict[int, Any],
            cur: int,
            transition_before: int,
        ) -> int:
            if str(kind) == "event" and visible_t is not None:
                base = int(visible_t) + 2 * int(transition_before)
                if int(event_offset) > 0:
                    overlapped_pre_t = int(base) - int(event_offset)
                    if int(overlapped_pre_t) < int(cur) <= int(base):
                        trimmed_t, _, _ = _trim_leading_xml_transitions_to_cursor(
                            overlapped_pre_t, _xml_component_total_length(_xml_part_nodes(node_xml)),
                            node_xml, cur, emitted_transitions=emitted_transitions,
                            transition_sources=transition_sources,
                        )
                        if trimmed_t >= int(cur):
                            return int(overlapped_pre_t)
                return int(base) + int(event_offset)
            return int(t)

        def first_unserializable_part(parts: list[_XmlLanePart]) -> Optional[_XmlLanePart]:
            cur = 0
            transition_before = 0
            emitted_transitions: dict[int, Any] = {}
            ordered = sorted(
                list(parts),
                key=lambda x: (
                    x[4] if x[3] == "event" and x[4] is not None else x[0],
                    0 if x[3] in ("structural", "transition") else 1,
                    x[1],
                ),
            )
            for item in ordered:
                (
                    t,
                    l,
                    node_xml,
                    kind,
                    visible_t,
                    _preferred_offset,
                    event_offset,
                    _class_kind,
                    _src_lane,
                    _top_idx,
                    _source_start,
                    _source_length,
                ) = item
                out_t = output_t_for_current_lane_cursor(
                    int(t),
                    str(kind),
                    visible_t,
                    int(event_offset),
                    node_xml=node_xml,
                    emitted_transitions=emitted_transitions,
                    cur=int(cur),
                    transition_before=int(transition_before),
                )
                out_l = int(l)
                out_node_xml = node_xml
                if out_t > int(cur):
                    cur = int(out_t)
                if out_t < int(cur):
                    out_t, out_l, out_node_xml = _trim_leading_xml_transitions_to_cursor(
                        out_t,
                        out_l,
                        out_node_xml,
                        cur,
                        emitted_transitions=emitted_transitions,
                        transition_sources=transition_sources,
                    )
                    if int(out_l) <= 0 or not _xml_part_nodes(out_node_xml):
                        continue
                    if out_t < int(cur):
                        return item
                record_emitted_transitions(out_t, out_node_xml, emitted_transitions)
                transition_before += _xml_transition_overlap_length(
                    _xml_part_nodes(out_node_xml)
                )
                cur = int(out_t) + int(out_l)
            return None

        def lane_is_serializable(parts: list[_XmlLanePart]) -> bool:
            return first_unserializable_part(parts) is None

        def relocate_to_later_lane(lane: int, part: _XmlLanePart) -> bool:
            if int(lane) in immutable_lane_set:
                return False
            (
                t,
                l,
                node_xml,
                kind,
                visible_t,
                preferred_offset,
                event_offset,
                class_kind,
                src_lane,
                top_idx,
                source_start,
                source_length,
            ) = part
            if str(kind) in ("structural", "transition"):
                return False
            for alt_lane in _event_candidate_lanes(
                current_lane=int(lane),
                n_lanes=int(n_lanes),
                class_kind=class_kind,
                candidate_lanes_by_kind=candidate_lanes_by_kind,
            ):
                if int(alt_lane) in immutable_lane_set:
                    continue
                if not lane_within_bounds(src_lane, top_idx, alt_lane) or lane_clocks[int(src_lane)] != lane_clocks[int(alt_lane)]:
                    continue
                if int(alt_lane) <= int(lane):
                    continue
                alt_t = part_t_for_lane(
                    int(alt_lane),
                    int(t),
                    node_xml,
                    str(kind),
                    visible_t,
                    int(preferred_offset),
                    int(event_offset),
                )
                if not lane_preserves_event_visible_time(
                    int(alt_lane),
                    int(alt_t),
                    node_xml,
                    str(kind),
                    visible_t,
                    int(preferred_offset),
                    int(event_offset),
                ):
                    continue
                conflict = False
                for a_t, a_l, *_a_rest in by_target.get(alt_lane, []):
                    if max(int(alt_t), int(a_t)) < min(
                        int(alt_t) + int(l), int(a_t) + int(a_l)
                    ):
                        conflict = True
                        break
                if not conflict:
                    moved_part: _XmlLanePart = (
                        int(alt_t),
                        int(l),
                        node_xml,
                        kind,
                        visible_t,
                        int(preferred_offset),
                        int(event_offset),
                        class_kind,
                        int(src_lane),
                        int(top_idx),
                        source_start,
                        source_length,
                    )
                    if not lane_is_serializable(
                        list(by_target.get(int(alt_lane), [])) + [moved_part]
                    ):
                        continue
                    by_target.setdefault(int(alt_lane), []).append(moved_part)
                    return True
            return False

        def repair_unserializable_event_lanes() -> int:
            moved = 0
            max_passes = max(1, sum(len(v) for v in by_target.values()) * max(1, int(n_lanes)))
            for _pass_idx in range(max_passes):
                changed = False
                for lane in range(n_lanes):
                    if int(lane) in immutable_lane_set:
                        continue
                    parts = list(by_target.get(int(lane), []))
                    bad = first_unserializable_part(parts)
                    if bad is None:
                        continue
                    (
                        t,
                        l,
                        node_xml,
                        kind,
                        visible_t,
                        preferred_offset,
                        event_offset,
                        class_kind,
                        src_lane,
                        top_idx,
                        source_start,
                        source_length,
                    ) = bad
                    if str(kind) != "event":
                        continue

                    source_without_bad = [item for item in parts if item is not bad]
                    repaired = False
                    for alt_lane in _event_candidate_lanes(
                        current_lane=int(lane),
                        n_lanes=int(n_lanes),
                        class_kind=class_kind,
                        candidate_lanes_by_kind=candidate_lanes_by_kind,
                    ):
                        if int(alt_lane) in immutable_lane_set:
                            continue
                        if not lane_within_bounds(src_lane, top_idx, alt_lane) or lane_clocks[int(src_lane)] != lane_clocks[int(alt_lane)]:
                            continue
                        if int(alt_lane) == int(lane):
                            continue
                        alt_t = part_t_for_lane(
                            int(alt_lane),
                            int(t),
                            node_xml,
                            str(kind),
                            visible_t,
                            int(preferred_offset),
                            int(event_offset),
                        )
                        if not lane_preserves_event_visible_time(
                            int(alt_lane),
                            int(alt_t),
                            node_xml,
                            str(kind),
                            visible_t,
                            int(preferred_offset),
                            int(event_offset),
                        ):
                            continue
                        moved_part: _XmlLanePart = (
                            int(alt_t),
                            int(l),
                            node_xml,
                            str(kind),
                            visible_t,
                            int(preferred_offset),
                            int(event_offset),
                            class_kind,
                            int(src_lane),
                            int(top_idx),
                            source_start,
                            source_length,
                        )
                        if not lane_is_serializable(
                            list(by_target.get(int(alt_lane), [])) + [moved_part]
                        ):
                            continue
                        by_target[int(lane)] = source_without_bad
                        by_target.setdefault(int(alt_lane), []).append(moved_part)
                        moved += 1
                        changed = True
                        repaired = True
                        break
                    if repaired:
                        break
                if not changed:
                    return int(moved)
            return int(moved)

        def raw_interval_conflicts(
            lane: int,
            t: int,
            l: int,
            placed: dict[int, list[tuple[int, int]]],
        ) -> bool:
            for a_t, a_l in placed.get(int(lane), []):
                if max(int(t), int(a_t)) < min(int(t) + int(l), int(a_t) + int(a_l)):
                    return True
            return False

        def rebalance_event_visible_times() -> bool:
            changed = False
            adjusted: dict[int, list[_XmlLanePart]] = {i: [] for i in range(n_lanes)}
            for lane in range(n_lanes):
                for (
                    t,
                    l,
                    node_xml,
                    kind,
                    visible_t,
                    preferred_offset,
                    event_offset,
                    class_kind,
                    src_lane,
                    top_idx,
                    source_start,
                    source_length,
                ) in by_target.get(lane, []):
                    out_t = int(t)
                    if str(kind) == "event" and visible_t is not None:
                        out_t = part_t_for_lane(
                            int(lane),
                            int(t),
                            node_xml,
                            str(kind),
                            visible_t,
                            int(preferred_offset),
                            int(event_offset),
                        )
                    if int(out_t) != int(t):
                        changed = True
                    adjusted.setdefault(int(lane), []).append(
                        (
                            int(out_t),
                            int(l),
                            node_xml,
                            kind,
                            visible_t,
                            int(preferred_offset),
                            int(event_offset),
                            class_kind,
                            int(src_lane),
                            int(top_idx),
                            source_start,
                            source_length,
                        )
                    )
            by_target.clear()
            by_target.update(adjusted)
            return changed

        def resolve_event_overlaps() -> int:
            # Resolve event-event underflows before rewriting any lane. During the final write pass
            # it is too late to move a conflicting event to an earlier lane because that lane may
            # already have been serialized. Structural transition anchors stay fixed; only movable
            # timeline events may be reassigned.
            fixed_by_lane: dict[int, list[_XmlLanePart]] = {
                i: [] for i in range(n_lanes)
            }
            movable_parts: list[tuple[int, _XmlLanePart]] = []
            for lane in range(n_lanes):
                for item in by_target.get(lane, []):
                    (
                        t,
                        _l,
                        _node_xml,
                        kind,
                        _visible_t,
                        _preferred_offset,
                        _event_offset,
                        _class_kind,
                        _src_lane,
                        _top_idx,
                        _source_start,
                        _source_length,
                    ) = item
                    if str(kind) == "event":
                        movable_parts.append((int(lane), item))
                    else:
                        fixed_by_lane[int(lane)].append(item)

            placed: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n_lanes)}
            rebuilt_by_target: dict[int, list[_XmlLanePart]] = {
                i: list(fixed_by_lane.get(i, [])) for i in range(n_lanes)
            }
            for lane in range(n_lanes):
                placed[int(lane)].extend(
                    _xml_lane_occupancy_intervals(fixed_by_lane.get(lane, []))
                )

            moved = 0
            for src_lane, item in sorted(
                movable_parts,
                key=lambda x: (
                    int(x[1][0]) if x[1][4] is None else int(x[1][4]),
                    int(x[0]),
                    -int(x[1][1]),
                ),
            ):
                (
                    t,
                    l,
                    node_xml,
                    kind,
                    visible_t,
                    preferred_offset,
                    event_offset,
                    class_kind,
                    original_src_lane,
                    top_idx,
                    source_start,
                    source_length,
                ) = item
                candidate_lanes = _event_candidate_lanes(
                    current_lane=int(src_lane),
                    n_lanes=int(n_lanes),
                    class_kind=class_kind,
                    candidate_lanes_by_kind=candidate_lanes_by_kind,
                )
                chosen: Optional[tuple[int, int]] = None
                for lane in candidate_lanes:
                    if int(lane) in immutable_lane_set:
                        continue
                    cand_t = part_t_for_lane(
                        int(lane),
                        int(t),
                        node_xml,
                        str(kind),
                        visible_t,
                        int(preferred_offset),
                        int(event_offset),
                    )
                    if not lane_preserves_event_visible_time(
                        int(lane),
                        int(cand_t),
                        node_xml,
                        str(kind),
                        visible_t,
                        int(preferred_offset),
                        int(event_offset),
                    ):
                        continue
                    if raw_interval_conflicts(int(lane), int(cand_t), int(l), placed):
                        continue
                    chosen = (int(lane), int(cand_t))
                    break
                if chosen is None:
                    chosen = (int(src_lane), int(t))
                lane, cand_t = chosen
                rebuilt_by_target.setdefault(int(lane), []).append(
                    (
                        int(cand_t),
                        int(l),
                        node_xml,
                        kind,
                        visible_t,
                        int(preferred_offset),
                        int(event_offset),
                        class_kind,
                        int(original_src_lane),
                        int(top_idx),
                        source_start,
                        source_length,
                    )
                )
                if int(l) > 0:
                    placed.setdefault(int(lane), []).append((int(cand_t), int(l)))
                if int(lane) != int(src_lane) or int(cand_t) != int(t):
                    moved += 1
            by_target.clear()
            by_target.update(rebuilt_by_target)
            return moved

        def class_lane_order(class_kind: Optional[str]) -> list[int]:
            if candidate_lanes_by_kind is None or class_kind is None:
                return [
                    lane
                    for lane in range(n_lanes)
                    if int(lane) not in immutable_lane_set
                ]
            raw = candidate_lanes_by_kind.get(str(class_kind))
            if not raw:
                return []
            out: list[int] = []
            for lane in raw:
                lane_i = int(lane)
                if (
                    0 <= lane_i < int(n_lanes)
                    and lane_i not in immutable_lane_set
                    and lane_i not in out
                ):
                    out.append(lane_i)
            return out

        def preserve_aligned_event_order() -> int:
            groups = {}
            for lane, parts in by_target.items():
                for item in parts:
                    if item[3] != "event" or item[4] is None or item[10] is None or item[11] is None:
                        continue
                    logical_length = int(item[1]) - _xml_transition_overlap_length(_xml_part_nodes(item[2]))
                    key = (int(item[4]), logical_length, int(item[10]), int(item[11]), str(item[7] or "unknown"))
                    groups.setdefault(key, []).append((int(lane), item))
            moved = 0
            for group in groups.values():
                ordered = sorted(group, key=lambda entry: (int(entry[1][8]), int(entry[1][9])))
                current_lanes = [lane for lane, _item in ordered]
                targets = sorted(current_lanes)
                if current_lanes == targets:
                    continue
                if len(set(targets)) != len(targets):
                    raise RuntimeError("SDK XML: aligned events share a destination lane")
                group_ids = {id(item) for _lane, item in group}
                released = {lane: [part for part in parts if id(part) not in group_ids]
                            for lane, parts in by_target.items()}
                candidates = []
                for current, item in ordered:
                    feasible = {}
                    for target in range(n_lanes):
                        if (current in immutable_lane_set or target in immutable_lane_set) and target != current:
                            continue
                        if lane_clocks[int(item[8])] != lane_clocks[target] or not lane_within_bounds(item[8], item[9], target):
                            continue
                        # The serializer derives raw placement from the full transition context.
                        moved_item = item if target in immutable_lane_set else (int(item[4]) + int(item[6]), *item[1:])
                        if lane_is_serializable(released.get(target, []) + [moved_item]):
                            feasible[target] = moved_item
                    candidates.append(feasible)
                assignment = ordered_lane_assignment(
                    [[lane for lane in feasible if lane in targets] for feasible in candidates], current_lanes)
                if assignment is None:
                    assignment = ordered_lane_assignment([list(feasible) for feasible in candidates], current_lanes)
                if assignment is None:
                    raise RuntimeError("SDK XML: aligned order has no compatible placement without overlap")
                for target, feasible in zip(assignment, candidates):
                    released.setdefault(target, []).append(feasible[target])
                by_target.clear()
                by_target.update(released)
                moved += sum(current != target for current, target in zip(current_lanes, assignment))
            return moved

        def compact_events_to_class_zones() -> int:
            occupancy: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n_lanes)}
            events_for_compaction: list[tuple[int, _XmlLanePart]] = []
            for lane in range(n_lanes):
                if int(lane) in immutable_lane_set:
                    continue
                for item in by_target.get(lane, []):
                    t, l, _node_xml, kind, _visible_t, _preferred_offset, _event_offset, class_kind, *_rest = item
                    if int(l) > 0:
                        occupancy.setdefault(int(lane), []).append((int(t), int(t) + int(l)))
                    if str(kind) == "event" and class_lane_order(class_kind):
                        events_for_compaction.append((int(lane), item))

            moved = 0
            for _pass_idx in range(8):
                changed = False
                for current_lane, item in sorted(
                    events_for_compaction,
                    key=lambda pair: (
                        int(pair[1][4]) if pair[1][4] is not None else int(pair[1][0]),
                        str(pair[1][7] or "unknown"),
                        int(pair[0]),
                        int(pair[1][1]),
                    ),
                ):
                    if item not in by_target.get(int(current_lane), []):
                        continue
                    (
                        t,
                        l,
                        node_xml,
                        kind,
                        visible_t,
                        preferred_offset,
                        event_offset,
                        class_kind,
                        src_lane,
                        top_idx,
                        source_start,
                        source_length,
                    ) = item
                    order = class_lane_order(class_kind)
                    if not order:
                        continue
                    try:
                        current_rank = order.index(int(current_lane))
                    except ValueError:
                        current_rank = len(order)
                    src_span = (int(t), int(t) + int(l))
                    try:
                        occupancy[int(current_lane)].remove(src_span)
                    except (KeyError, ValueError):
                        pass

                    chosen: Optional[tuple[int, int]] = None
                    for target_lane in order:
                        if int(target_lane) in immutable_lane_set:
                            continue
                        if int(target_lane) == int(current_lane):
                            continue
                        try:
                            target_rank = order.index(int(target_lane))
                        except ValueError:
                            continue
                        if int(target_rank) >= int(current_rank):
                            continue
                        cand_t = part_t_for_lane(
                            int(target_lane),
                            int(t),
                            node_xml,
                            str(kind),
                            visible_t,
                            int(preferred_offset),
                            int(event_offset),
                        )
                        if not lane_preserves_event_visible_time(
                            int(target_lane),
                            int(cand_t),
                            node_xml,
                            str(kind),
                            visible_t,
                            int(preferred_offset),
                            int(event_offset),
                        ):
                            continue
                        cand_span = (int(cand_t), int(cand_t) + int(l))
                        if any(
                            _intervals_overlap(cand_span[0], cand_span[1], occ0, occ1)
                            for occ0, occ1 in occupancy.get(int(target_lane), [])
                        ):
                            continue
                        chosen = (int(target_lane), int(cand_t))
                        break

                    if chosen is None:
                        occupancy.setdefault(int(current_lane), []).append(src_span)
                        continue

                    try:
                        by_target[int(current_lane)].remove(item)
                    except (KeyError, ValueError):
                        occupancy.setdefault(int(current_lane), []).append(src_span)
                        continue
                    target_lane, cand_t = chosen
                    moved_item: _XmlLanePart = (
                        int(cand_t),
                        int(l),
                        node_xml,
                        str(kind),
                        visible_t,
                        int(preferred_offset),
                        int(event_offset),
                        class_kind,
                        int(src_lane),
                        int(top_idx),
                        source_start,
                        source_length,
                    )
                    by_target.setdefault(int(target_lane), []).append(moved_item)
                    occupancy.setdefault(int(target_lane), []).append(
                        (int(cand_t), int(cand_t) + int(l))
                    )
                    events_for_compaction.remove((int(current_lane), item))
                    events_for_compaction.append((int(target_lane), moved_item))
                    moved += 1
                    changed = True
                if not changed:
                    break
            return moved

        # Lane selection is owned by the upstream planner. The XML writer must
        # serialize the plan without silently changing lanes or raw times; doing
        # another compaction pass here can alter DAW-visible timing on lanes with
        # different transition history.
        repair_unserializable_event_lanes()
        preserve_aligned_event_order()

        for lane in range(n_lanes):
            if cancel_check is not None:
                cancel_check()
            if int(lane) in immutable_lane_set:
                continue
            _tt, seq, comps, seq_wrapper = sound_tracks[lane]
            parts = sorted(
                by_target.get(lane, []),
                key=lambda x: (
                    x[4] if x[3] == "event" and x[4] is not None else x[0],
                    0 if x[3] in ("structural", "transition") else 1,
                    x[1],
                ),
            )

            total_len_el = seq.find(_xml_ns("ComponentLength"))
            original_declared_len = (
                original_declared_len_by_lane[lane]
                if lane < len(original_declared_len_by_lane)
                else 0
            )
            min_declared_len = max(
                int(original_declared_len),
                int(composition_declared_len),
            )

            new_children: list[Any] = []
            cur = 0
            transition_before = 0
            emitted_transitions: dict[int, Any] = {}

            for (
                T,
                L,
                node_xml,
                kind,
                visible_t,
                preferred_offset,
                event_offset,
                class_kind,
                src_lane,
                top_idx,
                source_start,
                source_length,
            ) in parts:
                out_T = output_t_for_current_lane_cursor(
                    int(T),
                    str(kind),
                    visible_t,
                    int(event_offset),
                    node_xml=node_xml,
                    emitted_transitions=emitted_transitions,
                    cur=int(cur),
                    transition_before=int(transition_before),
                )
                if out_T > cur:
                    new_children.append(make_filler(out_T - cur))
                    cur = out_T
                if out_T < cur:
                    out_T, L, node_xml = _trim_leading_xml_transitions_to_cursor(
                        out_T,
                        L,
                        node_xml,
                        cur,
                        emitted_transitions=emitted_transitions,
                        transition_sources=transition_sources,
                    )
                    if int(L) <= 0 or not _xml_part_nodes(node_xml):
                        continue
                    if out_T < cur:
                        if relocate_to_later_lane(
                            lane,
                            (
                                out_T,
                                L,
                                node_xml,
                                kind,
                                visible_t,
                                preferred_offset,
                                event_offset,
                                class_kind,
                                src_lane,
                                top_idx,
                                source_start,
                                source_length,
                            ),
                        ):
                            continue
                        raise RuntimeError(
                            "SDK XML: layout underflow "
                            f"lane={lane} T={out_T} cur={cur} L={L} "
                            f"kind={kind} class={class_kind or 'unknown'} visible={visible_t}"
                        )
                record_emitted_transitions(out_T, node_xml, emitted_transitions)
                new_children.extend(_xml_part_nodes(node_xml))
                transition_before += _xml_transition_overlap_length(
                    _xml_part_nodes(node_xml)
                )
                cur = out_T + L
            if not new_children:
                pad = max(1, int(min_declared_len or 1))
                new_children = [make_filler(pad)]
            else:
                final_declared_len = _xml_declared_sequence_length_from_children(
                    new_children
                )
                if final_declared_len < min_declared_len:
                    new_children.append(make_filler(min_declared_len - final_declared_len))
            final_declared_len = _xml_declared_sequence_length_from_children(new_children)

            for ch in list(comps):
                comps.remove(ch)
            for ch in new_children:
                comps.append(ch)

            if total_len_el is None:
                total_len_el = ET.SubElement(seq, _xml_ns("ComponentLength"))
            total_len_el.text = str(int(final_declared_len))
            if seq_wrapper is not None:
                wrapper_len_el = seq_wrapper.find(_xml_ns("ComponentLength"))
                if wrapper_len_el is None:
                    wrapper_len_el = ET.SubElement(seq_wrapper, _xml_ns("ComponentLength"))
                wrapper_len_el.text = str(int(final_declared_len))

        if order_tracks_by_class:
            kinds_by_lane = [set() for _ in range(n_lanes)]
            for event in events:
                if int(event.src_lane) in immutable_lane_set and event.kind == "event":
                    kinds_by_lane[int(event.src_lane)].add(event.class_kind or "unknown")
            for lane, parts in by_target.items():
                for part in parts:
                    if part[3] == "event":
                        kinds_by_lane[lane].add(part[7] or "unknown")
                    elif part[3] == "structural":
                        kinds_by_lane[lane].add("unknown")
            lane_order = class_ordered_lanes(kinds_by_lane, protected=immutable_lane_set)
            reorder_xml_tracks(tracks, [track for track, _seq, _comps, _wrapper in sound_tracks], lane_order)
            if result_out is not None:
                result_out["lane_order"] = lane_order
        ET.register_namespace("", "http://www.aafassociation.org/aafx/v1.1/20090617")
        ET.register_namespace("aaf", "http://www.aafassociation.org/aafx/v1.1/20090617")
        comp_xml = ET.tostring(comp, encoding="unicode")

        xml_text2 = splice_composition_package(xml_text, new_comp_xml=comp_xml, composition_id=composition_id)
        xml_path.write_text(xml_text2, encoding="utf-8")

        run_aaffmtconv_to_aaf(aaffmtconv, xml_path, output_aaf, cancel_check=cancel_check)
    finally:
        try:
            xml_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if streams_dir.exists():
                shutil.rmtree(streams_dir, ignore_errors=True)
        except Exception:
            pass
        # Best-effort cleanup of the default lane-layout temp directory.
        if created_default_lane_dir:
            try:
                shutil.rmtree(wd, ignore_errors=True)
            except Exception:
                pass
            # Prune empty parents: .../__aaf_tool_work
            try:
                parent = wd.parent
                if parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()
            except Exception:
                pass
