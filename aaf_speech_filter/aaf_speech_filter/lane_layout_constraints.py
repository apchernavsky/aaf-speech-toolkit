from __future__ import annotations

from typing import Any, Optional

from .lane_layout_placement import event_placement_start, event_target_raw_start_for_lane


def event_preserves_raw_placement_start(
    event: dict[str, Any],
    lane: int,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> bool:
    if transition_spans_by_lane is None:
        return True
    target_raw = event_target_raw_start_for_lane(event, int(lane), transition_spans_by_lane)
    return event_placement_start(event, int(target_raw)) >= 0


def target_lane_allowed_for_event(
    event: dict[str, Any],
    lane: int,
    *,
    blocked_target_lanes: Optional[set[int]] = None,
    lane_mutable_by_lane: Optional[dict[int, bool]] = None,
    lane_signature_by_lane: Optional[dict[int, tuple[Any, ...]]] = None,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]] = None,
) -> bool:
    src_lane = int(event["src_lane"])
    lane_i = int(lane)
    bounds = event.get("lane_bounds")
    if bounds is not None and not int(bounds[0]) <= lane_i <= int(bounds[1]):
        return False
    if blocked_target_lanes is not None and lane_i in blocked_target_lanes and lane_i != src_lane:
        return False
    if lane_mutable_by_lane is not None:
        if not bool(lane_mutable_by_lane.get(src_lane, False)):
            return False
        if not bool(lane_mutable_by_lane.get(lane_i, False)):
            return False
    if lane_signature_by_lane is not None and (
        lane_signature_by_lane.get(lane_i) != lane_signature_by_lane.get(src_lane)
    ):
        return False
    return event_preserves_raw_placement_start(event, lane_i, transition_spans_by_lane)


def apply_immutable_aligned_lane_bounds(events, immutable_lanes, lane_count):
    """Keep unknown events fixed and aligned peers on their side of fixed members.

    Bounds travel with each event through planning and both writers, so repair
    and final compaction cannot turn missing classification into a lane move.
    """
    groups = {}
    for event in events:
        event.pop("lane_bounds", None)
        if event.get("kind") == "unknown":
            source = int(event["src_lane"])
            event["lane_bounds"] = (source, source)
        if int(event.get("source_start", -1)) < 0 or int(event.get("source_length", 0)) <= 0:
            continue
        key = (int(event.get("T_edit", event.get("T_nuendo", event["T"]))),
               int(event["L"]), int(event["source_start"]), int(event["source_length"]),
               str(event.get("kind") or "unknown"))
        groups.setdefault(key, []).append(event)
    for group in groups.values():
        anchors = sorted({int(e["src_lane"]) for e in group if int(e["src_lane"]) in immutable_lanes})
        if not anchors:
            continue
        for event in group:
            source = int(event["src_lane"])
            if source in anchors or event.get("kind") == "unknown":
                event["lane_bounds"] = (source, source)
            else:
                lower = max((lane + 1 for lane in anchors if lane < source), default=0)
                upper = min((lane - 1 for lane in anchors if lane > source), default=lane_count - 1)
                event["lane_bounds"] = (lower, upper)
