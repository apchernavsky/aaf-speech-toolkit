from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .lane_layout_placement import (
    event_placement_length,
    event_placement_start,
    event_target_raw_start_for_lane,
    event_visible_occupancy_length,
    event_visible_start_for_placement,
)


@dataclass(frozen=True)
class LaneOccupancySnapshot:
    raw: list[list[tuple[int, int]]]
    visible: list[list[tuple[int, int]]]


def _current_event_lane(event: dict[str, Any], lane_count: int) -> Optional[int]:
    lane = int(event.get("target_lane", event["src_lane"]))
    if lane < 0 or lane >= int(lane_count):
        lane = int(event["src_lane"])
    if lane < 0 or lane >= int(lane_count):
        return None
    return int(lane)


def _current_event_raw_start(
    event: dict[str, Any],
    lane: int,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> int:
    if "target_T" in event:
        return int(event["target_T"])
    return event_target_raw_start_for_lane(event, int(lane), transition_spans_by_lane)


def build_lane_occupancy_snapshot(
    events: list[dict[str, Any]],
    lane_count: int,
    *,
    reserved_by_lane: Optional[dict[int, list[tuple[int, int]]]] = None,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]] = None,
) -> LaneOccupancySnapshot:
    n = int(lane_count)
    raw: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    visible: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    if reserved_by_lane:
        for lane, spans in reserved_by_lane.items():
            lane_i = int(lane)
            if lane_i < 0 or lane_i >= n:
                continue
            for t0, t1 in spans:
                if int(t1) > int(t0):
                    raw[lane_i].append((int(t0), int(t1)))
    for event in events:
        lane = _current_event_lane(event, n)
        if lane is None:
            continue
        event_t = _current_event_raw_start(event, lane, transition_spans_by_lane)
        t0 = event_placement_start(event, event_t)
        plen = event_placement_length(event)
        t1 = int(t0) + int(plen)
        if int(t1) <= int(t0):
            continue
        raw[lane].append((int(t0), int(t1)))
        v0 = event_visible_start_for_placement(event)
        visible[lane].append((int(v0), int(v0) + event_visible_occupancy_length(event)))
    return LaneOccupancySnapshot(raw=raw, visible=visible)
