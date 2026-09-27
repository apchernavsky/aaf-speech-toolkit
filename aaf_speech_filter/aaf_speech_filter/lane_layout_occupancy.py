from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Optional

from .lane_layout_model import intervals_overlap
from .lane_layout_placement import event_placement_length, event_placement_start, event_source_raw_start


def _overlap_is_covered_by_ignored_span(
    overlap_start: int,
    overlap_end: int,
    ignored_spans: Optional[list[tuple[int, int]]],
) -> bool:
    if not ignored_spans:
        return False
    return any(
        int(a) <= int(overlap_start) and int(overlap_end) <= int(b)
        for a, b in ignored_spans
    )


def spans_conflict_with_occupancy(
    occupancy: list[tuple[int, int]],
    start: int,
    end: int,
    ignored_spans: Optional[list[tuple[int, int]]] = None,
) -> bool:
    for u, v in occupancy:
        if intervals_overlap(start, end, u, v):
            ov0 = max(int(start), int(u))
            ov1 = min(int(end), int(v))
            if _overlap_is_covered_by_ignored_span(ov0, ov1, ignored_spans):
                continue
            return True
    return False


def lane_occupancy_conflicts(
    raw_occupancy: list[list[tuple[int, int]]],
    lane: int,
    raw_t0: int,
    raw_t1: int,
    *,
    lane_capacity_by_lane: Optional[dict[int, int]] = None,
    visible_occupancy: Optional[list[list[tuple[int, int]]]] = None,
    visible_t0: Optional[int] = None,
    visible_t1: Optional[int] = None,
    ignored_raw_spans: Optional[list[tuple[int, int]]] = None,
    ignored_visible_spans: Optional[list[tuple[int, int]]] = None,
) -> bool:
    lane_i = int(lane)
    if lane_capacity_by_lane is not None:
        cap = int(lane_capacity_by_lane.get(lane_i, 0) or 0)
        if cap > 0 and int(raw_t1) > cap:
            return True
    if spans_conflict_with_occupancy(
        raw_occupancy[lane_i],
        int(raw_t0),
        int(raw_t1),
        ignored_raw_spans,
    ):
        return True
    if visible_occupancy is None or visible_t0 is None or visible_t1 is None:
        return False
    return spans_conflict_with_occupancy(
        visible_occupancy[lane_i],
        int(visible_t0),
        int(visible_t1),
        ignored_visible_spans,
    )


@dataclass
class LaneOccupancyState:
    raw: list[list[tuple[int, int]]]
    visible: list[list[tuple[int, int]]]
    lane_capacity_by_lane: Optional[dict[int, int]] = None

    def conflicts(
        self,
        lane: int,
        raw_t0: int,
        raw_t1: int,
        *,
        visible_t0: Optional[int] = None,
        visible_t1: Optional[int] = None,
        ignored_raw_spans: Optional[list[tuple[int, int]]] = None,
        ignored_visible_spans: Optional[list[tuple[int, int]]] = None,
    ) -> bool:
        return lane_occupancy_conflicts(
            self.raw,
            int(lane),
            int(raw_t0),
            int(raw_t1),
            lane_capacity_by_lane=self.lane_capacity_by_lane,
            visible_occupancy=self.visible,
            visible_t0=visible_t0,
            visible_t1=visible_t1,
            ignored_raw_spans=ignored_raw_spans,
            ignored_visible_spans=ignored_visible_spans,
        )

    def add(
        self,
        lane: int,
        raw_t0: int,
        raw_t1: int,
        *,
        visible_t0: Optional[int] = None,
        visible_t1: Optional[int] = None,
    ) -> None:
        lane_i = int(lane)
        if int(raw_t1) > int(raw_t0):
            self.raw[lane_i].append((int(raw_t0), int(raw_t1)))
        if visible_t0 is not None and visible_t1 is not None and int(visible_t1) > int(visible_t0):
            self.visible[lane_i].append((int(visible_t0), int(visible_t1)))

    def remove(
        self,
        lane: int,
        raw_t0: int,
        raw_t1: int,
        *,
        visible_t0: Optional[int] = None,
        visible_t1: Optional[int] = None,
    ) -> bool:
        lane_i = int(lane)
        removed = False
        try:
            self.raw[lane_i].remove((int(raw_t0), int(raw_t1)))
            removed = True
        except ValueError:
            pass
        if visible_t0 is not None and visible_t1 is not None:
            try:
                self.visible[lane_i].remove((int(visible_t0), int(visible_t1)))
                removed = True
            except ValueError:
                pass
        return bool(removed)


@dataclass
class SourceSpanTracker:
    spans_by_event_id: dict[int, tuple[int, int, int]]
    occupancy: LaneOccupancyState

    @classmethod
    def from_events(
        cls,
        events: list[dict[str, Any]],
        lane_count: int,
        occupancy: LaneOccupancyState,
    ) -> "SourceSpanTracker":
        spans_by_event_id: dict[int, tuple[int, int, int]] = {}
        for event in events:
            src_lane = int(event["src_lane"])
            if src_lane < 0 or src_lane >= int(lane_count):
                continue
            t0 = event_placement_start(event, event_source_raw_start(event))
            t1 = int(t0) + event_placement_length(event)
            if int(t1) <= int(t0):
                continue
            span = (int(src_lane), int(t0), int(t1))
            spans_by_event_id[id(event)] = span
            occupancy.add(*span)
        return cls(spans_by_event_id=spans_by_event_id, occupancy=occupancy)

    def release(self, event: dict[str, Any]) -> Optional[tuple[int, int, int]]:
        span = self.spans_by_event_id.get(id(event))
        if span is None:
            return None
        lane, t0, t1 = span
        self.occupancy.remove(int(lane), int(t0), int(t1))
        return span

    def restore(self, span: Optional[tuple[int, int, int]]) -> None:
        if span is None:
            return
        lane, t0, t1 = span
        self.occupancy.add(int(lane), int(t0), int(t1))
