from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from .lane_layout_constraints import target_lane_allowed_for_event
from .lane_layout_model import (
    event_owned_transition_history_spans,
    event_pre_transition_len,
    event_raw_span_with_transitions,
    event_visible_start,
    intervals_overlap,
    planned_overlap_is_allowed_transition_overlap,
)
from .lane_layout_timing import (
    raw_from_visible_with_transition_spans,
    spans_without_exact_matches,
)
from .lane_layout_zones import class_zone_lane_order


def planned_event_raw_t_for_lane(
    event: dict[str, Any],
    lane: int,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> int:
    if transition_spans_by_lane is None:
        return int(event.get("target_T", event.get("T", 0)))
    transition_spans = spans_without_exact_matches(
        list(transition_spans_by_lane.get(int(lane), [])),
        event_owned_transition_history_spans(
            event,
            int(event.get("T", event.get("target_T", 0))),
        ),
    )
    return raw_from_visible_with_transition_spans(
        transition_spans,
        event_visible_start(event),
        event_pre_transition_len(event),
    )


def repair_lane_layout_plan_overlaps(
    events: list[dict[str, Any]],
    n_lanes: int,
    *,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]] = None,
    lane_signature_by_lane: Optional[dict[int, tuple[Any, ...]]] = None,
    blocked_target_lanes: Optional[set[int]] = None,
    lane_capacity_by_lane: Optional[dict[int, int]] = None,
    lane_mutable_by_lane: Optional[dict[int, bool]] = None,
) -> int:
    def target_allowed(event: dict[str, Any], lane: int) -> bool:
        lane_i = int(lane)
        if lane_i < 0 or lane_i >= int(n_lanes):
            return False
        if not target_lane_allowed_for_event(
            event, lane_i,
            blocked_target_lanes=blocked_target_lanes,
            lane_mutable_by_lane=lane_mutable_by_lane,
            lane_signature_by_lane=lane_signature_by_lane,
        ):
            return False
        event_t = planned_event_raw_t_for_lane(event, lane_i, transition_spans_by_lane)
        t0, t1 = event_raw_span_with_transitions(event, event_t)
        if t0 < 0 or t1 <= t0:
            return False
        if lane_capacity_by_lane is not None:
            cap = int(lane_capacity_by_lane.get(lane_i, 0) or 0)
            if cap > 0 and int(t1) > cap:
                return False
        return True

    def candidate_lanes(event: dict[str, Any]) -> list[int]:
        order = class_zone_lane_order(str(event.get("kind") or "unknown"), int(n_lanes))
        current = int(event.get("target_lane", event.get("src_lane", 0)))
        all_lanes = [*order, *range(int(n_lanes))]
        out: list[int] = []
        for lane in all_lanes:
            lane_i = int(lane)
            if lane_i == current or lane_i in out:
                continue
            out.append(lane_i)
        return out

    def conflicts_on_lane(event: dict[str, Any], lane: int, event_t: int) -> bool:
        cand_span = event_raw_span_with_transitions(event, event_t)
        for other in events:
            if other is event:
                continue
            if int(other.get("target_lane", other.get("src_lane", -1))) != int(lane):
                continue
            other_span = event_raw_span_with_transitions(other)
            if not intervals_overlap(cand_span[0], cand_span[1], other_span[0], other_span[1]):
                continue
            old_lane = event.get("target_lane")
            old_t = event.get("target_T")
            event["target_lane"] = int(lane)
            event["target_T"] = int(event_t)
            try:
                allowed = planned_overlap_is_allowed_transition_overlap(other, event)
            finally:
                if old_lane is None:
                    event.pop("target_lane", None)
                else:
                    event["target_lane"] = old_lane
                if old_t is None:
                    event.pop("target_T", None)
                else:
                    event["target_T"] = old_t
            if not allowed:
                return True
        return False

    def first_overlap() -> Optional[tuple[dict[str, Any], dict[str, Any]]]:
        by_lane: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            lane = int(event.get("target_lane", event.get("src_lane", -1)))
            if 0 <= lane < int(n_lanes):
                by_lane[lane].append(event)
        for lane_events in by_lane.values():
            ordered = sorted(
                lane_events,
                key=lambda event: (
                    event_raw_span_with_transitions(event)[0],
                    event_raw_span_with_transitions(event)[1],
                ),
            )
            prev: Optional[dict[str, Any]] = None
            for item in ordered:
                if prev is not None:
                    prev_span = event_raw_span_with_transitions(prev)
                    item_span = event_raw_span_with_transitions(item)
                    if item_span[0] < prev_span[1] and not planned_overlap_is_allowed_transition_overlap(prev, item):
                        return prev, item
                    if item_span[1] > prev_span[1]:
                        prev = item
                else:
                    prev = item
        return None

    moved = 0
    for _ in range(max(1, len(events) * max(1, int(n_lanes)))):
        pair = first_overlap()
        if pair is None:
            return int(moved)
        prev, item = pair
        repaired = False
        for victim in (item, prev):
            for lane in candidate_lanes(victim):
                if not target_allowed(victim, int(lane)):
                    continue
                event_t = planned_event_raw_t_for_lane(victim, int(lane), transition_spans_by_lane)
                if conflicts_on_lane(victim, int(lane), int(event_t)):
                    continue
                victim["target_lane"] = int(lane)
                victim["target_T"] = int(event_t)
                moved += 1
                repaired = True
                break
            if repaired:
                break
        if not repaired:
            return int(moved)
    return int(moved)


def validate_lane_layout_plan(events: list[dict[str, Any]], n_lanes: int) -> None:
    by_lane: dict[int, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for event in events:
        lane = int(event.get("target_lane", event.get("src_lane", -1)))
        if lane < 0 or lane >= int(n_lanes):
            raise RuntimeError(f"YAMNet lane layout produced out-of-range lane {lane}")
        bounds = event.get("lane_bounds")
        if bounds is not None and not int(bounds[0]) <= lane <= int(bounds[1]):
            raise RuntimeError(
                f"YAMNet lane layout produced lane {lane} outside bounds {bounds}"
            )
        t0, t1 = event_raw_span_with_transitions(event)
        if t1 <= t0:
            continue
        by_lane.setdefault(lane, []).append((t0, t1, event))

    for lane, spans in by_lane.items():
        spans_sorted = sorted(spans, key=lambda item: (item[0], item[1]))
        prev: Optional[tuple[int, int, dict[str, Any]]] = None
        for item in spans_sorted:
            if prev is not None and int(item[0]) < int(prev[1]):
                if planned_overlap_is_allowed_transition_overlap(prev[2], item[2]):
                    if int(item[1]) > int(prev[1]):
                        prev = item
                    continue
                prev_name = str(prev[2].get("display_name") or prev[2].get("name") or "clip")
                name = str(item[2].get("display_name") or item[2].get("name") or "clip")
                raise RuntimeError(
                    "YAMNet lane layout produced overlapping planned blocks "
                    f"on lane {lane}: "
                    f"{prev_name} [{prev[0]}, {prev[1]}) and "
                    f"{name} [{item[0]}, {item[1]})"
                )
            prev = item
