from __future__ import annotations

from typing import Any, Callable, Optional

from .lane_layout_model import event_owned_raw_transition_spans
from .lane_layout_occupancy import LaneOccupancyState
from .lane_layout_placement import EventPlacementSpan, event_owned_visible_transition_spans
from .lane_layout_snapshot import build_lane_occupancy_snapshot

TargetAllowed = Callable[[dict[str, Any], int], bool]


def _current_event_lane(event: dict[str, Any], lane_count: int) -> int:
    lane = int(event.get("target_lane", event["src_lane"]))
    if lane < 0 or lane >= int(lane_count):
        lane = int(event["src_lane"])
    return int(lane)


def _ordered_fallback_lanes(
    event: dict[str, Any],
    fallback_lanes: list[int],
) -> list[int]:
    ordered: list[int] = []
    for lane in (
        int(event.get("target_lane", event["src_lane"])),
        int(event["src_lane"]),
        *fallback_lanes,
    ):
        lane_i = int(lane)
        if lane_i not in ordered:
            ordered.append(lane_i)
    return ordered


def _remove_current_event_from_state(
    state: LaneOccupancyState,
    event: dict[str, Any],
    lane_count: int,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> int:
    current_lane = _current_event_lane(event, int(lane_count))
    if 0 <= current_lane < int(lane_count):
        span = EventPlacementSpan.from_event(
            event,
            transition_spans_by_lane=transition_spans_by_lane,
        )
        state.remove(
            current_lane,
            span.raw_t0,
            span.raw_t1,
            visible_t0=span.visible_t0,
            visible_t1=span.visible_t1,
        )
    return int(current_lane)


def _can_place_span(
    state: LaneOccupancyState,
    event: dict[str, Any],
    span: EventPlacementSpan,
) -> bool:
    return not state.conflicts(
        span.lane,
        span.raw_t0,
        span.raw_t1,
        visible_t0=span.visible_t0,
        visible_t1=span.visible_t1,
        ignored_raw_spans=event_owned_raw_transition_spans(event, span.event_t),
        ignored_visible_spans=event_owned_visible_transition_spans(event),
    )


def compact_events_to_preferred_lanes(
    all_events: list[dict[str, Any]],
    events_to_compact: list[dict[str, Any]],
    *,
    lane_count: int,
    preferred_lanes: list[int],
    fallback_lanes: list[int],
    target_allowed: TargetAllowed,
    reserved_by_lane: Optional[dict[int, list[tuple[int, int]]]],
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
    lane_capacity_by_lane: Optional[dict[int, int]],
) -> LaneOccupancyState:
    snapshot = build_lane_occupancy_snapshot(
        all_events,
        int(lane_count),
        reserved_by_lane=reserved_by_lane,
        transition_spans_by_lane=transition_spans_by_lane,
    )
    state = LaneOccupancyState(
        raw=snapshot.raw,
        visible=snapshot.visible,
        lane_capacity_by_lane=lane_capacity_by_lane,
    )
    for event in events_to_compact:
        current_lane = _remove_current_event_from_state(
            state,
            event,
            int(lane_count),
            transition_spans_by_lane,
        )
        chosen: Optional[int] = None
        chosen_span: Optional[EventPlacementSpan] = None
        for lane in preferred_lanes:
            if not target_allowed(event, int(lane)):
                continue
            span = EventPlacementSpan.from_event(
                event,
                lane=int(lane),
                transition_spans_by_lane=transition_spans_by_lane,
            )
            if _can_place_span(state, event, span):
                chosen = int(lane)
                chosen_span = span
                break
        if chosen is None:
            for lane in _ordered_fallback_lanes(event, fallback_lanes):
                is_source_lane = int(lane) == int(event["src_lane"])
                if lane < 0 or lane >= int(lane_count):
                    continue
                if not is_source_lane and not target_allowed(event, int(lane)):
                    continue
                span = EventPlacementSpan.from_event(
                    event,
                    lane=int(lane),
                    transition_spans_by_lane=transition_spans_by_lane,
                )
                if _can_place_span(state, event, span):
                    chosen = int(lane)
                    chosen_span = span
                    break
        if chosen is None or chosen_span is None:
            chosen = int(current_lane)
            chosen_span = EventPlacementSpan.from_event(
                event,
                lane=int(chosen),
                transition_spans_by_lane=transition_spans_by_lane,
            )
        event["target_lane"] = int(chosen)
        event["target_T"] = int(chosen_span.event_t)
        if 0 <= int(chosen) < int(lane_count) and chosen_span.raw_t1 > chosen_span.raw_t0:
            state.add(
                int(chosen),
                chosen_span.raw_t0,
                chosen_span.raw_t1,
                visible_t0=chosen_span.visible_t0,
                visible_t1=chosen_span.visible_t1,
            )
    return state
