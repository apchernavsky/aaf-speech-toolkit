from __future__ import annotations

from typing import Any, Callable, Optional

from .lane_layout_model import event_owned_raw_transition_spans, intervals_overlap
from .lane_layout_occupancy import LaneOccupancyState
from .lane_layout_placement import (
    EventPlacementSpan,
    event_owned_visible_transition_spans,
    event_placement_length,
    event_visible_start_for_placement,
)

TargetAllowed = Callable[[dict[str, Any], int], bool]


def _event_span(
    event: dict[str, Any],
    lane: Optional[int],
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> EventPlacementSpan:
    return EventPlacementSpan.from_event(
        event,
        lane=lane,
        transition_spans_by_lane=transition_spans_by_lane,
    )


def _remove_event_from_occupancy(
    occupancy: LaneOccupancyState,
    event: dict[str, Any],
    lane_count: int,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> None:
    span = _event_span(event, None, transition_spans_by_lane)
    if 0 <= span.lane < int(lane_count):
        occupancy.remove(
            span.lane,
            span.raw_t0,
            span.raw_t1,
            visible_t0=span.visible_t0,
            visible_t1=span.visible_t1,
        )


def _add_event_to_occupancy(
    occupancy: LaneOccupancyState,
    event: dict[str, Any],
    lane_count: int,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> None:
    span = _event_span(event, None, transition_spans_by_lane)
    if 0 <= span.lane < int(lane_count) and span.raw_t1 > span.raw_t0:
        occupancy.add(
            span.lane,
            span.raw_t0,
            span.raw_t1,
            visible_t0=span.visible_t0,
            visible_t1=span.visible_t1,
        )


def _can_place_event_on_lane(
    occupancy: LaneOccupancyState,
    event: dict[str, Any],
    lane: int,
    target_allowed: TargetAllowed,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> bool:
    if not target_allowed(event, int(lane)):
        return False
    span = _event_span(event, int(lane), transition_spans_by_lane)
    return not occupancy.conflicts(
        int(lane),
        span.raw_t0,
        span.raw_t1,
        visible_t0=span.visible_t0,
        visible_t1=span.visible_t1,
        ignored_raw_spans=event_owned_raw_transition_spans(event, span.event_t),
        ignored_visible_spans=event_owned_visible_transition_spans(event),
    )


def _try_swap_speech_lanes(
    upper: dict[str, Any],
    lower: dict[str, Any],
    *,
    lane_count: int,
    occupancy: LaneOccupancyState,
    target_allowed: TargetAllowed,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> bool:
    upper_lane = int(upper.get("target_lane", upper["src_lane"]))
    lower_lane = int(lower.get("target_lane", lower["src_lane"]))
    if lower_lane <= upper_lane:
        return False
    if event_placement_length(lower) <= event_placement_length(upper):
        return False
    lower_span = _event_span(lower, None, transition_spans_by_lane)
    upper_span = _event_span(upper, None, transition_spans_by_lane)
    if not (
        intervals_overlap(lower_span.raw_t0, lower_span.raw_t1, upper_span.raw_t0, upper_span.raw_t1)
        or intervals_overlap(
            lower_span.visible_t0,
            lower_span.visible_t1,
            upper_span.visible_t0,
            upper_span.visible_t1,
        )
    ):
        return False
    _remove_event_from_occupancy(occupancy, upper, lane_count, transition_spans_by_lane)
    _remove_event_from_occupancy(occupancy, lower, lane_count, transition_spans_by_lane)
    ok = _can_place_event_on_lane(
        occupancy,
        lower,
        upper_lane,
        target_allowed,
        transition_spans_by_lane,
    ) and _can_place_event_on_lane(
        occupancy,
        upper,
        lower_lane,
        target_allowed,
        transition_spans_by_lane,
    )
    if ok:
        lower["target_lane"] = int(upper_lane)
        lower["target_T"] = int(_event_span(lower, upper_lane, transition_spans_by_lane).event_t)
        upper["target_lane"] = int(lower_lane)
        upper["target_T"] = int(_event_span(upper, lower_lane, transition_spans_by_lane).event_t)
    _add_event_to_occupancy(occupancy, upper, lane_count, transition_spans_by_lane)
    _add_event_to_occupancy(occupancy, lower, lane_count, transition_spans_by_lane)
    return bool(ok)


def promote_longer_speech_events(
    speech_events: list[dict[str, Any]],
    *,
    lane_count: int,
    occupancy: LaneOccupancyState,
    target_allowed: TargetAllowed,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> None:
    for _ in range(max(1, len(speech_events))):
        changed = False
        ranked = sorted(
            speech_events,
            key=lambda event: (
                -event_placement_length(event),
                event_visible_start_for_placement(event),
                int(event["src_lane"]),
            ),
        )
        for lower in ranked:
            lower_lane = int(lower.get("target_lane", lower["src_lane"]))
            blockers = sorted(
                (
                    event for event in speech_events
                    if event is not lower
                    and int(event.get("target_lane", event["src_lane"])) < lower_lane
                ),
                key=lambda event: (
                    int(event.get("target_lane", event["src_lane"])),
                    event_placement_length(event),
                ),
            )
            for upper in blockers:
                if _try_swap_speech_lanes(
                    upper,
                    lower,
                    lane_count=int(lane_count),
                    occupancy=occupancy,
                    target_allowed=target_allowed,
                    transition_spans_by_lane=transition_spans_by_lane,
                ):
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
