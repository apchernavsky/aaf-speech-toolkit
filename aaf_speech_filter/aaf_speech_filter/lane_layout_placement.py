from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .lane_layout_model import event_owned_transition_history_spans, event_pre_transition_len
from .lane_layout_timing import raw_from_visible_with_transition_spans, spans_without_exact_matches


def event_source_raw_start(event: dict[str, Any]) -> int:
    return int(event["T"])


def event_visible_start_for_placement(event: dict[str, Any]) -> int:
    return int(event.get("T_edit", event["T"]))


def event_placement_length(event: dict[str, Any]) -> int:
    return (
        max(0, int(event.get("pre_transition_len", 0) or 0))
        + int(event["L"])
        + max(0, int(event.get("post_transition_len", 0) or 0))
    )


def event_visible_occupancy_length(event: dict[str, Any]) -> int:
    return max(0, int(event.get("L", 0) or 0))


def event_placement_start(event: dict[str, Any], event_t: int) -> int:
    return int(event_t) - max(0, int(event.get("pre_transition_len", 0) or 0))


def event_owned_visible_transition_spans(event: dict[str, Any]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pre = max(0, int(event.get("pre_transition_len", 0) or 0))
    post = max(0, int(event.get("post_transition_len", 0) or 0))
    visible_t = event_visible_start_for_placement(event)
    visible_len = event_placement_length(event)
    if pre > 0:
        spans.append((int(visible_t), int(visible_t) + 2 * int(pre)))
    if post > 0:
        spans.append(
            (
                int(visible_t) + int(visible_len) - 2 * int(post),
                int(visible_t) + int(visible_len),
            )
        )
    return spans


def event_target_raw_start_for_lane(
    event: dict[str, Any],
    lane: int,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> int:
    if transition_spans_by_lane is None:
        return event_source_raw_start(event)
    transition_spans = spans_without_exact_matches(
        list(transition_spans_by_lane.get(int(lane), [])),
        event_owned_transition_history_spans(event, event_source_raw_start(event)),
    )
    return raw_from_visible_with_transition_spans(
        transition_spans,
        event_visible_start_for_placement(event),
        event_pre_transition_len(event),
    )


@dataclass(frozen=True)
class EventPlacementSpan:
    lane: int
    event_t: int
    raw_t0: int
    raw_t1: int
    visible_t0: int
    visible_t1: int

    @classmethod
    def from_event(
        cls,
        event: dict[str, Any],
        *,
        lane: Optional[int] = None,
        transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]] = None,
    ) -> "EventPlacementSpan":
        lane_i = int(event.get("target_lane", event["src_lane"]) if lane is None else lane)
        if lane is None and "target_T" in event:
            event_t = int(event["target_T"])
        else:
            event_t = event_target_raw_start_for_lane(event, lane_i, transition_spans_by_lane)
        raw_t0 = event_placement_start(event, event_t)
        raw_t1 = int(raw_t0) + event_placement_length(event)
        visible_t0 = event_visible_start_for_placement(event)
        visible_t1 = int(visible_t0) + event_visible_occupancy_length(event)
        return cls(
            lane=int(lane_i),
            event_t=int(event_t),
            raw_t0=int(raw_t0),
            raw_t1=int(raw_t1),
            visible_t0=int(visible_t0),
            visible_t1=int(visible_t1),
        )

    @property
    def raw_span(self) -> tuple[int, int]:
        return int(self.raw_t0), int(self.raw_t1)

    @property
    def visible_span(self) -> tuple[int, int]:
        return int(self.visible_t0), int(self.visible_t1)
