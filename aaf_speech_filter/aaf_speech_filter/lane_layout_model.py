from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def event_visible_start(event: dict[str, Any]) -> int:
    return int(
        event.get(
            "T_edit",
            event.get("T_nuendo", event.get("target_T", event.get("T", 0))),
        )
    )


def event_length(event: dict[str, Any]) -> int:
    return max(0, int(event.get("L", 0) or 0))


def event_pre_transition_len(event: dict[str, Any]) -> int:
    return max(0, int(event.get("pre_transition_len", 0) or 0))


def event_post_transition_len(event: dict[str, Any]) -> int:
    return max(0, int(event.get("post_transition_len", 0) or 0))


@dataclass(frozen=True)
class TimelinePosition:
    raw_start: int
    visible_start: int
    length: int
    pre_transition_len: int = 0
    post_transition_len: int = 0

    @classmethod
    def from_event(
        cls,
        event: dict[str, Any],
        *,
        raw_start: Optional[int] = None,
    ) -> "TimelinePosition":
        raw = int(event.get("target_T", event.get("T", 0)) if raw_start is None else raw_start)
        return cls(
            raw_start=int(raw),
            visible_start=event_visible_start(event),
            length=event_length(event),
            pre_transition_len=event_pre_transition_len(event),
            post_transition_len=event_post_transition_len(event),
        )

    @property
    def visible_span(self) -> tuple[int, int]:
        return int(self.visible_start), int(self.visible_start) + int(self.length)

    @property
    def raw_span_with_transitions(self) -> tuple[int, int]:
        return (
            int(self.raw_start) - int(self.pre_transition_len),
            int(self.raw_start) + int(self.length) + int(self.post_transition_len),
        )


def event_visible_span(event: dict[str, Any]) -> tuple[int, int]:
    return TimelinePosition.from_event(event).visible_span


def event_raw_span_with_transitions(
    event: dict[str, Any],
    raw_start: Optional[int] = None,
) -> tuple[int, int]:
    return TimelinePosition.from_event(event, raw_start=raw_start).raw_span_with_transitions


def intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) < min(int(a1), int(b1))


def span_contains(container: tuple[int, int], span: tuple[int, int]) -> bool:
    return int(container[0]) <= int(span[0]) and int(span[1]) <= int(container[1])


def event_owned_raw_transition_spans(
    event: dict[str, Any],
    event_t: int,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pre = event_pre_transition_len(event)
    post = event_post_transition_len(event)
    if pre > 0:
        spans.append((int(event_t) - int(pre), int(event_t)))
    if post > 0:
        event_end = int(event_t) + event_length(event)
        spans.append((int(event_end), int(event_end) + int(post)))
    return spans


def event_owned_transition_history_spans(
    event: dict[str, Any],
    event_t: int,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pre = event_pre_transition_len(event)
    post = event_post_transition_len(event)
    if pre > 0:
        spans.append((int(event_t) - int(pre), int(pre)))
    if post > 0:
        spans.append((int(event_t) + event_length(event), int(post)))
    return spans


def events_share_same_kind_transition_edge(
    prev: dict[str, Any],
    item: dict[str, Any],
) -> bool:
    if int(prev.get("src_lane", -1)) != int(item.get("src_lane", -2)):
        return False
    if str(prev.get("kind") or "unknown") != str(item.get("kind") or "unknown"):
        return False
    post_idx = prev.get("post_transition_top_idx")
    pre_idx = item.get("pre_transition_top_idx")
    if post_idx is None or pre_idx is None:
        return False
    if int(post_idx) != int(pre_idx):
        return False
    return event_post_transition_len(prev) > 0 and event_pre_transition_len(item) > 0


def planned_overlap_is_allowed_transition_overlap(
    prev: dict[str, Any],
    item: dict[str, Any],
) -> bool:
    prev_span = event_raw_span_with_transitions(prev)
    item_span = event_raw_span_with_transitions(item)
    overlap = (max(prev_span[0], item_span[0]), min(prev_span[1], item_span[1]))
    if int(overlap[1]) <= int(overlap[0]):
        return False
    if events_share_same_kind_transition_edge(prev, item):
        return True
    prev_visible = event_visible_span(prev)
    item_visible = event_visible_span(item)
    if not intervals_overlap(
        prev_visible[0],
        prev_visible[1],
        item_visible[0],
        item_visible[1],
    ):
        return True
    prev_t = int(prev.get("target_T", prev.get("T", 0)))
    item_t = int(item.get("target_T", item.get("T", 0)))
    prev_owned = event_owned_raw_transition_spans(prev, prev_t)
    item_owned = event_owned_raw_transition_spans(item, item_t)
    return any(span_contains(span, overlap) for span in prev_owned) and any(
        span_contains(span, overlap) for span in item_owned
    )
