from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

from .lane_layout_model import intervals_overlap
from .lane_layout_placement import event_visible_occupancy_length, event_visible_start_for_placement

TargetRawResolver = Callable[[dict[str, Any], int], int]
AllowedOverlap = Callable[[dict[str, Any], dict[str, Any]], bool]


def _visible_span(event: dict[str, Any]) -> tuple[int, int]:
    v0 = event_visible_start_for_placement(event)
    return int(v0), int(v0) + event_visible_occupancy_length(event)


def target_lane_visible_conflicts(
    candidate: dict[str, Any],
    lane: int,
    events: Iterable[dict[str, Any]],
    *,
    target_raw_for_lane: TargetRawResolver,
    allowed_overlap: AllowedOverlap,
    ignored_event_ids: Optional[set[int]] = None,
) -> bool:
    cand_v0, cand_v1 = _visible_span(candidate)
    old_lane = candidate.get("target_lane")
    old_t = candidate.get("target_T")
    candidate["target_lane"] = int(lane)
    candidate["target_T"] = int(target_raw_for_lane(candidate, int(lane)))
    try:
        ignored_ids = ignored_event_ids or set()
        for other in events:
            if other is candidate:
                continue
            if id(other) in ignored_ids:
                continue
            if int(other.get("target_lane", other.get("src_lane", -1))) != int(lane):
                continue
            other_v0, other_v1 = _visible_span(other)
            if not intervals_overlap(cand_v0, cand_v1, other_v0, other_v1):
                continue
            if allowed_overlap(other, candidate):
                continue
            return True
        return False
    finally:
        if old_lane is None:
            candidate.pop("target_lane", None)
        else:
            candidate["target_lane"] = old_lane
        if old_t is None:
            candidate.pop("target_T", None)
        else:
            candidate["target_T"] = old_t
