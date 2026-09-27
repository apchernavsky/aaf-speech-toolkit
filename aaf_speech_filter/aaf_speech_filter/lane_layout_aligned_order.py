from __future__ import annotations

from typing import Any, Callable, Optional

from .lane_layout_groups import (
    aligned_group_source_order,
    aligned_group_target_lanes,
    aligned_group_target_order,
    cross_lane_event_groups,
)

TargetAllowed = Callable[[dict[str, Any], int], bool]
TargetRawForLane = Callable[[dict[str, Any], int], int]
AlignedKindKey = Callable[[dict[str, Any]], Optional[tuple[int, int, int, int]]]


def preserve_aligned_target_order_after_compaction(
    events: list[dict[str, Any]],
    *,
    target_allowed: TargetAllowed,
    target_raw_for_lane: TargetRawForLane,
    aligned_kind_group_key: AlignedKindKey,
) -> None:
    for group in cross_lane_event_groups(events, aligned_kind_group_key):
        target_lanes = aligned_group_target_lanes(group)
        if len(set(target_lanes)) != len(group):
            continue
        source_order = aligned_group_source_order(group)
        current_order = aligned_group_target_order(group)
        if [id(event) for event in source_order] == [id(event) for event in current_order]:
            continue
        if any(not target_allowed(event, lane) for event, lane in zip(source_order, target_lanes)):
            continue
        for event, lane in zip(source_order, target_lanes):
            event["target_lane"] = int(lane)
            event["target_T"] = int(target_raw_for_lane(event, int(lane)))
