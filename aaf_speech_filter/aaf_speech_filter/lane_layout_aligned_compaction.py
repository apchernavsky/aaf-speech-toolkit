from __future__ import annotations

from typing import Any, Callable, Optional

from .lane_layout_candidates import candidate_group_lane_sequences
from .lane_layout_conflicts import target_lane_visible_conflicts
from .lane_layout_groups import cross_lane_event_groups
from .lane_layout_zones import class_zone_lane_order

TargetAllowed = Callable[[dict[str, Any], int], bool]
TargetRawForLane = Callable[[dict[str, Any], int], int]
AlignedKindKey = Callable[[dict[str, Any]], Optional[tuple[int, int, int, int]]]
AllowedOverlap = Callable[[dict[str, Any], dict[str, Any]], bool]


def compact_aligned_groups_to_visible_gaps(
    events: list[dict[str, Any]],
    *,
    lane_count: int,
    target_allowed: TargetAllowed,
    target_raw_for_lane: TargetRawForLane,
    aligned_kind_group_key: AlignedKindKey,
    allowed_overlap: AllowedOverlap,
) -> None:
    groups = cross_lane_event_groups(events, aligned_kind_group_key)
    for group in sorted(
        groups,
        key=lambda group_items: (
            int(group_items[0].get("T_edit", group_items[0].get("T", 0))),
            len(group_items),
        ),
    ):
        kinds = {str(event.get("kind") or "unknown") for event in group}
        if len(kinds) != 1:
            continue
        kind = next(iter(kinds))
        ordered = sorted(group, key=lambda event: (int(event["src_lane"]), int(event.get("top_idx", 0))))
        group_ids = {id(event) for event in ordered}
        current_lanes = [int(event.get("target_lane", event["src_lane"])) for event in ordered]
        best_lanes: Optional[list[int]] = None
        for lane_seq in candidate_group_lane_sequences(
            kind,
            class_zone_lane_order(kind, int(lane_count)),
            len(ordered),
            lane_count=int(lane_count),
        ):
            if list(lane_seq) == current_lanes:
                best_lanes = list(lane_seq)
                break
            ok = True
            for event, lane in zip(ordered, lane_seq):
                if not target_allowed(event, int(lane)):
                    ok = False
                    break
                if target_lane_visible_conflicts(
                    event,
                    int(lane),
                    events,
                    target_raw_for_lane=target_raw_for_lane,
                    ignored_event_ids=group_ids,
                    allowed_overlap=allowed_overlap,
                ):
                    ok = False
                    break
            if ok:
                best_lanes = list(lane_seq)
                break
        if best_lanes is None:
            continue
        if tuple(best_lanes) >= tuple(current_lanes):
            continue
        for event, lane in zip(ordered, best_lanes):
            event["target_lane"] = int(lane)
            event["target_T"] = int(target_raw_for_lane(event, int(lane)))
