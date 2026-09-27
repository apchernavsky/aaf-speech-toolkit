from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from .lane_layout_conflicts import target_lane_visible_conflicts
from .lane_layout_model import (
    events_share_same_kind_transition_edge,
    planned_overlap_is_allowed_transition_overlap,
)
from .lane_layout_zones import class_zone_lane_order

TargetAllowed = Callable[[dict[str, Any], int], bool]
TargetRawForLane = Callable[[dict[str, Any], int], int]
AllowedOverlap = Callable[[dict[str, Any], dict[str, Any]], bool]


def same_kind_transition_chains(
    events: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    by_src_lane: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_src_lane[int(event["src_lane"])].append(event)

    chains: list[list[dict[str, Any]]] = []
    for lane_events in by_src_lane.values():
        ordered = sorted(lane_events, key=lambda event: int(event.get("top_idx", 0)))
        chain: list[dict[str, Any]] = []
        for event in ordered:
            if not chain:
                chain = [event]
                continue
            if events_share_same_kind_transition_edge(chain[-1], event):
                chain.append(event)
                continue
            if len(chain) > 1:
                chains.append(chain)
            chain = [event]
        if len(chain) > 1:
            chains.append(chain)
    return chains


def same_kind_chain_candidate_lanes(
    chain: list[dict[str, Any]],
    lane_count: int,
) -> list[int]:
    if not chain:
        return []
    current_lanes = [int(event.get("target_lane", event["src_lane"])) for event in chain]
    kind = str(chain[0].get("kind") or "unknown")
    candidates: list[int] = []
    for lane in [current_lanes[0], *current_lanes, *class_zone_lane_order(kind, int(lane_count))]:
        lane_i = int(lane)
        if 0 <= lane_i < int(lane_count) and lane_i not in candidates:
            candidates.append(lane_i)
    return candidates


def preserve_same_kind_xfade_chains(
    events: list[dict[str, Any]],
    *,
    lane_count: int,
    target_allowed: TargetAllowed,
    target_raw_for_lane: TargetRawForLane,
    allowed_overlap: AllowedOverlap,
) -> None:
    chains = same_kind_transition_chains(events)
    for chain in sorted(
        chains,
        key=lambda items: (
            int(items[0].get("T_edit", items[0].get("T", 0))),
            int(items[0]["src_lane"]),
        ),
    ):
        chain = sorted(chain, key=lambda event: int(event.get("top_idx", 0)))
        current_lanes = [int(event.get("target_lane", event["src_lane"])) for event in chain]
        if len(set(current_lanes)) == 1:
            continue
        candidates = same_kind_chain_candidate_lanes(chain, int(lane_count))
        chain_ids = {id(event) for event in chain}
        chosen: int | None = None
        for lane in candidates:
            ok = True
            for event in chain:
                if not target_allowed(event, int(lane)):
                    ok = False
                    break
                if target_lane_visible_conflicts(
                    event,
                    int(lane),
                    events,
                    target_raw_for_lane=target_raw_for_lane,
                    allowed_overlap=lambda other, item: (
                        (
                            id(other) in chain_ids
                            and str(other.get("kind") or "unknown") == str(item.get("kind") or "unknown")
                            and (
                                events_share_same_kind_transition_edge(other, item)
                                or events_share_same_kind_transition_edge(item, other)
                                or planned_overlap_is_allowed_transition_overlap(other, item)
                            )
                        )
                        or allowed_overlap(other, item)
                    ),
                ):
                    ok = False
                    break
            if ok:
                chosen = int(lane)
                break
        if chosen is None:
            continue
        for event in chain:
            event["target_lane"] = int(chosen)
            event["target_T"] = int(target_raw_for_lane(event, int(chosen)))
