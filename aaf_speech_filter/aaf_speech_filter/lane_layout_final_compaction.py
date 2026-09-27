from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .lane_layout_occupancy import LaneOccupancyState
from .lane_layout_placement import EventPlacementSpan
from .lane_layout_snapshot import build_lane_occupancy_snapshot
from .lane_layout_xfade import same_kind_transition_chains
from .lane_layout_zones import class_zone_lane_order
from .lane_layout_candidates import candidate_group_lane_sequences

TargetAllowed = Callable[[dict[str, Any], int], bool]
TargetRawForLane = Callable[[dict[str, Any], int], int]
AlignedGroupKey = Callable[[dict[str, Any]], Optional[tuple[Any, ...]]]


@dataclass(frozen=True)
class _CompactionUnit:
    events: tuple[dict[str, Any], ...]
    lane_offsets: tuple[int, ...]
    kind: str


def _current_lane(event: dict[str, Any], lane_count: int) -> int:
    lane = int(event.get("target_lane", event["src_lane"]))
    if 0 <= lane < int(lane_count):
        return int(lane)
    return int(event["src_lane"])


def _unit_kind(events: tuple[dict[str, Any], ...]) -> str:
    kinds = {str(event.get("kind") or "unknown") for event in events}
    if len(kinds) == 1:
        return next(iter(kinds))
    return "unknown"


def _dense_offsets_for_current_lanes(
    events: tuple[dict[str, Any], ...],
    lane_count: int,
) -> tuple[int, ...]:
    lanes = [_current_lane(event, int(lane_count)) for event in events]
    if len(set(lanes)) <= 1:
        return tuple(0 for _event in events)
    ordered = sorted(set(lanes))
    index_by_lane = {lane: idx for idx, lane in enumerate(ordered)}
    return tuple(index_by_lane[lane] for lane in lanes)


def _component_units(
    events: list[dict[str, Any]],
    lane_count: int,
    aligned_group_key: Optional[AlignedGroupKey] = None,
) -> list[_CompactionUnit]:
    parent: dict[int, int] = {id(event): id(event) for event in events}
    event_by_id = {id(event): event for event in events}

    def find(item_id: int) -> int:
        root = parent[item_id]
        while root != parent[root]:
            root = parent[root]
        while item_id != root:
            next_id = parent[item_id]
            parent[item_id] = root
            item_id = next_id
        return root

    def union(left: dict[str, Any], right: dict[str, Any]) -> None:
        left_root = find(id(left))
        right_root = find(id(right))
        if left_root != right_root:
            parent[right_root] = left_root

    for chain in same_kind_transition_chains(events):
        if len(chain) < 2:
            continue
        first = chain[0]
        for event in chain[1:]:
            union(first, event)

    if aligned_group_key is not None:
        keyed: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            key = aligned_group_key(event)
            if key is not None:
                keyed[key].append(event)
        for group in keyed.values():
            if len(group) < 2:
                continue
            if len({_current_lane(event, int(lane_count)) for event in group}) < 2:
                continue
            if len({str(event.get("kind") or "unknown") for event in group}) != 1:
                continue
            first = group[0]
            for event in group[1:]:
                union(first, event)

    by_root: dict[int, list[dict[str, Any]]] = {}
    for event_id, event in event_by_id.items():
        by_root.setdefault(find(event_id), []).append(event)

    units: list[_CompactionUnit] = []
    for group in by_root.values():
        ordered = tuple(
            sorted(
                group,
                key=lambda event: (
                    _current_lane(event, int(lane_count)),
                    int(event.get("T_edit", event.get("T", 0))),
                    int(event.get("src_lane", 0)),
                    int(event.get("top_idx", 0)),
                ),
            )
        )
        units.append(
            _CompactionUnit(
                events=ordered,
                lane_offsets=_dense_offsets_for_current_lanes(ordered, int(lane_count)),
                kind=_unit_kind(ordered),
            )
        )
    return units


def _aligned_groups_by_event_id(
    events: list[dict[str, Any]],
    aligned_group_key: Optional[AlignedGroupKey],
) -> dict[int, list[tuple[dict[str, Any], ...]]]:
    if aligned_group_key is None:
        return {}
    keyed: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        key = aligned_group_key(event)
        if key is not None:
            keyed[key].append(event)
    out: dict[int, list[tuple[dict[str, Any], ...]]] = defaultdict(list)
    for group in keyed.values():
        if len(group) < 2:
            continue
        if len({_current_lane(event, 10**9) for event in group}) < 2:
            continue
        frozen = tuple(group)
        for event in frozen:
            out[id(event)].append(frozen)
    return dict(out)


def _aligned_groups_for_atomic_compaction(
    events: list[dict[str, Any]],
    aligned_group_key: Optional[AlignedGroupKey],
) -> list[tuple[dict[str, Any], ...]]:
    if aligned_group_key is None:
        return []
    keyed: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        key = aligned_group_key(event)
        if key is not None:
            keyed[key].append(event)

    xfade_event_ids: set[int] = set()
    for chain in same_kind_transition_chains(events):
        if len(chain) > 1:
            xfade_event_ids.update(id(event) for event in chain)

    groups: list[tuple[dict[str, Any], ...]] = []
    for group in keyed.values():
        if len(group) < 2:
            continue
        if len({str(event.get("kind") or "unknown") for event in group}) != 1:
            continue
        if any(id(event) in xfade_event_ids for event in group):
            continue
        ordered = tuple(
            sorted(group, key=lambda event: (int(event["src_lane"]), int(event.get("top_idx", 0))))
        )
        if len({_current_lane(event, 10**9) for event in ordered}) < 2:
            continue
        groups.append(ordered)
    return groups


def _unit_preserves_aligned_order(
    unit: _CompactionUnit,
    base_lane: int,
    lane_count: int,
    aligned_groups_by_event_id: dict[int, list[tuple[dict[str, Any], ...]]],
) -> bool:
    if not aligned_groups_by_event_id:
        return True
    unit_target_lanes = {
        id(event): int(base_lane) + int(offset)
        for event, offset in zip(unit.events, unit.lane_offsets)
    }
    groups: dict[int, tuple[dict[str, Any], ...]] = {}
    for event in unit.events:
        for group in aligned_groups_by_event_id.get(id(event), []):
            groups[id(group)] = group
    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda event: (int(event["src_lane"]), int(event.get("top_idx", 0))),
        )
        for idx, left in enumerate(ordered):
            for right in ordered[idx + 1 :]:
                before_left = _current_lane(left, int(lane_count))
                before_right = _current_lane(right, int(lane_count))
                after_left = unit_target_lanes.get(id(left), before_left)
                after_right = unit_target_lanes.get(id(right), before_right)
                if before_left < before_right and after_left > after_right:
                    return False
                if before_left > before_right and after_left < after_right:
                    return False
    return True


def _remove_events_from_state(
    state: LaneOccupancyState,
    events: tuple[dict[str, Any], ...],
    lane_count: int,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> None:
    for event in events:
        lane = _current_lane(event, int(lane_count))
        span = EventPlacementSpan.from_event(
            event,
            transition_spans_by_lane=transition_spans_by_lane,
        )
        state.remove(
            lane,
            span.raw_t0,
            span.raw_t1,
            visible_t0=span.visible_t0,
            visible_t1=span.visible_t1,
        )


def _add_events_to_state(
    state: LaneOccupancyState,
    events: tuple[dict[str, Any], ...],
    lanes: tuple[int, ...],
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> None:
    for event, lane in zip(events, lanes):
        span = EventPlacementSpan.from_event(
            event,
            lane=int(lane),
            transition_spans_by_lane=transition_spans_by_lane,
        )
        if span.raw_t1 > span.raw_t0:
            state.add(
                int(lane),
                span.raw_t0,
                span.raw_t1,
                visible_t0=span.visible_t0,
                visible_t1=span.visible_t1,
            )


def _remove_unit_from_state(
    state: LaneOccupancyState,
    unit: _CompactionUnit,
    lane_count: int,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> None:
    _remove_events_from_state(state, unit.events, int(lane_count), transition_spans_by_lane)


def _add_unit_to_state(
    state: LaneOccupancyState,
    unit: _CompactionUnit,
    base_lane: int,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> None:
    lanes = tuple(int(base_lane) + int(offset) for offset in unit.lane_offsets)
    _add_events_to_state(state, unit.events, lanes, transition_spans_by_lane)


def _unit_can_move_to_base_lane(
    state: LaneOccupancyState,
    unit: _CompactionUnit,
    base_lane: int,
    lane_count: int,
    target_allowed: TargetAllowed,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
    aligned_groups_by_event_id: dict[int, list[tuple[dict[str, Any], ...]]],
) -> bool:
    if not _unit_preserves_aligned_order(
        unit,
        int(base_lane),
        int(lane_count),
        aligned_groups_by_event_id,
    ):
        return False
    for event, offset in zip(unit.events, unit.lane_offsets):
        lane = int(base_lane) + int(offset)
        if lane < 0 or lane >= int(lane_count):
            return False
        if not target_allowed(event, lane):
            return False
        span = EventPlacementSpan.from_event(
            event,
            lane=lane,
            transition_spans_by_lane=transition_spans_by_lane,
        )
        if state.conflicts(
            lane,
            span.raw_t0,
            span.raw_t1,
            visible_t0=span.visible_t0,
            visible_t1=span.visible_t1,
        ):
            return False
    return True


def _better_base_candidates(
    unit: _CompactionUnit, lane_count: int, pref: list[int],
) -> list[int]:
    if not unit.events:
        return []
    if not pref:
        return []
    current_base = min(
        _current_lane(event, int(lane_count)) - int(offset)
        for event, offset in zip(unit.events, unit.lane_offsets)
    )
    max_offset = max(unit.lane_offsets, default=0)
    current_rank = pref.index(current_base) if current_base in pref else None
    needs_normalization = any(
        _current_lane(event, int(lane_count)) != current_base + int(offset)
        for event, offset in zip(unit.events, unit.lane_offsets)
    )
    out: list[int] = []
    for rank, lane in enumerate(pref):
        if int(lane) + int(max_offset) >= int(lane_count):
            continue
        if current_rank is not None and (
            rank > current_rank or (rank == current_rank and not needs_normalization)
        ):
            continue
        if int(lane) not in out:
            out.append(int(lane))
    return out


def _unit_sort_key(unit: _CompactionUnit) -> tuple[int, int, int, int]:
    kind_order = {"speech": 0, "noise": 1, "unknown": 2, "music": 3}
    first = min(unit.events, key=lambda event: int(event.get("T_edit", event.get("T", 0))))
    return (
        kind_order.get(unit.kind, 2),
        int(first.get("T_edit", first.get("T", 0))),
        min(_current_lane(event, 10**9) for event in unit.events),
        min(int(event.get("src_lane", 0)) for event in unit.events),
    )


def _can_place_aligned_group_on_lanes(
    state: LaneOccupancyState,
    group: tuple[dict[str, Any], ...],
    lanes: list[int],
    lane_count: int,
    target_allowed: TargetAllowed,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> bool:
    if len(group) != len(lanes):
        return False
    for event, lane in zip(group, lanes):
        lane_i = int(lane)
        if lane_i < 0 or lane_i >= int(lane_count):
            return False
        if not target_allowed(event, lane_i):
            return False
        span = EventPlacementSpan.from_event(
            event,
            lane=lane_i,
            transition_spans_by_lane=transition_spans_by_lane,
        )
        if state.conflicts(
            lane_i,
            span.raw_t0,
            span.raw_t1,
            visible_t0=span.visible_t0,
            visible_t1=span.visible_t1,
        ):
            return False
    return True


def _compact_aligned_groups_as_atoms(
    events: list[dict[str, Any]],
    *,
    state: LaneOccupancyState,
    lane_count: int,
    target_allowed: TargetAllowed,
    target_raw_for_lane: TargetRawForLane,
    aligned_group_key: Optional[AlignedGroupKey],
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
    groups: Optional[list[tuple[dict[str, Any], ...]]] = None,
) -> int:
    moved = 0
    if groups is None:
        groups = _aligned_groups_for_atomic_compaction(events, aligned_group_key)
    groups = sorted(
        groups,
        key=lambda group: (
            int(group[0].get("T_edit", group[0].get("T", 0))),
            min(int(event["src_lane"]) for event in group),
        ),
    )
    for group in groups:
        _remove_events_from_state(state, group, int(lane_count), transition_spans_by_lane)
    for group in groups:
        kind = _unit_kind(group)
        current_lanes = tuple(_current_lane(event, int(lane_count)) for event in group)
        lane_pref = class_zone_lane_order(kind, int(lane_count))
        candidates = candidate_group_lane_sequences(
            kind,
            lane_pref,
            len(group),
            lane_count=int(lane_count),
        )
        if not candidates:
            _add_events_to_state(state, group, current_lanes, transition_spans_by_lane)
            continue
        chosen: Optional[list[int]] = None
        for lanes in candidates:
            lane_tuple = tuple(int(lane) for lane in lanes)
            if lane_tuple == current_lanes:
                chosen = list(lane_tuple)
                break
            if _can_place_aligned_group_on_lanes(
                state,
                group,
                list(lane_tuple),
                int(lane_count),
                target_allowed,
                transition_spans_by_lane,
            ):
                chosen = list(lane_tuple)
                break
        if chosen is None:
            _add_events_to_state(state, group, current_lanes, transition_spans_by_lane)
            continue
        for event, lane in zip(group, chosen):
            target_t = int(target_raw_for_lane(event, int(lane)))
            if (
                _current_lane(event, int(lane_count)) != int(lane)
                or int(event.get("target_T", event.get("T", 0))) != target_t
            ):
                moved += 1
            event["target_lane"] = int(lane)
            event["target_T"] = target_t
        _add_events_to_state(state, group, tuple(int(lane) for lane in chosen), transition_spans_by_lane)
    return int(moved)


def final_bounded_compact_class_events(
    events: list[dict[str, Any]],
    *,
    lane_count: int,
    target_allowed: TargetAllowed,
    target_raw_for_lane: TargetRawForLane,
    aligned_group_key: Optional[AlignedGroupKey] = None,
    reserved_by_lane: Optional[dict[int, list[tuple[int, int]]]] = None,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]] = None,
    lane_capacity_by_lane: Optional[dict[int, int]] = None,
) -> int:
    snapshot = build_lane_occupancy_snapshot(
        events,
        int(lane_count),
        reserved_by_lane=reserved_by_lane,
        transition_spans_by_lane=transition_spans_by_lane,
    )
    state = LaneOccupancyState(
        raw=snapshot.raw,
        visible=snapshot.visible,
        lane_capacity_by_lane=lane_capacity_by_lane,
    )
    aligned_groups = _aligned_groups_by_event_id(events, aligned_group_key)
    atomic_groups = _aligned_groups_for_atomic_compaction(events, aligned_group_key)
    moved = _compact_aligned_groups_as_atoms(
        events,
        state=state,
        lane_count=int(lane_count),
        target_allowed=target_allowed,
        target_raw_for_lane=target_raw_for_lane,
        aligned_group_key=aligned_group_key,
        transition_spans_by_lane=transition_spans_by_lane,
        groups=atomic_groups,
    )
    units = _component_units(events, int(lane_count), aligned_group_key)
    atomic_event_ids = {id(event) for group in atomic_groups for event in group}
    base_preferences = {}
    for unit in units:
        preference = class_zone_lane_order(unit.kind, int(lane_count))
        if all(id(event) in atomic_event_ids for event in unit.events):
            preference = [lanes[0] for lanes in candidate_group_lane_sequences(
                unit.kind, preference, len(unit.events), lane_count=int(lane_count),
            )]
        base_preferences[id(unit)] = preference
    # Each move improves (base preference rank, noncanonical lane offsets).
    # Finite ranks and one normalization step per rank guarantee convergence.
    while True:
        changed = False
        for unit in sorted(units, key=_unit_sort_key):
            candidates = _better_base_candidates(unit, int(lane_count), base_preferences[id(unit)])
            if not candidates:
                continue
            current_lanes = tuple(_current_lane(event, int(lane_count)) for event in unit.events)
            _remove_unit_from_state(state, unit, int(lane_count), transition_spans_by_lane)
            chosen: Optional[int] = None
            for base_lane in candidates:
                if _unit_can_move_to_base_lane(
                    state,
                    unit,
                    int(base_lane),
                    int(lane_count),
                    target_allowed,
                    transition_spans_by_lane,
                    aligned_groups,
                ):
                    chosen = int(base_lane)
                    break
            if chosen is None:
                _add_events_to_state(
                    state, unit.events, current_lanes, transition_spans_by_lane,
                )
                continue
            changed = True
            for event, offset in zip(unit.events, unit.lane_offsets):
                lane = int(chosen) + int(offset)
                old_lane = _current_lane(event, int(lane_count))
                target_t = int(target_raw_for_lane(event, lane))
                if old_lane != lane or int(event.get("target_T", event.get("T", 0))) != target_t:
                    moved += 1
                event["target_lane"] = lane
                event["target_T"] = target_t
            _add_unit_to_state(state, unit, int(chosen), transition_spans_by_lane)
        if not changed:
            break
    return int(moved)
