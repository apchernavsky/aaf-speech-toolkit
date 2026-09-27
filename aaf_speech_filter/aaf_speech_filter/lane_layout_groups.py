from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Optional

from .lane_layout_model import event_visible_start


SourceWindowResolver = Callable[[Any], Optional[tuple[int, int]]]
GroupKeyFunc = Callable[[dict[str, Any]], Optional[tuple[Any, ...]]]


def event_source_window_key(
    event: dict[str, Any],
    source_window_resolver: Optional[SourceWindowResolver] = None,
) -> tuple[int, int]:
    if "source_start" in event and "source_length" in event:
        return (int(event.get("source_start", -1)), int(event.get("source_length", -1)))
    if source_window_resolver is None:
        return (-1, -1)
    resolved = source_window_resolver(event.get("node"))
    if resolved is None:
        return (-1, -1)
    return (int(resolved[0]), int(resolved[1]))


def aligned_layout_group_key(
    event: dict[str, Any],
    source_window_resolver: Optional[SourceWindowResolver] = None,
) -> tuple[int, int, int, int]:
    source_start, source_length = event_source_window_key(event, source_window_resolver)
    return (
        int(event.get("T", 0)),
        int(event["L"]),
        int(source_start),
        int(source_length),
    )


def aligned_kind_group_key(
    event: dict[str, Any],
    source_window_resolver: Optional[SourceWindowResolver] = None,
) -> Optional[tuple[int, int, int, int]]:
    source_start, source_length = event_source_window_key(event, source_window_resolver)
    if int(source_start) < 0 or int(source_length) <= 0:
        return None
    return (
        event_visible_start(event),
        int(event["L"]),
        int(source_start),
        int(source_length),
    )


def dominant_aligned_group_kind(group: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for item in group:
        kind = str(item.get("kind") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    if counts.get("speech", 0) > 0 and counts.get("music", 0) <= 0:
        return "speech"
    priority = {"speech": 3, "music": 2, "noise": 1, "unknown": 0}
    return max(
        counts,
        key=lambda kind: (int(counts[kind]), int(priority.get(str(kind), -1))),
    )


def group_events_by_optional_key(
    events: list[dict[str, Any]],
    key_func: GroupKeyFunc,
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        key = key_func(event)
        if key is None:
            continue
        grouped[key].append(event)
    return dict(grouped)


def cross_lane_event_groups(
    events: list[dict[str, Any]],
    key_func: GroupKeyFunc,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for group in group_events_by_optional_key(events, key_func).values():
        if len(group) <= 1:
            continue
        if len({int(event["src_lane"]) for event in group}) <= 1:
            continue
        groups.append(group)
    return groups


def aligned_group_source_order(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(group, key=lambda event: (int(event["src_lane"]), int(event.get("top_idx", 0))))


def aligned_group_target_order(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assigned = [
        (int(event.get("target_lane", event["src_lane"])), event)
        for event in group
    ]
    return [event for _lane, event in sorted(assigned, key=lambda item: item[0])]


def aligned_group_target_lanes(group: list[dict[str, Any]]) -> list[int]:
    return sorted(int(event.get("target_lane", event["src_lane"])) for event in group)
