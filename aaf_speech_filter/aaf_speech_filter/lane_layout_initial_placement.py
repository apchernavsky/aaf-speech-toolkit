from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .lane_layout_candidates import candidate_group_lane_sequences
from .lane_layout_groups import group_events_by_optional_key
from .lane_layout_model import event_owned_raw_transition_spans
from .lane_layout_occupancy import LaneOccupancyState, SourceSpanTracker
from .lane_layout_placement import EventPlacementSpan

TargetAllowed = Callable[[dict[str, Any], int], bool]
SortKey = Callable[[dict[str, Any]], Any]
GroupKey = Callable[[dict[str, Any]], Optional[tuple[int, int, int, int]]]


@dataclass
class LaneInitialPlacementPlanner:
    lane_count: int
    raw_state: LaneOccupancyState
    source_spans: SourceSpanTracker
    lane_pref_all: list[int]
    target_allowed: TargetAllowed
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]]
    aligned_kind_group_key: GroupKey
    aligned_layout_group_key: GroupKey

    def _span_for_lane(self, event: dict[str, Any], lane: int) -> EventPlacementSpan:
        return EventPlacementSpan.from_event(
            event,
            lane=int(lane),
            transition_spans_by_lane=self.transition_spans_by_lane,
        )

    def _span_conflicts(self, event: dict[str, Any], span: EventPlacementSpan) -> bool:
        return self.raw_state.conflicts(
            span.lane,
            span.raw_t0,
            span.raw_t1,
            ignored_raw_spans=event_owned_raw_transition_spans(event, span.event_t),
        )

    def _take(self, span: EventPlacementSpan) -> None:
        self.raw_state.add(span.lane, span.raw_t0, span.raw_t1)

    def _fallback_lanes(self, event: dict[str, Any]) -> list[int]:
        fallback_lanes: list[int] = []
        for lane in (int(event["src_lane"]), *self.lane_pref_all):
            lane_i = int(lane)
            if 0 <= lane_i < int(self.lane_count) and lane_i not in fallback_lanes:
                fallback_lanes.append(lane_i)
        return fallback_lanes

    def place_event(self, event: dict[str, Any], lane_pref: list[int]) -> None:
        released_source = self.source_spans.release(event)
        for lane in lane_pref:
            if not self.target_allowed(event, int(lane)):
                continue
            span = self._span_for_lane(event, int(lane))
            if not self._span_conflicts(event, span):
                event["target_lane"] = int(lane)
                event["target_T"] = int(span.event_t)
                self._take(span)
                return
        for lane in self._fallback_lanes(event):
            if int(lane) != int(event["src_lane"]) and not self.target_allowed(event, int(lane)):
                continue
            span = self._span_for_lane(event, int(lane))
            if not self._span_conflicts(event, span):
                event["target_lane"] = int(lane)
                event["target_T"] = int(span.event_t)
                self._take(span)
                return
        self.source_spans.restore(released_source)
        name = str(event.get("display_name") or event.get("name") or "clip")
        raise RuntimeError(
            "YAMNet lane layout could not place block without overlap: "
            f"{name} kind={event.get('kind')} src_lane={int(event['src_lane'])} "
            f"T={event.get('T')} T_edit={event.get('T_edit')} L={event.get('L')}"
        )

    def place_event_group(
        self,
        group: list[dict[str, Any]],
        lane_pref: list[int],
        *,
        kind: str,
    ) -> bool:
        ordered = sorted(group, key=lambda event: (int(event["src_lane"]), int(event.get("top_idx", 0))))
        if len(ordered) <= 1:
            return False
        released_sources = [self.source_spans.release(event) for event in ordered]
        for lane_seq in candidate_group_lane_sequences(
            str(kind),
            lane_pref,
            len(ordered),
            lane_count=int(self.lane_count),
        ):
            planned: list[tuple[dict[str, Any], EventPlacementSpan]] = []
            ok = True
            for event, lane in zip(ordered, lane_seq):
                if not self.target_allowed(event, int(lane)):
                    ok = False
                    break
                span = self._span_for_lane(event, int(lane))
                if self._span_conflicts(event, span):
                    ok = False
                    break
                planned.append((event, span))
            if not ok:
                continue
            for event, span in planned:
                event["target_lane"] = int(span.lane)
                event["target_T"] = int(span.event_t)
                self._take(span)
            return True
        for span in released_sources:
            self.source_spans.restore(span)
        return False

    def place_events_preserving_aligned_order(
        self,
        class_events: list[dict[str, Any]],
        lane_pref: list[int],
        *,
        kind: str,
        sort_key: SortKey,
    ) -> None:
        placed_ids: set[int] = set()

        def place_grouped(key_func: GroupKey) -> None:
            grouped = group_events_by_optional_key(
                [event for event in class_events if id(event) not in placed_ids],
                key_func,
            )
            for _key, group in sorted(grouped.items(), key=lambda item: item[0]):
                if len(group) <= 1:
                    continue
                if self.place_event_group(group, lane_pref, kind=str(kind)):
                    placed_ids.update(id(event) for event in group)

        place_grouped(self.aligned_kind_group_key)
        place_grouped(self.aligned_layout_group_key)
        for event in sorted(class_events, key=sort_key):
            if id(event) in placed_ids:
                continue
            self.place_event(event, lane_pref)
