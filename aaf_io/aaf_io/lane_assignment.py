"""Deterministic order-preserving assignment with explicit feasible candidates."""
from __future__ import annotations


def ordered_lane_assignment(candidate_lanes_by_event, current_lanes):
    """Choose strictly increasing lanes, or None if no complete assignment exists.

    Callers own timing, effects and occupancy checks. Dynamic programming avoids
    greedy dead ends. Cost prefers fewer moves, then shorter distance, then a
    stable lexicographic assignment. Inputs are never mutated.
    """
    if len(candidate_lanes_by_event) != len(current_lanes):
        raise ValueError("Candidate and current-lane counts differ")
    states = {-1: (0, 0, ())}
    for candidates, current in zip(candidate_lanes_by_event, current_lanes):
        next_states = {}
        for lane in sorted(set(candidates)):
            if lane < 0:
                raise ValueError("Lane indices must be nonnegative")
            prefixes = [cost for previous, cost in states.items() if previous < lane]
            if not prefixes:
                continue
            moved, distance, assignment = min(prefixes)
            next_states[lane] = (moved + (lane != current), distance + abs(lane - current), (*assignment, lane))
        if not next_states:
            return None
        states = next_states
    return min(states.values())[2]
