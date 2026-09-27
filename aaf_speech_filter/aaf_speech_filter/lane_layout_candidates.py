from __future__ import annotations


def physical_class_band(kind: str, lane_count: int) -> list[int]:
    lanes_all = list(range(int(lane_count)))
    if not lanes_all:
        return []
    speech_end = max(1, int(lane_count) // 3)
    music_start = max(speech_end, (2 * int(lane_count)) // 3)
    if str(kind) == "music":
        return lanes_all[music_start:] or lanes_all[-1:]
    if str(kind) == "speech":
        return lanes_all[:speech_end]
    if str(kind) == "noise":
        return lanes_all[speech_end:music_start] or lanes_all[:speech_end]
    return lanes_all


def candidate_group_lane_sequences(
    kind: str,
    lane_pref: list[int],
    group_size: int,
    *,
    lane_count: int,
) -> list[list[int]]:
    size = int(group_size)
    if size <= 0:
        return []
    allowed = sorted({int(lane) for lane in lane_pref if 0 <= int(lane) < int(lane_count)})
    if len(allowed) < size:
        return []
    primary = set(physical_class_band(str(kind), int(lane_count)))
    if not primary:
        primary = set(allowed)
    primary_center = (
        (min(primary) + max(primary)) / 2.0 if primary else (int(lane_count) - 1) / 2.0
    )

    contiguous: list[list[int]] = []
    for start_idx in range(0, len(allowed) - size + 1):
        seq = allowed[start_idx:start_idx + size]
        if all(int(seq[i + 1]) == int(seq[i]) + 1 for i in range(len(seq) - 1)):
            contiguous.append(seq)

    def seq_rank(seq: list[int]) -> tuple[float, float, float]:
        outside = sum(1 for lane in seq if lane not in primary)
        if str(kind) == "music":
            return (float(outside), float(-seq[-1]), float(-seq[0]))
        if str(kind) == "noise":
            center = (seq[0] + seq[-1]) / 2.0
            return (float(outside), abs(center - primary_center), float(seq[0]))
        return (float(outside), float(seq[0]), float(seq[-1]))

    sequences = sorted(contiguous, key=seq_rank)
    if sequences:
        return [list(seq) for seq in sequences]

    ranked = sorted(allowed, key=lambda lane: seq_rank([lane]))
    fallback: list[list[int]] = []
    for start_idx in range(0, len(ranked) - size + 1):
        seq = sorted(ranked[start_idx:start_idx + size])
        if len(seq) == size and seq not in fallback:
            fallback.append(seq)
    return fallback
