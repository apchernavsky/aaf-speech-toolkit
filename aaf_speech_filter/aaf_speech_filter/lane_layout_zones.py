from __future__ import annotations


def lane_zone_preferences(
    lane_count: int,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    lanes_all = list(range(int(lane_count)))
    if not lanes_all:
        return [], [], [], [], []
    speech_end = max(1, int(lane_count) // 3)
    music_start = max(speech_end, (2 * int(lane_count)) // 3)
    speech_lanes = lanes_all[:speech_end]
    noise_lanes = lanes_all[speech_end:music_start] or speech_lanes
    music_lanes = lanes_all[music_start:] or lanes_all[-1:]
    center = (noise_lanes[0] + noise_lanes[-1]) / 2.0
    noise_pref = sorted(noise_lanes, key=lambda i: (abs(i - center), i))
    speech_pref = speech_lanes + [i for i in noise_lanes if i not in speech_lanes]
    noise_pref = noise_pref + [i for i in reversed(speech_lanes) if i not in noise_pref]
    music_pref = list(reversed(music_lanes))
    music_pref = music_pref + [i for i in reversed(noise_lanes) if i not in music_pref]
    unknown_pref = noise_pref + [i for i in speech_lanes if i not in noise_pref]
    return speech_pref, noise_pref, music_pref, unknown_pref, lanes_all


def class_zone_lane_order(kind: str, lane_count: int) -> list[int]:
    speech_pref, noise_pref, music_pref, unknown_pref, lanes_all = lane_zone_preferences(
        int(lane_count)
    )
    if not lanes_all:
        return []
    if str(kind) == "music":
        return list(music_pref)
    if str(kind) == "speech":
        return list(speech_pref)
    if str(kind) == "noise":
        return list(noise_pref)
    return list(unknown_pref)


def class_zone_lane_preference(kind: str, lane_count: int, current_lane: int) -> list[int]:
    pref = class_zone_lane_order(kind, lane_count)
    if int(current_lane) in pref:
        pref.remove(int(current_lane))
    pref.append(int(current_lane))
    return pref
