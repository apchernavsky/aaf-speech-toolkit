from __future__ import annotations


def dedupe_exact_transition_spans(
    spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for t, ln in spans:
        key = (int(t), int(ln))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def spans_without_exact_matches(
    spans: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    excluded_keys = {(int(t), int(ln)) for t, ln in excluded}
    out: list[tuple[int, int]] = []
    for span in spans:
        key = (int(span[0]), int(span[1]))
        if key not in excluded_keys:
            out.append(key)
    return out


def raw_from_visible_with_transition_spans(
    transition_spans: list[tuple[int, int]],
    visible_t: int,
    owned_pre_len: int = 0,
) -> int:
    transition_spans = dedupe_exact_transition_spans(list(transition_spans))
    raw = int(visible_t) + 2 * int(max(0, owned_pre_len))
    for _ in range(128):
        offset = 0
        for t, ln in sorted(transition_spans):
            if int(t) < int(raw):
                offset += int(ln)
        nxt = int(visible_t) + 2 * (int(offset) + int(max(0, owned_pre_len)))
        if int(nxt) == int(raw):
            return int(raw)
        raw = int(nxt)
    return int(raw)
