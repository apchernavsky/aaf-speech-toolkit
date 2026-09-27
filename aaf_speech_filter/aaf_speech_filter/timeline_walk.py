from __future__ import annotations

from typing import Any, Optional


def _rational_to_float(r) -> Optional[float]:
    try:
        return float(r)
    except Exception:
        try:
            n = getattr(r, "numerator", None)
            d = getattr(r, "denominator", None)
            if n is not None and d:
                return float(n) / float(d)
        except Exception:
            pass
    return None


def _aaf_property_int_optional(obj: Any, names: tuple[str, ...]) -> Optional[int]:
    for name in names:
        try:
            raw = obj[name].value
            rf = _rational_to_float(raw)
            if rf is not None:
                return int(round(rf))
            return int(raw)
        except Exception:
            continue
    return None


def _aaf_int_length(node: Any) -> int:
    v = _aaf_property_int_optional(node, ("Length",))
    return v if v is not None else int(getattr(node, "length", 0) or 0)


def _aaf_int_start(node: Any) -> int:
    v = _aaf_property_int_optional(node, ("StartPosition", "StartTime", "Start"))
    return v if v is not None else int(getattr(node, "start", 0) or 0)


def _slot_edit_rate(slot) -> Optional[float]:
    for attr in ("edit_rate", "editrate", "EditRate"):
        try:
            v = getattr(slot, attr, None)
            if v is None:
                continue
            try:
                return _rational_to_float(v)
            except Exception:
                return float(v)
        except Exception:
            pass
    try:
        return _rational_to_float(slot["EditRate"].value)
    except Exception:
        return None


def _slot_is_soundish(slot) -> bool:
    mk = getattr(slot, "media_kind", None)
    if mk is None:
        try:
            er = _slot_edit_rate(slot)
        except Exception:
            er = None
        try:
            if er is not None and float(er) > 0.0 and float(er) < 1000.0:
                return False
        except Exception:
            pass
        return True
    s = str(mk).lower()
    if (
        "picture" in s
        or "timecode" in s
        or "descriptive" in s
        or "metadata" in s
        or "aux" in s
        or "matte" in s
    ):
        return False
    return "sound" in s or "audio" in s


def _iter_segments(seg):
    if seg is None:
        return
    yield seg
    for attr in ("components", "segments", "alternates"):
        children = getattr(seg, attr, None)
        if children:
            for c in children:
                yield from _iter_segments(c)
    selected = getattr(seg, "selected", None)
    if selected is not None:
        yield from _iter_segments(selected)


def _is_sourceclip(obj) -> bool:
    if obj is None:
        return False
    n = obj.__class__.__name__
    return n == "SourceClip" or n.endswith("SourceClip")


def _is_filler(obj) -> bool:
    if obj is None:
        return False
    n = obj.__class__.__name__
    return n == "Filler" or n.endswith("Filler")


def _node_length_units(node) -> int:
    return _aaf_int_length(node)


def _inner_sourceclip_for_block(node: Any) -> Optional[Any]:
    if node is None:
        return None
    if _is_sourceclip(node):
        return node
    stack = [node]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        i = id(cur)
        if i in seen:
            continue
        seen.add(i)
        if _is_sourceclip(cur):
            return cur
        for attr in ("components", "segments"):
            ch = getattr(cur, attr, None)
            if not ch:
                continue
            try:
                n = len(ch)
            except Exception:
                continue
            for j in range(n):
                try:
                    stack.append(ch[j])
                except Exception:
                    pass
        try:
            ins = cur["InputSegments"]
            for j in range(len(ins)):
                try:
                    stack.append(ins[j])
                except Exception:
                    pass
        except Exception:
            pass
        sel = getattr(cur, "selected", None)
        if sel is not None:
            stack.append(sel)
    return None


def _vectors_on_container(container):
    seen: set[int] = set()
    for attr in ("components", "segments"):
        vec = getattr(container, attr, None)
        if vec is not None:
            try:
                if len(vec) > 0:
                    ident = id(vec)
                    if ident not in seen:
                        seen.add(ident)
                        yield vec
            except Exception:
                pass
    try:
        ins = container["InputSegments"]
        if ins is not None and len(ins) > 0:
            ident = id(ins)
            if ident not in seen:
                seen.add(ident)
                yield ins
    except Exception:
        pass
