from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ClipNameResolution:
    """
    Best-effort clip display name resolution.

    ``display_name`` is the preferred name for UI/logs/diagnostics.
    ``candidates`` keeps additional sources for debugging (mob name, locator basename, etc.).
    ``issues`` records potential problems (missing mob, missing locator, etc.).
    """

    display_name: str
    candidates: Tuple[Tuple[str, str], ...]
    issues: Tuple[str, ...]


def _safe_str(x: Any) -> str:
    try:
        if x is None:
            return ""
        return str(x)
    except Exception:
        return ""


def _mob_name(mob: Any) -> str:
    # PyAAF2: MasterMob.name is usually what Nuendo shows.
    for attr in ("name", "Name"):
        try:
            v = getattr(mob, attr)
            if v:
                return _safe_str(v)
        except Exception:
            pass
    try:
        v = mob["Name"].value
        if v:
            return _safe_str(v)
    except Exception:
        pass
    return ""


def _resolve_source_mob(aaf: Any, mob_id: Any) -> Any:
    try:
        mobs = aaf.content.mobs
        it = mobs.values() if hasattr(mobs, "values") else mobs
    except Exception:
        return None
    want_s = _safe_str(mob_id)
    for m in it:
        try:
            mid = getattr(m, "mob_id", None)
            if mid == mob_id:
                return m
            if want_s and _safe_str(mid) == want_s:
                return m
        except Exception:
            pass
        try:
            mid2 = m["MobID"].value
            if mid2 == mob_id:
                return m
            if want_s and _safe_str(mid2) == want_s:
                return m
        except Exception:
            pass
    return None


def _first_locator_url(desc: Any) -> str:
    """
    Return first URLString from descriptor Locator vector (if any).
    """
    try:
        loc_vec = desc["Locator"]
    except Exception:
        return ""
    try:
        first = next(iter(loc_vec), None)
    except Exception:
        try:
            first = loc_vec[0]
        except Exception:
            first = None
    if first is None:
        return ""
    try:
        u = first["URLString"].value
        return _safe_str(u)
    except Exception:
        return ""


def _basename_from_url_or_path(s: str) -> str:
    if not s:
        return ""
    t = str(s).strip()
    # Common: file:///
    if t.lower().startswith("file:"):
        # Keep it simple for diagnostics: just take last path-ish segment.
        t = t.replace("\\", "/")
        return t.split("/")[-1]
    # Windows / POSIX path
    try:
        return Path(t).name
    except Exception:
        t = t.replace("\\", "/")
        return t.split("/")[-1]


def _event_inner_sourceclip(node: Any) -> Any:
    """
    If node is OperationGroup, attempt to return the first SourceClip in InputSegments.
    If node is SourceClip, return it.
    """
    try:
        cn = node.__class__.__name__
    except Exception:
        return None
    if "SourceClip" in cn:
        return node

    # For OperationGroup (and other wrappers), SourceClip can be nested deeper than InputSegments[0].
    stack = [node]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        i = id(cur)
        if i in seen:
            continue
        seen.add(i)
        try:
            cn2 = cur.__class__.__name__
        except Exception:
            cn2 = ""
        if "SourceClip" in cn2:
            return cur
        # Common containers
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
        # AAF OperationGroup inputs
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


def resolve_clip_display_name(
    aaf: Any,
    timeline_node: Any,
    *,
    cache: Optional[Dict[Any, Any]] = None,
) -> ClipNameResolution:
    """
    Resolve display name for a timeline block (SourceClip or OperationGroup wrapping SourceClip).

    Priority:
    1) Resolved Mob name (MasterMob/SourceMob Name) — usually matches what Nuendo shows.
    2) Descriptor locator basename (if any).
    3) Fallback to class name / empty.
    """
    issues: list[str] = []
    candidates: list[tuple[str, str]] = []

    inner = _event_inner_sourceclip(timeline_node)
    if inner is None:
        issues.append("no_inner_sourceclip")
        nm = _safe_str(getattr(timeline_node, "__class__", type(timeline_node)).__name__)
        return ClipNameResolution(
            display_name=nm,
            candidates=tuple([("node_class", nm)]),
            issues=tuple(issues),
        )

    src_id = None
    try:
        src_id = inner["SourceID"].value
    except Exception:
        src_id = getattr(inner, "source_id", None)
    if src_id is None:
        issues.append("no_source_id")
        nm = _safe_str(getattr(timeline_node, "__class__", type(timeline_node)).__name__)
        return ClipNameResolution(
            display_name=nm,
            candidates=tuple([("node_class", nm)]),
            issues=tuple(issues),
        )

    mob = None
    if cache is not None and src_id in cache:
        mob = cache.get(src_id)
    else:
        mob = _resolve_source_mob(aaf, src_id)
        if cache is not None:
            cache[src_id] = mob

    if mob is None:
        issues.append("mob_not_found")
        return ClipNameResolution(
            display_name="",
            candidates=tuple([("source_id", _safe_str(src_id))]),
            issues=tuple(issues),
        )

    mob_nm = _mob_name(mob)
    if mob_nm:
        candidates.append(("mob_name", mob_nm))

    desc = getattr(mob, "descriptor", None)
    if desc is not None:
        url = _first_locator_url(desc)
        if url:
            candidates.append(("locator_url", url))
            bn = _basename_from_url_or_path(url)
            if bn:
                candidates.append(("locator_basename", bn))

    # Choose display_name: mob name first, then locator basename.
    if mob_nm:
        display = mob_nm
    else:
        display = ""
        for k, v in candidates:
            if k == "locator_basename" and v:
                display = v
                break
        if not display:
            issues.append("no_mob_name")

    return ClipNameResolution(
        display_name=display,
        candidates=tuple(candidates),
        issues=tuple(issues),
    )
