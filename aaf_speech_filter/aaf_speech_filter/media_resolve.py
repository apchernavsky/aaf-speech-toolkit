from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional

from aaf_io.heal.davinci import first_resolved_locator_media_path as _davinci_first_locator_media_path
from aaf_io.heal.premiere import file_url_to_path, resolve_import_sidecar_media_path

from .timeline_walk import _is_sourceclip


def _path_is_file_fast(p: Path) -> bool:
    """
    Windows guard: ``Path.is_file()`` against a missing drive letter can stall badly.
    """
    try:
        p = Path(p)
    except Exception:
        return False
    if os.name == "nt":
        try:
            drive = p.drive
        except Exception:
            drive = ""
        if drive and len(drive) >= 2 and drive[1] == ":":
            try:
                import ctypes

                dt = int(ctypes.windll.kernel32.GetDriveTypeW(drive + "\\"))
                if dt == 1:
                    return False
            except Exception:
                pass
    try:
        return bool(p.is_file())
    except Exception:
        return False


def _first_resolved_sidecar_path(desc, media_search_roots=None) -> Optional[Path]:
    try:
        loc_vec = desc["Locator"]
        first = next(iter(loc_vec), None)
        if first is None:
            return None
        url = first["URLString"].value
        if not url:
            return None
        return resolve_import_sidecar_media_path(file_url_to_path(str(url)), media_search_roots=media_search_roots)
    except Exception:
        return None


def _resolve_source_mob(aaf, mob_id):
    try:
        mobs = aaf.content.mobs
        indexed_get = getattr(mobs, "get", None)
        if callable(indexed_get):
            return indexed_get(mob_id)
        if hasattr(mobs, "values"):
            for m in mobs.values():
                if getattr(m, "mob_id", None) == mob_id:
                    return m
        else:
            for m in mobs:
                if getattr(m, "mob_id", None) == mob_id:
                    return m
    except Exception:
        return None
    return None


def _wave_path_from_source_mob(
    mob,
    runtime_essence_paths: Optional[Dict[Any, Path]] = None,
    *,
    media_search_roots: Optional[tuple[Path, ...]] = None,
) -> Optional[Path]:
    desc = getattr(mob, "descriptor", None)
    mid = getattr(mob, "mob_id", None)
    fb: Optional[Path] = None
    if runtime_essence_paths is not None and mid is not None:
        try:
            fb = runtime_essence_paths.get(mid)
        except Exception:
            fb = None
        if fb is not None and not _path_is_file_fast(fb):
            fb = None

    cname = desc.__class__.__name__ if desc is not None else ""
    if cname == "WAVEDescriptor":
        p = _first_resolved_sidecar_path(desc, media_search_roots)
        if p is not None and _path_is_file_fast(p):
            return p
        return fb
    if cname == "ImportDescriptor":
        p = _first_resolved_sidecar_path(desc, media_search_roots)
        if p is not None and _path_is_file_fast(p):
            return p
        return fb
    if cname in ("PCMDescriptor", "AIFCDescriptor"):
        p = _davinci_first_locator_media_path(desc, media_search_roots=media_search_roots)
        if p is not None and _path_is_file_fast(p):
            return p
        return fb
    return fb


class UnsupportedMediaMapping(ValueError):
    """The requested interval cannot be mapped to one continuous essence window."""


def _reference_slot(aaf, node):
    source_id = node["SourceID"].value
    mob = _resolve_source_mob(aaf, source_id)
    if mob is None:
        raise UnsupportedMediaMapping("SourceClip references an unavailable mob")
    slot_id = int(node["SourceMobSlotID"].value)
    slot = next((s for s in mob.slots if int(s.slot_id) == slot_id), None)
    if slot is None:
        raise UnsupportedMediaMapping("SourceClip references an unavailable slot")
    return mob, slot


def resolve_sourceclip_window(aaf, node, start_seconds, duration_seconds):
    """Return terminal SourceMob and exact seconds, composing every referenced slot.

    StartTime uses the owning slot's rate; each target Origin uses its target rate.
    AAF Object Specification 1.0.1 sections 7.7 and 26.3.3.
    """
    start = Fraction(str(start_seconds))
    duration = Fraction(str(duration_seconds))
    seen = set()
    for _ in range(64):
        mob, slot = _reference_slot(aaf, node)
        identity = (str(mob.mob_id), int(slot.slot_id))
        if identity in seen:
            raise UnsupportedMediaMapping("Cyclic media reference")
        seen.add(identity)
        rate = Fraction(str(slot.edit_rate))
        if rate <= 0:
            raise UnsupportedMediaMapping("Nonpositive referenced edit rate")
        start += Fraction(int(slot.origin), 1) / rate
        if start < 0 or duration < 0:
            raise UnsupportedMediaMapping("Media window precedes available essence")
        if mob.__class__.__name__ == "SourceMob":
            return mob, start, duration
        if mob.__class__.__name__ not in ("MasterMob", "CompositionMob"):
            raise UnsupportedMediaMapping("Unsupported referenced mob type")
        segment = slot.segment
        if segment.__class__.__name__ == "Sequence":
            # Transitions can overlap several sources; a single-file analysis is unsafe.
            components = list(segment.components)
            if any(c.__class__.__name__ == "Transition" for c in components):
                raise UnsupportedMediaMapping("Transition in referenced media window")
            selected = []
            for _, position, child in segment.positions():
                begin = Fraction(position, 1) / rate
                end = begin + Fraction(int(child.length), 1) / rate
                if begin <= start < end and start + duration <= end:
                    selected.append((child, start - begin))
            if len(selected) != 1:
                raise UnsupportedMediaMapping("Media window crosses a sequence boundary")
            segment, start = selected[0]
        if not _is_sourceclip(segment):
            raise UnsupportedMediaMapping("Referenced segment is not a linear SourceClip")
        if start + duration > Fraction(int(segment.length), 1) / rate:
            raise UnsupportedMediaMapping("Media window exceeds referenced SourceClip")
        start += Fraction(int(segment.start), 1) / rate
        node = segment
    raise UnsupportedMediaMapping("Media reference depth exceeds 64")


def _unique_reference_media_path(aaf, sourceclip, runtime_essence_paths, media_search_roots):
    """Find an unambiguous file before a malformed outer clock can be inferred.

    This is only media discovery, not permission to analyze a source interval.
    resolve_sourceclip_window must still prove the complete edited window.
    """
    pending = [sourceclip]
    visited = set()
    paths = set()
    for _ in range(256):
        if not pending:
            return next(iter(paths)) if len(paths) == 1 else None
        node = pending.pop()
        if _is_sourceclip(node):
            mob, slot = _reference_slot(aaf, node)
            identity = (str(mob.mob_id), int(slot.slot_id))
            if identity in visited:
                continue
            visited.add(identity)
            if type(mob).__name__ == "SourceMob":
                path = _wave_path_from_source_mob(mob, runtime_essence_paths, media_search_roots=media_search_roots)
                if path is None:
                    return None
                paths.add(path)
                if len(paths) > 1:
                    return None
            else:
                pending.append(slot.segment)
        elif type(node).__name__ == "Sequence":
            pending.extend(node.components)
        elif type(node).__name__ != "Filler":
            return None
    return None


def _resolve_wave_path_for_sourceclip(
    aaf,
    sourceclip,
    runtime_essence_paths: Optional[Dict[Any, Path]] = None,
    *,
    media_search_roots: Optional[tuple[Path, ...]] = None,
    edit_rate=None,
) -> Optional[Path]:
    try:
        mob, slot = _reference_slot(aaf, sourceclip)
        if (edit_rate is None or Fraction(str(edit_rate)) < 1000) and mob.__class__.__name__ != "SourceMob":
            return _unique_reference_media_path(aaf, sourceclip, runtime_essence_paths, media_search_roots)
        rate = Fraction(str(edit_rate if edit_rate is not None else slot.edit_rate))
        if rate <= 0:
            return None
        mob, _, _ = resolve_sourceclip_window(aaf, sourceclip, Fraction(int(sourceclip.start), 1) / rate, 0)
    except (UnsupportedMediaMapping, KeyError, AttributeError, TypeError, ValueError):
        return None
    return _wave_path_from_source_mob(mob, runtime_essence_paths, media_search_roots=media_search_roots)
