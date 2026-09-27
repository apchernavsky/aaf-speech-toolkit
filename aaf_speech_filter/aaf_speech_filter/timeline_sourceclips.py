from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional

from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient

from .media_resolve import _resolve_wave_path_for_sourceclip
from .timeline_timing import sourceclip_duration_sec
from .timeline_walk import (
    _is_sourceclip,
    _slot_is_soundish,
)


def _opgroup_gain_multiplier(node: Any) -> float:
    """Recognize constant standard gain by definition IDs, never display labels."""
    if "OperationGroup" not in type(node).__name__:
        return 1.0
    try:
        try:
            operation = node.operation
        except KeyError:
            legacy = node.get("OperationDefinition")
            operation = legacy.value if legacy is not None else None
        if str(operation.auid).lower() not in {
            "9d2ea894-0968-11d3-8a38-0050040ef7d2",
            "9d2ea895-0968-11d3-8a38-0050040ef7d2",
        }:
            return 1.0
        for parameter in node.parameters:
            if type(parameter).__name__ != "ConstantValue":
                continue
            if str(parameter.auid).lower() != "e4962321-2267-11d3-8a4c-0050040ef7d2":
                continue
            gain = float(parameter.value)
            return abs(gain) if math.isfinite(gain) else 1.0
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return 1.0
    return 1.0


@dataclass(frozen=True)
class TimelineSourceClip:
    composition_id: str
    slot_id: int
    path: tuple[tuple[str, int], ...]
    edit_rate: Optional[Fraction]
    gain_multiplier: float
    sourceclip: Any
    parent_vector: Any
    index: Optional[int]
    slot: Any


def iter_slot_sourceclip_occurrences(comp, slot):
    """Walk one slot with stable property/index paths and explicit mutation owners."""
    if not _slot_is_soundish(slot):
        return
    try:
        stored_rate = slot.edit_rate
        edit_rate = Fraction(str(stored_rate)) if stored_rate is not None else None
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
        edit_rate = None
    comp_id = str(getattr(comp, 'mob_id', ''))
    slot_id = int(getattr(slot, 'slot_id', 0))

    def walk(node, gain, path, parent_vector=None, index=None):
        if node is None:
            return
        if _is_sourceclip(node):
            yield TimelineSourceClip(comp_id, slot_id, path, edit_rate, gain,
                                     node, parent_vector, index, slot)
            return
        gain *= _opgroup_gain_multiplier(node)
        # These are AAF strong-reference properties, named as in SDK XML.
        for attribute, xml_property in (('components', 'ComponentObjects'),
                                        ('segments', 'InputSegments')):
            vector = getattr(node, attribute, None)
            if vector is None:
                continue
            for child_index in range(len(vector)):
                child = vector[child_index]
                yield from walk(child, gain, path + ((xml_property, child_index),),
                                vector, child_index)

    yield from walk(getattr(slot, 'segment', None), 1.0, ())


def iter_timeline_sourceclip_occurrences(aaf):
    for comp in aaf.content.compositionmobs():
        for slot in getattr(comp, 'slots', []) or []:
            yield from iter_slot_sourceclip_occurrences(comp, slot)


def iter_timeline_sourceclips(aaf):
    """Yield the established (edit rate, gain, clip) view of shared traversal."""
    for occurrence in iter_timeline_sourceclip_occurrences(aaf):
        yield occurrence.edit_rate, occurrence.gain_multiplier, occurrence.sourceclip


def total_timeline_clip_seconds(
    aaf,
    runtime_essence_paths: Optional[Dict[Any, Path]] = None,
    media_search_roots: Optional[tuple[Path, ...]] = None,
) -> float:
    total = 0.0
    for edit_rate, _gain, node in iter_timeline_sourceclips(aaf):
        wav_path = _resolve_wave_path_for_sourceclip(aaf, node, runtime_essence_paths, media_search_roots=media_search_roots, edit_rate=edit_rate)
        total += sourceclip_duration_sec(
            edit_rate,
            node,
            aaf=aaf,
            wav_path=wav_path,
            runtime_essence_paths=runtime_essence_paths,
            media_search_roots=media_search_roots,
        )
    return total


def total_timeline_sourceclips(aaf) -> int:
    return sum(1 for _edit_rate, _gain, _node in iter_timeline_sourceclips(aaf))


def count_resolvable_timeline_wavs(
    aaf_path: Path,
    *,
    runtime_essence_paths: Optional[Dict[Any, Path]] = None,
    media_search_roots: Optional[tuple[Path, ...]] = None,
    limit: int = 5000,
) -> int:
    """Count timeline SourceClips that resolve to a real media file on disk."""
    try:
        path = Path(aaf_path)
    except Exception:
        return 0
    if limit <= 0:
        limit = 1

    resolved = 0
    seen = 0
    with open_aaf_lenient(path, "r") as aaf:
        for _edit_rate, _gain, node in iter_timeline_sourceclips(aaf):
            seen += 1
            if seen > int(limit):
                break
            try:
                wav_path = _resolve_wave_path_for_sourceclip(aaf, node, runtime_essence_paths, media_search_roots=media_search_roots, edit_rate=_edit_rate)
                if wav_path is not None and wav_path.is_file():
                    resolved += 1
            except Exception:
                continue
    return int(resolved)


__all__ = [
    "count_resolvable_timeline_wavs",
    "iter_timeline_sourceclips",
    "iter_timeline_sourceclip_occurrences",
    "iter_slot_sourceclip_occurrences",
    "total_timeline_clip_seconds",
    "total_timeline_sourceclips",
]
