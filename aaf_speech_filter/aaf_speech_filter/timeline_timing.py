from __future__ import annotations

import os
import wave
from fractions import Fraction
from aaf_io.audio_format import pcm_container_kind
from pathlib import Path
from typing import Any, Dict, Optional

from .media_resolve import (
    UnsupportedMediaMapping,
    _first_resolved_sidecar_path,
    resolve_sourceclip_window,
    _resolve_source_mob,
    _resolve_wave_path_for_sourceclip,
)
from .timeline_walk import (
    _aaf_int_length,
    _aaf_int_start,
    _is_sourceclip,
    _rational_to_float,
)


_wav_sample_rate_cache: dict[tuple[str, float], float] = {}


def wav_sample_rate(path: Path) -> float:
    """Read WAV/AIFF sample rate with a small path+mtime cache."""
    cache_key: Optional[tuple[str, float]] = None
    try:
        if path.is_file():
            cache_key = (os.path.normcase(str(path.resolve())), float(path.stat().st_mtime))
            hit = _wav_sample_rate_cache.get(cache_key)
            if hit is not None:
                return hit
    except Exception:
        cache_key = None
    try:
        container_kind = pcm_container_kind(path)
    except (OSError, ValueError):
        return 0.0
    if container_kind == "aiff":
        try:
            import warnings
            import aifc

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                wf = aifc.open(str(path), "r")
            try:
                out = float(wf.getframerate())
            finally:
                wf.close()
        except Exception:
            out = 0.0
    else:
        try:
            import wave

            with wave.open(str(path), "rb") as wf:
                out = float(wf.getframerate())
        except Exception:
            out = 0.0
    if cache_key is not None:
        _wav_sample_rate_cache[cache_key] = out
    return out


def descriptor_sample_rate(desc: Any) -> float:
    if desc is None:
        return 0.0
    for key in ("SampleRate", "AudioSampleRate"):
        try:
            v = desc[key].value
            rf = _rational_to_float(v)
            if rf is not None and rf >= 8000.0:
                return float(rf)
        except Exception:
            pass
    if desc.__class__.__name__ == "ImportDescriptor":
        p = _first_resolved_sidecar_path(desc)
        if p is not None and p.is_file():
            sr = wav_sample_rate(p)
            if sr >= 8000.0:
                return sr
    return 0.0


def walk_segment_tree_for_essence_sample_rate(aaf: Any, seg: Any) -> float:
    if seg is None:
        return 0.0
    stack = [seg]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        i = id(cur)
        if i in seen:
            continue
        seen.add(i)
        if _is_sourceclip(cur):
            try:
                sid = cur["SourceID"].value
            except Exception:
                sid = getattr(cur, "source_id", None)
            if sid is None:
                continue
            mob = _resolve_source_mob(aaf, sid)
            if mob is None or mob.__class__.__name__ != "SourceMob":
                continue
            sr = descriptor_sample_rate(getattr(mob, "descriptor", None))
            if sr >= 8000.0:
                return sr
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
    return 0.0


def essence_sample_rate_for_sourceclip(
    aaf: Any,
    sourceclip: Any,
    runtime_essence_paths: Optional[Dict[Any, Path]] = None,
) -> float:
    try:
        src_id = sourceclip["SourceID"].value
    except Exception:
        src_id = getattr(sourceclip, "source_id", None)
    if src_id is None:
        return 0.0
    mob = _resolve_source_mob(aaf, src_id)
    if mob is None:
        return 0.0
    cname = mob.__class__.__name__
    if cname == "SourceMob":
        return descriptor_sample_rate(getattr(mob, "descriptor", None))
    if cname == "MasterMob":
        try:
            slot_id = sourceclip["SourceMobSlotID"].value
        except Exception:
            try:
                slot_id = int(sourceclip["SourceSlotID"].value)
            except Exception:
                slot_id = None
        if slot_id is None:
            return 0.0
        want = int(slot_id)
        for sl in getattr(mob, "slots", []) or []:
            got = getattr(sl, "slot_id", None)
            if got is None:
                try:
                    got = int(sl["SlotID"].value)
                except Exception:
                    got = None
            if got is None or int(got) != want:
                continue
            sub = getattr(sl, "segment", None)
            if sub is None:
                continue
            r = walk_segment_tree_for_essence_sample_rate(aaf, sub)
            if r >= 8000.0:
                return r
    return 0.0


def choose_timebase_rate_for_sourceclip(
    slot_edit_rate: float,
    *,
    length_units: int,
    start_units: int,
    wav_path: Optional[Path],
    essence_sr: float = 0.0,
) -> float:
    """Use the owning edit rate specified by the AAF SourceClip contract.

    AAF Object Specification 1.0.1 section 7.7 defines StartTime and Length in
    owning-slot units. A second numerically fitting sample interpretation is
    not evidence of malformed metadata. Incomplete declared windows remain
    unresolved; never infer sample units merely because they fit the file.
    """
    try:
        rate = Fraction(str(slot_edit_rate))
    except (ValueError, ZeroDivisionError) as exc:
        raise UnsupportedMediaMapping("Invalid owning edit rate") from exc
    if rate <= 0 or length_units <= 0 or start_units < 0:
        raise UnsupportedMediaMapping("Invalid SourceClip rate or window")
    if rate >= 1000:
        return float(rate)
    if wav_path is None:
        raise UnsupportedMediaMapping("Low-rate clock cannot be resolved without media")
    try:
        if pcm_container_kind(wav_path) == 'aiff':
            import aifc
            try:
                with aifc.open(str(wav_path), 'rb') as media:
                    sample_rate, frames = media.getframerate(), media.getnframes()
            except aifc.Error as exc:
                raise UnsupportedMediaMapping("AIFF clock or duration is unavailable") from exc
        else:
            with wave.open(str(wav_path), 'rb') as media:
                sample_rate, frames = media.getframerate(), media.getnframes()
    except (OSError, EOFError, ValueError, wave.Error) as exc:
        raise UnsupportedMediaMapping("Media clock or duration is unavailable") from exc
    if sample_rate <= 0:
        raise UnsupportedMediaMapping("Media sample rate is invalid")
    end_units = int(start_units) + int(length_units)
    media_duration = Fraction(frames, sample_rate)
    fits_declared = Fraction(end_units, 1) / rate <= media_duration
    fits_samples = end_units <= frames
    if fits_declared:
        return float(rate)
    if fits_samples:
        raise UnsupportedMediaMapping(
            "Incomplete owning-rate window does not prove sample units"
        )
    raise UnsupportedMediaMapping("Neither audio clock covers the requested media window")


def edited_audio_sample_rate(
    slot_edit_rate: float,
    wav_path: Optional[Path],
    essence_sr: float = 0.0,
) -> float:
    er = float(slot_edit_rate) if slot_edit_rate else 48000.0
    if er <= 0:
        er = 48000.0
    sr_file = 0.0
    if wav_path is not None and wav_path.is_file():
        sr_file = wav_sample_rate(wav_path)
    sr = sr_file if sr_file >= 8000.0 else float(essence_sr or 0.0)
    if sr >= 8000.0 and er < 1000.0:
        return sr
    return er


def sourceclip_duration_sec(
    edit_rate: float,
    node: Any,
    *,
    aaf: Optional[Any] = None,
    wav_path: Optional[Path] = None,
    runtime_essence_paths: Optional[Dict[Any, Path]] = None,
    media_search_roots: Optional[tuple[Path, ...]] = None,
) -> float:
    length = _aaf_int_length(node)
    if aaf is not None:
        wp = wav_path
        if wp is None:
            wp = _resolve_wave_path_for_sourceclip(aaf, node, runtime_essence_paths, media_search_roots=media_search_roots, edit_rate=edit_rate)
        ess = essence_sample_rate_for_sourceclip(aaf, node, runtime_essence_paths)
        st = _aaf_int_start(node)
        er = choose_timebase_rate_for_sourceclip(
            float(edit_rate) or 48000.0,
            length_units=length,
            start_units=st,
            wav_path=wp,
            essence_sr=ess,
        )
    else:
        er = float(edit_rate) if edit_rate else 48000.0
        if er <= 0:
            er = 48000.0
    return max(0.0, float(length) / er)


def sourceclip_audio_timing(
    aaf: Any,
    slot_edit_rate: float,
    node: Any,
    wav_path: Optional[Path],
    runtime_essence_paths: Optional[Dict[Any, Path]] = None,
    *,
    visible_duration_sec: Optional[float] = None,
) -> tuple[Fraction, Fraction]:
    """Return exact media seconds; ambiguous clocks raise UnsupportedMediaMapping."""
    ess = essence_sample_rate_for_sourceclip(aaf, node, runtime_essence_paths)
    length = _aaf_int_length(node)
    st = _aaf_int_start(node)
    er = choose_timebase_rate_for_sourceclip(
        slot_edit_rate,
        length_units=length,
        start_units=st,
        wav_path=wav_path,
        essence_sr=ess,
    )
    rate = Fraction(str(slot_edit_rate)) if slot_edit_rate and float(slot_edit_rate) == er else Fraction(str(er))
    dur = max(Fraction(0), Fraction(length, 1) / rate)
    if visible_duration_sec is not None:
        dur = min(dur, max(Fraction(0), Fraction(str(visible_duration_sec))))
    start_sec = Fraction(st, 1) / rate
    _, resolved_start, resolved_duration = resolve_sourceclip_window(aaf, node, start_sec, dur)
    return resolved_start, resolved_duration
