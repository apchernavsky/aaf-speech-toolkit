from __future__ import annotations

from aaf_io.diagnostic_log import DiagnosticMessage

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
from aaf_io.heal.davinci import ensure_wav_proxy_for_media
from aaf_io.sdk_xml import SourceClipOccurrence

from .config import FilterConfig
from .exceptions import SpeechFilterCancelled
from .media_resolve import (
    _path_is_file_fast,
    _resolve_wave_path_for_sourceclip,
)
from .speech_vad import quiet_clip_removal_decision
from .timeline_sourceclips import iter_timeline_sourceclip_occurrences, total_timeline_sourceclips
from .media_resolve import UnsupportedMediaMapping
from .timeline_timing import sourceclip_audio_timing
from .timeline_walk import _aaf_int_length, _aaf_int_start


RemovalKey = tuple[str, int, int, int]
_COOP_YIELD_INTERVAL_SEC = 5.0


def sourceclip_sdk_key_tuple(sourceclip: Any) -> Optional[RemovalKey]:
    """
    Source content identity shared by PyAAF2 and SDK XML; not an occurrence address.
    """
    try:
        sid = sourceclip["SourceID"].value
        src_str = str(sid).strip()
    except Exception:
        try:
            src_str = str(getattr(sourceclip, "source_id", None) or "").strip()
        except Exception:
            return None
    if not src_str:
        return None
    length = _aaf_int_length(sourceclip)
    start = _aaf_int_start(sourceclip)
    track = None
    try:
        track = int(sourceclip["SourceTrackID"].value)
    except Exception:
        track = None
    if track is None:
        try:
            track = int(sourceclip["SourceMobSlotID"].value)
        except Exception:
            track = None
    if track is None:
        try:
            track = int(sourceclip["SourceSlotID"].value)
        except Exception:
            track = None
    if track is None:
        try:
            track = int(getattr(sourceclip, "source_track_id", 0) or 0)
        except Exception:
            track = 0
    return (src_str.strip(), length, start, track)


def sourceclip_occurrence_key(occurrence) -> SourceClipOccurrence:
    key = sourceclip_sdk_key_tuple(occurrence.sourceclip)
    if key is None or not occurrence.composition_id:
        raise ValueError('Timeline SourceClip has no stable occurrence identity')
    return SourceClipOccurrence(occurrence.composition_id, occurrence.slot_id,
                                occurrence.path, key)


def collect_timeline_sourceclip_removals_for_sdk_xml(
    aaf_path: Path,
    cfg: FilterConfig,
    runtime_essence_paths: Optional[Dict[Any, Path]] = None,
    work_dir: Optional[Path] = None,
    cancel_event: Optional[threading.Event] = None,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    removal_stats: Optional[dict[str, int]] = None,
    max_unavailable_samples: int = 30,
) -> set[SourceClipOccurrence]:
    """
    Read-only pass returning guarded occurrence addresses for SourceClips to replace.

    The actual AAF rewrite happens later in the SDK XML layer. Keeping this pass here
    avoids importing the mutable PyAAF2 filter for SDK-based pipelines.
    """
    removals: set[SourceClipOccurrence] = set()
    removed_decision_n = 0
    removed_quiet_n = 0
    clips_no_wav_path = 0
    clips_wav_missing_on_disk = 0
    unavailable_samples: list[str] = []
    last_yield_t = time.monotonic()

    def check_cancel() -> None:
        nonlocal last_yield_t
        if cancel_event is not None and cancel_event.is_set():
            raise SpeechFilterCancelled()
        now = time.monotonic()
        if now - last_yield_t >= _COOP_YIELD_INTERVAL_SEC:
            last_yield_t = now
            time.sleep(0)

    def note_clip_done(total_sec: float, state: dict[str, float]) -> None:
        if progress_callback is None:
            return
        state["processed"] = min(total_sec, state["processed"] + 1.0)
        progress_callback(state["processed"], total_sec)

    with open_aaf_lenient(Path(aaf_path), "r") as aaf:
        if not list(aaf.content.compositionmobs()):
            raise RuntimeError("CompositionMob not found")

        if log_callback is not None:
            try:
                log_callback("AAF SDK XML: старт анализа таймлайна (тихие клипы)…")
            except Exception:
                pass

        total_sec = float(max(1, int(total_timeline_sourceclips(aaf))))
        progress_state = {"processed": 0.0}

        for occurrence in iter_timeline_sourceclip_occurrences(aaf):
            edit_rate, gain_mul, node = (occurrence.edit_rate, occurrence.gain_multiplier,
                                         occurrence.sourceclip)
            check_cancel()
            wav_path = _resolve_wave_path_for_sourceclip(aaf, node, runtime_essence_paths, media_search_roots=cfg.media_search_roots, edit_rate=edit_rate)
            if wav_path is not None and _path_is_file_fast(Path(wav_path)):
                wav_path = (
                    ensure_wav_proxy_for_media(
                        Path(wav_path),
                        work_dir=work_dir,
                        cancel_check=check_cancel,
                        log_callback=log_callback,
                    )
                    or wav_path
                )

            try:
                start_sec, duration_sec = sourceclip_audio_timing(
                    aaf, edit_rate, node, wav_path, runtime_essence_paths
                )
            except UnsupportedMediaMapping as exc:
                if log_callback is not None:
                    log_callback(DiagnosticMessage(
                        f"Media window unresolved; clip retained: {exc}",
                        category="Окно аудио: клип сохранён",
                    ))
                note_clip_done(total_sec, progress_state)
                continue
            if not wav_path:
                clips_no_wav_path += 1
                if log_callback is not None and len(unavailable_samples) < max_unavailable_samples:
                    unavailable_samples.append(
                        "нет пути к WAV (внешний файл не сопоставлен / нет embedded)"
                    )
                note_clip_done(total_sec, progress_state)
                continue
            if not wav_path.exists():
                clips_wav_missing_on_disk += 1
                if log_callback is not None and len(unavailable_samples) < max_unavailable_samples:
                    unavailable_samples.append(
                        f"WAV по пути отсутствует на диске: {wav_path}"
                    )
                note_clip_done(total_sec, progress_state)
                continue

            check_cancel()

            remove, removed_reason = quiet_clip_removal_decision(
                wav_path,
                start_sec,
                duration_sec,
                remove_quiet_clips=bool(cfg.remove_quiet_clips),
                quiet_peak_dbfs=float(cfg.quiet_peak_dbfs),
                gain_multiplier=gain_mul,
                cancel_check=check_cancel,
                log_callback=log_callback,
            )

            if remove:
                removals.add(sourceclip_occurrence_key(occurrence))
                removed_decision_n += 1
                if removed_reason == "quiet":
                    removed_quiet_n += 1

            note_clip_done(total_sec, progress_state)

        if progress_callback is not None and total_sec > 0:
            progress_callback(total_sec, total_sec)

    if log_callback is not None and (clips_no_wav_path or clips_wav_missing_on_disk):
        log_callback(
            "Таймлайн (анализ для удалений): часть SourceClip пропущена — нет доступного файла для "
            "YAMNet/VAD (ни путь из AAF/Locator, ни WAV в essence/ после выгрузки embedded). "
            f"Нет пути к медиа: {clips_no_wav_path}; путь в AAF есть, файла на диске нет: {clips_wav_missing_on_disk}."
        )
        if unavailable_samples:
            log_callback("  Примеры:")
            for line in unavailable_samples:
                log_callback(f"    {line}")

    if log_callback is not None:
        try:
            if removed_quiet_n > 0:
                log_callback(f"Удалено {removed_quiet_n} клипа (удаление тихих клипов).")
        except Exception:
            pass

    if removal_stats is not None:
        try:
            removal_stats["removed_decisions"] = int(removed_decision_n)
            removal_stats["removed_quiet"] = int(removed_quiet_n)
            removal_stats["unique_sdk_keys"] = int(len(removals))
        except Exception:
            pass

    return removals

__all__ = [
    "RemovalKey", "collect_timeline_sourceclip_removals_for_sdk_xml",
    "sourceclip_sdk_key_tuple", "sourceclip_occurrence_key",
]
