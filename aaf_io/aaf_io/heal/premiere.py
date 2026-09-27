"""
Repair Premiere-style linked audio: ``ImportDescriptor`` + locator only.

Nuendo (and other hosts) often warn about *unknown sample rate / bit depth* and
*invalid length* when the AAF carries almost no audio metadata. For typical
Premiere exports this shows up as many ``SourceMob`` objects whose descriptor is
only an ``ImportDescriptor`` with ``Locator`` URLs pointing at ``.aif`` / ``.wav``.

This module replaces those descriptors with a populated ``PCMDescriptor`` when
the linked file can be probed, and optionally rewrites ``SourceClip`` lengths on
the same mob to match the probed frame count (fixes many *invalid length* cases
where the AAF disagrees with the media header).

**Important:** PyAAF2 always calls ``save()`` when closing an ``r+`` file. To avoid
rewriting unrelated AAFs, use :func:`mend_premiere_import_audio_for_sound_pipeline`
which scans in ``r`` mode first and only opens ``r+`` when there is at least one
healable linked PCM file.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Type
from urllib.parse import unquote, urlparse

from aaf2.auid import AUID

from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
from aaf_io.diagnostic_log import BoundedDiagnosticLog, DiagnosticMessage

# AMA / MXF-style container label used by PyAAF2 for linked WAV/AIFF media.
_WAVE_AIFF_CONTAINER = AUID("3711d3cc-62d0-49d7-b0ae-c118101d1a16")


@dataclass
class PremiereImportScan:
    """Read-only scan of ``SourceMob`` + ``ImportDescriptor`` (sidecar audio)."""

    import_descriptor_mobs: int = 0
    no_locator: int = 0
    non_file_url: int = 0
    linked_file_missing: int = 0
    unsupported_extension: int = 0
    non_pcm_aiff: int = 0
    wav_probe_error: int = 0
    aiff_probe_error: int = 0
    healable_pcm_jobs: int = 0


@dataclass
class HealStats:
    """Counters from :func:`heal_import_descriptors_from_linked_media`."""

    pcm_replaced: int = 0
    skipped_missing_media: int = 0
    skipped_unsupported_extension: int = 0
    skipped_non_pcm_aiff: int = 0
    skipped_wav_probe_error: int = 0
    skipped_aiff_probe_error: int = 0
    errors: list[str] = field(default_factory=list)
    skipped_rplus_not_needed: bool = False


@dataclass
class PremiereAudioMendResult:
    """Scan + optional in-place heal (``r+`` only when ``scan.healable_pcm_jobs > 0``)."""

    scan: PremiereImportScan
    heal: HealStats


def file_url_to_path(url: str) -> Optional[Path]:
    """Turn ``file:///...`` / ``file://localhost/...`` into a local :class:`Path`."""
    try:
        parsed = urlparse(url)
        if parsed.scheme != "file":
            return None
        raw = unquote(parsed.path)
        if len(raw) >= 3 and raw[0] == "/" and raw[2] == ":":
            raw = raw[1:]
        return Path(raw)
    except Exception:
        return None


def _strip_premiere_untitled_path_prefix(path: Path) -> Optional[Path]:
    """
    Premiere often writes ``file:`` locators as ``.../Untitled/Users/...`` (sequence name),
    which resolves to a non-existent ``\\Untitled\\Users\\...`` on Windows. Drop that segment.

    Only applied when the path is clearly anchored (drive / root / UNC / ``\\\\?\\``), so a
    relative ``project/Untitled/...`` is left unchanged.
    """
    try:
        parts = path.parts
        if len(parts) < 3 or parts[1].lower() != "untitled":
            return None
        anchor = parts[0]
        root = anchor == os.sep or anchor == "/"
        drive = len(anchor) >= 2 and anchor[1] == ":"
        unc_or_device = anchor.startswith("\\\\")
        if not (root or drive or unc_or_device):
            return None
        return Path(anchor).joinpath(*parts[2:])
    except Exception:
        return None


def import_sidecar_path_candidates(path: Optional[Path]) -> list[Path]:
    """Ordered unique paths to probe for a linked sidecar (WAV/AIFF), including Premiere fixes."""
    if path is None:
        return []
    out: list[Path] = []
    seen: set[str] = set()

    def _push(p: Path) -> None:
        key = os.path.normcase(os.path.normpath(str(p)))
        if key not in seen:
            seen.add(key)
            out.append(p)

    _push(path)
    alt = _strip_premiere_untitled_path_prefix(path)
    if alt is not None:
        _push(alt)
    return out


def resolve_import_sidecar_media_path(
    path: Optional[Path], *, media_search_roots: Optional[tuple[Path, ...]] = None,
) -> Optional[Path]:
    """First existing filesystem path among :func:`import_sidecar_path_candidates`, or ``None``.

    Explicit roots take precedence over the legacy ``AAF_MEDIA_ROOTS`` environment
    fallback; an empty tuple disables that fallback. Resolve by basename under those
    roots for macOS Premiere exports where
    locators point to non-existent absolute paths (e.g. ``/Users/.../AAF Media Data/<uuid>.aif``).
    """
    for candidate in import_sidecar_path_candidates(path):
        if candidate.is_file():
            return candidate
    if path is None:
        return None
    roots = media_search_roots
    if roots is None:
        roots = tuple(Path(r.strip()) for r in os.environ.get("AAF_MEDIA_ROOTS", "").split(";") if r.strip())
    if not roots:
        return None
    base = path.name
    if not base:
        return None
    for r in roots:
        try:
            root = Path(r).expanduser()
            cand = root / base
            if cand.is_file():
                return cand
        except Exception:
            continue
    return None


def _first_locator_url(descriptor: Any) -> Optional[str]:
    try:
        locs = descriptor["Locator"]
        loc = next(iter(locs))
        return str(loc["URLString"].value)
    except Exception:
        return None


def _clone_locators(aaf: Any, src_desc: Any, dst_desc: Any) -> None:
    for loc in src_desc["Locator"]:
        n = aaf.create.NetworkLocator()
        n["URLString"].value = loc["URLString"].value
        dst_desc["Locator"].append(n)


def _apply_container_format(pcm: Any) -> None:
    try:
        pcm["ContainerFormat"].value = _WAVE_AIFF_CONTAINER
    except Exception:
        pass


def _fill_pcm_descriptor(
    pcm: Any,
    *,
    frames: int,
    sample_rate: int,
    channels: int,
    quantization_bits: int,
) -> None:
    sample_width = max(1, quantization_bits // 8)
    block_align = channels * sample_width
    pcm["Channels"].value = channels
    pcm["BlockAlign"].value = block_align
    pcm["SampleRate"].value = sample_rate
    pcm["AverageBPS"].value = sample_rate * channels * sample_width
    pcm["QuantizationBits"].value = quantization_bits
    pcm["AudioSamplingRate"].value = sample_rate
    pcm.length = int(frames)
    _apply_container_format(pcm)


def _probe_wav(path: Path) -> Optional[dict[str, Any]]:
    try:
        from aaf2.audio import WaveReader
    except Exception:
        return None
    try:
        with open(path, "rb") as handle:
            w = WaveReader(handle)
            frames = w.getnframes()
            rate = int(w.getframerate())
            channels = w.getnchannels()
            bits = w.getsampwidth() * 8
        return {"frames": frames, "rate": rate, "channels": channels, "bits": bits}
    except Exception:
        return None


def _probe_aiff(path: Path) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Returns ``(info, reason)`` where ``reason`` is ``non_pcm_aiff``, ``aiff_error``,
    or ``None`` when ``info`` is set.
    """
    try:
        import aifc
    except Exception:
        return None, "aiff_error"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            handle = aifc.open(str(path), "r")
        try:
            comptype = handle.getcomptype()
            if comptype not in ("NONE", "not compressed", ""):
                return None, "non_pcm_aiff"
            frames = handle.getnframes()
            rate = int(handle.getframerate())
            channels = handle.getnchannels()
            bits = handle.getsampwidth() * 8
        finally:
            handle.close()
        return (
            {"frames": frames, "rate": rate, "channels": channels, "bits": bits},
            None,
        )
    except Exception:
        return None, "aiff_error"


def _probe_sidecar_media(path: Path) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Returns ``(info, skip_reason)``.

    ``skip_reason`` is one of ``unsupported_ext``, ``non_pcm_aiff``, ``wav_error``,
    ``aiff_error``, or ``None`` when ``info`` is populated.
    """
    ext = path.suffix.lower()
    if ext in (".wav", ".wave", ".bwf"):
        info = _probe_wav(path)
        if info is None:
            return None, "wav_error"
        return info, None
    if ext in (".aif", ".aiff", ".aifc"):
        info, reason = _probe_aiff(path)
        if info is None:
            return None, reason or "aiff_error"
        return info, None
    return None, "unsupported_ext"


def _sync_source_clip_lengths(mob: Any, frames: int) -> None:
    for slot in mob.slots:
        seg = slot.segment
        if seg is None:
            continue
        if seg.__class__.__name__ == "SourceClip":
            seg.length = int(frames)


def _mob_label(mob: Any) -> str:
    try:
        n = mob.name
        if n:
            return str(n)
    except Exception:
        pass
    return str(getattr(mob, "mob_id", "?"))


def _scan_one_import_descriptor(
    mob: Any,
    desc: Any,
    scan: PremiereImportScan,
    *,
    samples: list[str],
    max_samples: int,
    suppress_missing_file_samples: bool = False,
    media_search_roots: Optional[tuple[Path, ...]] = None,
) -> None:
    scan.import_descriptor_mobs += 1
    url = _first_locator_url(desc)
    if not url:
        scan.no_locator += 1
        if len(samples) < max_samples:
            samples.append(f"[ImportDescriptor] {_mob_label(mob)}: нет URL в Locator.")
        return
    path = file_url_to_path(url)
    if path is None:
        scan.non_file_url += 1
        if len(samples) < max_samples:
            udisp = url if len(url) <= 160 else url[:160] + "…"
            samples.append(f"[ImportDescriptor] {_mob_label(mob)}: не file:// URL ({udisp})")
        return
    resolved = resolve_import_sidecar_media_path(path, media_search_roots=media_search_roots)
    if resolved is None:
        scan.linked_file_missing += 1
        if (not suppress_missing_file_samples) and len(samples) < max_samples:
            samples.append(f"[ImportDescriptor] {_mob_label(mob)}: файл не найден: {path}")
        return
    info, reason = _probe_sidecar_media(resolved)
    if reason == "unsupported_ext":
        scan.unsupported_extension += 1
        if len(samples) < max_samples:
            samples.append(
                f"[ImportDescriptor] {_mob_label(mob)}: расширение не для PCM-зонда "
                f"({resolved.suffix}): {resolved}"
            )
        return
    if reason == "non_pcm_aiff":
        scan.non_pcm_aiff += 1
        if len(samples) < max_samples:
            samples.append(f"[ImportDescriptor] {_mob_label(mob)}: AIFC не PCM: {resolved}")
        return
    if reason == "wav_error":
        scan.wav_probe_error += 1
        if len(samples) < max_samples:
            samples.append(f"[ImportDescriptor] {_mob_label(mob)}: не читается как WAV: {resolved}")
        return
    if reason == "aiff_error":
        scan.aiff_probe_error += 1
        if len(samples) < max_samples:
            samples.append(f"[ImportDescriptor] {_mob_label(mob)}: не читается как AIFF: {resolved}")
        return
    if info:
        scan.healable_pcm_jobs += 1


def scan_premiere_import_linked_audio(
    aaf_path: Path,
    *,
    log_callback: Optional[Callable[[str], None]] = None,
    max_samples: int = 40,
    suppress_missing_file_samples: bool = False,
    cancel_event: Any = None,
    cancel_exc_type: Optional[Type[BaseException]] = None,
    media_search_roots: Optional[tuple[Path, ...]] = None,
) -> tuple[PremiereImportScan, list[str]]:
    """
    Read-only pass: count ``ImportDescriptor`` sidecars and how many can be probed as PCM.

    Returns ``(scan, sample_lines)``. Does not modify the AAF.
    """
    scan = PremiereImportScan()
    samples: list[str] = []

    def _check_cancel() -> None:
        if cancel_event is not None and cancel_exc_type is not None and cancel_event.is_set():
            raise cancel_exc_type()

    with open_aaf_lenient(Path(aaf_path), "r") as aaf:
        for mob in aaf.content.mobs:
            _check_cancel()
            if mob.__class__.__name__ != "SourceMob":
                continue
            desc = getattr(mob, "descriptor", None)
            if desc is None or desc.__class__.__name__ != "ImportDescriptor":
                continue
            _scan_one_import_descriptor(
                mob,
                desc,
                scan,
                samples=samples,
                max_samples=max_samples,
                suppress_missing_file_samples=bool(suppress_missing_file_samples),
                media_search_roots=media_search_roots,
            )

    lines: list[str] = []
    if scan.import_descriptor_mobs == 0:
        lines.append("Скан звука (Premiere-style ImportDescriptor): не найдено — шаг пропущен.")
    else:
        lines.append(
            "Скан звука (Premiere-style ImportDescriptor, только внешние клипы): "
            f"мобов с ImportDescriptor={scan.import_descriptor_mobs}; "
            f"готово к починке PCM (файл на диске + .wav/.aif PCM)={scan.healable_pcm_jobs}."
        )
        lines.append(
            "  Детали: нет Locator="
            f"{scan.no_locator}; не file://={scan.non_file_url}; файл не найден="
            f"{scan.linked_file_missing}; расширение не WAV/AIFF="
            f"{scan.unsupported_extension}; AIFC не PCM={scan.non_pcm_aiff}; "
            f"ошибка чтения WAV={scan.wav_probe_error}; ошибка чтения AIFF={scan.aiff_probe_error}."
        )
        if samples:
            lines.append("  Примеры (первые строки):")
            lines.extend(f"    {s}" for s in samples)
    if log_callback is not None:
        for ln in lines:
            log_callback(ln)
    return scan, samples


def format_heal_stats_lines(
    stats: HealStats,
    *,
    title: str = "Починка PCM (ImportDescriptor→PCMDescriptor)",
    include_error_samples: bool = True,
) -> list[str]:
    """Human-readable summary lines for GUI logs."""
    lines = [title + ":"]
    lines.append(f"  заменено на PCMDescriptor: {stats.pcm_replaced}")
    if stats.skipped_rplus_not_needed:
        lines.append("  перезапись AAF (r+) не выполнялась — нечего чинить по диску.")
    lines.append(
        "  пропуски при записи: нет файла/URL="
        f"{stats.skipped_missing_media}; не .wav/.aif="
        f"{stats.skipped_unsupported_extension}; AIFC не PCM="
        f"{stats.skipped_non_pcm_aiff}; ошибка WAV="
        f"{stats.skipped_wav_probe_error}; ошибка AIFF={stats.skipped_aiff_probe_error}."
    )
    if stats.errors:
        lines.append(f"  ошибки PyAAF2/дескриптора: {len(stats.errors)}")
        if include_error_samples:
            cap = 10
            for e in stats.errors[:cap]:
                lines.append(f"    {e}")
            if len(stats.errors) > cap:
                lines.append(f"    … скрыто {len(stats.errors) - cap} сообщений.")
    return lines


def heal_import_descriptors_from_linked_media(
    aaf: Any,
    *,
    fix_source_clip_lengths: bool = True,
    on_progress: Optional[Callable[[int, int], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    max_log_lines: int = 10,
    media_search_roots: Optional[tuple[Path, ...]] = None,
) -> HealStats:
    """
    Walk all mobs; for each ``SourceMob`` whose descriptor is ``ImportDescriptor``,
    probe linked ``.wav`` / ``.aif`` / ``.aiff`` / ``.aifc`` and replace the descriptor
    with ``PCMDescriptor`` carrying rate / bit depth / channel / length metadata.

    Operates on an already-open ``aaf2`` file (typically ``r+``). The file is saved
    when the surrounding ``aaf2.open`` / :func:`aaf_io.compat.pyaaf2_lenient.open_aaf_lenient`
    context exits normally.
    """
    stats = HealStats()
    mobs = list(aaf.content.mobs)
    total = len(mobs)
    # This public helper also runs outside the pipeline; own its warning budget.
    diagnostics = BoundedDiagnosticLog(log_callback or (lambda _: None), limit=max_log_lines)

    def _log(msg: str) -> None:
        diagnostics(DiagnosticMessage(msg, category="Восстановление PCM"))

    with diagnostics:
        for idx, mob in enumerate(mobs):
            if on_progress is not None:
                on_progress(idx + 1, total)
            if mob.__class__.__name__ != "SourceMob":
                continue
            desc = getattr(mob, "descriptor", None)
            if desc is None or desc.__class__.__name__ != "ImportDescriptor":
                continue
            url = _first_locator_url(desc)
            if not url:
                stats.skipped_missing_media += 1
                _log(f"[heal] {_mob_label(mob)}: нет URL в Locator, пропуск.")
                continue
            path = file_url_to_path(url)
            resolved = resolve_import_sidecar_media_path(path, media_search_roots=media_search_roots)
            if path is None or resolved is None:
                stats.skipped_missing_media += 1
                _log(f"[heal] {_mob_label(mob)}: нет файла по локатору, пропуск.")
                continue
            info, reason = _probe_sidecar_media(resolved)
            if reason == "unsupported_ext":
                stats.skipped_unsupported_extension += 1
                _log(f"[heal] {_mob_label(mob)}: не WAV/AIFF ({resolved.suffix}), пропуск.")
                continue
            if reason == "non_pcm_aiff":
                stats.skipped_non_pcm_aiff += 1
                _log(f"[heal] {_mob_label(mob)}: AIFC не PCM, пропуск: {resolved}")
                continue
            if reason == "wav_error":
                stats.skipped_wav_probe_error += 1
                _log(f"[heal] {_mob_label(mob)}: ошибка чтения WAV: {resolved}")
                continue
            if reason == "aiff_error":
                stats.skipped_aiff_probe_error += 1
                _log(f"[heal] {_mob_label(mob)}: ошибка чтения AIFF: {resolved}")
                continue
            if not info:
                detail = f"unexpected empty probe for {resolved}"
                stats.errors.append(detail)
                _log(f"[heal] Ошибка: {detail}")
                continue
            try:
                pcm = aaf.create.PCMDescriptor()
                _clone_locators(aaf, desc, pcm)
                _fill_pcm_descriptor(
                    pcm,
                    frames=int(info["frames"]),
                    sample_rate=int(info["rate"]),
                    channels=int(info["channels"]),
                    quantization_bits=int(info["bits"]),
                )
                mob.descriptor = pcm
                if fix_source_clip_lengths:
                    _sync_source_clip_lengths(mob, int(info["frames"]))
                stats.pcm_replaced += 1
            except Exception as exc:
                stats.errors.append(f"{mob.mob_id}: {exc}")
                _log(f"[heal] Ошибка {_mob_label(mob)}: {exc}")
    return stats


def mend_premiere_import_audio_for_sound_pipeline(
    work_aaf: Path,
    *,
    fix_source_clip_lengths: bool = True,
    log_callback: Optional[Callable[[str], None]] = None,
    emit_scan_summary: bool = True,
    on_progress: Optional[Callable[[int, int], None]] = None,
    max_scan_samples: int = 40,
    suppress_missing_file_samples: bool = False,
    cancel_event: Any = None,
    cancel_exc_type: Optional[Type[BaseException]] = None,
    media_search_roots: Optional[tuple[Path, ...]] = None,
) -> PremiereAudioMendResult:
    """
    1) Scan in read-only mode and log explainers / samples.
    2) If at least one linked PCM file is readable, open ``r+`` and run
       :func:`heal_import_descriptors_from_linked_media` (triggers PyAAF2 save on close).

    If nothing is healable, the AAF is **not** opened for write (avoids an unnecessary
    full-file save on unrelated projects).
    """
    scan_log = log_callback if bool(emit_scan_summary) else None
    scan, _samples = scan_premiere_import_linked_audio(
        work_aaf,
        log_callback=scan_log,
        max_samples=max_scan_samples,
        suppress_missing_file_samples=bool(suppress_missing_file_samples),
        cancel_event=cancel_event,
        cancel_exc_type=cancel_exc_type,
        media_search_roots=media_search_roots,
    )
    if scan.healable_pcm_jobs <= 0:
        heal = HealStats()
        heal.skipped_rplus_not_needed = True
        return PremiereAudioMendResult(scan=scan, heal=heal)

    if log_callback is not None:
        log_callback(
            f"Запись PCM-метаданных в AAF (PyAAF2, r+), задач: {scan.healable_pcm_jobs} …"
        )

    with open_aaf_lenient(Path(work_aaf), "r+") as aaf:
        heal = heal_import_descriptors_from_linked_media(
            aaf,
            fix_source_clip_lengths=fix_source_clip_lengths,
            media_search_roots=media_search_roots,
            on_progress=on_progress,
            log_callback=log_callback,
        )
    if log_callback is not None:
        for ln in format_heal_stats_lines(heal, include_error_samples=False):
            log_callback(ln)
    return PremiereAudioMendResult(scan=scan, heal=heal)


def heal_premiere_import_audio_inplace(
    work_copy: Path,
    *,
    fix_source_clip_lengths: bool = True,
    on_progress: Optional[Callable[[int, int], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    cancel_event: Any = None,
    cancel_exc_type: Optional[Type[BaseException]] = None,
    media_search_roots: Optional[tuple[Path, ...]] = None,
) -> HealStats:
    """
    Обёртка над :func:`mend_premiere_import_audio_for_sound_pipeline` (возвращает только
    статистику записи). Не открывает ``r+``, если по скану нечего чинить.
    """
    return mend_premiere_import_audio_for_sound_pipeline(
        work_copy,
        fix_source_clip_lengths=fix_source_clip_lengths,
        on_progress=on_progress,
        log_callback=log_callback,
        cancel_event=cancel_event,
        cancel_exc_type=cancel_exc_type,
        media_search_roots=media_search_roots,
    ).heal
