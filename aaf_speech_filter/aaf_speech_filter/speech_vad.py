from __future__ import annotations

from aaf_io.diagnostic_log import DiagnosticMessage

import audioop
import math
import struct
import warnings
import wave
from fractions import Fraction
from aaf_io.audio_format import pcm_container_kind
from pathlib import Path
from typing import Callable, Optional

from .thresholds import validate_quiet_peak_dbfs


def _normalized_aifc_comptype(comptype: object) -> str:
    if isinstance(comptype, bytes):
        return comptype.decode("ascii", errors="replace").strip().lower()
    return str(comptype).strip().lower()


class IncompleteAudioWindow(ValueError):
    """The decoder cannot prove complete coverage of the requested interval."""


def _pcm_frame_window(start_seconds, duration_seconds, sample_rate, total_frames):
    """Resolve a half-open window without silently clipping it to available media.

    Exact rational inputs cover every intersecting source frame (floor/ceil).
    Legacy float inputs denote nearest source-frame boundaries, ties to even.
    Callers needing sub-frame precision must pass Fraction values.
    """
    start = Fraction(start_seconds)
    duration = Fraction(duration_seconds)
    if sample_rate <= 0 or start < 0 or duration <= 0:
        raise IncompleteAudioWindow("invalid or empty requested audio window")
    if isinstance(start_seconds, float) or isinstance(duration_seconds, float):
        first = round(float(start_seconds) * sample_rate)
        end = round((float(start_seconds) + float(duration_seconds)) * sample_rate)
    else:
        first = math.floor(start * sample_rate)
        end = math.ceil((start + duration) * sample_rate)
    if first < 0 or end > total_frames or end <= first:
        raise IncompleteAudioWindow(
            f"requested frames [{first}, {end}) exceed available [0, {total_frames}) or are empty"
        )
    return first, end


def _require_complete_pcm(raw, frame_count, channels, sample_width):
    expected = frame_count * channels * sample_width
    if channels <= 0 or len(raw) != expected:
        raise IncompleteAudioWindow(f"expected {expected} PCM bytes, decoded {len(raw)}")


def _aiff_declared_sample_width(path):
    """Read the COMM width because aifc overwrites it for sowt decoding."""
    with Path(path).open('rb') as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b'FORM':
            raise ValueError("invalid AIFF FORM header")
        form_end = 8 + int.from_bytes(header[4:8], 'big')
        while stream.tell() + 8 <= form_end:
            header = stream.read(8)
            if len(header) != 8:
                break
            size = int.from_bytes(header[4:8], 'big')
            if header[:4] == b'COMM':
                fields = stream.read(min(size, 8))
                if len(fields) != 8:
                    break
                bits = int.from_bytes(fields[6:8], 'big')
                if bits not in (8, 16, 24, 32):
                    raise ValueError(f"unsupported AIFF sample size: {bits}")
                return bits // 8
            stream.seek(size + size % 2, 1)
    raise ValueError("AIFF COMM sample size is unavailable")


def _read_wav_segment_raw_16le(
    wav_path: Path,
    start_seconds: float | Fraction,
    duration_seconds: float | Fraction,
) -> tuple[bytes, int, int]:
    """
    Фрагмент PCM в 16-bit little-endian (mono — подряд сэмплы, стерео — interleaved), число каналов, SR.
    Вход: 8/16/24/32-bit integer PCM в WAV; иное приводится через ``audioop.lin2lin``.
    """
    with wave.open(str(wav_path), "rb") as wf:
        ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        nframes = wf.getnframes()

        if sampwidth not in (1, 2, 3, 4):
            raise ValueError(
                f"Unsupported integer PCM WAV (sampwidth={sampwidth}); need 8/16/24/32-bit."
            )

        start_frame, end_frame = _pcm_frame_window(start_seconds, duration_seconds, sr, nframes)
        wf.setpos(start_frame)
        raw = wf.readframes(end_frame - start_frame)
        _require_complete_pcm(raw, end_frame - start_frame, ch, sampwidth)

    if sampwidth == 1:
        raw = audioop.bias(raw, 1, -128)
    if sampwidth != 2:
        raw = audioop.lin2lin(raw, sampwidth, 2)

    return raw, ch, sr


def _read_aiff_segment_raw_16le(
    aiff_path: Path,
    start_seconds: float | Fraction,
    duration_seconds: float | Fraction,
) -> tuple[bytes, int, int]:
    """
    AIFF/AIFC PCM (несжатый): фрагмент как 16-bit **little-endian** interleaved, каналы, SR.
    """
    import aifc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            wf = aifc.open(str(aiff_path), "r")
        except aifc.Error as exc:
            raise ValueError(f"AIFF decoder could not open media: {exc}") from exc
    try:
        compression = _normalized_aifc_comptype(wf.getcomptype())
        if compression not in ('none', 'not compressed', '', 'sowt'):
            raise ValueError(f"unsupported AIFC compression: {compression!r}")
        declared_width = _aiff_declared_sample_width(aiff_path)
        if compression == 'sowt' and declared_width != 2:
            raise ValueError("aifc sowt decoding supports only 16-bit PCM")
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        if sw not in (1, 2, 3, 4):
            raise ValueError(f"unsupported AIFF sampwidth={sw}")
        nframes = wf.getnframes()
        start_frame, end_frame = _pcm_frame_window(start_seconds, duration_seconds, sr, nframes)
        wf.setpos(start_frame)
        raw = wf.readframes(end_frame - start_frame)
        _require_complete_pcm(raw, end_frame - start_frame, ch, sw)
    finally:
        wf.close()

    # aifc.readframes returns big-endian PCM, including normalized sowt data.
    raw = audioop.byteswap(raw, sw)
    if sw != 2:
        raw = audioop.lin2lin(raw, sw, 2)
    return raw, ch, sr


def _read_media_segment_raw_16le(
    media_path: Path,
    start_seconds: float | Fraction,
    duration_seconds: float | Fraction,
) -> tuple[bytes, int, int]:
    if pcm_container_kind(media_path) == "aiff":
        return _read_aiff_segment_raw_16le(media_path, start_seconds, duration_seconds)
    return _read_wav_segment_raw_16le(media_path, start_seconds, duration_seconds)


def downmix_pcm16le(raw: bytes, channels: int) -> bytes:
    """Average all channels within each complete PCM frame, preserving frame count."""
    if channels <= 0 or len(raw) % (2 * channels):
        raise ValueError("PCM data must contain complete frames with a positive channel count")
    if channels == 1:
        return raw
    if channels == 2:
        return audioop.tomono(raw, 2, 0.5, 0.5)
    frame_struct = struct.Struct("<" + "h" * channels)
    mono = bytearray(len(raw) // channels)
    for index, frame in enumerate(frame_struct.iter_unpack(raw)):
        struct.pack_into("<h", mono, 2 * index, sum(frame) // channels)
    return bytes(mono)


def read_media_segment_pcm16_mono(
    media_path: Path,
    start_seconds: float | Fraction,
    duration_seconds: float | Fraction,
) -> tuple[bytes, int]:
    """Return a complete PCM16 LE mono window and rate, or raise ValueError.

    Fraction windows cover intersecting frames; float boundaries round to the
    nearest source frame (ties to even). Requests outside media are rejected.
    """
    raw, ch, sr = _read_media_segment_raw_16le(media_path, start_seconds, duration_seconds)
    return downmix_pcm16le(raw, ch), sr


def _read_wav_segment_pcm16_mono(
    wav_path: Path,
    start_seconds: float | Fraction,
    duration_seconds: float | Fraction,
) -> tuple[bytes, int]:
    return read_media_segment_pcm16_mono(wav_path, start_seconds, duration_seconds)


def _linear_abs_peak_limit(peak_dbfs_threshold: float) -> float:
    """
    Максимальное |int16| такое, что 20*log10(mx/32768) <= peak_dbfs_threshold.
    Сравнение float с целочисленным пиком даёт тот же результат, что и старый расчёт через log10.
    """
    peak_dbfs_threshold = validate_quiet_peak_dbfs(peak_dbfs_threshold)
    if peak_dbfs_threshold <= -160.0:
        return 0.0
    return min(32768.0, 32768.0 * (10.0 ** (peak_dbfs_threshold / 20.0)))


def peak_dbfs_16le_mono(pcm16_mono: bytes) -> float:
    """
    Sample peak of 16-bit little-endian mono PCM relative to digital full scale, in dBFS (≤ 0).
    """
    if not pcm16_mono or len(pcm16_mono) < 2:
        return -120.0
    mx = audioop.max(pcm16_mono, 2)
    if mx <= 0:
        return -120.0
    return 20.0 * math.log10(mx / 32768.0)


def _mono_segment_is_quiet_or_empty(raw: bytes, limit: float) -> bool:
    if not raw:
        return True
    return float(audioop.max(raw, 2)) <= limit


def _stereo_interleaved_segment_is_quiet(raw: bytes, limit: float) -> bool:
    """
    Пик после downmix (l+r)//2 — совпадает с audioop.tomono(..., 0.5, 0.5) для 16-bit LE.
    Ранний выход без отдельного буфера mono.
    """
    mv = memoryview(raw)
    n = len(mv)
    i = 0
    while i + 4 <= n:
        l = int.from_bytes(mv[i : i + 2], "little", signed=True)
        r = int.from_bytes(mv[i + 2 : i + 4], "little", signed=True)
        if abs((l + r) // 2) > limit:
            return False
        i += 4
    return True


def _multichannel_interleaved_segment_is_quiet(raw: bytes, channels: int, limit: float) -> bool:
    """Среднее по каналам в кадре (типичный downmix); ранний выход."""
    if channels < 1:
        return True
    frame_b = 2 * channels
    mv = memoryview(raw)
    n = len(mv)
    i = 0
    while i + frame_b <= n:
        s = 0
        base = i
        for c in range(channels):
            s += int.from_bytes(mv[base + 2 * c : base + 2 * c + 2], "little", signed=True)
        if abs(s // channels) > limit:
            return False
        i += frame_b
    return True


def measure_segment_peak_dbfs(
    wav_path: Path,
    start_seconds: float | Fraction,
    duration_seconds: float | Fraction,
    *,
    sample_dur: float = 2.0,
) -> float:
    """
    Quick peak measurement for diagnostic logging only.

    Reads up to *sample_dur* seconds from *start_seconds* and returns the sample
    peak in dBFS (≤ 0), or -120.0 if the file is unreadable / the region is empty.
    Channels > 1 are downmixed via (L+R+…)/N before peak measurement.
    """
    try:
        dur = min(float(duration_seconds), float(sample_dur))
        if dur < 0.05:
            dur = min(float(duration_seconds), 0.5)
        if dur <= 0:
            return -120.0
        raw, ch, _sr = _read_media_segment_raw_16le(wav_path, start_seconds, dur)
        if not raw:
            return -120.0
        mono = downmix_pcm16le(raw, ch)
        return peak_dbfs_16le_mono(mono)
    except Exception:
        return -120.0


def wav_file_info(wav_path: Path) -> tuple[float, float, int, int]:
    """
    Return (total_duration_sec, peak_at_start_dbfs, sample_rate, channels) for diagnostic logging.
    peak_at_start_dbfs — peak of the first 1 second (from position 0).
    Returns (-1.0, -120.0, 0, 0) on error.
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            nframes = wf.getnframes()
            total_sec = float(nframes) / sr if sr > 0 else 0.0
            # Read first second (or all if shorter)
            to_read = min(nframes, sr)
            wf.setpos(0)
            raw = wf.readframes(to_read)
        if sw == 1:
            raw = audioop.bias(raw, 1, -128)
        if sw != 2:
            raw = audioop.lin2lin(raw, sw, 2)
        if not raw:
            return total_sec, -120.0, sr, ch
        mono = downmix_pcm16le(raw, ch)
        return total_sec, peak_dbfs_16le_mono(mono), sr, ch
    except Exception:
        return -1.0, -120.0, 0, 0


def is_quiet_clip(
    wav_path: Path,
    start_seconds: float | Fraction,
    duration_seconds: float | Fraction,
    peak_dbfs_threshold: float,
    *,
    cancel_check: Optional[Callable[[], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    True, если **цифровой пик сэмпла** (16-bit PCM после decode) не выше порога в dBFS.

    Стерео/мульти: сравнение с порогом по кадру после downmix ((L+R)/2 и аналог для N каналов).
    Это sample peak по выборкам, не inter-sample True Peak (BS.1770) из Nuendo — там пик может
    читаться на доли dB выше на тех же данных.
    """
    limit = _linear_abs_peak_limit(peak_dbfs_threshold)

    if duration_seconds <= 0:
        return False

    def _segment_is_quiet(seg_start: float, seg_dur: float) -> Optional[bool]:
        if seg_dur <= 0:
            return None
        if cancel_check is not None:
            cancel_check()
        try:
            raw, ch, _sr = _read_media_segment_raw_16le(wav_path, seg_start, seg_dur)
        except (OSError, EOFError, ValueError, wave.Error, audioop.error) as exc:
            if log_callback is not None:
                log_callback(DiagnosticMessage(
                    f"Audio analysis incomplete; clip retained: {exc}",
                    category="Неполный анализ аудио: клип сохранён",
                ))
            return None
        if not raw or ch <= 0 or len(raw) % (2 * ch):
            # Can't measure a peak on an empty decode window → unknown.
            return None
        if cancel_check is not None:
            cancel_check()
        if ch == 1:
            return _mono_segment_is_quiet_or_empty(raw, limit)
        if ch == 2:
            return _stereo_interleaved_segment_is_quiet(raw, limit)
        return _multichannel_interleaved_segment_is_quiet(raw, ch, limit)

    # Fast path: probe a few short windows (start/middle/end). If any window is loud, clip is not quiet.
    # If all probed windows look quiet, fall back to the full-window check for safety.
    # Ultra-fast precheck: a tiny slice at the start. If it's already loud, we can skip other probes.
    pre_win = min(0.12, float(duration_seconds))
    if pre_win >= 0.05:
        r0 = _segment_is_quiet(float(start_seconds), float(pre_win))
        if r0 is False:
            return False

    quick_win = min(0.35, max(0.05, duration_seconds, 0.05))
    if duration_seconds > 0.80:
        quick_win = min(0.35, max(0.05, duration_seconds * 0.25))
    else:
        quick_win = duration_seconds

    if duration_seconds > quick_win and duration_seconds >= 1.0:
        a = float(start_seconds)
        b = float(start_seconds) + max(0.0, float(duration_seconds) * 0.5 - float(quick_win) * 0.5)
        c = float(start_seconds) + max(0.0, float(duration_seconds) - float(quick_win))
        for s in (a, b, c):
            r = _segment_is_quiet(s, quick_win)
            if r is False:
                return False

    # Slow, "careful" check: full segment as before.
    r_full = _segment_is_quiet(start_seconds, duration_seconds)
    if r_full is False:
        return False

    # The decision is about the edited timeline segment, not the whole source take.
    # A Foley take may contain loud material elsewhere; that must not keep a silent
    # slice on the timeline from being removed.
    return bool(r_full) if r_full is not None else False


def quiet_clip_removal_decision(
    media_path: Path,
    start_seconds: float | Fraction,
    duration_seconds: float | Fraction,
    *,
    remove_quiet_clips: bool,
    quiet_peak_dbfs: float,
    gain_multiplier: object = 1.0,
    cancel_check: Optional[Callable[[], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> tuple[bool, str]:
    """
    Shared quiet-removal rule for SDK XML analysis and PyAAF2 fallback.

    Clip gain does not shift the peak threshold. A zero-or-negative gain is
    treated as mute because there is no audible edited segment to preserve.
    """
    if not bool(remove_quiet_clips):
        return False, ""
    try:
        gain = float(gain_multiplier) if gain_multiplier is not None else 1.0
    except Exception:
        gain = 1.0
    if gain <= 0.0:
        return True, "quiet"
    remove = bool(
        is_quiet_clip(
            media_path,
            start_seconds,
            duration_seconds,
            float(quiet_peak_dbfs),
            cancel_check=cancel_check,
            log_callback=log_callback,
        )
    )
    return (remove, "quiet" if remove else "")
