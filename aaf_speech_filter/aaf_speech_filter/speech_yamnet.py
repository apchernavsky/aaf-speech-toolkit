"""
Речь / не речь через YAMNet (TensorFlow Hub), без распознавания текста.

Зависимости: ``tensorflow``, ``tensorflow-hub`` (см. зависимости пакета ``aaf-speech-filter``).
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import types
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np

from .speech_vad import read_media_segment_pcm16_mono
from .speech_audio import _pcm16_mono_to_float32_16k

logger = logging.getLogger(__name__)

# Индексы классов YAMNet (порядок AudioSet в class map модели /1) для грубой тройной классификации.
MUSIC_CLASS_INDICES: tuple[int, ...] = (
    24,
    25,
    26,
    27,
    29,
    30,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    147,
    148,
    149,
    157,
    158,
    159,
    160,
    162,
    163,
    170,
    179,
    180,
    182,
    184,
    186,
    190,
    192,
    209,
    211,
    212,
    214,
    222,
    225,
    228,
    229,
    232,
    233,
    234,
    235,
    238,
    240,
    241,
    242,
    243,
    244,
    247,
    248,
    249,
    250,
    251,
    253,
    254,
    255,
    256,
    257,
    259,
    260,
    262,
    263,
    264,
    265,
    267,
    268,
    269,
    270,
    271,
    272,
    273,
    274,
    275,
    276,
)
VOCAL_MUSIC_CLASS_INDICES: tuple[int, ...] = (
    24,   # Singing
    25,   # Choir
    26,   # Yodeling
    27,   # Chant
    29,   # Child singing
    30,   # Synthetic singing
    249,  # Vocal music
    250,  # A capella
)
DIALOG_SPEECH_CLASS_INDICES: tuple[int, ...] = (
    1,  # Child speech, kid speaking
    2,  # Conversation
    3,  # Narration, monologue
    4,  # Babbling
    5,  # Speech synthesizer
)
NOISE_CLASS_INDICES: tuple[int, ...] = (
    65,
    278,
    279,
    293,
    321,
    # Broad SFX / ambience / non-musical sound classes. These matter for lane layout:
    # YAMNet can give a car signal high "Silence" / "Sound effect" confidence and a small
    # "Speech synthesizer" score; without these competitors the clip is mislabeled speech.
    488,
    494,
    498,
    500,
    501,
    503,
    506,
    507,
    508,
    509,
    510,
    513,
    514,
    515,
)


def _broad_non_speech_class_indices(stop: int, nclass: int) -> list[int]:
    music = set(i for i in MUSIC_CLASS_INDICES if i < nclass)
    speech = set(range(0, max(1, min(int(stop), int(nclass)))))
    return [i for i in range(int(nclass)) if i not in speech and i not in music]


_model = None
_model_lock = threading.Lock()
_model_url: Optional[str] = None
YAMNET_MODEL_DIR_ENV = "AAF_YAMNET_MODEL_DIR"
YAMNET_ALLOW_DOWNLOAD_ENV = "AAF_YAMNET_ALLOW_DOWNLOAD"


class YamnetModelUnavailableError(RuntimeError):
    """Raised when lane classification needs YAMNet but no local/loadable model exists."""


def is_yamnet_model_unavailable_error(exc: BaseException) -> bool:
    if isinstance(exc, YamnetModelUnavailableError):
        return True
    msg = str(exc)
    return msg.startswith("YAMNet model is not available")


def _ensure_pkg_resources_for_tensorflow_hub() -> None:
    """
    ``tensorflow_hub`` still does ``from pkg_resources import parse_version`` in its
    import path. Setuptools 82+ no longer ships the ``pkg_resources`` module, so a real
    ``pip install setuptools`` does not help — register a tiny shim backed by
    ``packaging.version`` (same comparison semantics TF Hub needs for version checks).
    """
    if "pkg_resources" in sys.modules:
        return
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        return
    try:
        from packaging.version import InvalidVersion, Version

        def parse_version(v: str) -> Version:
            try:
                return Version(v)
            except InvalidVersion:
                return Version("0")

        mod = types.ModuleType("pkg_resources")
        mod.parse_version = parse_version
        sys.modules["pkg_resources"] = mod
    except Exception as exc:
        raise ImportError(
            "Для YAMNet нужен ``pkg_resources`` (его вызывает tensorflow_hub при импорте). "
            "В setuptools 82+ модуль удалён — установите ``packaging`` и обновите "
            "aaf-speech-filter, либо выполните: pip install 'setuptools<82'"
        ) from exc


@dataclass(frozen=True)
class YamnetConfig:
    """Порог и агрегация по кадрам YAMNet (≈0.48 s на кадр)."""

    # Официальный модуль в туториалах TF Hub — /1; /4 и др. могут не резолвиться как SavedModel.
    hub_url: str = "https://tfhub.dev/google/yamnet/1"
    allow_download: bool = False
    # Классы AudioSet 0 .. speech_class_stop-1 (Speech … Whispering).
    speech_class_stop: int = 13
    # По кадру: max softmax(P(class)) среди классов 0..stop-1 (521 класс → значения часто 1e-3…5e-2).
    # По клипу: max или mean по кадрам (см. frame_aggregate). Порог 0.15 здесь был бы заведомо
    # слишком высоким и отрезал бы почти всю речь; типичный рабочий диапазон ~0.0025…0.02.
    score_threshold: float = 0.0035
    frame_aggregate: str = "max"  # max | mean
    # For clip classification (speech/noise/music lanes): sample several windows inside the
    # timeline interval to stabilize results on diverse material.
    clip_samples: int = 3
    clip_sample_window_sec: float = 1.0
    clip_samples_min_clip_sec: float = 1.2
    clip_edge_trim_sec: float = 0.1
    # Generic Speech can have a small non-zero score on non-speech material. Treat it as
    # speech-present only when it is strong, competitive with the dominant background,
    # or backed by a dialog-specific speech subclass. Vocalization classes such as
    # Screaming, Whoop, Yell, and Whispering are intentionally not dialog proof:
    # sound effects can trigger them without containing spoken words.
    strong_speech_score: float = 0.20
    mixed_speech_score: float = 0.15
    mixed_speech_competitor_ratio: float = 0.45
    speech_competitor_ratio: float = 0.50
    dialog_speech_score: float = 0.012
    generic_speech_dialog_detail_score: float = 0.030
    generic_speech_noise_score: float = 0.10
    generic_speech_noise_competitor_ratio: float = 0.45
    generic_speech_strong_noise_escape_score: float = 0.80
    generic_speech_strong_noise_escape_ratio: float = 3.5
    # If the top-2 group scores are too close, treat as "unknown" (goes to center lanes).
    ambiguous_margin_abs: float = 0.0006
    ambiguous_margin_rel: float = 0.15


def _truthy_env(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _tfhub_cached_model_dir(hub_url: str) -> Optional[Path]:
    import hashlib
    import tempfile

    cache_root_str = os.environ.get("TFHUB_CACHE_DIR") or os.path.join(
        tempfile.gettempdir(), "tfhub_modules"
    )
    cache_root = Path(cache_root_str)
    if not cache_root.is_dir():
        return None
    model_dir = cache_root / hashlib.sha1(hub_url.encode("utf-8")).hexdigest()
    has_pb = (model_dir / "saved_model.pb").is_file()
    has_pbtxt = (model_dir / "saved_model.pbtxt").is_file()
    if model_dir.is_dir() and (has_pb or has_pbtxt):
        return model_dir
    return None


def _local_yamnet_model_handle(cfg: YamnetConfig) -> Optional[str]:
    local_model = (os.environ.get(YAMNET_MODEL_DIR_ENV) or "").strip().strip('"')
    if local_model:
        model_dir = Path(local_model)
        has_pb = (model_dir / "saved_model.pb").is_file()
        has_pbtxt = (model_dir / "saved_model.pbtxt").is_file()
        if not model_dir.is_dir() or not (has_pb or has_pbtxt):
            raise YamnetModelUnavailableError(
                f"YAMNet local model directory from {YAMNET_MODEL_DIR_ENV} is invalid: "
                f"{model_dir}. It must contain saved_model.pb or saved_model.pbtxt."
            )
        return str(model_dir)

    _clear_corrupt_tfhub_cache(str(cfg.hub_url))
    cached_model = _tfhub_cached_model_dir(str(cfg.hub_url))
    if cached_model is not None:
        return str(cached_model)
    return None


def yamnet_model_available_locally(cfg: Optional[YamnetConfig] = None) -> bool:
    return _local_yamnet_model_handle(cfg or YamnetConfig()) is not None


def _yamnet_model_handle(cfg: YamnetConfig) -> str:
    local_handle = _local_yamnet_model_handle(cfg)
    if local_handle is not None:
        return local_handle

    if bool(getattr(cfg, "allow_download", False)) or _truthy_env(YAMNET_ALLOW_DOWNLOAD_ENV):
        return str(cfg.hub_url)

    raise YamnetModelUnavailableError(
        "YAMNet model is not available locally. "
        f"Set {YAMNET_MODEL_DIR_ENV} to a local YAMNet SavedModel directory "
        f"or set {YAMNET_ALLOW_DOWNLOAD_ENV}=1 to explicitly allow TensorFlow Hub "
        "network download before running YAMNet lane layout."
    )


def _same_handle(handle: str, hub_url: str) -> bool:
    return handle == hub_url


def _maybe_clear_corrupt_url_cache(handle: str, hub_url: str) -> None:
    if _same_handle(handle, hub_url):
        _clear_corrupt_tfhub_cache(hub_url)


def _yamnet_model_unavailable_error(handle: str, exc: BaseException) -> YamnetModelUnavailableError:
    return YamnetModelUnavailableError(
        "YAMNet model is not available for local processing. "
        f"TensorFlow Hub handle/path: {handle}. "
        f"Loader error: {exc.__class__.__name__}: {exc}. "
        f"Cache or install the YAMNet SavedModel locally and set {YAMNET_MODEL_DIR_ENV} "
        "to that directory. The pipeline cannot silently continue because YAMNet lane "
        "layout requires real speech/noise/music classification."
    )


def _load_yamnet_model_from_handle(hub, handle: str):
    try:
        return hub.load(handle)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise _yamnet_model_unavailable_error(handle, exc) from exc


def _clear_corrupt_tfhub_cache(hub_url: str) -> bool:
    """
    Удалить директорию кэша TF Hub для *hub_url*, если она существует, но повреждена
    (нет ``saved_model.pb`` / ``saved_model.pbtxt``).

    Возвращает True, если повреждённый кэш найден и удалён.
    """
    import hashlib
    import shutil
    import tempfile

    cache_root_str = os.environ.get("TFHUB_CACHE_DIR") or os.path.join(
        tempfile.gettempdir(), "tfhub_modules"
    )
    cache_root = Path(cache_root_str)
    if not cache_root.is_dir():
        return False

    # TF Hub (legacy) кэширует по SHA1 URL-строки.
    url_hash = hashlib.sha1(hub_url.encode("utf-8")).hexdigest()
    model_dir = cache_root / url_hash
    if not model_dir.is_dir():
        return False

    has_pb = (model_dir / "saved_model.pb").is_file()
    has_pbtxt = (model_dir / "saved_model.pbtxt").is_file()
    if has_pb or has_pbtxt:
        return False

    logger.warning(
        "YAMNet: обнаружен повреждённый кэш модели (нет saved_model.pb/pbtxt), удаляю: %s",
        model_dir,
    )
    try:
        shutil.rmtree(str(model_dir), ignore_errors=True)
    except Exception as _e:
        logger.warning("YAMNet: не удалось удалить повреждённый кэш %s: %s", model_dir, _e)
        return False
    return True


def _get_yamnet_model(cfg: YamnetConfig):
    global _model, _model_url
    handle = _yamnet_model_handle(cfg)
    with _model_lock:
        if _model is not None and _model_url != handle:
            _model = None
        if _model is None:
            _ensure_pkg_resources_for_tensorflow_hub()
            try:
                import tensorflow_hub as hub
            except ImportError as e:
                raise ImportError(
                    "Для YAMNet установите: pip install tensorflow tensorflow-hub packaging"
                ) from e
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            _model_url = handle

            # Pre-flight: clear corrupt cache before the first load attempt.
            _maybe_clear_corrupt_url_cache(handle, cfg.hub_url)
            try:
                _model = _load_yamnet_model_from_handle(hub, handle)
            except ValueError as _ve:
                _msg = str(_ve)
                if "saved_model.pb" in _msg or "saved_model.pbtxt" in _msg or "incompatible/unknown type" in _msg:
                    # Cache may have been written between our check and hub.load() — try once more.
                    logger.warning(
                        "YAMNet: hub.load() вернул ValueError (повреждённый кэш?), "
                        "очищаю кэш и повторяю загрузку: %s",
                        _msg,
                    )
                    _maybe_clear_corrupt_url_cache(handle, cfg.hub_url)
                    try:
                        _model = _load_yamnet_model_from_handle(hub, handle)
                    except (urllib.error.URLError, OSError, TimeoutError) as exc:
                        raise _yamnet_model_unavailable_error(handle, exc) from exc
                else:
                    raise
            logger.info("YAMNet загружен: %s", cfg.hub_url)
        return _model


def ensure_yamnet_model_available(cfg: YamnetConfig) -> None:
    _get_yamnet_model(cfg)


def has_speech_yamnet(
    wav_path: Path,
    start_seconds: float,
    duration_seconds: float,
    cfg: YamnetConfig,
    cancel_check: Optional[Callable[[], None]] = None,
) -> bool:
    """
    True если по YAMNet в клипе достаточно высокая softmax-вероятность хотя бы одного из
    классов речи (0..12) на одном из кадров (после агрегации max/mean), см. ``score_threshold``.

    При нулевой длительности, пустом PCM, слишком коротком буфере для кадра YAMNet или пустом
    выходе модели возвращает **True** (считаем «есть речь») — это защищает пайплайн от
    агрессивных решений на ошибках тайминга/формата файла.

    Любая непойманная ошибка (float-WAV, обрезанный файл, сбой TF) → **True** (не удалять).
    """
    def _speech_present_relaxed(s: float, m: float, n: float) -> bool:
        """
        Conservative speech-presence decision for noisy/long clips.

        We prefer false-positives over false-negatives because this function is used to decide
        whether speech is present in a clip.

        Rationale (Premiere AAF with long/noisy production audio):
        even when speech is clearly audible, YAMNet group maxima can be close (speech≈noise or
        speech≈music). If speech is "close enough" to the dominant group and above a low floor,
        we treat it as speech-present.
        """
        thr = float(cfg.score_threshold)
        if s >= thr:
            return True
        soft_floor = min(0.0020, max(0.0, thr * 0.55))
        dom = max(float(m), float(n))
        # If speech is reasonably strong and not far below the dominant group, keep it.
        return (s >= soft_floor) and (dom <= 0.0 or (s >= 0.80 * dom))
    if cancel_check is not None:
        cancel_check()

    if duration_seconds <= 0:
        return True

    try:
        pcm, sr = read_media_segment_pcm16_mono(wav_path, start_seconds, duration_seconds)
        if not pcm or len(pcm) < 4:
            return True

        audio = _pcm16_mono_to_float32_16k(pcm, sr)
        if len(audio) < 400:
            return True

        min_samples = int(0.35 * 16000)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))

        if cancel_check is not None:
            cancel_check()

        model = _get_yamnet_model(cfg)
        import tensorflow as tf

        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)

        scores, _emb, _spec = model(waveform)
        if cancel_check is not None:
            cancel_check()

        # YAMNet returns per-class scores/probabilities already, not logits.
        # Applying softmax here flattens confident detections toward ~1/521
        # and makes obvious music/dialog look "unknown".
        probs = scores.numpy()
        if probs.size == 0:
            return True
        stop = max(1, min(int(cfg.speech_class_stop), probs.shape[-1]))
        frame_scores = np.max(probs[:, :stop], axis=-1)
        agg = (cfg.frame_aggregate or "max").strip().lower()
        if agg == "mean":
            s = float(np.mean(frame_scores))
        else:
            s = float(np.max(frame_scores))

        # Compute competing groups for a more conservative speech-presence decision.
        nclass = probs.shape[-1]
        mi = [i for i in MUSIC_CLASS_INDICES if i < nclass]
        ni = [i for i in NOISE_CLASS_INDICES if i < nclass]
        music_frames = np.max(probs[:, mi], axis=-1) if mi else np.zeros_like(frame_scores)
        noise_frames = np.max(probs[:, ni], axis=-1) if ni else np.zeros_like(frame_scores)
        if agg == "mean":
            m = float(np.mean(music_frames))
            n = float(np.mean(noise_frames))
        else:
            m = float(np.max(music_frames))
            n = float(np.max(noise_frames))

        return _speech_present_relaxed(s, m, n)
    except Exception as exc:
        logger.warning(
            "YAMNet: пропуск классификации (считаем «есть речь»): %s start=%.4f dur=%.4f — %s: %s",
            wav_path,
            start_seconds,
            duration_seconds,
            type(exc).__name__,
            exc,
        )
        return True


def yamnet_clip_kind_with_scores(
    wav_path: Path,
    start_seconds: float,
    duration_seconds: float,
    cfg: YamnetConfig,
    cancel_check: Optional[Callable[[], None]] = None,
) -> tuple[Literal["speech", "noise", "music", "unknown"], float, float, float]:
    """
    Как ``yamnet_clip_kind``, плюс агрегаты softmax по группам: speech (0..stop-1), music, noise.
    """
    if cancel_check is not None:
        cancel_check()
    if duration_seconds <= 0:
        return "unknown", 0.0, 0.0, 0.0

    def _scores_for_window(st: float, du: float) -> tuple[float, float, float, float, float, float, float]:
        pcm, sr = read_media_segment_pcm16_mono(wav_path, float(st), float(du))
        if not pcm or len(pcm) < 4:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        audio = _pcm16_mono_to_float32_16k(pcm, sr)
        if len(audio) < 400:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        min_samples = int(0.35 * 16000)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))

        if cancel_check is not None:
            cancel_check()

        model = _get_yamnet_model(cfg)
        import tensorflow as tf

        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
        scores, _emb, _spec = model(waveform)
        if cancel_check is not None:
            cancel_check()

        # YAMNet returns per-class scores/probabilities already, not logits.
        probs = scores.numpy()
        if probs.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        nclass = probs.shape[-1]
        stop = max(1, min(int(cfg.speech_class_stop), nclass))
        speech_frames = np.max(probs[:, :stop], axis=-1)
        generic_speech_frames = probs[:, 0] if nclass > 0 else np.zeros_like(speech_frames)
        di = [i for i in DIALOG_SPEECH_CLASS_INDICES if i < stop]
        dialog_speech_frames = np.max(probs[:, di], axis=-1) if di else np.zeros_like(speech_frames)
        mi = [i for i in MUSIC_CLASS_INDICES if i < nclass]
        vi = [i for i in VOCAL_MUSIC_CLASS_INDICES if i < nclass]
        ni = sorted(set(i for i in NOISE_CLASS_INDICES if i < nclass) | set(_broad_non_speech_class_indices(stop, nclass)))
        music_frames = np.max(probs[:, mi], axis=-1) if mi else np.zeros_like(speech_frames)
        vocal_music_frames = np.max(probs[:, vi], axis=-1) if vi else np.zeros_like(speech_frames)
        noise_frames = np.max(probs[:, ni], axis=-1) if ni else np.zeros_like(speech_frames)
        agg = (cfg.frame_aggregate or "max").strip().lower()

        # Speech can benefit from `max` (catch brief speech), while music/noise should be more
        # robust to brief clicks/tones, so we use `mean` for them.
        def _agg_speech(frames: np.ndarray) -> float:
            if agg == "mean":
                return float(np.mean(frames))
            return float(np.max(frames))

        def _agg_bg(frames: np.ndarray) -> float:
            return float(np.mean(frames))

        dialog_floor = max(float(cfg.dialog_speech_score), float(cfg.score_threshold) * 3.0)
        dialog_support = float(np.mean(dialog_speech_frames >= dialog_floor))
        return (
            _agg_speech(speech_frames),
            _agg_bg(music_frames),
            _agg_bg(noise_frames),
            _agg_speech(vocal_music_frames),
            _agg_speech(generic_speech_frames),
            _agg_speech(dialog_speech_frames),
            dialog_support,
        )

    # Multi-sample inside the visible timeline window, for stability.
    ns = max(1, int(getattr(cfg, "clip_samples", 1) or 1))
    min_clip = float(getattr(cfg, "clip_samples_min_clip_sec", 1.2) or 1.2)
    win = float(getattr(cfg, "clip_sample_window_sec", 1.0) or 1.0)
    trim = float(getattr(cfg, "clip_edge_trim_sec", 0.1) or 0.0)
    # Trim first/last 100ms (user requirement) to reduce edge artifacts; keep safe fallback.
    st_eff = float(start_seconds)
    du_eff = float(duration_seconds)
    if trim > 0.0 and du_eff > (2.0 * trim + 0.05):
        st_eff = st_eff + trim
        du_eff = du_eff - 2.0 * trim

    win = max(0.35, min(win, float(du_eff)))
    # Short clips need one full-window pass: splitting them into overlapping peak-ranked
    # windows distorts temporal speech support and adds redundant model calls.
    if ns <= 1 or float(du_eff) <= 2.1 or float(du_eff) < min_clip or float(du_eff) <= win + 1e-6:
        s, m, n, v, generic_speech, dialog_speech, dialog_support = _scores_for_window(
            float(st_eff), float(du_eff)
        )
    else:
        # Pick representative audible windows inside the visible timeline interval.
        # Long music cues often contain long silent gaps; fixed start/mid/end sampling
        # can land entirely in silence and classify obvious music as "unknown".
        st0 = float(st_eff)
        span = max(0.0, float(du_eff) - win)
        cand_n = max(ns * 3, 9)
        if span <= 1e-6 or cand_n <= 1:
            starts = [st0]
        else:
            starts = [st0 + span * (i / float(cand_n - 1)) for i in range(cand_n)]
        ranked_starts: list[tuple[float, float]] = []
        for st in starts:
            try:
                pcm, _sr = read_media_segment_pcm16_mono(wav_path, float(st), float(win))
                if pcm:
                    # audioop.max would also work, but numpy avoids adding another import.
                    peak = float(np.max(np.abs(np.frombuffer(pcm, dtype=np.int16))))
                else:
                    peak = 0.0
            except Exception:
                peak = 0.0
            ranked_starts.append((peak, float(st)))
        peak_by_start = {float(st): float(peak) for peak, st in ranked_starts}
        audible_starts = [
            st
            for _peak, st in sorted(ranked_starts, key=lambda x: (-x[0], x[1]))[:ns]
        ]
        ss: list[float] = []
        ms: list[float] = []
        ns_: list[float] = []
        vs: list[float] = []
        generic_speech_scores: list[float] = []
        dialog_speech_scores: list[float] = []
        dialog_support_scores: list[float] = []
        bg_ms: list[float] = []
        bg_ns: list[float] = []
        bg_ws: list[float] = []
        for st in starts:
            if cancel_check is not None:
                cancel_check()
            _s, _m, _n, _v, _generic_speech, _dialog_speech, _dialog_support = _scores_for_window(
                float(st), float(win)
            )
            ss.append(float(_s))
            ms.append(float(_m))
            ns_.append(float(_n))
            vs.append(float(_v))
            generic_speech_scores.append(float(_generic_speech))
            dialog_speech_scores.append(float(_dialog_speech))
            dialog_support_scores.append(float(_dialog_support))
            if float(st) in audible_starts:
                bg_ms.append(float(_m))
                bg_ns.append(float(_n))
                bg_ws.append(max(float(win), min(1.0, peak_by_start.get(float(st), 0.0))))
        # Multi-sample aggregation:
        # - For speech, use temporal coverage across the visible clip window; quiet dialog must
        #   not disappear just because louder non-speech windows exist elsewhere in the clip.
        # - For music/noise, keep the audible-window mean to avoid reacting to silence or brief
        #   low-energy artifacts when deciding background ownership.
        wsum = float(sum(bg_ws) or 0.0)
        if wsum > 0.0:
            s = float(max(ss) if ss else 0.0)
            m = float(sum(w * v for w, v in zip(bg_ws, bg_ms)) / wsum)
            n = float(sum(w * v for w, v in zip(bg_ws, bg_ns)) / wsum)
            v = float(max(vs) if vs else 0.0)
            generic_speech = float(max(generic_speech_scores) if generic_speech_scores else 0.0)
            dialog_speech = float(max(dialog_speech_scores) if dialog_speech_scores else 0.0)
            dialog_support = float(max(dialog_support_scores) if dialog_support_scores else 0.0)
        else:
            s, m, n, v, generic_speech, dialog_speech = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            dialog_support = 0.0

    # If nothing could be computed (IO/format issues), be conservative.
    if max(float(s), float(m), float(n)) <= 0.0:
        return "unknown", float(s), float(m), float(n)

    # If all group scores are very low and nearly equal, the model is effectively not confident.
    # Treat as unknown (center lanes) rather than forcing "speech".
    _vals = [float(s), float(m), float(n)]
    _vals_sorted = sorted(_vals, reverse=True)
    top1 = _vals_sorted[0]
    top2 = _vals_sorted[1]
    if top1 < 0.0023 and (top1 - top2) < 0.00025:
        # Very low overall confidence: usually treat as unknown.
        # However, if speech is close to the dominant group, prefer speech for lane layout.
        thr = float(cfg.score_threshold)
        soft_floor = min(0.0020, max(0.0, thr * 0.55))
        dom = max(float(m), float(n))
        if s >= soft_floor and (dom <= 0.0 or s >= 0.70 * dom):
            return "speech", float(s), float(m), float(n)
        return "unknown", float(s), float(m), float(n)

    # For layout (speech/noise/music lanes), prefer classifying as speech when speech is close
    # to the dominant group and above a low floor. This matches user expectation: if there is
    # speech in the active AAF interval, keep it on "speech" lanes even with noise present.
    thr = float(cfg.score_threshold)
    soft_floor = min(0.0020, max(0.0, thr * 0.55))
    dom = max(float(m), float(n))
    trace_speech_floor = max(0.012, thr * 3.5)
    if m >= 0.15 and m >= 20.0 * max(float(s), 1e-9) and s < trace_speech_floor:
        return "music", float(s), float(m), float(n)
    # Singing/vocal-music classes are music ownership, not dialog ownership.
    # A strong vocal-music score can coexist with generic Speech, but the clip belongs
    # to the music lane unless dialog/narration evidence clearly dominates.
    if v >= 0.05 and m >= 0.08 and m >= 0.50 * max(float(s), 1e-9):
        return "music", float(s), float(m), float(n)
    dialog_floor = max(float(cfg.dialog_speech_score), thr * 3.0)
    dialog_speech_present = dialog_speech >= dialog_floor and (
        (dom < 0.10 and dialog_support >= 0.20)
        or dialog_speech >= max(0.05, dialog_floor * 3.0)
    )
    dialog_detail_floor = max(float(cfg.generic_speech_dialog_detail_score), dialog_floor * 2.5)
    generic_dominant_speech = generic_speech >= 0.80 * max(float(s), 1e-9)
    generic_only_speech = (
        generic_dominant_speech
        and not dialog_speech_present
        and dialog_speech < dialog_detail_floor
    )
    dominant_generic_speech = (
        s >= float(cfg.strong_speech_score)
        and (dom <= 0.0 or s >= 1.75 * dom)
    )
    decisive_generic_speech = (
        generic_only_speech
        and s >= float(cfg.generic_speech_strong_noise_escape_score)
    ) or (
        dominant_generic_speech
        and dom > 0.0
        and s >= float(cfg.generic_speech_strong_noise_escape_ratio) * dom
    )
    strong_non_dialog_noise = (
        generic_only_speech
        and not dialog_speech_present
        and n >= max(0.15, float(cfg.generic_speech_noise_score))
        and n >= m
        and not decisive_generic_speech
    )
    if strong_non_dialog_noise:
        return "noise", float(s), float(m), float(n)
    if (
        generic_only_speech
        and not dominant_generic_speech
        and n >= float(cfg.generic_speech_noise_score)
        and n >= float(cfg.generic_speech_noise_competitor_ratio) * max(float(s), 1e-9)
        and n >= m
        and not decisive_generic_speech
    ):
        return "noise", float(s), float(m), float(n)
    # Music can produce a non-trivial generic Speech score without any dialog evidence.
    # Keep strong music-dominant material in music lanes, while allowing dialog speech
    # subclasses to preserve dialog over music.
    if generic_only_speech and not decisive_generic_speech and m >= 0.25 and m >= s:
        return "music", float(s), float(m), float(n)
    # Speech decision:
    # - If speech clears the main threshold: speech.
    # - Otherwise, if speech is above a low floor and is "close enough" to the dominant group,
    #   treat as speech-present. This is intentionally permissive for "process stream" style clips
    #   where speech exists but may not dominate.
    #
    # To avoid turning strong engine/tonal noise into speech, tighten the ratio when the dominant
    # group confidence is itself high.
    strong_speech = s >= float(cfg.strong_speech_score)
    mixed_speech = (
        s >= float(cfg.mixed_speech_score)
        and (dom <= 0.0 or s >= float(cfg.mixed_speech_competitor_ratio) * dom)
    )
    competitive_speech = dom <= 0.0 or s >= float(cfg.speech_competitor_ratio) * dom
    non_dialog_noise_false_positive = (
        n >= float(cfg.generic_speech_noise_score)
        and n >= float(cfg.generic_speech_noise_competitor_ratio) * max(float(s), 1e-9)
        and n >= m
        and not generic_dominant_speech
        and not dialog_speech_present
    )
    if non_dialog_noise_false_positive:
        return "noise", float(s), float(m), float(n)
    if s >= thr and (strong_speech or mixed_speech or competitive_speech or dialog_speech_present):
        return "speech", float(s), float(m), float(n)
    if dom >= 0.035 and s < float(cfg.speech_competitor_ratio) * dom:
        if m >= n:
            return "music", float(s), float(m), float(n)
        return "noise", float(s), float(m), float(n)
    if s >= soft_floor:
        # Low/medium confidence overall: allow speech when reasonably close.
        if dom < 0.010 and (dom <= 0.0 or s >= 0.75 * dom):
            return "speech", float(s), float(m), float(n)
        # High confidence non-speech: require speech to be very close.
        if dom >= 0.010 and (dom <= 0.0 or s >= 0.90 * dom):
            return "speech", float(s), float(m), float(n)
        # Fall through to non-speech decision.

    # Vocal music (singing): when speech evidence is only soft-floor level, keep music-dominant
    # material in music lanes. Above-threshold speech has already won as speech-present.
    music_floor = max(0.0022, soft_floor * 1.05)
    if m >= music_floor and s >= soft_floor and m >= 0.90 * s and m >= 1.05 * n:
        return "music", float(s), float(m), float(n)

    # Ambiguous: if top-2 are too close, treat as unknown (center lanes).
    confs = sorted([("speech", float(s)), ("music", float(m)), ("noise", float(n))], key=lambda x: x[1], reverse=True)
    top1 = confs[0][1]
    top2 = confs[1][1]
    try:
        mar_abs = float(getattr(cfg, "ambiguous_margin_abs", 0.0006) or 0.0006)
        mar_rel = float(getattr(cfg, "ambiguous_margin_rel", 0.15) or 0.15)
    except Exception:
        mar_abs, mar_rel = 0.0006, 0.15
    if (top1 - top2) < max(mar_abs, mar_rel * max(1e-9, top1)):
        # If it's "music vs speech" and both are above floor, prefer music (vocal music).
        if m >= music_floor and s >= soft_floor:
            return "music", float(s), float(m), float(n)
        # If speech is present but close to the dominant group, prefer speech over "unknown"
        # for lane layout (user expectation: audible dialog should go to speech lanes).
        dom = max(float(m), float(n))
        if s >= soft_floor and (dom <= 0.0 or s >= 0.70 * dom):
            return "speech", float(s), float(m), float(n)
        return "unknown", float(s), float(m), float(n)

    if s >= m and s >= n:
        return "speech", float(s), float(m), float(n)
    if m >= n:
        return "music", float(s), float(m), float(n)
    return "noise", float(s), float(m), float(n)


def yamnet_clip_kind(
    wav_path: Path,
    start_seconds: float,
    duration_seconds: float,
    cfg: YamnetConfig,
    cancel_check: Optional[Callable[[], None]] = None,
) -> Literal["speech", "noise", "music"]:
    """
    Грубая классификация по softmax YAMNet: речь (классы 0..speech_class_stop-1), музыка, шум.
    При равенстве счётов приоритет: speech > music > noise.
    """
    k, _s, _m, _n = yamnet_clip_kind_with_scores(
        wav_path, start_seconds, duration_seconds, cfg, cancel_check=cancel_check
    )
    return k
