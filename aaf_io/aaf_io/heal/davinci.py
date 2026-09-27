"""
DaVinci Resolve (unembedded) AAF helpers.

DaVinci often exports unembedded AAF where linked audio points to container media (e.g. MXF)
via ``Locator`` URLs on descriptors like ``PCMDescriptor``. The speech toolkit primarily
operates on WAV/AIFF paths; this module provides best-effort resolution and proxy WAV decoding
via ffmpeg without changing the higher-level workflow.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Optional

from aaf_io.subprocess_hidden import run_hidden

from .premiere import file_url_to_path, resolve_import_sidecar_media_path

_SUPPORTED_WAV_EXTS = {".wav", ".wave", ".aif", ".aiff"}


def first_resolved_locator_media_path(
    descriptor: Any, *, media_search_roots: Optional[tuple[Path, ...]] = None,
) -> Optional[Path]:
    """
    Return the first existing filesystem path referenced by ``descriptor['Locator']``.

    Works for DaVinci PCMDescriptor locators (typically ``file:///.../*.mxf``) and also for
    WAVEDescriptor/ImportDescriptor style locators.
    """
    try:
        loc_vec = descriptor["Locator"]
    except Exception:
        return None
    try:
        for loc in loc_vec:
            try:
                url = str(loc["URLString"].value)
            except Exception:
                continue
            p = resolve_import_sidecar_media_path(file_url_to_path(str(url)), media_search_roots=media_search_roots)
            if p is not None and Path(p).is_file():
                return Path(p)
    except Exception:
        return None
    return None


def _ffmpeg_exe() -> Optional[str]:
    try:
        from shutil import which

        p = which("ffmpeg")
        return str(p) if p else None
    except Exception:
        return None


def _wav_proxy_path_for_media(media_path: Path, proxy_dir: Path) -> Path:
    try:
        s = str(Path(media_path).resolve())
    except Exception:
        s = str(media_path)
    h = hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:10]
    stem = (Path(media_path).stem or "media").strip()[:80]
    safe = "".join((ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_") for ch in stem).strip("._") or "media"
    return Path(proxy_dir) / f"{safe}.{h}.wav"


def ensure_wav_proxy_for_media(
    media_path: Path,
    *,
    work_dir: Optional[Path],
    cancel_check: Optional[Callable[[], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Optional[Path]:
    """
    Ensure we have a WAV file path usable by WaveReader/YAMNet/VAD.

    - If `media_path` is already a WAV/AIFF: return it.
    - Otherwise, if ffmpeg is available and work_dir is provided: decode to a cached WAV proxy.
    """
    try:
        p = Path(media_path)
    except Exception:
        return None
    if not p.is_file():
        return None
    if p.suffix.lower() in _SUPPORTED_WAV_EXTS:
        return p
    if work_dir is None:
        return None
    ff = _ffmpeg_exe()
    if not ff:
        return None
    try:
        proxy_dir = Path(work_dir) / "__media_wav_cache"
        proxy_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    out = _wav_proxy_path_for_media(p, proxy_dir)
    if out.is_file():
        return out
    if log_callback is not None:
        try:
            log_callback(f"Медиа: декодирование в WAV через ffmpeg… ({p.name})")
        except Exception:
            pass
    if cancel_check is not None:
        cancel_check()
    try:
        cmd = [
            ff,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(p),
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(out),
        ]
        run_hidden(cmd, check=True)
    except Exception:
        try:
            out.unlink(missing_ok=True)
        except Exception:
            pass
        return None
    # If cancellation is requested right after decode, treat as cancelled and cleanup proxy.
    try:
        if cancel_check is not None:
            cancel_check()
    except BaseException:
        try:
            out.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    return out if out.is_file() else None
