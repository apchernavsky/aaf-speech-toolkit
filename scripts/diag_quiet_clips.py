#!/usr/bin/env python3
"""
Diagnostic for quiet-clip detection on a specific AAF.

Goal: explain why some clips are NOT considered quiet by our detector even if Nuendo meters look low.
This script reports, for each timeline SourceClip:
  - resolved media path (sidecar or runtime essence map if provided)
  - computed (start_sec, dur_sec) used for reading audio
  - sample peak (dBFS) measured on decoded PCM16 after optional downmix
  - whether the clip sits under an OperationGroup chain (potential gain/effects we currently ignore)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional


def _ensure_toolkit_on_path() -> None:
    here = Path(__file__).resolve()
    toolkit = here.parents[1]
    aaf_io_root = toolkit / "aaf_io"
    aaf_speech_root = toolkit / "aaf_speech_filter"
    for p in (toolkit, aaf_io_root, aaf_speech_root):
        if p.is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)


_ensure_toolkit_on_path()

from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient  # noqa: E402
from aaf_io.converter import AAFConverter  # noqa: E402
from aaf_speech_filter.media_resolve import _resolve_wave_path_for_sourceclip  # noqa: E402
from aaf_speech_filter.timeline_timing import sourceclip_audio_timing  # noqa: E402
from aaf_speech_filter.timeline_walk import (  # noqa: E402
    _is_sourceclip,
    _slot_edit_rate,
    _slot_is_soundish,
    _vectors_on_container,
)
from aaf_speech_filter.speech_vad import peak_dbfs_16le_mono, read_media_segment_pcm16_mono  # noqa: E402


def _iter_sourceclips_with_stack(aaf: Any):
    compositions = list(aaf.content.compositionmobs())

    def walk(container: Any, edit_rate: float, stack: list[str]):
        if container is None:
            return
        # record node type in stack for diagnostics
        cur_name = container.__class__.__name__
        stack2 = stack + [cur_name]
        for vec in _vectors_on_container(container):
            i = 0
            while True:
                try:
                    n = len(vec)
                except Exception:
                    break
                if i >= n:
                    break
                try:
                    node = vec[i]
                except IndexError:
                    break
                if _is_sourceclip(node):
                    yield edit_rate, node, stack2
                else:
                    yield from walk(node, edit_rate, stack2)
                i += 1
        selected = getattr(container, "selected", None)
        if selected is not None:
            yield from walk(selected, edit_rate, stack2)

    for comp in compositions:
        for slot in getattr(comp, "slots", []) or []:
            if not _slot_is_soundish(slot):
                continue
            er = float(_slot_edit_rate(slot) or 48000.0)
            seg = getattr(slot, "segment", None)
            if seg is None:
                continue
            yield from walk(seg, er, stack=[comp.__class__.__name__, "Slot"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose quiet detection for a single AAF.")
    ap.add_argument("aaf", type=Path)
    ap.add_argument("--threshold", type=float, default=-1.0, help="dBFS threshold (default -1)")
    ap.add_argument("--limit", type=int, default=60, help="print at most N clips (default 60)")
    ap.add_argument("--only-not-quiet", action="store_true", help="print only clips our detector deems NOT quiet")
    ap.add_argument("--min-dur", type=float, default=0.02, help="skip very short durations (seconds)")
    ap.add_argument(
        "--prepare-embedded",
        action="store_true",
        help="Extract embedded essence via AAFConverter and use runtime_essence_paths (recommended for embedded AAF).",
    )
    args = ap.parse_args()

    aaf_path = Path(args.aaf).resolve()
    if not aaf_path.is_file():
        print(f"AAF not found: {aaf_path}")
        return 2

    printed = 0
    total = 0
    no_media = 0
    errors = 0
    runtime_essence_paths = None
    work_path: Optional[Path] = None
    conv: Optional[AAFConverter] = None

    if args.prepare_embedded:
        conv = AAFConverter(aaf_path)
        work_path = conv.output_dir / f"{aaf_path.stem}.__diag_quiet_work{aaf_path.suffix}"
        try:
            work_path.unlink(missing_ok=True)
        except Exception:
            pass
        ok = conv.prepare_external_media_copy(work_path)
        runtime_essence_paths = conv.runtime_essence_paths_for_filter()
        if not ok or not work_path.is_file() or runtime_essence_paths is None:
            print("Failed to prepare embedded essence (no runtime_essence_paths).")
            try:
                if conv is not None:
                    conv._unlink_mapped_essence()
            except Exception:
                pass
            try:
                if work_path is not None:
                    work_path.unlink(missing_ok=True)
            except Exception:
                pass
            return 1

    open_path = work_path if work_path is not None else aaf_path
    try:
        with open_aaf_lenient(open_path, "r") as aaf:
            for er, node, stack in _iter_sourceclips_with_stack(aaf):
                total += 1
                wp = _resolve_wave_path_for_sourceclip(
                    aaf, node, runtime_essence_paths=runtime_essence_paths
                )
                if wp is None or not Path(wp).is_file():
                    no_media += 1
                    continue
                wav_path = Path(wp)

                start_sec, dur_sec = sourceclip_audio_timing(
                    aaf,
                    float(er or 48000.0),
                    node,
                    wav_path,
                    runtime_essence_paths=runtime_essence_paths,
                )
                if dur_sec < float(args.min_dur):
                    continue

                under_opg = any("OperationGroup" in s for s in stack)
                try:
                    pcm, _sr = read_media_segment_pcm16_mono(wav_path, start_sec, dur_sec)
                    pk = peak_dbfs_16le_mono(pcm)
                    is_quiet = pk <= float(args.threshold)
                except Exception as exc:
                    errors += 1
                    pk = None
                    is_quiet = False
                    if printed < args.limit and not args.only_not_quiet:
                        print(f"[ERR] {type(exc).__name__}: {exc} :: {wav_path}")
                        printed += 1
                    continue

                if args.only_not_quiet and is_quiet:
                    continue

                if printed >= args.limit:
                    continue

                pk_s = f"{pk:.2f} dBFS" if pk is not None else "n/a"
                print(
                    f"[{'QUIET' if is_quiet else 'LOUD '}] pk={pk_s} thr={args.threshold:.2f} "
                    f"start={start_sec:.3f}s dur={dur_sec:.3f}s opg={under_opg} :: {wav_path.name}"
                )
                printed += 1
    finally:
        if args.prepare_embedded:
            try:
                if conv is not None:
                    conv._unlink_mapped_essence()
            except Exception:
                pass
            try:
                if work_path is not None:
                    work_path.unlink(missing_ok=True)
            except Exception:
                pass

    print(
        f"\nSummary: timeline_sourceclips={total}, no_media_path={no_media}, errors={errors}, printed={printed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
