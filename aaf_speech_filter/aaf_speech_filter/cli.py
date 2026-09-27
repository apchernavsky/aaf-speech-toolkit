from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from aaf_io.subprocess_hidden import run_hidden

from .pyaaf2_filter import filter_aaf_speech_only
from .config import FilterConfig
from .thresholds import validate_quiet_peak_dbfs
from .speech_yamnet import YamnetConfig


def _find_aaf_pipeline() -> Path | None:
    env = os.environ.get("AAF_PIPELINE_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    here = Path(__file__).resolve()
    for p in (here.parents[2] / "aaf_pipeline.py", here.parents[3] / "aaf_pipeline.py"):
        if p.is_file():
            return p
    return None


def main() -> int:
    # aaf2 may emit noisy CFB warnings via root logger; keep CLI output clean.
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("aaf2").setLevel(logging.ERROR)

    ap = argparse.ArgumentParser(description="Keep only speech clips in an unembedded AAF (Nuendo-friendly).")
    ap.add_argument("input", type=Path, help="Path to input .aaf")
    ap.add_argument(
        "--prepare",
        action="store_true",
        help="Only run aaf_pipeline (builds <stem>_processed.aaf); no second filter pass.",
    )
    ap.add_argument(
        "--experimental-yamnet-lanes",
        dest="experimental_yamnet_lane_layout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "YAMNet lane layout (speech up, noise mid, music down). "
            "Runs last after quiet removal when using filter_aaf_speech_only."
        ),
    )
    ap.add_argument(
        "--yamnet-score-threshold",
        type=float,
        default=0.0035,
        help=(
            "YAMNet: speech if aggregate (max/mean over frames) of max_j softmax(P(j)) for speech "
            "classes j in 0..12 >= this. With 521 classes values are often ~2.5e-3..2e-2, not a 0..1 confidence."
        ),
    )
    ap.add_argument(
        "--yamnet-frame-aggregate",
        type=str,
        choices=("max", "mean"),
        default="max",
        help="YAMNet: aggregate per-frame scores over clip with max or mean (default max).",
    )
    ap.add_argument(
        "--remove-quiet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove clips at or below --quiet-peak-dbfs (default: on).",
    )
    ap.add_argument(
        "--quiet-peak-dbfs",
        type=validate_quiet_peak_dbfs,
        default=-40.0,
        help="Finite peak threshold <= 0 dBFS (default -40); 0 includes full-scale PCM.",
    )

    args = ap.parse_args()

    media_search_roots = None
    aaf_tools_dir = None
    try:
        from aaf_tool_config import resolve_media_search_roots, effective_aaf_tools_dir
    except ModuleNotFoundError as exc:
        if exc.name != "aaf_tool_config":
            raise
    else:
        media_search_roots = resolve_media_search_roots(args.input)
        aaf_tools_dir = effective_aaf_tools_dir()

    if (
        (not args.remove_quiet)
        and (not args.experimental_yamnet_lane_layout)
    ):
        print(
            "Enable at least one of --remove-quiet or --experimental-yamnet-lanes.",
            file=sys.stderr,
        )
        return 2

    orig = args.input.resolve()
    if args.prepare:
        pipeline = _find_aaf_pipeline()
        if pipeline is None:
            print("aaf_pipeline.py not found. Set AAF_PIPELINE_PATH.", file=sys.stderr)
            return 2
        command = [
            sys.executable, str(pipeline), str(orig),
            '--remove-quiet' if args.remove_quiet else '--no-remove-quiet',
            '--experimental-yamnet-lanes' if args.experimental_yamnet_lane_layout
            else '--no-experimental-yamnet-lanes',
            f'--quiet-peak-dbfs={args.quiet_peak_dbfs!r}',
            f'--yamnet-score-threshold={args.yamnet_score_threshold!r}',
            f'--yamnet-frame-aggregate={args.yamnet_frame_aggregate}',
            # Preparation must preserve the package entry point's duplicate policy.
            '--no-remove-duplicates',
        ]
        if aaf_tools_dir is not None:
            command.extend(['--aaf-tools', str(aaf_tools_dir)])
        run_hidden(command, check=True)
        out = orig.with_name(f"{orig.stem}_processed{orig.suffix}")
        if not out.is_file():
            print(f"expected after pipeline: {out}", file=sys.stderr)
            return 2
        return 0

    inp = orig
    output = orig.with_name(f"{orig.stem}_processed{orig.suffix}")

    cfg = FilterConfig(
        media_search_roots=media_search_roots,
        aaf_tools_dir=aaf_tools_dir,
        yamnet=YamnetConfig(
            score_threshold=float(args.yamnet_score_threshold),
            frame_aggregate=str(args.yamnet_frame_aggregate).strip().lower(),
        ),
        experimental_yamnet_lane_layout=args.experimental_yamnet_lane_layout,
        remove_quiet_clips=args.remove_quiet,
        quiet_peak_dbfs=args.quiet_peak_dbfs,
    )

    try:
        removed = filter_aaf_speech_only(inp, output, cfg)
    except KeyboardInterrupt:
        print("Остановка.", file=sys.stderr)
        return 2
    except BaseException as e:
        # Avoid noisy tracebacks on cancellation in CLI.
        try:
            from .exceptions import SpeechFilterCancelled

            if isinstance(e, SpeechFilterCancelled):
                print("Остановка.", file=sys.stderr)
                return 2
        except Exception:
            pass
        raise
    print(f"Удалено клипов: {removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
