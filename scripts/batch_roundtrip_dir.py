#!/usr/bin/env python3
"""
Batch AAF SDK roundtrip for an arbitrary directory of .aaf files.

Runs: aaffmtconv -xml -> (optional sanitize) -> aaffmtconv -ss, via aaf_io.roundtrip.sdk_roundtrip,
with the same fallbacks used by the main pipeline (CFB header heal retry, binary copy on -xml crash).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path


def _ensure_aaf_io_on_path() -> None:
    here = Path(__file__).resolve()
    toolkit = here.parents[1]
    aaf_io_root = toolkit / "aaf_io"
    if (aaf_io_root / "aaf_io" / "__init__.py").is_file():
        s = str(aaf_io_root)
        if s not in sys.path:
            sys.path.insert(0, s)


_ensure_aaf_io_on_path()

from aaf_io.roundtrip import sdk_roundtrip  # noqa: E402
from aaf_io.sdk_tools import default_aaf_tools_dir  # noqa: E402


def _safe_stem(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
        s = str(rel).replace("\\", "_").replace("/", "_")
    except Exception:
        s = path.name
    for ch in '<>:"|?*':
        s = s.replace(ch, "_")
    if len(s) > 180:
        s = s[:180]
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch SDK roundtrip for all *.aaf under a directory.")
    ap.add_argument("aaf_root", type=Path, help="Directory to scan recursively for *.aaf")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for __rt.xml/__rt.aaf and roundtrip_log.txt (default: <aaf_root>/__roundtrip_out)",
    )
    ap.add_argument(
        "--aaf-tools",
        type=Path,
        default=None,
        help="Directory with aaffmtconv.exe / ComAAFInfo.exe (default: sdk_bin near toolkit, then env AAF_TOOLS/AAF_TOOLS_DIR).",
    )
    args = ap.parse_args()

    aaf_root = Path(args.aaf_root).resolve()
    if not aaf_root.is_dir():
        print(f"Not a directory: {aaf_root}")
        return 2

    out_root = Path(args.out).resolve() if args.out else (aaf_root / "__roundtrip_out")
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "roundtrip_log.txt"

    tools_dir = (
        Path(args.aaf_tools).resolve()
        if args.aaf_tools is not None
        else default_aaf_tools_dir(extra_sdk_bin=Path(__file__).resolve().parents[1] / "sdk_bin")
    )

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    jobs = sorted({p.resolve() for p in aaf_root.rglob("*.aaf") if p.is_file()}, key=lambda p: str(p).lower())
    lines: list[str] = []
    lines.append(f"aaf_tools_dir: {tools_dir}")
    lines.append(f"aaf_root: {aaf_root}")
    lines.append(f"out: {out_root}")
    lines.append(f"jobs: {len(jobs)}")

    errors = 0
    t0 = time.perf_counter()
    for i, src in enumerate(jobs, 1):
        stem = _safe_stem(src, aaf_root)
        xml_out = out_root / f"{stem}.__rt.xml"
        aaf_out = out_root / f"{stem}.__rt.aaf"
        for p in (xml_out, aaf_out):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        t_file = time.perf_counter()
        try:
            res = sdk_roundtrip(
                src,
                aaf_out,
                aaf_tools_dir=tools_dir,
                strip_this_namespace=True,
                xml_sibling=xml_out,
                on_xml_crash="binary_copy",
            )
            dt = time.perf_counter() - t_file
            sz = aaf_out.stat().st_size if aaf_out.is_file() else 0
            if res.ok:
                line = (
                    f"OK [{i}/{len(jobs)}] {src.name} :: {dt:.1f}s :: "
                    f"{res.method.value} :: {sz} B :: {res.message}"
                )
            else:
                errors += 1
                line = f"FAIL [{i}/{len(jobs)}] {src} :: {res.method.value} :: {res.message}"
            lines.append(line)
            print(line, flush=True)
        except Exception as e:
            errors += 1
            dt = time.perf_counter() - t_file
            tb = traceback.format_exc()
            lines.append(f"FAIL [{i}/{len(jobs)}] {src}")
            lines.append(f"  {type(e).__name__}: {e} ({dt:.1f}s)")
            lines.append(tb[-4000:] if len(tb) > 4000 else tb)
            print(lines[-3], flush=True)

    total = time.perf_counter() - t0
    lines.append(f"TOTAL {len(jobs)} files, {errors} failures, {total:.1f}s wall")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8", errors="replace")
    print(log_path)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
