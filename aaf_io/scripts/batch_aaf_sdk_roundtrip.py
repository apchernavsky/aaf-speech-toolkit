#!/usr/bin/env python3
"""
AAF SDK roundtrip: aaffmtconv -xml → -ss для набора файлов.
Список: все *.aaf под LibAAF test/aaf, все имена из test.py (test/extract),
опционально дополнительные AAF (если переданы явно).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

# пакет aaf_io: родитель scripts → корень проекта aaf_io
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aaf_io.roundtrip import sdk_roundtrip


def _names_from_libaaf_test_py(test_py: Path) -> set[str]:
    text = test_py.read_text(encoding="utf-8", errors="replace")
    names: set[str] = set()
    for m in re.finditer(r"\btest\s*\(\s*\"([^\"]+\.aaf)\"", text):
        names.add(m.group(1))
    for m in re.finditer(r"\bextract\s*\(\s*\"([^\"]+\.aaf)\"", text):
        names.add(m.group(1))
    return names


def _collect_aaf_paths(aaf_root: Path) -> list[Path]:
    if not aaf_root.is_dir():
        return []
    return sorted({p.resolve() for p in aaf_root.rglob("*.aaf") if p.is_file()})


def _union_jobs(
    aaf_root: Path,
    test_py: Path,
    extra_aaf: list[Path],
) -> tuple[list[Path], list[tuple[str, str]]]:
    """Возвращает (список_существующих_aaf, список_(имя_из_таблицы, причина_пропуска))."""
    on_disk = _collect_aaf_paths(aaf_root)
    on_disk_set = {p.resolve() for p in on_disk}

    table = _names_from_libaaf_test_py(test_py)
    missing: list[tuple[str, str]] = []
    for name in sorted(table):
        cand = (aaf_root / name).resolve()
        if cand.is_file():
            if cand not in on_disk_set:
                on_disk.append(cand)
                on_disk_set.add(cand)
        else:
            found = [p.resolve() for p in aaf_root.rglob(name) if p.is_file()]
            if found:
                p0 = found[0]
                if p0 not in on_disk_set:
                    on_disk.append(p0)
                    on_disk_set.add(p0)
            else:
                missing.append((name, "нет файла под test/aaf"))

    for p in extra_aaf:
        p = p.resolve()
        if p.is_file() and p not in on_disk_set:
            on_disk.append(p)
            on_disk_set.add(p)

    on_disk.sort(key=lambda x: str(x).lower())
    return on_disk, missing


def _safe_job_stem(path: Path, aaf_root: Path) -> str:
    try:
        rel = path.relative_to(aaf_root.resolve())
        s = str(rel).replace("\\", "_").replace("/", "_")
    except ValueError:
        s = path.parent.name + "_" + path.name if path.parent != path.anchor else path.name
    for ch in '<>:"|?*':
        s = s.replace(ch, "_")
    return s[:180] if len(s) > 180 else s


def run_batch(
    aaf_tools_dir: Path,
    aaf_root: Path,
    test_py: Path,
    out_root: Path,
    extra: Iterable[Path],
) -> int:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    extra_list = [Path(x) for x in extra]
    jobs, missing = _union_jobs(aaf_root, test_py, extra_list)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "roundtrip_log.txt"
    lines: list[str] = []
    lines.append(f"aaf_tools_dir: {aaf_tools_dir}")
    lines.append(f"aaf_root: {aaf_root}")
    lines.append(f"jobs: {len(jobs)}, table_missing: {len(missing)}")
    for name, why in missing:
        lines.append(f"MISSING_TABLE {name} :: {why}")

    errors = 0
    t0 = time.perf_counter()
    for i, src in enumerate(jobs, 1):
        stem = _safe_job_stem(src, aaf_root)
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
                aaf_tools_dir=aaf_tools_dir,
                strip_this_namespace=True,
                xml_sibling=xml_out,
                on_xml_crash="binary_copy",
            )
            dt = time.perf_counter() - t_file
            sz = aaf_out.stat().st_size if aaf_out.is_file() else 0
            if res.ok:
                lines.append(
                    f"OK [{i}/{len(jobs)}] {src.name} :: {dt:.1f}s :: {res.method.value} :: {sz} B :: {res.message}"
                )
            else:
                errors += 1
                lines.append(f"FAIL [{i}/{len(jobs)}] {src} :: {res.method.value} :: {res.message}")
        except Exception as e:
            errors += 1
            dt = time.perf_counter() - t_file
            tb = traceback.format_exc()
            lines.append(f"FAIL [{i}/{len(jobs)}] {src}")
            lines.append(f"  {type(e).__name__}: {e} ({dt:.1f}s)")
            lines.append(tb[-4000:] if len(tb) > 4000 else tb)
        print(lines[-1], flush=True)

    total = time.perf_counter() - t0
    lines.append(f"TOTAL {len(jobs)} files, {errors} failures, {total:.1f}s wall")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8", errors="replace")
    print(log_path)
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch AAF SDK XML roundtrip.")
    ap.add_argument(
        "--aaf-tools",
        type=Path,
        default=None,
        help="Каталог с aaffmtconv.exe",
    )
    ap.add_argument(
        "--aaf-root",
        type=Path,
        default=None,
        help="Корень LibAAF test/aaf",
    )
    ap.add_argument(
        "--test-py",
        type=Path,
        default=None,
        help="test.py со списком test(...)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("roundtrip_batch_out"),
        help="Каталог для XML/AAF и лога",
    )
    ap.add_argument(
        "--extra",
        type=Path,
        nargs="*",
        default=[],
        help="Дополнительные AAF (если переданы).",
    )
    args = ap.parse_args()
    if args.aaf_tools is None or args.aaf_root is None or args.test_py is None:
        ap.error("Provide --aaf-tools, --aaf-root and --test-py explicitly.")
    extras = [p for p in (args.extra or []) if p is not None]
    return run_batch(
        args.aaf_tools.resolve(),
        args.aaf_root.resolve(),
        args.test_py.resolve(),
        args.out.resolve(),
        extras,
    )


if __name__ == "__main__":
    raise SystemExit(main())
