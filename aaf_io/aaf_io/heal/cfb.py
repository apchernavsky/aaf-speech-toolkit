"""
Fixups for typical CFB/OLE inconsistencies inside AAF containers.

Nuendo and some other hosts sometimes write a ``fat_sector_count`` in the OLE header
that does not match the DIFAT/FAT chain. PyAAF2 corrects it in memory; ``olefile`` and
AAF SDK tools may fail on the original bytes.

This module only performs a conservative fix: write **4 bytes at offset 44**
(little-endian ``fat_sector_count``), aligned with what PyAAF2 derived from the chain.
"""

from __future__ import annotations

import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import aaf2


CFB_HEADER_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


@dataclass(frozen=True)
class CfbBookkeepingRestoreResult:
    applied: bool
    restored_ranges: int = 0
    restored_bytes: int = 0
    skipped_ranges: int = 0
    truncated_to_source_size: bool = False
    reason: str = ""


def read_fat_sector_count_on_disk(path: Path) -> int:
    with open(path, "rb") as f:
        f.seek(44)
        return struct.unpack("<I", f.read(4))[0]


def _is_cfb_container(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(CFB_HEADER_SIGNATURE)) == CFB_HEADER_SIGNATURE
    except OSError:
        return False


def compute_fat_sector_count_from_chain(path: Path, *, sector_size: int = 4096) -> int:
    """Value PyAAF2 converges to after reading DIFAT/FAT."""
    with aaf2.open(str(path), "rb", sector_size=sector_size) as a:
        return int(a.cfb.fat_sector_count)


def sync_fat_sector_count_header(path: Path, *, sector_size: int = 4096) -> bool:
    """
    Write corrected ``fat_sector_count`` to the header if it differs from the chain.

    Returns ``True`` when the header bytes were modified.
    """
    path = Path(path)
    if not _is_cfb_container(path):
        return False
    correct = compute_fat_sector_count_from_chain(path, sector_size=sector_size)
    with open(path, "r+b") as f:
        f.seek(44)
        cur = struct.unpack("<I", f.read(4))[0]
        if cur == correct:
            return False
        f.seek(44)
        f.write(struct.pack("<I", correct))
    return True


def copy_and_sync_fat_header(src: Path, dst: Path, *, sector_size: int = 4096) -> bool:
    """
    Copy an AAF and align ``fat_sector_count`` in the copy.

    Returns ``True`` when the header in the copy was modified.
    """
    import shutil

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return sync_fat_sector_count_header(dst, sector_size=sector_size)


def _diff_ranges(
    left: Path,
    right: Path,
    *,
    compare_size: int,
    max_ranges: int,
    max_changed_bytes: int,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    changed = 0
    pos = 0
    current_start: int | None = None
    current_end = -1
    with open(left, "rb") as fa, open(right, "rb") as fb:
        remaining = int(compare_size)
        while remaining > 0:
            chunk_len = min(1024 * 1024, remaining)
            a = fa.read(chunk_len)
            b = fb.read(chunk_len)
            n = min(len(a), len(b), chunk_len)
            if n <= 0:
                break
            for i in range(n):
                if a[i] == b[i]:
                    continue
                off = pos + i
                changed += 1
                if changed > max_changed_bytes:
                    raise RuntimeError("CFB bookkeeping diff is too large")
                if current_start is None:
                    current_start = off
                    current_end = off + 1
                elif off == current_end:
                    current_end = off + 1
                else:
                    ranges.append((current_start, current_end - current_start))
                    if len(ranges) > max_ranges:
                        raise RuntimeError("CFB bookkeeping diff has too many ranges")
                    current_start = off
                    current_end = off + 1
            pos += n
            remaining -= n
    if current_start is not None:
        ranges.append((current_start, current_end - current_start))
    if len(ranges) > max_ranges:
        raise RuntimeError("CFB bookkeeping diff has too many ranges")
    return ranges


def _read_exact_at(handle, offset: int, size: int) -> bytes:
    handle.seek(int(offset))
    data = handle.read(int(size))
    if len(data) != int(size):
        raise EOFError("short read")
    return data


def _range_equals(path: Path, offset: int, expected: bytes) -> bool:
    try:
        with open(path, "rb") as f:
            return _read_exact_at(f, offset, len(expected)) == expected
    except Exception:
        return False


def _tail_matches(path_a: Path, path_b: Path, start: int, end: int) -> bool:
    if end <= start:
        return True
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        fa.seek(start)
        fb.seek(start)
        remaining = end - start
        while remaining > 0:
            n = min(1024 * 1024, remaining)
            a = fa.read(n)
            b = fb.read(n)
            if a != b:
                return False
            if not a:
                return False
            remaining -= len(a)
    return True


def restore_pyAAF2_noop_bookkeeping_from_source(
    source_aaf: Path,
    output_aaf: Path,
    *,
    work_dir: Path,
    max_ranges: int = 4096,
    max_changed_bytes: int = 1024 * 1024,
) -> CfbBookkeepingRestoreResult:
    """
    Restore CFB/OLE bookkeeping bytes that PyAAF2 changes on a no-op save.

    The repair is deliberately guarded:
    1. Make a temporary no-op PyAAF2 resave of ``source_aaf``.
    2. Diff ``source_aaf`` vs that no-op copy.
    3. Copy source bytes into ``output_aaf`` only for ranges where ``output_aaf``
       still exactly matches the no-op copy. If output differs, the range may
       contain real timeline/object changes and is skipped.
    4. Truncate to source size only when the output tail equals the no-op tail.

    This preserves logical AAF edits while undoing PyAAF2's container-level
    bookkeeping rewrite for strict hosts that reject it.
    """
    source_aaf = Path(source_aaf)
    output_aaf = Path(output_aaf)
    work_dir = Path(work_dir)
    if not source_aaf.is_file() or not output_aaf.is_file():
        return CfbBookkeepingRestoreResult(False, reason="missing source/output")
    try:
        if source_aaf.resolve() == output_aaf.resolve():
            return CfbBookkeepingRestoreResult(False, reason="source equals output")
    except Exception:
        if str(source_aaf) == str(output_aaf):
            return CfbBookkeepingRestoreResult(False, reason="source equals output")

    work_dir.mkdir(parents=True, exist_ok=True)
    noop = work_dir / f"{output_aaf.name}.__pyaaf2_noop_save.aaf"
    try:
        noop.unlink(missing_ok=True)
    except Exception:
        pass

    try:
        shutil.copyfile(str(source_aaf), str(noop))
        from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient

        with open_aaf_lenient(noop, "r+") as aaf:
            _ = getattr(aaf, "content", None)

        source_size = int(source_aaf.stat().st_size)
        output_size = int(output_aaf.stat().st_size)
        noop_size = int(noop.stat().st_size)
        compare_size = min(source_size, noop_size)
        ranges = _diff_ranges(
            source_aaf,
            noop,
            compare_size=compare_size,
            max_ranges=max_ranges,
            max_changed_bytes=max_changed_bytes,
        )
        if not ranges and output_size == source_size:
            return CfbBookkeepingRestoreResult(False, reason="no noop bookkeeping diff")

        restored_ranges = 0
        restored_bytes = 0
        skipped_ranges = 0
        with open(source_aaf, "rb") as fs, open(noop, "rb") as fn, open(output_aaf, "r+b") as fo:
            for offset, size in ranges:
                source_bytes = _read_exact_at(fs, offset, size)
                noop_bytes = _read_exact_at(fn, offset, size)
                if not _range_equals(output_aaf, offset, noop_bytes):
                    skipped_ranges += 1
                    continue
                fo.seek(offset)
                fo.write(source_bytes)
                restored_ranges += 1
                restored_bytes += int(size)

        truncated = False
        if output_size > source_size and noop_size == output_size:
            if _tail_matches(output_aaf, noop, source_size, output_size):
                with open(output_aaf, "r+b") as fo:
                    fo.truncate(source_size)
                truncated = True

        applied = bool(restored_ranges > 0 or truncated)
        return CfbBookkeepingRestoreResult(
            applied,
            restored_ranges=restored_ranges,
            restored_bytes=restored_bytes,
            skipped_ranges=skipped_ranges,
            truncated_to_source_size=truncated,
            reason="ok" if applied else "no safe ranges matched output",
        )
    finally:
        try:
            noop.unlink(missing_ok=True)
        except Exception:
            pass
