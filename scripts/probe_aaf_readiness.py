#!/usr/bin/env python3
"""
Пакетная проверка AAF: открытие через aaf_io.open_aaf_lenient (как в пайплайне),
подсчёт CompositionMob, звуковых слотов с Sequence, таймлайновых SourceClip.

Референсные AAF (ручные прогоны):
  локальные большие AAF (Nuendo/SDK-стиль) + тяжёлые кейсы Premiere. Дефолтные каталоги — ``DEFAULT_PROBE_AAF_DIRS``.

Использование:
  python scripts/probe_aaf_readiness.py
  python scripts/probe_aaf_readiness.py C:\\path\\to\\AAF ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Дефолтные каталоги для ``main()`` (без аргументов): референс + тесты LibAAF.
DEFAULT_PROBE_AAF_DIRS: tuple[Path, ...] = ()


def _toolkit_roots() -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    toolkit = here.parent
    aaf_io = toolkit / "aaf_io"
    aaf_speech = toolkit / "aaf_speech_filter"
    return aaf_io, aaf_speech


def _ensure_path() -> None:
    aaf_io, aaf_speech = _toolkit_roots()
    for p in (aaf_io, aaf_speech):
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)


def _slot_is_soundish(slot) -> bool:
    mk = getattr(slot, "media_kind", None)
    if mk is None:
        return True
    s = str(mk).lower()
    if "picture" in s or "timecode" in s:
        return False
    return True


def _pick_main_composition(comp_mobs: list) -> object:
    best = comp_mobs[0]
    best_n = -1
    for comp in comp_mobs:
        n_seq = 0
        for slot in getattr(comp, "slots", []) or []:
            if not _slot_is_soundish(slot):
                continue
            seg = getattr(slot, "segment", None)
            if seg is not None and "Sequence" in seg.__class__.__name__:
                n_seq += 1
        if n_seq > best_n:
            best_n = n_seq
            best = comp
    return best


def _count_seq_sound_slots(comp) -> int:
    n = 0
    for slot in getattr(comp, "slots", []) or []:
        if not _slot_is_soundish(slot):
            continue
        seg = getattr(slot, "segment", None)
        if seg is not None and "Sequence" in seg.__class__.__name__:
            n += 1
    return n


def _rational_to_float(r) -> float | None:
    try:
        return float(r)
    except Exception:
        return None


def _slot_edit_rate(slot) -> float | None:
    try:
        return _rational_to_float(slot["EditRate"].value)
    except Exception:
        return None


def _slot_is_soundish(slot) -> bool:
    mk = getattr(slot, "media_kind", None)
    if mk is None:
        return True
    s = str(mk).lower()
    if "picture" in s or "timecode" in s:
        return False
    return True


def _vectors_on_container(container):
    for attr in ("components", "segments"):
        vec = getattr(container, attr, None)
        if vec is not None:
            try:
                if len(vec) > 0:
                    yield vec
            except Exception:
                pass
    try:
        ins = container["InputSegments"]
        if ins is not None and len(ins) > 0:
            yield ins
    except Exception:
        pass


def _is_sourceclip(obj) -> bool:
    if obj is None:
        return False
    n = obj.__class__.__name__
    return n == "SourceClip" or n.endswith("SourceClip")


def _iter_timeline_sourceclips_same_as_filter(aaf):
    """Тот же обход, что ``pyaaf2_filter._iter_timeline_sourceclips`` (без импорта tensorflow)."""
    compositions = list(aaf.content.compositionmobs())

    def walk(container, edit_rate):
        if container is None:
            return
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
                    yield edit_rate, node
                else:
                    yield from walk(node, edit_rate)
                i += 1

    for comp in compositions:
        for slot in getattr(comp, "slots", []) or []:
            if not _slot_is_soundish(slot):
                continue
            er = float(_slot_edit_rate(slot) or 48000.0)
            seg = getattr(slot, "segment", None)
            if seg is None:
                continue
            yield from walk(seg, er)


def probe_one(aaf_path: Path) -> tuple[str, str, dict[str, object]]:
    _ensure_path()
    from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient

    info: dict[str, object] = {}
    try:
        with open_aaf_lenient(aaf_path, "r") as aaf:
            comps = list(aaf.content.compositionmobs())
            info["composition_mobs"] = len(comps)
            if not comps:
                return "OK", "no CompositionMob", info
            comp = _pick_main_composition(comps)
            info["seq_sound_slots"] = _count_seq_sound_slots(comp)
            info["timeline_sourceclips"] = sum(
                1 for _ in _iter_timeline_sourceclips_same_as_filter(aaf)
            )
        return "OK", "", info
    except Exception as e:
        return "FAIL", f"{type(e).__name__}: {e}", info


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe all .aaf in given directories.")
    ap.add_argument(
        "dirs",
        nargs="*",
        default=[str(p) for p in DEFAULT_PROBE_AAF_DIRS],
        help="Directories to scan (default: none; pass directories explicitly).",
    )
    args = ap.parse_args()
    _ensure_path()

    any_fail = False
    for d in args.dirs:
        root = Path(d)
        if not root.is_dir():
            print(f"SKIP (not a dir): {root}")
            continue
        files = sorted(root.glob("*.aaf"))
        print(f"\n=== {root} ({len(files)} files) ===")
        for p in files:
            status, err, info = probe_one(p)
            lane_ok = int(info.get("seq_sound_slots") or 0) >= 2
            extra = ""
            if status == "OK":
                extra = (
                    f" comps={info.get('composition_mobs')} seq_sound={info.get('seq_sound_slots')} "
                    f"clips={info.get('timeline_sourceclips')} lane_eligible={lane_ok}"
                )
            else:
                any_fail = True
            print(f"{status:4} {p.name}{extra}")
            if err:
                print(f"     {err}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
