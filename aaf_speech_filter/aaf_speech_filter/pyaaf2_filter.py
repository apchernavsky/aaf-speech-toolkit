from __future__ import annotations

import importlib.util
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import aaf2


def _ensure_aaf_io_on_path() -> None:
    if importlib.util.find_spec("aaf_io") is not None:
        return
    here = Path(__file__).resolve()
    for anc in here.parents:
        root = anc / "aaf_io"
        if (root / "aaf_io" / "__init__.py").is_file():
            s = str(root)
            if s not in sys.path:
                sys.path.insert(0, s)
            return


_ensure_aaf_io_on_path()
from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient

from aaf_io.path_safety import require_distinct_files
from aaf_io.output_transaction import output_candidate

from .timeline_walk import _aaf_int_length, _slot_is_soundish

from .progress import (
    progress_set_global,
    progress_stage_wrap,
)
from .config import FilterConfig
from .exceptions import SpeechFilterCancelled
from .aaf_mutation import (
    replace_sourceclip_with_filler as _replace_sourceclip_with_filler,
)
from .duplicate_filter import remove_duplicate_timeline_blocks_inplace
from .sdk_removals import (
    collect_timeline_sourceclip_removals_for_sdk_xml as _collect_quiet_removal_keys,
    sourceclip_occurrence_key,
)
from .timeline_sourceclips import (
    iter_slot_sourceclip_occurrences,
    total_timeline_sourceclips as _total_timeline_sourceclips,
)

logger = logging.getLogger(__name__)

_COOP_YIELD_INTERVAL_SEC = 5.0


def _output_aaf_lightly_readable(path: Path) -> bool:
    """
    Проверка выхода без ``len(mobs)``/``metadict`` — на больших Nuendo-AAF PyAAF2 там часто падает,
    хотя файл открывается в Nuendo. Достаточно успешного открытия и доступа к ``content``.
    """
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        with aaf2.open(str(path), "r") as a:
            try:
                _ = getattr(a, "content", None)
            except Exception:
                pass
            try:
                _ = a.metadict
            except Exception:
                pass
        return True
    except Exception:
        return False


def filter_aaf_speech_only(
    input_aaf: Path,
    output_aaf: Path,
    cfg: FilterConfig,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    runtime_essence_paths: Optional[Dict[Any, Path]] = None,
    work_dir: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    lane_layout_result_out: Optional[dict[str, Any]] = None,
) -> int:
    """Filter a private candidate and atomically publish a validated, closed AAF."""
    def check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise SpeechFilterCancelled()

    with output_candidate(input_aaf, output_aaf, cancel_check=check_cancel) as candidate:
        removed = _filter_aaf_candidate(
            input_aaf, candidate, cfg, progress_callback=progress_callback,
            cancel_event=cancel_event, runtime_essence_paths=runtime_essence_paths,
            work_dir=work_dir, log_callback=log_callback,
            lane_layout_result_out=lane_layout_result_out,
        )
    return removed


def _filter_aaf_candidate(
    input_aaf: Path,
    output_aaf: Path,
    cfg: FilterConfig,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    runtime_essence_paths: Optional[Dict[Any, Path]] = None,
    work_dir: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    lane_layout_result_out: Optional[dict[str, Any]] = None,
) -> int:
    """
    Копия ``input_aaf`` в ``output_aaf`` с опциональными правками.

    Порядок: **удаление тихих** (если включено) → **раскладка по дорожкам** (если включена).
    """
    require_distinct_files(input_aaf, output_aaf)
    output_aaf.parent.mkdir(parents=True, exist_ok=True)

    if log_callback is not None:
        try:
            log_callback("PyAAF2: старт обработки AAF…")
        except Exception:
            pass

    def _check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise SpeechFilterCancelled()
        nonlocal _last_yield_t
        now = time.monotonic()
        if now - _last_yield_t >= _COOP_YIELD_INTERVAL_SEC:
            _last_yield_t = now
            time.sleep(0)

    # Copy file first to preserve any metadata/layout Nuendo is picky about.
    #
    # IMPORTANT (Nuendo + huge embedded AAF):
    # A single long-lived `r+` session with thousands of object edits sometimes produces
    # an AAF that Nuendo won't open, even though smaller projects work.
    # Empirically, reopening the file between timeline slots reduces the likelihood of
    # a bad compound rewrite in PyAAF2 for "huge" sessions.
    import shutil

    _last_yield_t = time.monotonic()
    _check_cancel()
    shutil.copyfile(str(input_aaf), str(output_aaf))

    if log_callback is not None:
        try:
            if bool(cfg.remove_quiet_clips):
                log_callback("PyAAF2: удаление тихих клипов — начало…")
            else:
                log_callback("PyAAF2: удаление тихих клипов — пропуск (опция выключена).")
        except Exception:
            pass

    def _iter_comp_slot_keys(aaf) -> list[tuple[Any, int]]:
        keys: list[tuple[Any, int]] = []
        for comp in list(aaf.content.compositionmobs()):
            comp_id = getattr(comp, "mob_id", None)
            if comp_id is None:
                continue
            for slot in getattr(comp, "slots", []) or []:
                sid = getattr(slot, "slot_id", None)
                if sid is None:
                    try:
                        sid = int(slot["SlotID"].value)
                    except Exception:
                        sid = None
                if sid is None:
                    continue
                keys.append((comp_id, int(sid)))
        return keys

    def _find_comp_slot(aaf, comp_id, slot_id: int):
        for comp in list(aaf.content.compositionmobs()):
            if getattr(comp, "mob_id", None) != comp_id:
                continue
            for slot in getattr(comp, "slots", []) or []:
                sid = getattr(slot, "slot_id", None)
                if sid is None:
                    try:
                        sid = int(slot["SlotID"].value)
                    except Exception:
                        sid = None
                if sid is None or int(sid) != int(slot_id):
                    continue
                return comp, slot
        return None, None

    try:
        w_layout = 0.15 if cfg.experimental_yamnet_lane_layout else 0.0
        w_removals = 1.0 - w_layout
        clip_cb = progress_stage_wrap(progress_callback, 0.0, w_removals)

        # First, do a lightweight read pass to gather slot keys and estimate total work.
        with open_aaf_lenient(output_aaf, "r") as aaf_ro:
            compositions = list(aaf_ro.content.compositionmobs())
            if not compositions:
                raise RuntimeError("CompositionMob not found")
            comp_slot_keys = _iter_comp_slot_keys(aaf_ro)
            total_sec = float(max(1, int(_total_timeline_sourceclips(aaf_ro))))

        prog_state = {"processed": 0.0}
        removed_clips = 0
        removed_quiet_n = 0
        quiet_removal_keys = set()
        if bool(cfg.remove_quiet_clips):
            quiet_removal_keys = _collect_quiet_removal_keys(
                output_aaf,
                cfg,
                runtime_essence_paths=runtime_essence_paths,
                work_dir=work_dir,
                cancel_event=cancel_event,
                progress_callback=None,
                log_callback=log_callback,
            )

        def _note_clip_done(dur_sec: float) -> None:
            if clip_cb is None:
                return
            # Progress is in "clips processed" units to avoid an expensive duration pre-pass.
            prog_state["processed"] = min(total_sec, prog_state["processed"] + 1.0)
            clip_cb(prog_state["processed"], total_sec)

        # Patch slot-by-slot with reopen in between to reduce PyAAF2 rewrite fragility
        # on very large embedded Nuendo sessions.
        for comp_id, slot_id in comp_slot_keys:
            _check_cancel()
            with open_aaf_lenient(output_aaf, "r+") as aaf:
                _check_cancel()
                _comp, slot = _find_comp_slot(aaf, comp_id, slot_id)
                if slot is None:
                    continue
                if not _slot_is_soundish(slot):
                    continue
                seg = getattr(slot, "segment", None)
                if seg is None:
                    continue

                # Snapshot references before mutation; addresses are from the original tree.
                occurrences = list(iter_slot_sourceclip_occurrences(_comp, slot))
                for occurrence in occurrences:
                    _check_cancel()
                    if quiet_removal_keys and sourceclip_occurrence_key(occurrence) in quiet_removal_keys:
                        node = occurrence.sourceclip
                        length = _aaf_int_length(node)
                        if occurrence.parent_vector is None:
                            filler = aaf.create.Filler(length=length, media_kind='sound')
                            filler['DataDefinition'].value = node['DataDefinition'].value
                            slot.segment = filler
                        else:
                            _replace_sourceclip_with_filler(
                                occurrence.parent_vector, occurrence.index, aaf, length, node
                            )
                        removed_clips += 1
                        removed_quiet_n += 1
                    _note_clip_done(1.0)

        progress_set_global(progress_callback, w_removals)

        if log_callback is not None:
            try:
                if removed_quiet_n > 0:
                    log_callback(f"Удалено {removed_quiet_n} клипа (удаление тихих клипов).")
            except Exception:
                pass

        # After removals, optionally drop exact duplicates before lane layout.
        if cfg.remove_duplicates:
            if log_callback is not None:
                log_callback("Duplicates: removing exact timeline duplicates.")
            removed_clips += remove_duplicate_timeline_blocks_inplace(
                output_aaf,
                cfg=cfg,
                log_callback=log_callback,
                cancel_event=cancel_event,
            )

        if cfg.experimental_yamnet_lane_layout:
            from .aaf_yamnet_lane_layout import apply_experimental_yamnet_lane_layout

            layout_cb = progress_stage_wrap(progress_callback, w_removals, w_layout)
            if log_callback is not None:
                try:
                    log_callback("YAMNet дорожки: запуск раскладки (PyAAF2 режим).")
                except Exception:
                    pass
            moved_out = apply_experimental_yamnet_lane_layout(
                output_aaf,
                cfg,
                runtime_essence_paths=runtime_essence_paths,
                cancel_event=cancel_event,
                progress_callback=layout_cb,
                log_callback=log_callback,
                result_out=lane_layout_result_out,
            )
            if log_callback is not None:
                try:
                    log_callback(
                        f"YAMNet дорожки: раскладка завершена (перенесено={int(moved_out or 0)})."
                    )
                except Exception:
                    pass

        progress_set_global(progress_callback, 1.0)
    except SpeechFilterCancelled:
        try:
            output_aaf.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    if log_callback is not None:
        try:
            log_callback("PyAAF2: основной проход завершён.")
        except Exception:
            pass

    if not _output_aaf_lightly_readable(output_aaf):
        raise RuntimeError("Processed AAF candidate could not be read; output was not published.")

    return removed_clips
