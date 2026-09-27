from __future__ import annotations

import threading
from fractions import Fraction
from pathlib import Path
from typing import Callable, Optional

from aaf_io.composition import CompositionSelectionError, select_aaf_composition
from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient

from .aaf_mutation import (
    replace_block_with_filler,
    replace_first_sourceclip_in_block_with_filler,
)
from .render_semantics import block_render_signature, operation_render_signature
from .config import FilterConfig
from .exceptions import SpeechFilterCancelled
from .sdk_removals import sourceclip_sdk_key_tuple
from .timeline_walk import (
    _inner_sourceclip_for_block,
    _is_filler,
    _node_length_units,
    _slot_edit_rate,
    _slot_is_soundish,
)


def duplicate_timeline_key(
    sourceclip_key: tuple[str, int, int, int],
    *,
    timeline_pos_units: int,
    slot_edit_rate: float,
) -> tuple:
    """
    Key for *timeline duplicates*.

    A duplicate is the same source content with the same source range and length
    stacked at the same timeline position. Reusing the same source clip later in
    the edit is intentional montage, not a duplicate.
    """
    src, length, start, track = sourceclip_key
    rate = Fraction(str(slot_edit_rate))
    if rate <= 0:
        raise ValueError("Duplicate comparison requires a positive edit rate")
    return (str(src), int(start), int(length), int(track),
            Fraction(int(timeline_pos_units), 1) / rate, rate)



def remove_duplicate_timeline_blocks_inplace(
    aaf_path: Path,
    *,
    cfg: FilterConfig,
    log_callback: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    dry_run: bool = False,
) -> int:
    """
    Remove duplicate timeline blocks by SourceClip identity key.

    This operates in-place on ``aaf_path`` unless ``dry_run`` is true.
    """
    if not bool(cfg.remove_duplicates):
        return 0

    def _check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise SpeechFilterCancelled()

    removed = 0
    seen: set[tuple] = set()
    with open_aaf_lenient(Path(aaf_path), "r" if dry_run else "r+") as aaf:
        comps = list(aaf.content.compositionmobs())
        if not comps:
            return 0
        try:
            comp = select_aaf_composition(comps)
        except CompositionSelectionError as exc:
            if log_callback is not None:
                log_callback(f"Duplicate removal skipped: {exc}")
            return 0
        for slot in getattr(comp, "slots", []) or []:
            _check_cancel()
            if not _slot_is_soundish(slot):
                continue
            er = Fraction(str(slot.edit_rate))
            if er <= 0:
                continue
            seg = getattr(slot, "segment", None)
            if seg is None:
                continue
            seq = None
            name = seg.__class__.__name__
            if "Sequence" in name:
                seq = seg
            elif "OperationGroup" in name:
                try:
                    inner = getattr(seg, "segments", None)
                    if inner is not None and len(inner) >= 1 and "Sequence" in inner[0].__class__.__name__:
                        seq = inner[0]
                except Exception:
                    seq = None
            if seq is None:
                continue
            track_effect = () if seg is seq else operation_render_signature(seg)
            if track_effect is None:
                continue
            try:
                top = seq.components
            except Exception:
                continue
            # PyAAF2 Sequence.positions() subtracts transition overlap from the cursor.
            for i, pos, node in seq.positions():
                _check_cancel()
                if 'Transition' in node.__class__.__name__ or _is_filler(node):
                    continue
                if any(type(top[j]).__name__ == "Transition" for j in (i - 1, i + 1) if 0 <= j < len(top)):
                    continue
                ln = _node_length_units(node)
                inner_sc = _inner_sourceclip_for_block(node)
                if inner_sc is None:
                    continue
                k = sourceclip_sdk_key_tuple(inner_sc)
                if k is None:
                    continue
                rendering = block_render_signature(node, inner_sc)
                if rendering is None:
                    continue
                key = (duplicate_timeline_key(k, timeline_pos_units=pos - int(slot.origin), slot_edit_rate=er),
                       (track_effect, pos) if track_effect else (), rendering)
                if key in seen:
                    if not dry_run:
                        if not replace_first_sourceclip_in_block_with_filler(node, aaf, inner_sc):
                            replace_block_with_filler(top, i, aaf, ln, inner_sc)
                    removed += 1
                else:
                    seen.add(key)

    if log_callback is not None:
        try:
            if removed > 0:
                log_callback(
                    f"Дубли: удалено {removed} клипов(блоков) по ключу SourceID/Start/Length/SlotID + позиция таймлайна."
                )
            else:
                log_callback("Дубли: не найдено.")
        except Exception:
            pass
    return int(removed)


__all__ = ["duplicate_timeline_key", "remove_duplicate_timeline_blocks_inplace"]
