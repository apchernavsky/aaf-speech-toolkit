"""
YAMNet: классификация клипов (речь / шум / музыка) и раскладка по звуковым дорожкам Nuendo.

Логика (ориентир — вручную ``*_sorted.aaf``):
  • **Речь** — на минимальный индекс дорожки, где интервал ``[T, T+L)`` не пересекается с уже
    размещёнными клипами (любого типа).
  • **Шум** — на дорожку с минимальным ``|i − (n−1)/2|`` при отсутствии пересечений.
  • **Музыка** — вниз: перебор дорожек в порядке ``n−2, n−1, n−3, …`` (типичный Nuendo: основной
    низ на предпоследней дорожке, нижняя под стерео/дубли).

Порядок назначения: **шум → речь → музыка**. Классификация: YAMNet с чуть более строгим порогом речи
(``_lane_yamnet_cfg``), очень короткие «речи» (<~0,55 с) считаются музыкой для раскладки,
низкоуверенные равные softmax (s≈n) — шумом.
Если WAV недоступен (``unknown``), блоки в начале таймлайна можно перевести в ``music`` см.
``lane_layout_unknown_opening_music_sec`` (типичный случай: Premiere + нечитаемый embedded essence).

Реализация: только верхний уровень ``Sequence`` (``Filler`` / ``OperationGroup`` / ``SourceClip``).
Сохраняются исходные ``T``, ``L``; блоки ``pop`` с дорожек и пересобираются в ``components.value``.
Пустая целевая дорожка после переноса — один ``Filler`` на **ненулевую** длину (как у исходного
слота или по максимуму композиции): ``Filler(0)`` как единственный компонент часто ломает Nuendo/SDK.

Each timeline occurrence keeps its own edit coordinate. Reused source cuts do not
imply that their occurrences should be aligned.

См. ``aaf_pipeline``: раскладка после SDK на ``*_processed.aaf``.
"""

from __future__ import annotations

from aaf_io.diagnostic_log import DiagnosticMessage
from aaf_io.lane_order import class_ordered_lanes, reorder_pyaaf2_slots
from aaf_io.lane_assignment import ordered_lane_assignment

from .lane_layout_constraints import apply_immutable_aligned_lane_bounds

import logging
import threading
import copy
from collections import defaultdict
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import FilterConfig
from .exceptions import SpeechFilterCancelled
from .aaf_mutation import datadef_for_filler_fallback as _datadef_for_filler_fallback
from .lane_layout_model import (
    event_owned_raw_transition_spans as _model_event_owned_raw_transition_spans,
    event_owned_transition_history_spans as _model_event_owned_transition_history_spans,
    event_raw_span_with_transitions as _model_event_raw_span_with_transitions,
    event_visible_span as _model_event_visible_span,
    event_visible_start as _model_event_visible_start,
    events_share_same_kind_transition_edge as _model_events_share_same_kind_transition_edge,
    planned_overlap_is_allowed_transition_overlap as _model_planned_overlap_is_allowed_transition_overlap,
)
from .lane_layout_plan import (
    planned_event_raw_t_for_lane as _model_planned_event_raw_t_for_lane,
    repair_lane_layout_plan_overlaps as _model_repair_lane_layout_plan_overlaps,
    validate_lane_layout_plan as _model_validate_lane_layout_plan,
)
from .lane_layout_groups import (
    aligned_kind_group_key as _model_aligned_kind_group_key,
    aligned_layout_group_key as _model_aligned_layout_group_key,
    event_source_window_key as _model_event_source_window_key,
)
from .lane_layout_initial_placement import LaneInitialPlacementPlanner as _ModelLaneInitialPlacementPlanner
from .lane_layout_candidates import physical_class_band as _model_physical_class_band
from .lane_layout_constraints import (
    target_lane_allowed_for_event as _model_target_lane_allowed_for_event,
)
from .lane_layout_compaction import (
    compact_events_to_preferred_lanes as _model_compact_events_to_preferred_lanes,
)
from .lane_layout_aligned_compaction import (
    compact_aligned_groups_to_visible_gaps as _model_compact_aligned_groups_to_visible_gaps,
)
from .lane_layout_aligned_order import (
    preserve_aligned_target_order_after_compaction as _model_preserve_aligned_target_order_after_compaction,
)
from .lane_layout_placement import (
    event_target_raw_start_for_lane as _model_event_target_raw_start_for_lane,
    event_visible_start_for_placement as _model_event_visible_start_for_placement,
)
from .lane_layout_speech_swap import promote_longer_speech_events as _model_promote_longer_speech_events
from .lane_layout_occupancy import (
    LaneOccupancyState as _ModelLaneOccupancyState,
    SourceSpanTracker as _ModelSourceSpanTracker,
)
from .lane_layout_timing import (
    dedupe_exact_transition_spans as _model_dedupe_exact_transition_spans,
    raw_from_visible_with_transition_spans as _model_raw_from_visible_with_transition_spans,
    spans_without_exact_matches as _model_spans_without_exact_matches,
)
from .lane_layout_xfade import preserve_same_kind_xfade_chains as _model_preserve_same_kind_xfade_chains
from .lane_layout_final_compaction import (
    final_bounded_compact_class_events as _model_final_bounded_compact_class_events,
)
from .lane_layout_writer import (
    write_lane_layout_with_sdk_or_pyaaf2_fallback as _write_lane_layout_with_writer_facade,
)
from .lane_layout_zones import (
    class_zone_lane_order as _model_class_zone_lane_order,
    class_zone_lane_preference as _model_class_zone_lane_preference,
    lane_zone_preferences as _model_lane_zone_preferences,
)
from .render_semantics import MONO_PAN, operation_render_signature
from .media_resolve import UnsupportedMediaMapping
from .timeline_timing import (
    edited_audio_sample_rate as _edited_audio_sample_rate,
    essence_sample_rate_for_sourceclip as _essence_sample_rate_for_sourceclip,
    sourceclip_audio_timing as _sourceclip_audio_timing,
    sourceclip_duration_sec as _sourceclip_duration_sec,
)
from .media_resolve import _resolve_wave_path_for_sourceclip
from .timeline_walk import (
    _aaf_int_length,
    _aaf_int_start,
    _is_filler,
    _is_sourceclip,
    _node_length_units,
    _slot_edit_rate,
    _slot_is_soundish,
)
from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
from .speech_yamnet import YamnetConfig, yamnet_clip_kind_with_scores

from aaf_io.heal.davinci import ensure_wav_proxy_for_media as _ensure_wav_proxy_for_media

from aaf_io.composition import CompositionSelectionError, select_aaf_composition
from aaf_io.sdk_lane_layout import LaneLayoutEvent, apply_lane_layout_via_aaf_sdk_xml
from aaf_io.audio_silence import silence_replacement_signature
from aaf_io.clip_naming import resolve_clip_display_name

logger = logging.getLogger(__name__)

_RebuiltLanePart = tuple[int, int, Any, str, Optional[int], Optional[str], int]


def _configured_media_roots(input_aaf: Path) -> Optional[tuple[Path, ...]]:
    """Read toolkit settings lazily for standalone library calls, without mutation."""
    try:
        from aaf_tool_config import resolve_media_search_roots
    except ModuleNotFoundError as exc:
        if exc.name != "aaf_tool_config":
            raise
        return None
    return resolve_media_search_roots(input_aaf)


def _clip_name(aaf_obj: Any, node: Any) -> str:
    try:
        r = resolve_clip_display_name(aaf_obj, node, cache={})
        return (r.display_name or "").strip()
    except Exception:
        return ""


"""
SDK-XML lane layout write-back is implemented in :mod:`aaf_io.sdk_lane_layout`.
This module (aaf_speech_filter) should remain analysis/decision-focused.
"""


def _og_segments_vector(op_group) -> Optional[Any]:
    try:
        segs = op_group.segments
        if segs is not None and len(segs) >= 1:
            return segs
    except Exception:
        pass
    try:
        segs = op_group["Segments"]
        if segs is not None and len(segs) >= 1:
            return segs
    except Exception:
        pass
    return None


def _og_first_sourceclip(
    op_group,
    *,
    _depth: int = 0,
    _seen: Optional[set[int]] = None,
) -> Optional[Any]:
    """
    Первый ``SourceClip`` внутри ``OperationGroup`` (сегменты ``segments`` / ``Segments``).

    В Nuendo-AAF часто встречается **вложенный** ``OperationGroup`` (обёртка вокруг обёртки);
    без рекурсии внутренний ``SourceClip`` не находится — ломаются классификация YAMNet,
    ключ ``T_edit`` и стабильность раскладки.
    """
    if _depth > 64:
        return None
    if _seen is None:
        _seen = set()
    oid = id(op_group)
    if oid in _seen:
        return None
    _seen.add(oid)
    inner = _og_segments_vector(op_group)
    if inner is None:
        return None
    try:
        n = len(inner)
    except Exception:
        return None
    for j in range(n):
        try:
            ch = inner[j]
        except Exception:
            continue
        if _is_sourceclip(ch):
            return ch
        if _is_operation_group(ch):
            nested = _og_first_sourceclip(ch, _depth=_depth + 1, _seen=_seen)
            if nested is not None:
                return nested
    return None


def _is_operation_group(node) -> bool:
    return "OperationGroup" in node.__class__.__name__


def _event_inner_sourceclip(node: Any) -> Optional[Any]:
    if _is_operation_group(node):
        return _og_first_sourceclip(node)
    if _is_sourceclip(node):
        return node
    return None


def _sourceclip_edit_align_key(inner: Any) -> Optional[Tuple[str, int, int]]:
    """
    Ключ «одна и та же подрезка в источнике» без SourceMobSlotID — для стерео/дублей
    на разных дорожках, где локальный ``T`` в Sequence может различаться из‑за разной
    длины ведущих Filler при одной шкале времени в Nuendo.
    """
    if inner is None:
        return None
    try:
        sid = str(inner["SourceID"].value).strip()
    except Exception:
        sid = str(getattr(inner, "source_id", None) or "").strip()
    if not sid:
        return None
    length = _aaf_int_length(inner)
    start = _aaf_int_start(inner)
    return (sid, start, length)


def _sourceclip_source_id(inner: Any) -> str:
    try:
        return str(inner["SourceID"].value).strip()
    except Exception:
        return str(getattr(inner, "source_id", None) or "").strip()


def _sourceclip_source_track(inner: Any) -> int:
    for prop in ("SourceMobSlotID", "SourceSlotID"):
        try:
            return int(inner[prop].value)
        except Exception:
            pass
    try:
        return int(getattr(inner, "source_track_id", 0) or 0)
    except Exception:
        return 0


def _lane_layout_plan_identity_key(
    *,
    src_lane: int,
    visible_t: int,
    length: int,
    source_id: str,
    source_track: int,
    source_start: int,
    source_length: int,
) -> Optional[tuple[object, ...]]:
    sid = str(source_id or "").strip()
    if not sid:
        return None
    return (
        int(src_lane),
        sid,
        int(source_track),
        int(source_start),
        int(source_length),
        int(visible_t),
        int(length),
    )


def _lane_layout_plan_identity_key_for_event(e: dict[str, Any]) -> Optional[tuple[object, ...]]:
    source_id = str(e.get("source_id") or "").strip()
    source_track = int(e.get("source_track", 0) or 0)
    if not source_id:
        inner = _event_inner_sourceclip(e.get("node"))
        if inner is None:
            return None
        source_id = _sourceclip_source_id(inner)
        source_track = _sourceclip_source_track(inner)
    if not source_id:
        return None
    visible_t = e.get("T_edit")
    if visible_t is None:
        visible_t = e.get("T_nuendo")
    if visible_t is None:
        visible_t = e["T"]
    return _lane_layout_plan_identity_key(
        src_lane=int(e["src_lane"]),
        visible_t=int(visible_t),
        length=int(e["L"]),
        source_id=source_id,
        source_track=source_track,
        source_start=int(e.get("source_start", -1)),
        source_length=int(e.get("source_length", -1)),
    )


def _index_lane_layout_plan_event(
    ev: dict[str, Any],
    *,
    planned_by_identity: dict[tuple[object, ...], list[dict[str, Any]]],
    planned_by_top: dict[tuple[int, int], dict[str, Any]],
) -> None:
    planned_by_top[(int(ev["src_lane"]), int(ev["top_idx"]))] = ev
    identity_key = _lane_layout_plan_identity_key_for_event(ev)
    if identity_key is not None:
        planned_by_identity.setdefault(identity_key, []).append(ev)


def _match_lane_layout_plan_event(
    *,
    src_lane: int,
    top_idx: int,
    visible_t: int,
    length: int,
    source_id: str,
    source_track: int,
    source_start: int,
    source_length: int,
    planned_by_identity: dict[tuple[object, ...], list[dict[str, Any]]],
    planned_by_top: dict[tuple[int, int], dict[str, Any]],
    matched_event_ids: set[int],
) -> Optional[dict[str, Any]]:
    identity_key = _lane_layout_plan_identity_key(
        src_lane=int(src_lane),
        visible_t=int(visible_t),
        length=int(length),
        source_id=str(source_id or ""),
        source_track=int(source_track),
        source_start=int(source_start),
        source_length=int(source_length),
    )
    if identity_key is not None:
        candidates = planned_by_identity.get(identity_key, [])
        while candidates:
            ev = candidates.pop(0)
            if id(ev) in matched_event_ids:
                continue
            matched_event_ids.add(id(ev))
            return ev

    ev = planned_by_top.get((int(src_lane), int(top_idx)))
    if ev is not None and id(ev) not in matched_event_ids:
        matched_event_ids.add(id(ev))
        return ev
    return None


def _apply_unknown_opening_music_fallback(
    events: List[dict[str, Any]],
    head_sec: float,
) -> None:
    """
    Если медиа недоступно, YAMNet не вызывается → ``unknown`` и клипы не «падают» вниз.

    Для блоков в начале таймлайна (часто заставка/музыка) задаём ``music``, чтобы
    :func:`_assign_target_lanes` мог разместить их на нижних дорожках.
    """
    if float(head_sec) <= 0.0:
        return
    h = float(head_sec)
    for e in events:
        if e.get("kind") != "unknown":
            continue
        try:
            er = float(e.get("er") or 48000.0)
        except Exception:
            er = 48000.0
        if er <= 0.0:
            continue
        try:
            t_ed = int(e.get("T_edit", e.get("T", 0)))
        except Exception:
            t_ed = 0
        start_s = float(t_ed) / er
        if start_s < h:
            e["kind"] = "music"


def _apply_edit_timeline_unification(events: List[dict[str, Any]]) -> None:
    """Initialize each occurrence's edit coordinate without aligning source cuts."""
    for event in events:
        event["T_edit"] = int(event.get("T_nuendo", event["T"]))


def _pick_main_composition(comp_mobs: List[Any]) -> Any:
    return select_aaf_composition(comp_mobs)


def _sound_sequence_timeline_slots(
    comp: Any,
    *,
    cancel_check: Callable[[], None],
) -> tuple[list[tuple[Any, Any]], int]:
    """
    Звуковые слоты композиции, у которых сегмент содержит плоский ``Sequence`` (как требует раскладка).

    Поддерживаем два варианта:
    - ``slot.segment`` это ``Sequence`` (обычный Nuendo/AAF SDK стиль)
    - ``slot.segment`` это ``OperationGroup`` (например Premiere `Mono Audio Pan`), внутри которого
      единственный сегмент — ``Sequence``. Это типично для некоторых экспортов с эффектами на дорожке.

    Возвращает ``(lanes, skipped_non_sequence_sound)``, где lane = (slot, sequence_segment).
    """
    sound_lanes: list[tuple[Any, Any]] = []
    skipped_non_sequence_sound = 0
    for slot in getattr(comp, "slots", []) or []:
        cancel_check()
        if not _slot_is_soundish(slot):
            continue
        seg = getattr(slot, "segment", None)
        if seg is None:
            skipped_non_sequence_sound += 1
            continue
        seg_name = seg.__class__.__name__
        if "Sequence" in seg_name:
            sound_lanes.append((slot, seg))
            continue
        if "OperationGroup" in seg_name:
            # Common for Premiere exports: slot.segment is an OperationGroup wrapping a Sequence.
            seq = None
            try:
                inner = getattr(seg, "segments", None)
                if inner is not None and len(inner) >= 1:
                    seq = inner[0]
            except Exception:
                seq = None
            if seq is None:
                try:
                    inner = seg["Segments"]
                    if inner is not None and len(inner) >= 1:
                        seq = inner[0]
                except Exception:
                    seq = None
            if seq is not None and "Sequence" in seq.__class__.__name__:
                sound_lanes.append((slot, seq))
                continue
        skipped_non_sequence_sound += 1
    return sound_lanes, skipped_non_sequence_sound


def _count_top_sequence_layout_blocks(sound_lanes: list[tuple[Any, Any]]) -> int:
    """Число верхнеуровневых ``SourceClip`` / ``OperationGroup`` (не ``Filler``) в ``Sequence``."""
    n_blocks = 0
    for _slot, seq in sound_lanes:
        try:
            top = seq.components
            tn = len(top)
        except Exception:
            continue
        for ti in range(tn):
            try:
                node = top[ti]
            except Exception:
                break
            if _is_filler(node):
                continue
            if _is_operation_group(node) or _is_sourceclip(node):
                n_blocks += 1
    return n_blocks


def _sequence_top_total_length(top) -> int:
    pos = 0
    try:
        n = len(top)
    except Exception:
        return 0
    for i in range(n):
        try:
            node = top[i]
        except Exception:
            break
        pos += _node_length_units(node)
    return int(pos)


def _sequence_declared_length(seq: Any, fallback: int) -> int:
    try:
        ln = _aaf_int_length(seq)
    except Exception:
        ln = 0
    return int(ln) if int(ln) > 0 else int(fallback)


def _components_transition_overlap_units(components: list[Any]) -> int:
    total = 0
    for node in components:
        if _is_transition_node(node):
            total += int(_node_length_units(node))
    return int(total)


def _rebuilt_lane_declared_length(component_total: int, components: list[Any]) -> int:
    transition_overlap = _components_transition_overlap_units(components)
    return max(0, int(component_total) - 2 * int(transition_overlap))


def _set_component_length(obj: Any, length: int) -> None:
    ln = int(max(0, length))
    try:
        obj.length = ln
        return
    except Exception:
        pass
    try:
        obj["Length"].value = ln
    except Exception:
        pass


def _rebuilt_lane_total_length(original_total: int, placed_end: int) -> int:
    return max(int(original_total), int(placed_end))


def _slot_segment_sequence_wrapper(slot: Any, seq: Any) -> Optional[Any]:
    try:
        seg = getattr(slot, "segment", None)
    except Exception:
        return None
    if seg is None or seg is seq or "OperationGroup" not in seg.__class__.__name__:
        return None
    try:
        inner = getattr(seg, "segments", None)
        if inner is not None and len(inner) >= 1 and inner[0] is seq:
            return seg
    except Exception:
        return None
    return None


def _operation_group_name(op_group: Any) -> str:
    try:
        op = op_group["Operation"].value
        name = getattr(op, "name", None)
        if name:
            return str(name).strip()
        return str(op).strip()
    except Exception:
        pass
    try:
        op = getattr(op_group, "operation", None)
        if op is not None:
            name = getattr(op, "name", None)
            if name:
                return str(name).strip()
            return str(op).strip()
    except Exception:
        pass
    return ""


def _lane_layout_signature(slot: Any, seq: Any) -> tuple[Any, ...]:
    rate = getattr(slot, "edit_rate", None)
    clock = (Fraction(str(rate)) if rate is not None else None, int(getattr(slot, "origin", 0)))
    wrapper = _slot_segment_sequence_wrapper(slot, seq)
    if wrapper is None:
        return ("Sequence", "", *clock)
    semantics = operation_render_signature(wrapper)
    if semantics is None:
        return ("UnsupportedOperation", getattr(slot, "slot_id", None), *clock)
    if semantics[0] == MONO_PAN and all(type(p).__name__ == "ConstantValue" for p in wrapper.parameters):
        # Constant pan does not depend on total track duration; automation does.
        semantics = (semantics[0], None, *semantics[2:])
    return ("OperationGroup", semantics, *clock)


def _lane_layout_wrapper_is_rebuildable(wrapper: Any) -> bool:
    semantics = operation_render_signature(wrapper)
    return semantics is not None and semantics[0] == MONO_PAN


def _lane_layout_mutable(slot: Any, seq: Any) -> bool:
    nodes = list(seq.components)
    for index, node in enumerate(nodes):
        if _is_filler(node) or _is_transition_node(node) or silence_replacement_signature(nodes, index) is not None:
            continue
        if not (_is_sourceclip(node) or _is_operation_group(node)):
            return False
        if _event_inner_sourceclip(node) is None:
            return False
    wrapper = _slot_segment_sequence_wrapper(slot, seq)
    if wrapper is None:
        return True
    return _lane_layout_wrapper_is_rebuildable(wrapper)


def _sound_filler(aaf, length: int, dd_ref: Optional[Any], inner_sc: Optional[Any]) -> Any:
    ln = max(0, int(length))
    f = aaf.create.Filler(length=ln, media_kind="Sound")
    try:
        if inner_sc is not None:
            f["DataDefinition"].value = inner_sc["DataDefinition"].value
        elif dd_ref is not None:
            f["DataDefinition"].value = dd_ref
    except Exception:
        dd = _datadef_for_filler_fallback(aaf, inner_sc) if inner_sc is not None else None
        if dd is not None:
            try:
                f["DataDefinition"].value = dd
            except Exception:
                pass
    return f


def _lane_yamnet_cfg(cfg: FilterConfig) -> YamnetConfig:
    """
    Lane layout must NOT apply file/clip-specific heuristics.

    Keep the same YAMNet configuration as the main pipeline. Raising the speech threshold here
    can systematically misclassify borderline speech as music/unknown on diverse material.
    """
    return cfg.yamnet


def _classify_timeline_block(
    aaf,
    node: Any,
    runtime_essence_paths: Optional[Dict[Any, Path]],
    yamnet_cfg: YamnetConfig,
    edit_rate: float,
    cancel_check: Callable[[], None],
    *,
    work_dir: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    media_search_roots: Optional[tuple[Path, ...]] = None,
) -> str:
    kind, _s, _m, _n = _classify_timeline_block_with_scores(
        aaf,
        node,
        runtime_essence_paths,
        yamnet_cfg,
        edit_rate,
        cancel_check,
        work_dir=work_dir,
        log_callback=log_callback,
        media_search_roots=media_search_roots,
    )
    return str(kind)


def _classify_timeline_block_with_scores(
    aaf,
    node: Any,
    runtime_essence_paths: Optional[Dict[Any, Path]],
    yamnet_cfg: YamnetConfig,
    edit_rate: float,
    cancel_check: Callable[[], None],
    *,
    work_dir: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    media_search_roots: Optional[tuple[Path, ...]] = None,
) -> tuple[str, float, float, float]:
    inner = None
    if _is_operation_group(node):
        inner = _og_first_sourceclip(node)
    elif _is_sourceclip(node):
        inner = node
    if inner is None:
        return "unknown", 0.0, 0.0, 0.0
    wav = _resolve_wave_path_for_sourceclip(aaf, inner, runtime_essence_paths, media_search_roots=media_search_roots, edit_rate=edit_rate)
    if not wav or not wav.is_file():
        # Media unavailable: classification must not depend on clip/display names.
        return "unknown", 0.0, 0.0, 0.0
    # Unembedded AAFs (e.g. DaVinci) can point to MXF/other containers; create a WAV proxy if needed.
    wav2 = _ensure_wav_proxy_for_media(
        Path(wav),
        work_dir=work_dir,
        cancel_check=cancel_check,
        log_callback=log_callback,
    )
    if wav2 is not None and wav2.is_file():
        wav = wav2
    er = float(edit_rate) or 48000.0
    visible_duration_sec = max(Fraction(0), Fraction(_node_length_units(node), 1) / Fraction(str(edit_rate or 48000)))
    try:
        start_sec, dur = _sourceclip_audio_timing(
            aaf,
            edit_rate,
            inner,
            wav,
            runtime_essence_paths,
            visible_duration_sec=visible_duration_sec,
        )
    except UnsupportedMediaMapping as exc:
        if log_callback is not None:
            log_callback(DiagnosticMessage(
                f"Media window unresolved; classification skipped: {exc}",
                category="Окно аудио: классификация пропущена",
            ))
        return "unknown", 0.0, 0.0, 0.0
    kind, s, m, n = yamnet_clip_kind_with_scores(
        wav,
        start_sec,
        dur,
        yamnet_cfg,
        cancel_check=cancel_check,
    )
    kind = str(kind)
    # Не даунгрейдим «speech» из yamnet_clip_kind_with_scores по мелкой разнице s vs n:
    # у Premiere embedded на разные MobID дают разные audio_XXX.wav; для стерео/дублей одного
    # кадра старый abs(s-n)<1e-4 превращал часть дорожек в «шум», часть — в «речь» при одинаковом тайминге.
    # Оставляем только случай явно доминирующего шума по абсолютной уверенности.
    fs = float(s)
    fm = float(m)
    fn = float(n)
    return kind, fs, fm, fn


def _sdk_xml_event_preferred_transition_offset(e: dict[str, Any]) -> int:
    return max(0, int(e.get("pre_transition_len", 0) or 0))


def _build_sdk_lane_layout_events(
    events: list[dict[str, Any]],
    structural_events: list[LaneLayoutEvent],
) -> list[LaneLayoutEvent]:
    layout_events = list(structural_events)
    for e in events:
        if "target_lane" not in e:
            continue
        layout_events.append(
            LaneLayoutEvent(
                src_lane=int(e["src_lane"]),
                top_idx=int(e["top_idx"]),
                T_edit=int(e.get("target_T", e["T"])),
                L=int(e["L"]),
                target_lane=int(e["target_lane"]),
                class_kind=str(e.get("kind") or "unknown"),
                lane_bounds=e.get("lane_bounds"),
                visible_t=int(e.get("T_edit", e.get("T_nuendo", e["T"]))),
                preferred_transition_offset=_sdk_xml_event_preferred_transition_offset(e),
                pre_top_idx=(
                    None
                    if e.get("pre_transition_top_idx") is None
                    else int(e["pre_transition_top_idx"])
                ),
                pre_len=int(e.get("pre_transition_len", 0) or 0),
                post_top_idx=(
                    None
                    if e.get("post_transition_top_idx") is None
                    else int(e["post_transition_top_idx"])
                ),
                post_len=int(e.get("post_transition_len", 0) or 0),
                source_start=(
                    None if e.get("source_start") is None else int(e["source_start"])
                ),
                source_length=(
                    None if e.get("source_length") is None else int(e["source_length"])
                ),
                raw_t_authoritative=bool(e.get("raw_t_authoritative", False)),
            )
        )
    return layout_events


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) < min(a1, b1)


def _event_owned_raw_transition_spans(e: dict[str, Any], event_t: int) -> list[tuple[int, int]]:
    return _model_event_owned_raw_transition_spans(e, int(event_t))


def _event_owned_transition_history_spans(
    e: dict[str, Any],
    event_t: int,
) -> list[tuple[int, int]]:
    return _model_event_owned_transition_history_spans(e, int(event_t))


def _span_contains(container: tuple[int, int], span: tuple[int, int]) -> bool:
    return int(container[0]) <= int(span[0]) and int(span[1]) <= int(container[1])


def _lane_zone_preferences(
    lane_count: int,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    return _model_lane_zone_preferences(int(lane_count))


def _class_zone_lane_order(kind: str, lane_count: int) -> list[int]:
    return _model_class_zone_lane_order(kind, int(lane_count))


def _class_zone_lane_preference(kind: str, lane_count: int, current_lane: int) -> list[int]:
    return _model_class_zone_lane_preference(kind, int(lane_count), int(current_lane))


def _rebuilt_lane_in_bounds(part, lane, bounds_by_node_id):
    bounds = (bounds_by_node_id or {}).get(id(_rebuilt_event_primary_node(part)))
    return bounds is None or int(bounds[0]) <= int(lane) <= int(bounds[1])


def _compact_rebuilt_parts_to_class_zones_without_visible_time_shift(
    rebuilt: dict[int, list[_RebuiltLanePart]],
    lane_count: int,
    raw_for_visible_on_lane: Optional[Callable[[int, int, int], int]] = None,
    immutable_lanes: Optional[set[int]] = None,
    composition_id: Optional[str] = None,
    lane_signatures: Optional[list[tuple[Any, ...]]] = None,
    lane_bounds_by_node_id: Optional[dict[int, tuple[int, int]]] = None,
) -> int:
    """Move rebuilt event parts to class zones without changing visible time."""
    immutable_lane_set = {int(lane) for lane in (immutable_lanes or set())}
    occupancy: dict[int, list[tuple[int, int]]] = {i: [] for i in range(int(lane_count))}
    event_items: list[tuple[int, _RebuiltLanePart]] = []
    for lane_idx, parts_on_lane in rebuilt.items():
        for item in parts_on_lane:
            t0, l0, _node0, kind0, _visible0, class_kind0, _event_offset0 = item
            if int(l0) > 0 and kind0 != "transition":
                occupancy.setdefault(int(lane_idx), []).append((int(t0), int(t0) + int(l0)))
            if kind0 == "event" and int(lane_idx) not in immutable_lane_set:
                event_items.append((int(lane_idx), item))

    moved = 0
    for _pass_idx in range(4):
        changed = False
        for lane_idx, item in sorted(
            event_items,
            key=lambda pair: (
                str(pair[1][5] or "unknown"),
                int(pair[1][4]) if pair[1][4] is not None else int(pair[1][0]),
                int(pair[0]),
                int(pair[1][1]),
            ),
        ):
            t0, l0, node0, kind0, visible0, class_kind0, event_offset0 = item
            if item not in rebuilt.get(int(lane_idx), []):
                continue
            src_span = (int(t0), int(t0) + int(l0))
            try:
                occupancy[int(lane_idx)].remove(src_span)
            except (KeyError, ValueError):
                pass
            chosen: Optional[tuple[int, int]] = None
            lane_order = [
                lane
                for lane in _class_zone_lane_order(str(class_kind0 or "unknown"), int(lane_count))
                if _rebuilt_lane_in_bounds(item, lane, lane_bounds_by_node_id)
                and int(lane) not in immutable_lane_set
                and (lane_signatures is None or lane_signatures[lane_idx] == lane_signatures[lane])
            ]
            try:
                current_rank = lane_order.index(int(lane_idx))
            except ValueError:
                current_rank = len(lane_order)
            for target_lane in lane_order:
                if int(target_lane) == int(lane_idx):
                    continue
                try:
                    target_rank = lane_order.index(int(target_lane))
                except ValueError:
                    continue
                if int(target_rank) >= int(current_rank):
                    continue
                cand_t = int(t0)
                if visible0 is not None and raw_for_visible_on_lane is not None:
                    cand_t = (
                        int(
                            raw_for_visible_on_lane(
                                int(target_lane),
                                int(visible0),
                                int(event_offset0),
                            )
                        )
                        - int(event_offset0)
                    )
                cand_span = (int(cand_t), int(cand_t) + int(l0))
                if any(
                    _intervals_overlap(cand_span[0], cand_span[1], occ0, occ1)
                    for occ0, occ1 in occupancy.get(int(target_lane), [])
                ):
                    continue
                chosen = (int(target_lane), int(cand_t))
                break
            if chosen is None:
                occupancy.setdefault(int(lane_idx), []).append(src_span)
                continue
            try:
                rebuilt[int(lane_idx)].remove(item)
            except ValueError:
                occupancy.setdefault(int(lane_idx), []).append(src_span)
                continue
            target_lane, cand_t = chosen
            moved_item: _RebuiltLanePart = (
                int(cand_t),
                int(l0),
                node0,
                str(kind0),
                visible0,
                str(class_kind0 or "unknown"),
                int(event_offset0),
            )
            rebuilt.setdefault(int(target_lane), []).append(moved_item)
            occupancy.setdefault(int(target_lane), []).append((int(cand_t), int(cand_t) + int(l0)))
            event_items.remove((int(lane_idx), item))
            event_items.append((int(target_lane), moved_item))
            moved += 1
            changed = True
        if not changed:
            break
    return moved


def _rebuilt_part_transition_spans(T: int, nodes: Any) -> list[tuple[int, int]]:
    seq_nodes = nodes if isinstance(nodes, list) else [nodes]
    spans: list[tuple[int, int]] = []
    cur_part = int(T)
    for nd in seq_nodes:
        ln = int(_node_length_units(nd))
        if _is_transition_node(nd) and ln > 0:
            spans.append((int(cur_part), int(ln)))
        cur_part += max(0, ln)
    return spans


def _trim_leading_rebuilt_transitions_to_cursor(
    t: int,
    length: int,
    node_or_nodes: Any,
    cursor: int,
    *,
    emitted_transitions: Optional[dict[int, Any]] = None,
    transition_sources: Optional[dict[int, Any]] = None,
) -> tuple[int, int, Any]:
    """Coalesce only an original edge already emitted at this raw position."""
    out_t = int(t)
    out_l = int(length)
    if out_t >= int(cursor):
        return out_t, out_l, node_or_nodes
    nodes = list(node_or_nodes) if isinstance(node_or_nodes, list) else [node_or_nodes]
    if not nodes:
        return out_t, out_l, node_or_nodes
    original_was_list = isinstance(node_or_nodes, list)
    changed = False
    while nodes and out_t < int(cursor):
        node = nodes[0]
        node_len = max(0, int(_node_length_units(node)))
        if not _is_transition_node(node) or node_len <= 0:
            break
        if out_t + node_len > int(cursor):
            break
        source = (transition_sources or {}).get(id(node), node)
        if (emitted_transitions or {}).get(out_t) is not source:
            break
        out_t += node_len
        out_l -= node_len
        nodes.pop(0)
        changed = True
    if not changed:
        return int(t), int(length), node_or_nodes
    if original_was_list:
        return int(out_t), max(0, int(out_l)), nodes
    if len(nodes) == 1:
        return int(out_t), max(0, int(out_l)), nodes[0]
    return int(out_t), max(0, int(out_l)), nodes


def _raw_from_visible_with_transition_spans(
    transition_spans: list[tuple[int, int]],
    visible_t: int,
    owned_pre_len: int = 0,
) -> int:
    return _model_raw_from_visible_with_transition_spans(
        transition_spans,
        int(visible_t),
        int(owned_pre_len),
    )


def _dedupe_exact_transition_spans(
    spans: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    return _model_dedupe_exact_transition_spans(spans)


def _spans_without_exact_matches(
    spans: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    return _model_spans_without_exact_matches(spans, excluded)


def _planned_event_visible_t(e: dict[str, Any]) -> int:
    return _model_event_visible_start(e)


def _planned_event_visible_span(e: dict[str, Any]) -> tuple[int, int]:
    return _model_event_visible_span(e)


def _planned_event_raw_t_for_lane(
    e: dict[str, Any],
    lane: int,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]],
) -> int:
    return _model_planned_event_raw_t_for_lane(e, int(lane), transition_spans_by_lane)


def _planned_event_span(
    e: dict[str, Any],
    event_t: Optional[int] = None,
) -> tuple[int, int]:
    return _model_event_raw_span_with_transitions(e, event_t)


def _events_share_same_kind_transition_edge(
    prev: dict[str, Any],
    item: dict[str, Any],
) -> bool:
    return _model_events_share_same_kind_transition_edge(prev, item)


def _planned_overlap_is_shared_transition_edge(
    prev: dict[str, Any],
    item: dict[str, Any],
) -> bool:
    return _model_planned_overlap_is_allowed_transition_overlap(prev, item)


def _repair_lane_layout_plan_overlaps(
    events: list[dict[str, Any]],
    n_lanes: int,
    *,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]] = None,
    lane_signature_by_lane: Optional[dict[int, tuple[Any, ...]]] = None,
    blocked_target_lanes: Optional[set[int]] = None,
    lane_capacity_by_lane: Optional[dict[int, int]] = None,
    lane_mutable_by_lane: Optional[dict[int, bool]] = None,
) -> int:
    return _model_repair_lane_layout_plan_overlaps(
        events,
        int(n_lanes),
        transition_spans_by_lane=transition_spans_by_lane,
        lane_signature_by_lane=lane_signature_by_lane,
        blocked_target_lanes=blocked_target_lanes,
        lane_capacity_by_lane=lane_capacity_by_lane,
        lane_mutable_by_lane=lane_mutable_by_lane,
    )


def _rebalance_rebuilt_event_parts_to_visible_times(
    rebuilt: dict[int, list[_RebuiltLanePart]],
    lane_count: int,
    transition_history_by_lane: Optional[dict[int, list[tuple[int, int]]]] = None,
) -> None:
    for _ in range(8):
        spans_by_lane: dict[int, list[tuple[int, int]]] = {}
        for lane_idx, parts_on_lane in rebuilt.items():
            spans: list[tuple[int, int]] = list(
                (transition_history_by_lane or {}).get(int(lane_idx), [])
            )
            for t, l, node, kind, _visible, _class_kind, _event_offset in parts_on_lane:
                if kind == "transition" and int(l) > 0:
                    spans.append((int(t), int(l)))
                elif kind == "event":
                    spans.extend(_rebuilt_part_transition_spans(int(t), node))
            spans_by_lane[int(lane_idx)] = sorted(spans)

        changed = False
        next_rebuilt: dict[int, list[_RebuiltLanePart]] = {i: [] for i in range(int(lane_count))}
        for lane_idx in range(int(lane_count)):
            for t, l, node, kind, visible, class_kind, event_offset in rebuilt.get(int(lane_idx), []):
                out_t = int(t)
                if kind == "event" and visible is not None:
                    own_transition_spans = _rebuilt_part_transition_spans(int(t), node)
                    external_spans = _spans_without_exact_matches(
                        spans_by_lane.get(int(lane_idx), []),
                        own_transition_spans,
                    )
                    event_raw = _raw_from_visible_with_transition_spans(
                        external_spans,
                        int(visible),
                        int(event_offset),
                    )
                    out_t = int(event_raw) - int(event_offset)
                    if int(out_t) != int(t):
                        changed = True
                next_rebuilt[int(lane_idx)].append(
                    (
                        int(out_t),
                        int(l),
                        node,
                        str(kind),
                        visible,
                        None if class_kind is None else str(class_kind),
                        int(event_offset),
                    )
                )
        rebuilt.clear()
        rebuilt.update(next_rebuilt)
        if not changed:
            break


def _rebuilt_placement_is_feasible(
    rebuilt: dict[int, list[_RebuiltLanePart]],
    lane_count: int,
    transition_history_by_lane: Optional[dict[int, list[tuple[int, int]]]],
    lane_bounds_by_node_id: Optional[dict[int, tuple[int, int]]],
) -> bool:
    for lane, parts in rebuilt.items():
        if not 0 <= int(lane) < int(lane_count):
            return False
        spans = _rebuilt_transition_spans_for_lane(
            rebuilt, int(lane), transition_history_by_lane,
        )
        occupied: list[tuple[int, int]] = []
        for part in parts:
            raw, length, nodes, kind, visible, _class_kind, offset = part
            if kind == "event":
                if not _rebuilt_lane_in_bounds(part, int(lane), lane_bounds_by_node_id):
                    return False
                if visible is not None:
                    external = _spans_without_exact_matches(
                        spans, _rebuilt_part_transition_spans(int(raw), nodes),
                    )
                    expected = _raw_from_visible_with_transition_spans(
                        external, int(visible), int(offset),
                    ) - int(offset)
                    if int(raw) != int(expected):
                        return False
            if kind != "transition" and int(length) > 0:
                if int(raw) < 0:
                    return False
                occupied.append((int(raw), int(raw) + int(length)))
        occupied.sort()
        if any(left[1] > right[0] for left, right in zip(occupied, occupied[1:])):
            return False
    return True


def _resolve_rebuilt_event_overlaps_without_visible_time_shift(
    rebuilt: dict[int, list[_RebuiltLanePart]],
    lane_count: int,
    transition_history_by_lane: Optional[dict[int, list[tuple[int, int]]]] = None,
    immutable_lanes: Optional[set[int]] = None,
    composition_id: Optional[str] = None,
    lane_signatures: Optional[list[tuple[Any, ...]]] = None,
    lane_bounds_by_node_id: Optional[dict[int, tuple[int, int]]] = None,
) -> int:
    # Repair must not turn an already feasible assignment into a greedy dead end.
    if _rebuilt_placement_is_feasible(
        rebuilt, lane_count, transition_history_by_lane, lane_bounds_by_node_id,
    ):
        return 0
    immutable_lane_set = {int(lane) for lane in (immutable_lanes or set())}
    if any(not 0 <= int(lane) < int(lane_count) for lane in rebuilt):
        raise RuntimeError("YAMNet lane layout cannot repair an out-of-range rebuilt lane")
    working = {lane: list(rebuilt.get(lane, [])) for lane in range(int(lane_count))}
    # Tokens identify occurrences even when rebalance replaces their tuples.
    tokens: dict[int, list[int]] = {}
    original: dict[int, tuple[int, _RebuiltLanePart]] = {}
    for lane, parts in working.items():
        tokens[lane] = []
        for part in parts:
            token = len(original)
            tokens[lane].append(token)
            original[token] = (lane, part)

    def placement_problems(
        state: dict[int, list[_RebuiltLanePart]],
        occurrence_tokens: dict[int, list[int]],
    ) -> set[tuple[str, int, int]]:
        problems: set[tuple[str, int, int]] = set()
        for lane, parts in state.items():
            spans = _rebuilt_transition_spans_for_lane(
                state, lane, transition_history_by_lane,
            )
            occupied = []
            for token, part in zip(occurrence_tokens[lane], parts):
                raw, length, nodes, kind, visible, _class_kind, offset = part
                if kind == "event":
                    if not _rebuilt_lane_in_bounds(part, lane, lane_bounds_by_node_id):
                        problems.add(("bounds", token, token))
                    if visible is not None:
                        external = _spans_without_exact_matches(
                            spans, _rebuilt_part_transition_spans(int(raw), nodes),
                        )
                        expected = _raw_from_visible_with_transition_spans(
                            external, int(visible), int(offset),
                        ) - int(offset)
                        if int(raw) != int(expected):
                            problems.add(("timing", token, token))
                original_lane, original_part = original[token]
                if original_lane in immutable_lane_set and (
                    lane != original_lane or int(raw) != int(original_part[0])
                ):
                    problems.add(("immutable", token, token))
                if kind != "transition" and int(length) > 0:
                    if int(raw) < 0:
                        problems.add(("negative", token, token))
                    occupied.append((int(raw), int(raw) + int(length), token))
            active = []
            for start, end, token in sorted(occupied):
                active = [(stop, other) for stop, other in active if stop > start]
                for _stop, other in active:
                    problems.add(("overlap", min(token, other), max(token, other)))
                active.append((end, token))
        return problems

    problems = placement_problems(working, tokens)
    while problems:
        locations = {
            token: (lane, index)
            for lane, lane_tokens in tokens.items()
            for index, token in enumerate(lane_tokens)
        }
        victims = []
        for _kind, left, right in sorted(problems):
            for token in (right, left):
                if token not in victims:
                    victims.append(token)
        accepted = False
        for token in victims:
            src_lane, index = locations[token]
            item = working[src_lane][index]
            if item[3] != "event" or src_lane in immutable_lane_set:
                continue
            candidates = dict.fromkeys((
                src_lane,
                *_class_zone_lane_order(str(item[5] or "unknown"), int(lane_count)),
                *range(int(lane_count)),
            ))
            for lane in candidates:
                if (
                    not 0 <= lane < int(lane_count)
                    or lane in immutable_lane_set
                    or not _rebuilt_lane_in_bounds(item, lane, lane_bounds_by_node_id)
                    or (lane_signatures is not None and lane_signatures[src_lane] != lane_signatures[lane])
                ):
                    continue
                proposal = {key: list(parts) for key, parts in working.items()}
                proposal_tokens = {key: list(ids) for key, ids in tokens.items()}
                if lane != src_lane:
                    proposal[src_lane].pop(index)
                    proposal_tokens[src_lane].pop(index)
                    proposal[lane].append(item)
                    proposal_tokens[lane].append(token)
                _rebalance_rebuilt_event_parts_to_visible_times(
                    proposal, lane_count, transition_history_by_lane,
                )
                proposed_problems = placement_problems(proposal, proposal_tokens)
                # Strict subset prevents new conflicts and guarantees termination.
                if not proposed_problems < problems:
                    continue
                if any(token in problem[1:] for problem in proposed_problems):
                    continue
                working, tokens, problems = proposal, proposal_tokens, proposed_problems
                accepted = True
                break
            if accepted:
                break
        if not accepted:
            src_lane, index = locations[victims[0]]
            item = working[src_lane][index]
            raise RuntimeError(
                "YAMNet lane layout could not resolve rebuilt overlap without moving time: "
                f"src_lane={src_lane} T={item[0]} L={item[1]} "
                f"visible={item[4]} class={item[5] or 'unknown'}"
            )

    moved = sum(
        lane != original[token][0] or int(part[0]) != int(original[token][1][0])
        for lane, parts in working.items()
        for token, part in zip(tokens[lane], parts)
        if part[3] == "event"
    )
    rebuilt.clear()
    rebuilt.update(working)
    return int(moved)


def _rebuilt_transition_spans_for_lane(
    rebuilt: dict[int, list[_RebuiltLanePart]],
    lane_idx: int,
    transition_history_by_lane: Optional[dict[int, list[tuple[int, int]]]] = None,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = list(
        (transition_history_by_lane or {}).get(int(lane_idx), [])
    )
    for t, l, node, kind, _visible, _class_kind, _event_offset in rebuilt.get(int(lane_idx), []):
        if kind == "transition" and int(l) > 0:
            spans.append((int(t), int(l)))
        elif kind == "event":
            spans.extend(_rebuilt_part_transition_spans(int(t), node))
    return sorted(_dedupe_exact_transition_spans(spans))


def _prune_rebuilt_transition_parts_not_serializable(
    rebuilt: dict[int, list[_RebuiltLanePart]],
    lane_count: int,
) -> int:
    removed = 0
    for lane_idx in range(int(lane_count)):
        parts = list(rebuilt.get(int(lane_idx), []))
        if not parts:
            continue
        event_edges: set[int] = set()
        event_intervals: list[tuple[int, int]] = []
        for t, l, _node, kind, _visible, _class_kind, _event_offset in parts:
            if kind != "event":
                continue
            event_edges.add(int(t))
            event_edges.add(int(t) + int(l))
            event_intervals.append((int(t), int(t) + int(l)))
        kept: list[_RebuiltLanePart] = []
        for item in parts:
            t, l, _node, kind, _visible, _class_kind, _event_offset = item
            if kind == "empty_audio_wrapper":
                removed += 1
                continue
            if kind == "transition" and not _transition_part_is_serializable(
                int(t),
                int(l),
                event_edges,
                event_intervals,
            ):
                removed += 1
                continue
            kept.append(item)
        rebuilt[int(lane_idx)] = kept
    return int(removed)


def _rebuilt_event_primary_sourceclip(item: _RebuiltLanePart) -> Optional[Any]:
    _t, _l, node, _kind, _visible, _class_kind, _event_offset = item
    nodes = node if isinstance(node, list) else [node]
    for nd in nodes:
        inner = _event_inner_sourceclip(nd)
        if inner is not None:
            return inner
    return None


def _rebuilt_event_group_key(item: _RebuiltLanePart) -> tuple[int, int, str, int, int]:
    _t, l, _node, _kind, visible, class_kind, _event_offset = item
    inner = _rebuilt_event_primary_sourceclip(item)
    source_start = -1
    source_length = -1
    if inner is not None:
        try:
            source_start = int(_aaf_int_start(inner))
            source_length = int(_aaf_int_length(inner))
        except Exception:
            source_start = -1
            source_length = -1
    return (
        int(visible) if visible is not None else int(item[0]),
        int(l),
        str(class_kind or "unknown"),
        int(source_start),
        int(source_length),
    )


def _rebuilt_event_primary_node(item: _RebuiltLanePart) -> Any:
    _t, _l, node, _kind, _visible, _class_kind, _event_offset = item
    if isinstance(node, list):
        for nd in node:
            if _is_operation_group(nd) or _is_sourceclip(nd):
                return nd
        return node[0] if node else None
    return node


def _rebuilt_event_group_order_key(
    lane_idx: int,
    item: _RebuiltLanePart,
    event_order_by_node_id: Optional[dict[int, int]] = None,
) -> tuple[int, int]:
    if event_order_by_node_id is not None:
        primary = _rebuilt_event_primary_node(item)
        order = event_order_by_node_id.get(id(primary))
        if order is not None:
            return (int(order), int(lane_idx))
    inner = _rebuilt_event_primary_sourceclip(item)
    if inner is not None:
        track = _sourceclip_source_track(inner)
        if int(track) > 0:
            return (int(track), int(lane_idx))
    return (int(lane_idx), int(lane_idx))


def _contiguous_class_lane_sequences(kind: str, lane_count: int, group_size: int) -> list[list[int]]:
    size = int(group_size)
    if size <= 0:
        return []
    lanes_all = list(range(int(lane_count)))
    if len(lanes_all) < size:
        return []
    speech_end = max(1, int(lane_count) // 3)
    music_start = max(speech_end, (2 * int(lane_count)) // 3)
    if str(kind) == "speech":
        primary = lanes_all[:speech_end]
    elif str(kind) == "music":
        primary = lanes_all[music_start:] or lanes_all[-1:]
    else:
        primary = lanes_all[speech_end:music_start] or lanes_all[:speech_end]
    allowed = list(primary)
    if len(allowed) < size:
        allowed = lanes_all
    sequences: list[list[int]] = []
    for i in range(0, len(allowed) - size + 1):
        seq = allowed[i : i + size]
        if len(seq) == size and all(int(seq[j + 1]) == int(seq[j]) + 1 for j in range(len(seq) - 1)):
            sequences.append(list(seq))
    if not sequences:
        return []
    center = (primary[0] + primary[-1]) / 2.0 if primary else (int(lane_count) - 1) / 2.0

    def rank(seq: list[int]) -> tuple[float, float, float]:
        if str(kind) == "speech":
            return (0.0, float(seq[0]), float(seq[-1]))
        if str(kind) == "music":
            return (0.0, float(-seq[-1]), float(-seq[0]))
        seq_center = (seq[0] + seq[-1]) / 2.0
        return (0.0, abs(seq_center - center), float(seq[0]))

    return sorted(sequences, key=rank)


def _compact_rebuilt_aligned_groups_without_visible_time_shift(
    rebuilt: dict[int, list[_RebuiltLanePart]],
    lane_count: int,
    event_order_by_node_id: Optional[dict[int, int]] = None,
    transition_history_by_lane: Optional[dict[int, list[tuple[int, int]]]] = None,
    immutable_lanes: Optional[set[int]] = None,
    composition_id: Optional[str] = None,
    lane_signatures: Optional[list[tuple[Any, ...]]] = None,
    lane_bounds_by_node_id: Optional[dict[int, tuple[int, int]]] = None,
) -> int:
    immutable_lane_set = {int(lane) for lane in (immutable_lanes or set())}
    event_entries: list[tuple[int, _RebuiltLanePart]] = []
    occupancy: dict[int, list[tuple[int, int]]] = {i: [] for i in range(int(lane_count))}
    for lane_idx, parts in rebuilt.items():
        for item in parts:
            t, l, _node, kind, _visible, _class_kind, _event_offset = item
            if int(l) > 0 and kind != "transition":
                occupancy.setdefault(int(lane_idx), []).append((int(t), int(t) + int(l)))
            if kind == "event" and int(lane_idx) not in immutable_lane_set:
                event_entries.append((int(lane_idx), item))

    grouped: dict[tuple[int, int, str, int, int], list[tuple[int, _RebuiltLanePart]]] = defaultdict(list)
    for lane_idx, item in event_entries:
        grouped[_rebuilt_event_group_key(item)].append((int(lane_idx), item))

    moved = 0
    for key, group in sorted(grouped.items(), key=lambda pair: pair[0]):
        if len(group) <= 1:
            continue
        visible, _length, class_kind, _source_start, _source_length = key
        ordered = sorted(
            group,
            key=lambda pair: _rebuilt_event_group_order_key(
                pair[0],
                pair[1],
                event_order_by_node_id,
            ),
        )
        released: list[tuple[int, tuple[int, int]]] = []
        released_transition_spans_by_lane: dict[int, list[tuple[int, int]]] = {}
        for lane_idx, item in group:
            span = (int(item[0]), int(item[0]) + int(item[1]))
            try:
                occupancy[int(lane_idx)].remove(span)
                released.append((int(lane_idx), span))
            except (KeyError, ValueError):
                pass
            spans = _rebuilt_part_transition_spans(int(item[0]), item[2])
            if spans:
                released_transition_spans_by_lane.setdefault(int(lane_idx), []).extend(spans)

        chosen: Optional[list[tuple[int, int, _RebuiltLanePart]]] = None
        for lane_seq in _contiguous_class_lane_sequences(str(class_kind), int(lane_count), len(ordered)):
            if any(int(lane) in immutable_lane_set for lane in lane_seq):
                continue
            planned: list[tuple[int, int, _RebuiltLanePart]] = []
            ok = True
            for target_lane, (_src_lane, item) in zip(lane_seq, ordered):
                if not _rebuilt_lane_in_bounds(item, target_lane, lane_bounds_by_node_id):
                    ok = False
                    break
                if lane_signatures is not None and lane_signatures[_src_lane] != lane_signatures[target_lane]:
                    ok = False
                    break
                t, l, _node, _kind, item_visible, _item_class, event_offset = item
                item_visible_i = int(item_visible) if item_visible is not None else int(visible)
                lane_transition_spans = _rebuilt_transition_spans_for_lane(
                    rebuilt,
                    int(target_lane),
                    transition_history_by_lane,
                )
                lane_transition_spans = _spans_without_exact_matches(
                    lane_transition_spans,
                    released_transition_spans_by_lane.get(int(target_lane), []),
                )
                cand_event_t = _raw_from_visible_with_transition_spans(
                    lane_transition_spans,
                    item_visible_i,
                    int(event_offset),
                )
                cand_t = int(cand_event_t) - int(event_offset)
                cand_span = (int(cand_t), int(cand_t) + int(l))
                if any(
                    _intervals_overlap(cand_span[0], cand_span[1], occ0, occ1)
                    for occ0, occ1 in occupancy.get(int(target_lane), [])
                ):
                    ok = False
                    break
                planned.append((int(target_lane), int(cand_t), item))
            if ok:
                chosen = planned
                break

        if chosen is None:
            for lane_idx, span in released:
                occupancy.setdefault(int(lane_idx), []).append(span)
            continue

        for lane_idx, item in group:
            try:
                rebuilt[int(lane_idx)].remove(item)
            except ValueError:
                pass
        for target_lane, cand_t, item in chosen:
            _old_t, l, node, kind, item_visible, item_class, event_offset = item
            moved_item: _RebuiltLanePart = (
                int(cand_t),
                int(l),
                node,
                str(kind),
                item_visible,
                None if item_class is None else str(item_class),
                int(event_offset),
            )
            rebuilt.setdefault(int(target_lane), []).append(moved_item)
            occupancy.setdefault(int(target_lane), []).append((int(cand_t), int(cand_t) + int(l)))
            if int(target_lane) != int(next(src for src, original in group if original is item)) or int(cand_t) != int(item[0]):
                moved += 1
    return int(moved)


def _restore_rebuilt_aligned_source_order(
    rebuilt: dict[int, list[_RebuiltLanePart]],
    event_order_by_node_id: Optional[dict[int, int]],
    transition_history_by_lane: dict[int, list[tuple[int, int]]],
    immutable_lanes: set[int],
    lane_signatures: Optional[list[tuple[Any, ...]]] = None,
    lane_bounds_by_node_id: Optional[dict[int, tuple[int, int]]] = None,
) -> int:
    """Repair inverted groups using compatible lanes and an atomic group plan.

    Alignment belongs to the logical event, not its attached transition padding.
    Every proposed permutation is validated before any group member is replaced.
    """
    if not event_order_by_node_id:
        return 0
    grouped: dict[tuple[Any, ...], list[tuple[int, _RebuiltLanePart, int]]] = defaultdict(list)
    for lane, parts in rebuilt.items():
        for part in parts:
            if part[3] != "event":
                continue
            primary = _rebuilt_event_primary_node(part)
            source_lane = event_order_by_node_id.get(id(primary))
            inner = _rebuilt_event_primary_sourceclip(part)
            if source_lane is None or inner is None:
                continue
            key = (
                part[4], int(_node_length_units(primary)), str(part[5] or "unknown"),
                int(_aaf_int_start(inner)), int(_aaf_int_length(inner)),
            )
            if key[0] is not None and key[1] > 0 and key[3] >= 0 and key[4] > 0:
                grouped[key].append((int(lane), part, int(source_lane)))

    moved = 0
    for group in grouped.values():
        ordered = sorted(group, key=lambda entry: entry[2])
        source_lanes = [entry[2] for entry in ordered]
        current_lanes = [entry[0] for entry in ordered]
        if len(set(source_lanes)) != len(group) or current_lanes == sorted(current_lanes):
            continue
        target_lanes = sorted(current_lanes)
        if len(set(target_lanes)) != len(group):
            raise RuntimeError("Aligned group shares a destination lane")
        group_ids = {id(part) for _lane, part, _source in group}
        released_spans: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for lane, part, _source in group:
            released_spans[lane].extend(_rebuilt_part_transition_spans(int(part[0]), part[2]))
        candidates = []
        available_lanes = range(len(lane_signatures)) if lane_signatures is not None else sorted(rebuilt)
        for old_lane, part, source_lane in ordered:
            feasible = {}
            for target_lane in available_lanes:
                if not _rebuilt_lane_in_bounds(part, target_lane, lane_bounds_by_node_id):
                    continue
                if (old_lane in immutable_lanes or target_lane in immutable_lanes) and target_lane != old_lane:
                    continue
                if lane_signatures is not None and lane_signatures[source_lane] != lane_signatures[target_lane]:
                    continue
                _raw, length, nodes, kind, visible, class_kind, offset = part
                spans = _rebuilt_transition_spans_for_lane(rebuilt, target_lane, transition_history_by_lane)
                spans = _spans_without_exact_matches(spans, released_spans.get(target_lane, []))
                raw = _raw_from_visible_with_transition_spans(spans, int(visible), int(offset)) - int(offset)
                if raw < 0 or any(
                    id(other) not in group_ids and other[3] != "transition" and
                    _intervals_overlap(raw, raw + int(length), int(other[0]), int(other[0]) + int(other[1]))
                    for other in rebuilt.get(target_lane, [])
                ):
                    continue
                feasible[target_lane] = part if target_lane in immutable_lanes else (raw, length, nodes, kind, visible, class_kind, offset)
            candidates.append(feasible)
        assignment = ordered_lane_assignment(
            [[lane for lane in feasible if lane in target_lanes] for feasible in candidates], current_lanes)
        if assignment is None:
            assignment = ordered_lane_assignment([list(feasible) for feasible in candidates], current_lanes)
        if assignment is None:
            raise RuntimeError("Aligned source order has no compatible placement without overlap")
        for lane, part, _source in group:
            rebuilt[lane] = [other for other in rebuilt[lane] if other is not part]
        for target_lane, feasible in zip(assignment, candidates):
            rebuilt.setdefault(target_lane, []).append(feasible[target_lane])
        moved += sum(old_lane != target for old_lane, target in zip(current_lanes, assignment))
    return moved


def _finalize_rebuilt_parts_without_visible_time_shift(
    rebuilt: dict[int, list[_RebuiltLanePart]],
    lane_count: int,
    event_order_by_node_id: Optional[dict[int, int]] = None,
    immutable_lanes: Optional[set[int]] = None,
    composition_id: Optional[str] = None,
    lane_signatures: Optional[list[tuple[Any, ...]]] = None,
    lane_bounds_by_node_id: Optional[dict[int, tuple[int, int]]] = None,
) -> int:
    """Compact rebuilt events into class zones without changing visible starts."""
    immutable_lane_set = {int(lane) for lane in (immutable_lanes or set())}
    moved = _prune_rebuilt_transition_parts_not_serializable(rebuilt, int(lane_count))
    moved += _compact_rebuilt_aligned_groups_without_visible_time_shift(
        rebuilt,
        int(lane_count),
        event_order_by_node_id,
        None,
        immutable_lane_set,
        lane_signatures=lane_signatures,
        lane_bounds_by_node_id=lane_bounds_by_node_id,
    )
    _rebalance_rebuilt_event_parts_to_visible_times(
        rebuilt,
        int(lane_count),
        None,
    )
    for _pass_idx in range(4):
        def raw_for_visible_on_lane(lane: int, visible_t: int, event_offset: int) -> int:
            return _raw_from_visible_with_transition_spans(
                _rebuilt_transition_spans_for_lane(
                    rebuilt,
                    int(lane),
                    None,
                ),
                int(visible_t),
                int(event_offset),
            )

        compacted = _compact_rebuilt_parts_to_class_zones_without_visible_time_shift(
            rebuilt,
            int(lane_count),
            raw_for_visible_on_lane,
            immutable_lane_set,
            lane_signatures=lane_signatures,
            lane_bounds_by_node_id=lane_bounds_by_node_id,
        )
        if int(compacted) <= 0:
            break
        moved += int(compacted)
        _rebalance_rebuilt_event_parts_to_visible_times(
            rebuilt,
            int(lane_count),
            None,
        )
        moved += _resolve_rebuilt_event_overlaps_without_visible_time_shift(
            rebuilt,
            int(lane_count),
            None,
            immutable_lane_set,
            lane_signatures=lane_signatures,
            lane_bounds_by_node_id=lane_bounds_by_node_id,
        )
    moved += _compact_rebuilt_aligned_groups_without_visible_time_shift(
        rebuilt,
        int(lane_count),
        event_order_by_node_id,
        None,
        immutable_lane_set,
        lane_signatures=lane_signatures,
        lane_bounds_by_node_id=lane_bounds_by_node_id,
    )
    _rebalance_rebuilt_event_parts_to_visible_times(
        rebuilt,
        int(lane_count),
        None,
    )
    repaired_order = _restore_rebuilt_aligned_source_order(
        rebuilt, event_order_by_node_id, {},
        immutable_lane_set, lane_signatures, lane_bounds_by_node_id,
    )
    if repaired_order:
        moved += repaired_order
        _rebalance_rebuilt_event_parts_to_visible_times(
            rebuilt, int(lane_count),
        )
    return int(moved)


def _validate_serialized_event_visible_positions(
    nodes: list[Any], expected_by_node_id: dict[int, int],
) -> None:
    """Validate actual component positions, independent of placement metadata."""
    remaining = dict(expected_by_node_id)
    visible_position = 0
    for node in nodes:
        expected = remaining.pop(id(node), None)
        if expected is not None and visible_position != int(expected):
            raise RuntimeError(
                "YAMNet lane layout: serialized event visible position changed "
                f"({visible_position} != {expected})"
            )
        length = int(_node_length_units(node))
        visible_position += -length if _is_transition_node(node) else length
    if remaining:
        raise RuntimeError("YAMNet lane layout: serialized events are missing")


def _transition_part_is_serializable(
    t: int,
    l: int,
    event_edges: set[int],
    event_intervals: list[tuple[int, int]],
) -> bool:
    t0 = int(t)
    t1 = int(t) + int(l)
    if t0 not in event_edges and t1 not in event_edges:
        return False
    return not any(
        _intervals_overlap(t0, t1, int(ev_t0), int(ev_t1))
        for ev_t0, ev_t1 in event_intervals
    )


def _is_transition_node(node: Any) -> bool:
    return "Transition" in node.__class__.__name__


def _event_owned_transition_indices(nodes: list[Any]) -> dict[int, int]:
    owners: dict[int, int] = {}

    def is_eventish(node: Any) -> bool:
        return _is_operation_group(node) or _is_sourceclip(node)

    for idx, node in enumerate(nodes):
        if not _is_transition_node(node):
            continue
        prev_is_event = idx > 0 and is_eventish(nodes[idx - 1])
        next_is_event = idx + 1 < len(nodes) and is_eventish(nodes[idx + 1])
        if next_is_event:
            owners[int(idx)] = int(idx + 1)
        elif prev_is_event:
            owners[int(idx)] = int(idx - 1)
    return owners


def _event_transition_edge_owners(nodes: list[Any]) -> tuple[dict[int, int], dict[int, int]]:
    pre_owner_by_transition: dict[int, int] = {}
    post_owner_by_transition: dict[int, int] = {}

    def is_eventish(node: Any) -> bool:
        return _is_operation_group(node) or _is_sourceclip(node)

    for idx, node in enumerate(nodes):
        if not _is_transition_node(node):
            continue
        if idx > 0 and is_eventish(nodes[idx - 1]):
            post_owner_by_transition[int(idx)] = int(idx - 1)
        if idx + 1 < len(nodes) and is_eventish(nodes[idx + 1]):
            pre_owner_by_transition[int(idx)] = int(idx + 1)
    return pre_owner_by_transition, post_owner_by_transition


def _copy_transition_node_for_shared_edge(node: Any) -> Any:
    if not _is_transition_node(node):
        return node
    copier = getattr(node, "copy", None)
    if callable(copier):
        return copier()
    try:
        return copy.deepcopy(node)
    except Exception as exc:
        raise RuntimeError(
            "YAMNet lane layout PyAAF2 fallback cannot duplicate a shared transition"
        ) from exc


def _sequence_transition_protected_parts(
    seq: Any,
) -> tuple[list[tuple[int, int, Any]], set[int]]:
    entries: list[tuple[int, int, Any]] = []
    transition_indices: list[int] = []
    try:
        top = seq.components
        tn = len(top)
    except Exception:
        return [], set()
    pos = 0
    for ti in range(tn):
        try:
            node = top[ti]
        except Exception:
            break
        ln = _node_length_units(node)
        entries.append((int(pos), int(ln), node))
        if _is_transition_node(node):
            transition_indices.append(ti)
        pos += int(ln)
    protected: set[int] = set()
    for ti in transition_indices:
        for idx in (ti - 1, ti, ti + 1):
            if 0 <= idx < len(entries):
                protected.add(idx)
    return [entries[i] for i in sorted(protected)], protected


def _sequence_transition_protected_groups(
    seq: Any,
) -> list[tuple[int, int, list[Any], set[int]]]:
    entries, protected = _sequence_transition_protected_parts(seq)
    if not entries or not protected:
        return []
    by_index: dict[int, tuple[int, int, Any]] = {}
    try:
        top = seq.components
        pos = 0
        for ti in range(len(top)):
            node = top[ti]
            ln = _node_length_units(node)
            if ti in protected:
                by_index[ti] = (int(pos), int(ln), node)
            pos += int(ln)
    except Exception:
        return []
    groups: list[tuple[int, int, list[Any], set[int]]] = []
    current: list[int] = []
    for idx in sorted(by_index):
        if current and idx != current[-1] + 1:
            groups.append(_transition_group_from_indices(current, by_index))
            current = []
        current.append(idx)
    if current:
        groups.append(_transition_group_from_indices(current, by_index))
    return groups


def _transition_group_from_indices(
    indices: list[int],
    by_index: dict[int, tuple[int, int, Any]],
) -> tuple[int, int, list[Any], set[int]]:
    start = by_index[indices[0]][0]
    nodes = [by_index[i][2] for i in indices]
    length = sum(max(0, int(by_index[i][1])) for i in indices)
    return int(start), int(length), nodes, set(indices)


def _primary_layout_node(nodes: list[Any]) -> Optional[Any]:
    for node in nodes:
        if _is_operation_group(node) or _is_sourceclip(node):
            return node
    return None


def _assign_target_lanes(
    events: List[dict[str, Any]],
    n: int,
    reserved_by_lane: Optional[dict[int, list[tuple[int, int]]]] = None,
    blocked_target_lanes: Optional[set[int]] = None,
    transition_spans_by_lane: Optional[dict[int, list[tuple[int, int]]]] = None,
    lane_capacity_by_lane: Optional[dict[int, int]] = None,
    lane_signature_by_lane: Optional[dict[int, tuple[Any, ...]]] = None,
    lane_mutable_by_lane: Optional[dict[int, bool]] = None,
) -> None:
    """
    Заполняет ``target_lane``; при отсутствии места оставляет ``src_lane``.

    Порядок: **шум → речь → музыка**, чтобы короткие/сомнительные «речевые» hits YAMNet
    (свист, SFX) не занимали верхние дорожки до размещения явного шума у центра.
    """
    apply_immutable_aligned_lane_bounds(
        events, {lane for lane, mutable in (lane_mutable_by_lane or {}).items() if not mutable}, n
    )
    raw_occ: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    if reserved_by_lane:
        for lane, spans in reserved_by_lane.items():
            if lane < 0 or lane >= n:
                continue
            for t0, t1 in spans:
                if int(t1) > int(t0):
                    raw_occ[lane].append((int(t0), int(t1)))
    raw_state = _ModelLaneOccupancyState(
        raw=raw_occ,
        visible=[[] for _ in range(n)],
        lane_capacity_by_lane=lane_capacity_by_lane,
    )

    def target_allowed(e: dict[str, Any], lane: int) -> bool:
        return _model_target_lane_allowed_for_event(
            e,
            int(lane),
            blocked_target_lanes=blocked_target_lanes,
            lane_mutable_by_lane=lane_mutable_by_lane,
            lane_signature_by_lane=lane_signature_by_lane,
            transition_spans_by_lane=transition_spans_by_lane,
        )

    def _t_edit(e: dict[str, Any]) -> int:
        return _model_event_visible_start_for_placement(e)

    def _target_raw_t(e: dict[str, Any], lane: int) -> int:
        return _model_event_target_raw_start_for_lane(e, int(lane), transition_spans_by_lane)

    source_spans = _ModelSourceSpanTracker.from_events(events, n, raw_state)

    def class_lane_preferences() -> tuple[list[int], list[int], list[int], list[int]]:
        speech_pref, noise_pref, music_pref, _unknown_pref, lanes = _lane_zone_preferences(n)
        return speech_pref, noise_pref, music_pref, lanes

    def _physical_class_band(kind: str) -> list[int]:
        return _model_physical_class_band(kind, int(n))

    def _source_window_from_node(node: Any) -> Optional[tuple[int, int]]:
        inner = _event_inner_sourceclip(node)
        if inner is None:
            return None
        try:
            return (int(_aaf_int_start(inner)), int(_aaf_int_length(inner)))
        except Exception:
            return None

    def _event_source_window_key(e: dict[str, Any]) -> tuple[int, int]:
        return _model_event_source_window_key(e, _source_window_from_node)

    def _aligned_layout_group_key(e: dict[str, Any]) -> tuple[int, int, int, int]:
        return _model_aligned_layout_group_key(e, _source_window_from_node)

    def _aligned_kind_group_key(e: dict[str, Any]) -> Optional[tuple[int, int, int, int]]:
        return _model_aligned_kind_group_key(e, _source_window_from_node)

    def _same_kind_aligned_group_key(e: dict[str, Any]) -> Optional[tuple[object, ...]]:
        key = _aligned_kind_group_key(e)
        if key is None:
            return None
        return (*key, str(e.get("kind") or "unknown"))

    lane_pref_speech, lane_pref_noise, music_lane_pref, lane_pref_all = class_lane_preferences()
    initial_placement = _ModelLaneInitialPlacementPlanner(
        lane_count=n,
        raw_state=raw_state,
        source_spans=source_spans,
        lane_pref_all=lane_pref_all,
        target_allowed=target_allowed,
        transition_spans_by_lane=transition_spans_by_lane,
        aligned_kind_group_key=_aligned_kind_group_key,
        aligned_layout_group_key=_aligned_layout_group_key,
    )
    speech = [e for e in events if e["kind"] == "speech"]
    initial_placement.place_events_preserving_aligned_order(
        speech,
        lane_pref_speech,
        kind="speech",
        sort_key=lambda e: (_t_edit(e), int(e["L"])),
    )

    noise = [e for e in events if e["kind"] == "noise"]
    initial_placement.place_events_preserving_aligned_order(
        noise,
        lane_pref_noise,
        kind="noise",
        sort_key=lambda e: (_t_edit(e), int(e["L"])),
    )

    unknown = [e for e in events if e.get("kind") == "unknown"]

    # Unknown events carry source-lane bounds; their occupied intervals stay fixed.
    lane_pref_unknown = lane_pref_noise + [i for i in lane_pref_speech if i not in lane_pref_noise]
    if not lane_pref_unknown:
        lane_pref_unknown = lane_pref_all
    initial_placement.place_events_preserving_aligned_order(
        unknown,
        lane_pref_unknown,
        kind="unknown",
        sort_key=lambda e: (_t_edit(e), int(e["L"])),
    )

    music = [e for e in events if e["kind"] == "music"]
    # For equal T stereo/duplicate music, keep the original vertical order while packing downward.
    music.sort(key=lambda e: (int(e["T_edit"]), -int(e["src_lane"]), int(e["L"])))
    initial_placement.place_events_preserving_aligned_order(
        music,
        music_lane_pref,
        kind="music",
        sort_key=lambda e: (int(e["T_edit"]), -int(e["src_lane"]), int(e["L"])),
    )

    speech_events = [e for e in events if e.get("kind") == "speech"]
    speech_events.sort(key=lambda e: (int(e["T"]), int(e["src_lane"]), -int(e["L"])))
    packed_speech_state = _model_compact_events_to_preferred_lanes(
        events,
        speech_events,
        lane_count=n,
        preferred_lanes=lane_pref_speech,
        fallback_lanes=lane_pref_all,
        target_allowed=target_allowed,
        reserved_by_lane=reserved_by_lane,
        transition_spans_by_lane=transition_spans_by_lane,
        lane_capacity_by_lane=lane_capacity_by_lane,
    )

    _model_promote_longer_speech_events(
        speech_events,
        lane_count=n,
        occupancy=packed_speech_state,
        target_allowed=target_allowed,
        transition_spans_by_lane=transition_spans_by_lane,
    )

    _model_compact_events_to_preferred_lanes(
        events,
        music,
        lane_count=n,
        preferred_lanes=music_lane_pref,
        fallback_lanes=lane_pref_all,
        target_allowed=target_allowed,
        reserved_by_lane=reserved_by_lane,
        transition_spans_by_lane=transition_spans_by_lane,
        lane_capacity_by_lane=lane_capacity_by_lane,
    )

    _model_compact_aligned_groups_to_visible_gaps(
        events,
        lane_count=n,
        target_allowed=target_allowed,
        target_raw_for_lane=lambda event, lane_i: _target_raw_t(event, int(lane_i)),
        aligned_kind_group_key=_same_kind_aligned_group_key,
        allowed_overlap=lambda other, item: (
            str(other.get("kind") or "unknown") == str(item.get("kind") or "unknown")
            and _planned_overlap_is_shared_transition_edge(other, item)
        ),
    )

    _model_preserve_aligned_target_order_after_compaction(
        events,
        target_allowed=target_allowed,
        target_raw_for_lane=lambda event, lane_i: _target_raw_t(event, int(lane_i)),
        aligned_kind_group_key=_same_kind_aligned_group_key,
    )
    _model_preserve_same_kind_xfade_chains(
        events,
        lane_count=n,
        target_allowed=target_allowed,
        target_raw_for_lane=lambda event, lane_i: _target_raw_t(event, int(lane_i)),
        allowed_overlap=lambda _other, _item: False,
    )
    _model_final_bounded_compact_class_events(
        events,
        lane_count=n,
        target_allowed=target_allowed,
        target_raw_for_lane=lambda event, lane_i: _target_raw_t(event, int(lane_i)),
        aligned_group_key=_same_kind_aligned_group_key,
        reserved_by_lane=reserved_by_lane,
        transition_spans_by_lane=transition_spans_by_lane,
        lane_capacity_by_lane=lane_capacity_by_lane,
    )


def _validate_lane_layout_plan(events: list[dict[str, Any]], n_lanes: int) -> None:
    _model_validate_lane_layout_plan(events, int(n_lanes))


def _is_sdk_xml_exporter_unavailable_error(exc: BaseException) -> bool:
    msg = str(exc)
    if "aaffmtconv -xml failed" not in msg:
        return False
    crash_codes = (
        "3221225477",
        "3221226519",
        "-1073741819",
        "-1073740777",
        "0XC0000005",
        "C0000005",
        "0XC0000375",
        "C0000375",
    )
    upper = msg.upper()
    return any(code in upper for code in crash_codes)


def _set_sequence_components_value(seq: Any, values: list[Any]) -> None:
    top = seq.components
    try:
        top.value = values
        return
    except Exception:
        pass
    try:
        seq.components.value = values
        return
    except Exception:
        pass
    try:
        seq.components = values
        return
    except Exception:
        pass
    raise RuntimeError("YAMNet lane layout could not replace Sequence components")


def _apply_lane_layout_via_pyaaf2_rebuild(
    *,
    aaf_path: Path,
    events: list[dict[str, Any]],
    structural_events: list[LaneLayoutEvent],
    n_lanes: int,
    cfg: FilterConfig,
    runtime_essence_paths: Optional[Dict[Any, Path]],
    work_dir: Optional[Path],
    cancel_check: Callable[[], None],
    progress_callback: Optional[Callable[[float, float], None]],
    log_callback: Optional[Callable[[str], None]],
    immutable_lanes: Optional[set[int]] = None,
    composition_id: Optional[str] = None,
    order_tracks_by_class: bool = False,
    result_out: Optional[dict[str, Any]] = None,
) -> int:
    del cfg, runtime_essence_paths, work_dir, progress_callback

    def _emit(msg: str) -> None:
        if log_callback is not None:
            log_callback(msg)
        else:
            logger.info("%s", msg)

    immutable_lane_set = {int(lane) for lane in (immutable_lanes or set())}
    planned_by_top: dict[tuple[int, int], dict[str, Any]] = {}
    planned_by_identity: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for ev in events:
        if "target_lane" not in ev:
            continue
        _index_lane_layout_plan_event(
            ev,
            planned_by_identity=planned_by_identity,
            planned_by_top=planned_by_top,
        )

    if not planned_by_top:
        return 0

    with open_aaf_lenient(aaf_path, "r+") as aaf:
        comps = list(aaf.content.compositionmobs())
        if not comps:
            raise RuntimeError("YAMNet lane layout PyAAF2 fallback: CompositionMob not found")
        comp = select_aaf_composition(comps, composition_id)
        sound_lanes, _skipped = _sound_sequence_timeline_slots(comp, cancel_check=cancel_check)
        if len(sound_lanes) != int(n_lanes):
            raise RuntimeError(
                "YAMNet lane layout PyAAF2 fallback: sound lane count changed "
                f"({len(sound_lanes)} != {int(n_lanes)})"
            )

        silent_indices: set[tuple[int, int]] = set()
        for item in structural_events:
            if item.kind != "silence":
                continue
            lane, index = int(item.src_lane), int(item.top_idx)
            if not (0 <= lane < n_lanes) or item.target_lane != lane:
                raise RuntimeError("YAMNet lane layout: planned silence lane mismatch")
            nodes = list(sound_lanes[lane][1].components)
            key = (lane, index)
            if (not 0 <= index < len(nodes) or key in silent_indices or key in planned_by_top
                    or item.silence_signature is None
                    or silence_replacement_signature(nodes, index) != item.silence_signature
                    or _node_length_units(nodes[index]) != int(item.L)
                    or sum(_node_length_units(nd) for nd in nodes[:index]) != int(item.T_edit)):
                raise RuntimeError("YAMNet lane layout: planned silence identity mismatch")
            silent_indices.add(key)

        lane_signatures = [_lane_layout_signature(slot, seq) for slot, seq in sound_lanes]
        for event in events:
            source_lane = int(event["src_lane"])
            target_lane = int(event.get("target_lane", source_lane))
            if not (0 <= source_lane < n_lanes and 0 <= target_lane < n_lanes):
                raise RuntimeError("YAMNet lane layout: planned lane out of range")
            if source_lane != target_lane and (
                    source_lane in immutable_lane_set or target_lane in immutable_lane_set):
                raise RuntimeError("YAMNet lane layout: move touches an immutable lane")
            if lane_signatures[source_lane] != lane_signatures[target_lane]:
                raise RuntimeError("YAMNet lane layout: incompatible lane clock or wrapper")

        orig_totals: list[int] = []
        orig_declared_lengths: list[int] = []
        dd_sample: Optional[Any] = None
        inner_sample: Optional[Any] = None
        for _slot, seq in sound_lanes:
            total = _sequence_top_total_length(seq.components)
            orig_totals.append(int(total))
            orig_declared_lengths.append(_sequence_declared_length(seq, int(total)))

        rebuilt: dict[int, list[_RebuiltLanePart]] = {i: [] for i in range(int(n_lanes))}
        event_order_by_node_id: dict[int, int] = {}
        lane_bounds_by_node_id: dict[int, tuple[int, int]] = {}
        matched_event_ids: set[int] = set()
        transition_sources: dict[int, Any] = {}
        moved = 0

        for lane_idx, (_slot, seq) in enumerate(sound_lanes):
            cancel_check()
            if int(lane_idx) in immutable_lane_set:
                continue
            top = seq.components
            try:
                nodes = [top[i] for i in range(len(top))]
            except Exception:
                nodes = []
            try:
                pre_owner_by_transition, post_owner_by_transition = _event_transition_edge_owners(nodes)
            except Exception:
                pre_owner_by_transition = {}
                post_owner_by_transition = {}
            owned_transition_indices = set(pre_owner_by_transition) | set(post_owner_by_transition)

            pos_by_idx: dict[int, int] = {}
            len_by_idx: dict[int, int] = {}
            transition_before_by_idx: dict[int, int] = {}
            pos = 0
            transition_before = 0
            for ti, node in enumerate(nodes):
                ln = max(0, int(_node_length_units(node)))
                pos_by_idx[int(ti)] = int(pos)
                len_by_idx[int(ti)] = int(ln)
                transition_before_by_idx[int(ti)] = int(transition_before)
                if _is_transition_node(node):
                    transition_before += int(ln)
                pos += int(ln)

            for ti, node in enumerate(nodes):
                cancel_check()
                ln = int(len_by_idx.get(int(ti), 0))
                if ln <= 0 or _is_filler(node) or (int(lane_idx), int(ti)) in silent_indices:
                    continue
                if _is_transition_node(node):
                    if int(ti) not in owned_transition_indices:
                        rebuilt[int(lane_idx)].append(
                            (int(pos_by_idx[ti]), int(ln), node, "transition", None, None, 0)
                        )
                    continue
                if not (_is_operation_group(node) or _is_sourceclip(node)):
                    rebuilt[int(lane_idx)].append(
                        (int(pos_by_idx[ti]), int(ln), node, "structural", None, None, 0)
                    )
                    continue

                inner = _event_inner_sourceclip(node)
                if inner is None:
                    rebuilt[int(lane_idx)].append(
                        (
                            int(pos_by_idx[ti]),
                            int(ln),
                            node,
                            "structural",
                            None,
                            None,
                            0,
                        )
                    )
                    continue
                if inner is not None and dd_sample is None:
                    try:
                        dd_sample = inner["DataDefinition"].value
                    except Exception:
                        dd_sample = None
                    inner_sample = inner

                source_id = ""
                source_track = 0
                source_start = -1
                source_length = -1
                if inner is not None:
                    source_id = _sourceclip_source_id(inner)
                    source_track = _sourceclip_source_track(inner)
                    source_start = int(_aaf_int_start(inner))
                    source_length = int(_aaf_int_length(inner))
                visible_t = int(pos_by_idx[ti]) - 2 * int(transition_before_by_idx.get(int(ti), 0))
                ev = _match_lane_layout_plan_event(
                    src_lane=int(lane_idx),
                    top_idx=int(ti),
                    visible_t=int(visible_t),
                    length=int(ln),
                    source_id=source_id,
                    source_track=int(source_track),
                    source_start=int(source_start),
                    source_length=int(source_length),
                    planned_by_identity=planned_by_identity,
                    planned_by_top=planned_by_top,
                    matched_event_ids=matched_event_ids,
                )
                target_lane = int(lane_idx)
                event_t = int(pos_by_idx[ti])
                class_kind = "unknown"
                if ev is not None:
                    target_lane = int(ev.get("target_lane", lane_idx))
                    event_t = int(ev.get("target_T", ev.get("T", pos_by_idx[ti])))
                    class_kind = str(ev.get("kind") or "unknown")
                    if target_lane != int(lane_idx):
                        moved += 1
                if target_lane < 0 or target_lane >= int(n_lanes):
                    raise RuntimeError(
                        f"YAMNet lane layout PyAAF2 fallback: target lane out of range ({target_lane})"
                    )
                event_order_by_node_id[id(node)] = int(ev.get("src_lane", lane_idx)) if ev is not None else int(lane_idx)

                if ev is not None and ev.get("lane_bounds") is not None:
                    lane_bounds_by_node_id[id(node)] = tuple(ev["lane_bounds"])

                part_nodes: list[Any] = [node]
                part_t = int(event_t)
                part_l = int(ln)
                event_offset = 0
                pre_idx = int(ti) - 1
                if pre_idx >= 0 and pre_owner_by_transition.get(pre_idx) == int(ti):
                    pre_l = int(len_by_idx.get(pre_idx, 0))
                    if pre_l > 0:
                        pre_node = nodes[pre_idx]
                        if post_owner_by_transition.get(pre_idx) is not None:
                            pre_node = _copy_transition_node_for_shared_edge(pre_node)
                            transition_sources[id(pre_node)] = nodes[pre_idx]
                        part_nodes = [pre_node] + part_nodes
                        part_t -= int(pre_l)
                        part_l += int(pre_l)
                        event_offset = int(pre_l)
                post_idx = int(ti) + 1
                if post_idx < len(nodes) and post_owner_by_transition.get(post_idx) == int(ti):
                    post_l = int(len_by_idx.get(post_idx, 0))
                    if post_l > 0:
                        part_nodes = part_nodes + [nodes[post_idx]]
                        part_l += int(post_l)
                if part_t < 0:
                    first_name = _clip_name(aaf, node)
                    raise RuntimeError(
                        "YAMNet lane layout PyAAF2 fallback produced negative placement"
                        + (f": {first_name}" if first_name else "")
                    )
                rebuilt[int(target_lane)].append(
                    (
                        int(part_t),
                        int(part_l),
                        part_nodes,
                        "event",
                        int(ev.get("T_edit", ev.get("T_nuendo", ev.get("T", event_t))))
                        if ev is not None
                        else int(pos_by_idx[ti]),
                        class_kind,
                        int(event_offset),
                    )
                )

        missing = [
            (int(ev.get("src_lane", -1)), int(ev.get("top_idx", -1)))
            for ev in events
            if "target_lane" in ev and id(ev) not in matched_event_ids
            and int(ev.get("src_lane", -1)) not in immutable_lane_set
        ]
        missing = sorted(missing)
        if missing:
            raise RuntimeError(
                "YAMNet lane layout PyAAF2 fallback: planned clips were not found in writable AAF "
                f"({len(missing)} missing)"
            )

        _rebalance_rebuilt_event_parts_to_visible_times(rebuilt, int(n_lanes))
        moved += _resolve_rebuilt_event_overlaps_without_visible_time_shift(
            rebuilt,
            int(n_lanes),
            immutable_lanes=immutable_lane_set,
            lane_signatures=lane_signatures,
            lane_bounds_by_node_id=lane_bounds_by_node_id,
        )
        moved += _finalize_rebuilt_parts_without_visible_time_shift(
            rebuilt,
            int(n_lanes),
            event_order_by_node_id,
            immutable_lanes=immutable_lane_set,
            lane_signatures=lane_signatures,
            lane_bounds_by_node_id=lane_bounds_by_node_id,
        )

        # Writer repair must not bypass the planner's lane compatibility boundary.
        for target_lane, parts in rebuilt.items():
            for item in parts:
                if not _rebuilt_lane_in_bounds(item, target_lane, lane_bounds_by_node_id):
                    raise RuntimeError("YAMNet lane layout: repair crossed a fixed aligned member")
                _t, _length, nodes, _kind, _visible, _class, _offset = item
                for node in nodes if isinstance(nodes, list) else [nodes]:
                    source_lane = event_order_by_node_id.get(id(node))
                    if source_lane is not None and lane_signatures[source_lane] != lane_signatures[target_lane]:
                        raise RuntimeError("YAMNet lane layout: repair crossed incompatible lane clock or wrapper")

        for _lane_idx, (_slot, seq) in enumerate(sound_lanes):
            if int(_lane_idx) in immutable_lane_set:
                continue
            _set_sequence_components_value(seq, [])

        for lane_idx, (slot, seq) in enumerate(sound_lanes):
            cancel_check()
            if int(lane_idx) in immutable_lane_set:
                continue
            total_len = int(orig_totals[lane_idx]) if lane_idx < len(orig_totals) else 0
            declared_len = (
                int(orig_declared_lengths[lane_idx])
                if lane_idx < len(orig_declared_lengths)
                else int(total_len)
            )
            parts = sorted(
                rebuilt.get(int(lane_idx), []),
                key=lambda item: (
                    int(item[0]),
                    0 if item[3] in ("transition", "structural") else 1,
                    int(item[1]),
                ),
            )
            event_edges: set[int] = set()
            event_intervals: list[tuple[int, int]] = []
            for t_edge, l_edge, _node_edge, kind_edge, _visible_edge, _class_kind_edge, _event_offset_edge in parts:
                if kind_edge == "event":
                    event_edges.add(int(t_edge))
                    event_edges.add(int(t_edge) + int(l_edge))
                    event_intervals.append((int(t_edge), int(t_edge) + int(l_edge)))
            parts = [
                item
                for item in parts
                if item[3] != "transition"
                or _transition_part_is_serializable(
                    int(item[0]),
                    int(item[1]),
                    event_edges,
                    event_intervals,
                )
            ]

            new_list: list[Any] = []
            cur = 0
            emitted_transitions: dict[int, Any] = {}
            for T, L, node_or_nodes, kind, _visible, _class_kind, _event_offset in parts:
                out_T = int(T)
                out_L = int(L)
                out_nodes = node_or_nodes
                if out_T > int(cur):
                    new_list.append(_sound_filler(aaf, out_T - int(cur), dd_sample, inner_sample))
                    cur = int(out_T)
                if out_T < int(cur):
                    out_T, out_L, out_nodes = _trim_leading_rebuilt_transitions_to_cursor(
                        out_T,
                        out_L,
                        out_nodes,
                        cur,
                        emitted_transitions=emitted_transitions,
                        transition_sources=transition_sources,
                    )
                    seq_nodes = out_nodes if isinstance(out_nodes, list) else [out_nodes]
                    if int(out_L) <= 0 or not seq_nodes:
                        continue
                if out_T < int(cur):
                    first_node = (
                        out_nodes[0]
                        if isinstance(out_nodes, list) and out_nodes
                        else out_nodes
                    )
                    nm = _clip_name(aaf, first_node)
                    raise RuntimeError(
                        f"YAMNet lane layout produced overlapping blocks on lane {lane_idx}"
                        + (f": {nm}" if nm else "")
                    )
                position = int(out_T)
                for node in out_nodes if isinstance(out_nodes, list) else [out_nodes]:
                    if _is_transition_node(node):
                        emitted_transitions[position] = transition_sources.get(id(node), node)
                    position += max(0, int(_node_length_units(node)))
                if isinstance(out_nodes, list):
                    new_list.extend(out_nodes)
                else:
                    new_list.append(out_nodes)
                cur = int(out_T) + int(out_L)

            # Padding is measured in visible units; moved transitions change the
            # raw component sum without changing the lane's required duration.
            final_declared_len = _rebuilt_lane_declared_length(int(cur), new_list)
            if final_declared_len < declared_len:
                new_list.append(_sound_filler(aaf, declared_len - final_declared_len, dd_sample, inner_sample))
                final_declared_len = declared_len
            if not new_list:
                pad = max(1, declared_len)
                new_list = [_sound_filler(aaf, pad, dd_sample, inner_sample)]
                final_declared_len = pad
            expected_positions = {
                id(_rebuilt_event_primary_node(item)): int(item[4])
                for item in parts if item[3] == "event" and item[4] is not None
            }
            _validate_serialized_event_visible_positions(new_list, expected_positions)
            _set_sequence_components_value(seq, new_list)
            _set_component_length(seq, final_declared_len)
            try:
                seg0 = getattr(slot, "segment", None)
                if seg0 is not None and seg0 is seq:
                    _set_component_length(seg0, final_declared_len)
                wrapper = _slot_segment_sequence_wrapper(slot, seq)
                if wrapper is not None:
                    _set_component_length(wrapper, final_declared_len)
            except Exception:
                pass

        if order_tracks_by_class:
            kinds_by_lane = [set() for _ in range(n_lanes)]
            for event in events:
                if int(event["src_lane"]) in immutable_lane_set:
                    kinds_by_lane[int(event["src_lane"])].add(event.get("kind") or "unknown")
            for lane, parts in rebuilt.items():
                for part in parts:
                    if part[3] == "event":
                        kinds_by_lane[lane].add(part[5] or "unknown")
                    elif part[3] == "structural":
                        kinds_by_lane[lane].add("unknown")
            lane_order = class_ordered_lanes(kinds_by_lane, protected=immutable_lane_set)
            reorder_pyaaf2_slots(comp, [slot for slot, _seq in sound_lanes], lane_order)
            if result_out is not None:
                result_out["lane_order"] = lane_order
        _emit("YAMNet дорожки: применено через PyAAF2 writer без SDK XML exporter.")
        return int(moved)


def _write_lane_layout_with_sdk_or_pyaaf2_fallback(
    *,
    input_aaf: Path,
    output_aaf: Path,
    events: list[dict[str, Any]],
    structural_events: list[LaneLayoutEvent],
    n_lanes: int,
    cfg: FilterConfig,
    runtime_essence_paths: Optional[Dict[Any, Path]],
    work_dir: Optional[Path],
    cancel_check: Callable[[], None],
    log_callback: Optional[Callable[[str], None]],
    result_out: Optional[dict[str, Any]],
    progress_callback: Optional[Callable[[float, float], None]] = None,
    immutable_lanes: Optional[set[int]] = None,
    composition_id: Optional[str] = None,
    order_tracks_by_class: bool = False,
) -> int:
    return _write_lane_layout_with_writer_facade(
        input_aaf=input_aaf,
        output_aaf=output_aaf,
        events=events,
        structural_events=structural_events,
        n_lanes=n_lanes,
        cfg=cfg,
        runtime_essence_paths=runtime_essence_paths,
        work_dir=work_dir,
        cancel_check=cancel_check,
        log_callback=log_callback,
        result_out=result_out,
        progress_callback=progress_callback,
        order_tracks_by_class=order_tracks_by_class,
        build_sdk_events=_build_sdk_lane_layout_events,
        class_zone_lane_order=_class_zone_lane_order,
        pyaaf2_rebuild=_apply_lane_layout_via_pyaaf2_rebuild,
        sdk_xml_writer=apply_lane_layout_via_aaf_sdk_xml,
        composition_id=composition_id,
        sdk_exporter_unavailable_error=_is_sdk_xml_exporter_unavailable_error,
        immutable_lanes=set(immutable_lanes or set()),
        pyaaf2_output_validator=lambda output: _validate_pyaaf2_lane_layout_output_for_sdk_open(
            output,
            work_dir=(
                Path(work_dir) / "lane_layout_pyaaf2_validate"
                if work_dir is not None
                else (
                    Path(output_aaf).parent
                    / "__aaf_tool_work"
                    / f"{Path(output_aaf).stem}.__pyaaf2_validate"
                )
            ),
            cancel_check=cancel_check,
            aaf_tools_dir=cfg.aaf_tools_dir,
        ),
    )


def _validate_pyaaf2_lane_layout_output_for_sdk_open(
    output_aaf: Path,
    *,
    work_dir: Path,
    cancel_check: Optional[Callable[[], None]] = None,
    aaf_tools_dir: Optional[Path] = None,
) -> None:
    """
    Reject PyAAF2 lane-layout output that strict SDK readers cannot open.

    XML export is not a container-readability test: the SDK can reject valid
    binary AAF graphs during XML serialization. Require a binary SDK read/write.

    A PyAAF2 write can return successfully while leaving a physically corrupted
    OLE/CFB container. That output must be treated as a failed writer backend so
    the facade can fall back to the SDK XML writer instead of returning a broken
    processed AAF.
    """
    from aaf_io.sdk_tools import (
        find_aaffmtconv,
        find_comaafinfo,
        run_aaffmtconv_to_structured_storage,
        run_comaafinfo,
    )

    output_aaf = Path(output_aaf)
    cinfo = find_comaafinfo(aaf_tools_dir)
    code, out = run_comaafinfo(cinfo, output_aaf, cancel_check=cancel_check)
    if int(code) != 0:
        detail = (out or "").strip()
        if detail:
            detail = ": " + detail.splitlines()[0][:300]
        raise RuntimeError(f"ComAAFInfo failed after PyAAF2 lane layout (exit {code}){detail}")

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    validation_aaf = work_dir / f"{output_aaf.stem}.__pyaaf2_sdk_validate.aaf"
    try:
        conv = find_aaffmtconv(aaf_tools_dir)
        run_aaffmtconv_to_structured_storage(
            conv,
            output_aaf,
            validation_aaf,
            cancel_check=cancel_check,
        )
    finally:
        try:
            validation_aaf.unlink(missing_ok=True)
        except Exception:
            pass


def apply_experimental_yamnet_lane_layout(
    aaf_path: Path,
    cfg: FilterConfig,
    runtime_essence_paths: Optional[Dict[Any, Path]] = None,
    work_dir: Optional[Path] = None,
    cancel_event: Optional[threading.Event] = None,
    progress_callback: Optional[Callable[[float, float], None]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    result_out: Optional[dict[str, Any]] = None,
) -> int:
    _last_yield_t = time.monotonic()
    _yield_interval_sec = 5.0

    def _emit(msg: str) -> None:
        if log_callback is not None:
            log_callback(msg)
        else:
            logger.info("%s", msg)

    def _check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise SpeechFilterCancelled()
        nonlocal _last_yield_t
        now = time.monotonic()
        if now - _last_yield_t >= _yield_interval_sec:
            _last_yield_t = now
            time.sleep(0)

    aaf_path = Path(aaf_path)
    if not cfg.experimental_yamnet_lane_layout:
        if result_out is not None:
            try:
                result_out["moved"] = 0
                result_out["sdk_normalized"] = False
                result_out["sdk_xml_fallback"] = False
            except Exception:
                pass
        return 0
    if cfg.media_search_roots is None:
        from dataclasses import replace

        cfg = replace(cfg, media_search_roots=_configured_media_roots(aaf_path))

    if result_out is not None:
        try:
            result_out["moved"] = 0
            result_out["sdk_normalized"] = False
            result_out["sdk_xml_fallback"] = False
        except Exception:
            pass

    # Cache for resolving Nuendo-style clip names (MasterMob["Name"]).
    _name_cache: dict[Any, Any] = {}

    def _clip_name(aaf_obj: Any, node: Any) -> str:
        try:
            r = resolve_clip_display_name(aaf_obj, node, cache=_name_cache)
            return (r.display_name or "").strip()
        except Exception:
            return ""

    def _reserved_transition_intervals(
        anchors_by_lane: dict[int, list[tuple[int, int, Any]]],
    ) -> dict[int, list[tuple[int, int]]]:
        reserved: dict[int, list[tuple[int, int]]] = {}
        for lane, anchors in anchors_by_lane.items():
            spans = [(int(t), int(t) + int(l)) for t, l, _node in anchors if int(l) > 0]
            if spans:
                reserved[int(lane)] = spans
        return reserved


    def _transition_spans_from_anchors(
        anchors_by_lane: dict[int, list[tuple[int, int, Any]]],
    ) -> dict[int, list[tuple[int, int]]]:
        spans: dict[int, list[tuple[int, int]]] = {}
        for lane, anchors in anchors_by_lane.items():
            lane_spans: list[tuple[int, int]] = []
            for t, _l, node_or_nodes in anchors:
                nodes = node_or_nodes if isinstance(node_or_nodes, list) else [node_or_nodes]
                cur = int(t)
                for node in nodes:
                    ln = int(_node_length_units(node))
                    if _is_transition_node(node) and ln > 0:
                        lane_spans.append((int(cur), int(ln)))
                    cur += max(0, ln)
            if lane_spans:
                spans[int(lane)] = sorted(lane_spans)
        return spans

    # If caller did not provide runtime_essence_paths (embedded extraction map),
    # try a read-only extraction to enable YAMNet classification on embedded AAF.
    _tmp_conv = None
    try:
        if runtime_essence_paths is None:
            try:
                from aaf_io.converter import AAFConverter

                _tmp_conv = AAFConverter(aaf_path, work_dir=work_dir)
                # Extract essence without rewriting the AAF (read-only).
                with open_aaf_lenient(aaf_path, "r") as _aaf_ro2:
                    _tmp_conv.essence_map = {}
                    _tmp_conv._extract_all(_aaf_ro2)
                runtime_essence_paths = _tmp_conv.runtime_essence_paths_for_filter()
            except Exception:
                runtime_essence_paths = None
    except Exception:
        runtime_essence_paths = runtime_essence_paths

    # Analyze in read-only mode, then write the final layout through SDK XML.
    # The writer must move whole event subtrees, not PyAAF2 copies, so fades,
    # automation, and host metadata survive lane changes.
    def _cleanup_runtime_essence() -> None:
        if _tmp_conv is not None:
            try:
                _tmp_conv._unlink_mapped_essence()
            except Exception:
                pass

    with open_aaf_lenient(aaf_path, "r") as aaf_ro:
        comps_ro = list(aaf_ro.content.compositionmobs())
        if not comps_ro:
            _emit("YAMNet дорожки: CompositionMob не найден.")
            _cleanup_runtime_essence()
            return 0
        try:
            comp_ro = _pick_main_composition(comps_ro)
        except CompositionSelectionError as exc:
            _emit(f"YAMNet lanes: layout retained: {exc}")
            _cleanup_runtime_essence()
            return 0
        selected_composition_id = str(comp_ro.mob_id)
        sound_lanes_ro, skipped_ro = _sound_sequence_timeline_slots(
            comp_ro, cancel_check=_check_cancel
        )
        if skipped_ro:
            _emit(
                "YAMNet дорожки: пропущено звуковых слотов без плоского "
                f"Sequence (OperationGroup и т.п.): {skipped_ro}."
            )
        n_ro = len(sound_lanes_ro)
        if n_ro < 3:
            _emit(
                "YAMNet дорожки: нужно ≥3 звуковых дорожек с сегментом Sequence — "
                f"найдено {n_ro}, пропуск (AAF не меняется, режим только чтение)."
            )
            _cleanup_runtime_essence()
            return 0
        if _count_top_sequence_layout_blocks(sound_lanes_ro) == 0:
            _emit(
                "YAMNet дорожки: на Sequence-дорожках нет верхнеуровневых "
                "SourceClip/OperationGroup — пропуск (AAF не меняется)."
            )
            _cleanup_runtime_essence()
            return 0

        _emit(
            f"YAMNet дорожки: анализ таймлайна и классификация клипов "
            f"({n_ro} дорожек)…"
        )
        # Second read-only pass: if nothing would move, do NOT open with "r+"
        # because PyAAF2 rewrites the file on close and this can break Nuendo
        # compatibility even when moved==0.
        events_ro: list[dict[str, Any]] = []
        transition_anchors_ro: dict[int, list[tuple[int, int, Any]]] = {}
        transition_spans_ro: dict[int, list[tuple[int, int]]] = {}
        sdk_structural_events_ro: list[LaneLayoutEvent] = []
        wav_ok_ro = 0
        lane_yamnet_ro = _lane_yamnet_cfg(cfg)
        for lane_idx, (slot, seq) in enumerate(sound_lanes_ro):
            _check_cancel()
            top = list(seq.components)
            pos = 0
            trans_before = 0
            try:
                tn = len(top)
            except Exception:
                continue
            try:
                pre_owner_by_transition, post_owner_by_transition = _event_transition_edge_owners(
                    [top[i] for i in range(tn)]
                )
            except Exception:
                pre_owner_by_transition = {}
                post_owner_by_transition = {}
            owned_transition_indices = set(pre_owner_by_transition) | set(post_owner_by_transition)
            for ti in range(tn):
                try:
                    node = top[ti]
                except Exception:
                    break
                ln = _node_length_units(node)
                silence = silence_replacement_signature(top, ti) if not _is_filler(node) else None
                if silence is not None:
                    sdk_structural_events_ro.append(LaneLayoutEvent(
                        src_lane=int(lane_idx), top_idx=int(ti), T_edit=int(pos),
                        L=int(ln), target_lane=int(lane_idx), kind="silence",
                        silence_signature=silence,
                    ))
                    pos += ln
                    continue
                if _is_filler(node):
                    pos += ln
                    continue
                if _is_transition_node(node):
                    if int(ln) > 0:
                        transition_spans_ro.setdefault(int(lane_idx), []).append(
                            (int(pos), int(ln))
                        )
                    if int(ti) not in owned_transition_indices:
                        transition_anchors_ro.setdefault(int(lane_idx), []).append(
                            (int(pos), int(ln), node)
                        )
                        sdk_structural_events_ro.append(
                            LaneLayoutEvent(
                                src_lane=int(lane_idx),
                                top_idx=int(ti),
                                T_edit=int(pos),
                                L=int(ln),
                                target_lane=int(lane_idx),
                                kind="transition",
                            )
                        )
                    trans_before += int(ln)
                    pos += ln
                    continue
                if _is_operation_group(node) or _is_sourceclip(node):
                    inner_sc = (
                        _og_first_sourceclip(node) if _is_operation_group(node) else node
                    )
                    if inner_sc is None:
                        pos += ln
                        continue
                    er = float(_slot_edit_rate(slot) or 48000.0)
                    dur_sec: Optional[float] = None
                    wav_ok_this = False
                    try:
                        wav_sc = _resolve_wave_path_for_sourceclip(
                            aaf_ro, inner_sc, runtime_essence_paths,
                            media_search_roots=cfg.media_search_roots,
                            edit_rate=slot.edit_rate,
                        )
                        if wav_sc is not None and wav_sc.is_file():
                            wav_ok_ro += 1
                            wav_ok_this = True
                        dur_sec = _sourceclip_duration_sec(
                            er,
                            inner_sc,
                            aaf=aaf_ro,
                            wav_path=wav_sc,
                            runtime_essence_paths=runtime_essence_paths,
                            media_search_roots=cfg.media_search_roots,
                        )
                    except Exception:
                        dur_sec = None
                    kind, ys, ym, yn = _classify_timeline_block_with_scores(
                        aaf_ro,
                        node,
                        runtime_essence_paths,
                        lane_yamnet_ro,
                        slot.edit_rate,
                        _check_cancel,
                        work_dir=work_dir,
                        log_callback=log_callback,
                        media_search_roots=cfg.media_search_roots,
                    )
                    post_transition_len = 0
                    post_transition_top_idx: Optional[int] = None
                    pre_transition_len = 0
                    pre_transition_top_idx: Optional[int] = None
                    try:
                        if (
                            ti - 1 >= 0
                            and _is_transition_node(top[ti - 1])
                            and pre_owner_by_transition.get(int(ti - 1)) == int(ti)
                        ):
                            pre_transition_len = int(_node_length_units(top[ti - 1]))
                            pre_transition_top_idx = int(ti - 1)
                        if (
                            ti + 1 < tn
                            and _is_transition_node(top[ti + 1])
                            and post_owner_by_transition.get(int(ti + 1)) == int(ti)
                        ):
                            post_transition_len = int(_node_length_units(top[ti + 1]))
                            post_transition_top_idx = int(ti + 1)
                    except Exception:
                        pre_transition_len = 0
                        pre_transition_top_idx = None
                        post_transition_len = 0
                        post_transition_top_idx = None
                    events_ro.append(
                        {
                            "src_lane": lane_idx,
                            "top_idx": ti,
                            "T": int(pos),
                            "T_nuendo": int(pos) - 2 * int(trans_before),
                            "transition_offset_before": int(trans_before),
                            "L": int(ln),
                            "source_start": int(_aaf_int_start(inner_sc)),
                            "source_length": int(_aaf_int_length(inner_sc)),
                            "source_id": _sourceclip_source_id(inner_sc),
                            "source_track": _sourceclip_source_track(inner_sc),
                            "node": node,
                            "display_name": resolve_clip_display_name(aaf_ro, node),
                            "kind": kind,
                            "yamnet_s": float(ys),
                            "yamnet_m": float(ym),
                            "yamnet_n": float(yn),
                            "er": er,
                            "dur_sec": dur_sec,
                            "pre_transition_len": pre_transition_len,
                            "pre_transition_top_idx": pre_transition_top_idx,
                            "post_transition_len": post_transition_len,
                            "post_transition_top_idx": post_transition_top_idx,
                        }
                    )
                    pos += ln
                    continue
                pos += ln

        # Safety: dedupe events by original timeline identity.
        # On some malformed AAFs the same logical block can be observed more than once
        # (e.g. wrapper components). Never allow writing duplicates.
        deduped_ro: list[dict[str, Any]] = []
        seen_ro: set[tuple[Any, ...]] = set()
        for e in events_ro:
            inner = _event_inner_sourceclip(e.get("node"))
            k_align = _sourceclip_edit_align_key(inner)
            if k_align is not None:
                k = (int(e["src_lane"]), int(e["top_idx"]), k_align, int(e["L"]))
            else:
                k = (int(e["src_lane"]), int(e["top_idx"]), int(e["T"]), int(e["L"]))
            if k in seen_ro:
                continue
            seen_ro.add(k)
            deduped_ro.append(e)
        events_ro = deduped_ro

        # If analysis media is unavailable, YAMNet cannot run and everything stays "unknown".
        # We allow continuing only if the user enabled an opening-music fallback window.
        if wav_ok_ro <= 0:
            try:
                head_sec = float(cfg.lane_layout_unknown_opening_music_sec)
            except Exception:
                head_sec = 0.0
            if head_sec <= 0.0:
                _emit("YAMNet lanes: no available analysis media; layout retained without classification.")
                if result_out is not None:
                    result_out["analysis_skipped"] = "media_unavailable"
                _cleanup_runtime_essence()
                return 0
            _emit(
                "YAMNet дорожки: WAV не найдены — YAMNet недоступен; применяем только "
                f"fallback для начала таймлайна (unknown→music в первые {head_sec:g} сек)."
            )
        _apply_edit_timeline_unification(events_ro)
        _apply_unknown_opening_music_fallback(
            events_ro, float(cfg.lane_layout_unknown_opening_music_sec)
        )
        unknown_count = sum(event["kind"] == "unknown" for event in events_ro)
        if unknown_count:
            _emit(f"YAMNet: не распознано {unknown_count}/{len(events_ro)} клипов; они сохраняют исходные дорожки.")
        lane_signature_by_lane = {
            i: _lane_layout_signature(slot, seq)
            for i, (slot, seq) in enumerate(sound_lanes_ro)
        }
        lane_mutable_by_lane = {
            i: _lane_layout_mutable(slot, seq)
            for i, (slot, seq) in enumerate(sound_lanes_ro)
        }
        immutable_lanes = {
            int(lane)
            for lane, mutable in lane_mutable_by_lane.items()
            if not bool(mutable)
        }
        if immutable_lanes:
            protected = ", ".join(str(lane + 1) for lane in sorted(immutable_lanes))
            _emit(f"YAMNet lanes: protected tracks {protected}; unsupported structures or silence adjacent to transitions.")
        _assign_target_lanes(
            events_ro,
            n_ro,
            reserved_by_lane=_reserved_transition_intervals(transition_anchors_ro),
            transition_spans_by_lane=transition_spans_ro,
            lane_signature_by_lane=lane_signature_by_lane,
            lane_mutable_by_lane=lane_mutable_by_lane,
        )
        repaired_overlaps = _repair_lane_layout_plan_overlaps(
            events_ro,
            n_ro,
            transition_spans_by_lane=transition_spans_ro,
            lane_signature_by_lane=lane_signature_by_lane,
            lane_mutable_by_lane=lane_mutable_by_lane,
        )
        if repaired_overlaps > 0:
            _emit(
                "YAMNet дорожки: исправлены конфликтующие xfade/transition blocks "
                f"в плане раскладки: {int(repaired_overlaps)}."
            )
        _validate_lane_layout_plan(events_ro, n_ro)
        moved_ro = sum(1 for e in events_ro if int(e["target_lane"]) != int(e["src_lane"]))
        lane_order = list(range(n_ro))
        order_tracks_by_class = True
        if order_tracks_by_class:
            kinds_by_lane = [set() for _ in range(n_ro)]
            for event in events_ro:
                kinds_by_lane[int(event["target_lane"])].add(event["kind"])
            lane_order = class_ordered_lanes(kinds_by_lane, protected=immutable_lanes)
        tracks_reordered = lane_order != list(range(n_ro))
        if moved_ro == 0 and not tracks_reordered:
            # Extra diagnostics for "VL-like" cases: helps understand why nothing moved.
            try:
                kinds: dict[str, int] = {}
                for e in events_ro:
                    k = str(e.get("kind") or "unknown")
                    kinds[k] = kinds.get(k, 0) + 1
                kinds_s = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
            except Exception:
                kinds_s = ""
            try:
                _emit(
                    "YAMNet дорожки: перемещений нет — "
                    f"WAV_для_анализа={int(wav_ok_ro)}/{len(events_ro)}; "
                    + (f"классы: {kinds_s}." if kinds_s else "")
                )
            except Exception:
                pass
            _emit(
                f"YAMNet дорожки: блоков {len(events_ro)}, перенесено на другую дорожку: 0."
            )
            _cleanup_runtime_essence()
            return 0

    backup_path = aaf_path.with_suffix(aaf_path.suffix + ".__pre_lane_layout.bak")
    try:
        import shutil

        _emit("YAMNet дорожки: план готов; запись изменений в AAF…")
        shutil.copy2(aaf_path, backup_path)
        moved_written = _write_lane_layout_with_sdk_or_pyaaf2_fallback(
            input_aaf=backup_path,
            output_aaf=aaf_path,
            events=events_ro,
            structural_events=sdk_structural_events_ro,
            n_lanes=n_ro,
            cfg=cfg,
            runtime_essence_paths=runtime_essence_paths,
            work_dir=work_dir,
            cancel_check=_check_cancel,
            log_callback=log_callback,
            result_out=result_out,
            progress_callback=progress_callback,
            immutable_lanes=immutable_lanes,
            composition_id=selected_composition_id,
            order_tracks_by_class=order_tracks_by_class,
        )
        _emit(
            f"YAMNet дорожки: блоков {len(events_ro)}, перенесено на другую дорожку: {moved_ro}."
        )
        if result_out is None or result_out.get("sdk_xml_primary") or result_out.get("sdk_xml_fallback"):
            _emit("YAMNet дорожки: применено через AAF SDK XML.")
        if result_out is not None:
            try:
                result_out["moved"] = int(moved_written)
            except Exception:
                pass
        return int(moved_written)
    except Exception:
        try:
            import shutil

            if backup_path.exists():
                shutil.copy2(backup_path, aaf_path)
        except Exception:
            pass
        raise
    finally:
        _cleanup_runtime_essence()
        try:
            backup_path.unlink(missing_ok=True)
        except Exception:
            pass
