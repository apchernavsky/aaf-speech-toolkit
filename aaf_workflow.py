from __future__ import annotations

"""
Workflow stage helpers shared by the GUI/CLI pipeline.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from aaf_io.errors import OperationCancelled
from aaf_workflow_logging import StageLogger


LogFn = Callable[[str], None]

MEDIA_UNAVAILABLE_FOR_REMOVALS_ERROR = (
    "Анализ невозможен: не найдено ни одного медиа-файла для клипов таймлайна. "
    "Похоже, embedded essence не удалось распаковать и/или локаторы на внешние файлы отсутствуют. "
    "Процесс остановлен без сохранения результата."
)


@dataclass(frozen=True)
class WorkflowPlan:
    """
    Normalized processing modes for one run.

    Keep mode decisions here so SDK and PyAAF2 paths do not drift when a flag is
    added or when media is unavailable for only some stages.
    """

    remove_quiet_clips: bool
    remove_duplicates: bool
    experimental_yamnet_lane_layout: bool

    @property
    def lane_layout_only(self) -> bool:
        return self.experimental_yamnet_lane_layout and not (
            self.remove_quiet_clips or self.remove_duplicates
        )

    @property
    def requires_timeline_media(self) -> bool:
        return bool(self.remove_quiet_clips)

    @property
    def can_continue_without_timeline_media(self) -> bool:
        return bool(self.remove_duplicates or self.experimental_yamnet_lane_layout)

    def without_quiet_analysis(self) -> "WorkflowPlan":
        return replace(self, remove_quiet_clips=False)


def make_workflow_plan(
    *,
    remove_quiet_clips: bool,
    remove_duplicates: bool,
    experimental_yamnet_lane_layout: bool,
) -> WorkflowPlan:
    return WorkflowPlan(
        remove_quiet_clips=bool(remove_quiet_clips),
        remove_duplicates=bool(remove_duplicates),
        experimental_yamnet_lane_layout=bool(experimental_yamnet_lane_layout),
    )


def workflow_plan_after_missing_media(plan: WorkflowPlan) -> WorkflowPlan:
    """
    Disable only media-dependent stages when other selected stages can still run.
    Raise when the selected run would otherwise do nothing useful.
    """
    if not plan.requires_timeline_media:
        return plan
    if plan.can_continue_without_timeline_media:
        return plan.without_quiet_analysis()
    raise RuntimeError(MEDIA_UNAVAILABLE_FOR_REMOVALS_ERROR)


def flatten_top_level_empty_sound_operationgroups(aaf_path: Path) -> int:
    """
    Replace only proven, transition-independent gain-of-silence wrappers.

    These are produced when removals mute an OperationGroup input by replacing
    the inner SourceClip with Filler. Nuendo tolerates that wrapper, but strict
    importers can still initialize it as an audio operation and then fail because
    there is no media source. A top-level Filler expresses the same timeline
    silence without pretending to be an audio effect.
    """
    from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
    from aaf_speech_filter.aaf_mutation import replace_block_with_filler
    from aaf_io.audio_silence import silence_replacement_signature
    from aaf_speech_filter.timeline_walk import (
        _node_length_units,
        _slot_is_soundish,
    )

    def _sound_sequence(seg):
        if seg is None:
            return None
        name = seg.__class__.__name__
        if "Sequence" in name:
            return seg
        if "OperationGroup" not in name:
            return None
        try:
            inner = getattr(seg, "segments", None)
            if inner is not None and len(inner) >= 1 and "Sequence" in inner[0].__class__.__name__:
                return inner[0]
        except Exception:
            return None
        return None

    changed = 0
    with open_aaf_lenient(Path(aaf_path), "r+") as aaf:
        for comp in aaf.content.compositionmobs():
            for slot in getattr(comp, "slots", []) or []:
                if not _slot_is_soundish(slot):
                    continue
                seq = _sound_sequence(getattr(slot, "segment", None))
                if seq is None:
                    continue
                try:
                    top = seq.components
                    top_nodes = list(top)
                    n = len(top_nodes)
                except Exception:
                    continue
                for i in range(n):
                    try:
                        node = top[i]
                    except Exception:
                        break
                    if "OperationGroup" not in node.__class__.__name__:
                        continue
                    if silence_replacement_signature(top_nodes, i) is None:
                        continue
                    replace_block_with_filler(top, i, aaf, _node_length_units(node), None)
                    changed += 1
    return int(changed)


def lane_layout_result_already_sdk_safe(result: Optional[dict[str, object]]) -> bool:
    """
    True when lane layout already produced an SDK-normalized output.
    """
    if not result:
        return False
    return bool(
        result.get("sdk_normalized")
        or result.get("sdk_xml_fallback")
        or result.get("pyaaf2_lane_layout_primary")
        or result.get("pyaaf2_lane_layout_fallback")
        or result.get("sdk_structured_storage_normalized")
        or result.get("sdk_final_structured_storage_normalized")
    )


def lane_layout_result_written_by_pyaaf2(result: Optional[dict[str, object]]) -> bool:
    """
    True when lane layout wrote timeline changes through PyAAF2.

    This is intentionally separate from ``lane_layout_result_already_sdk_safe``:
    PyAAF2 may write a logically correct AAF that still needs an AAF SDK
    AAF-to-AAF structured-storage pass for strict OLE/CFB readers.
    """
    if not result:
        return False
    if result.get("sdk_final_structured_storage_normalized"):
        return False
    return bool(
        result.get("pyaaf2_lane_layout_primary")
        or result.get("pyaaf2_lane_layout_fallback")
    )


def finalize_pyaaf2_lane_layout_container(
    *,
    processed_aaf: Path,
    work_dir: Path,
    lane_layout_result_out: Optional[dict[str, Any]],
    tools_root: Optional[Path] = None,
    emit: Optional[LogFn] = None,
    cancel_check=None,
) -> bool:
    """
    Normalize the physical AAF container after PyAAF2 wrote lane-layout edits.

    The pass is AAF-to-AAF ``aaffmtconv -ss``. It does not rebuild from XML and
    therefore must not localize or otherwise rewrite media locator URLs.
    """
    if not lane_layout_result_written_by_pyaaf2(lane_layout_result_out):
        return False
    processed_aaf = Path(processed_aaf)
    work_dir = Path(work_dir)
    if not processed_aaf.is_file():
        return False
    work_dir.mkdir(parents=True, exist_ok=True)
    norm_out = work_dir / f"{processed_aaf.name}.__pyaaf2_lane_layout_sdk_ss.aaf"
    try:
        from aaf_io.sdk_tools import find_aaffmtconv, run_aaffmtconv_to_structured_storage

        if emit is not None:
            emit(
                "PyAAF2 lane layout: normalizing output AAF container with AAF SDK "
                "structured-storage pass..."
            )
        conv = find_aaffmtconv(Path(tools_root) if tools_root is not None else None)
        run_aaffmtconv_to_structured_storage(
            conv,
            processed_aaf,
            norm_out,
            cancel_check=cancel_check,
        )
        import shutil

        from aaf_io.heal.cfb import sync_fat_sector_count_header

        sync_fat_sector_count_header(norm_out)
        if cancel_check is not None:
            cancel_check()
        norm_out.replace(processed_aaf)
        if lane_layout_result_out is not None:
            lane_layout_result_out["sdk_final_structured_storage_normalized"] = True
        if emit is not None:
            emit("PyAAF2 lane layout: output AAF container normalization complete.")
        return True
    finally:
        try:
            norm_out.unlink(missing_ok=True)
        except Exception:
            pass


def emit_comaafinfo_header(
    *,
    input_aaf: Path,
    tools_root: Path,
    emit: LogFn,
    cancel_check: Optional[Callable[[], None]] = None,
) -> None:
    """
    Emit ComAAFInfo header diagnostics; cancellation always propagates.
    Behavior is kept identical to the previous inlined implementation.
    """
    try:
        from aaf_io.sdk_tools import find_comaafinfo, run_comaafinfo

        _cinfo = find_comaafinfo(tools_root)
        _code, _out = run_comaafinfo(_cinfo, Path(input_aaf), cancel_check=cancel_check)
        emit("--- ComAAFInfo (входной AAF) ---")
        for _line in (_out if _out else "(нет вывода)").splitlines():
            emit(_line)
        if _code != 0:
            emit(f"ComAAFInfo: код выхода {_code}")
    except OperationCancelled:
        raise
    except Exception as e:
        emit(f"ComAAFInfo: не выполнен — {e}")


def warn_if_strict_aaf_validation_fails(
    *,
    processed_aaf: Path,
    tools_root: Path,
    emit: LogFn,
    cancel_check: Optional[Callable[[], None]] = None,
) -> None:
    """
    Best-effort strict-reader validation for hosts that behave like AAF SDK.

    Nuendo may import PyAAF2-written files that AAF SDK/Samplitude reject. Warn
    explicitly so this compatibility loss is never silent.
    """
    try:
        from aaf_io.sdk_tools import find_comaafinfo, run_comaafinfo

        cinfo = find_comaafinfo(tools_root)
        code, out = run_comaafinfo(cinfo, Path(processed_aaf), cancel_check=cancel_check)
        if int(code) == 0:
            return
        detail = (out or "").strip()
        if detail:
            detail = " " + detail.splitlines()[0][:300]
        emit(
            "WARNING: strict AAF validation failed for processed AAF "
            f"(ComAAFInfo exit {code}). Nuendo may still import it, but stricter "
            f"hosts such as Samplitude may reject it.{detail}"
        )
    except OperationCancelled:
        raise
    except Exception as exc:
        emit(
            "WARNING: strict AAF validation could not be run for processed AAF "
            f"({exc.__class__.__name__}: {exc})."
        )


def compute_processed_path(input_path: Path) -> Path:
    input_path = Path(input_path)
    return input_path.parent / f"{input_path.stem}_processed{input_path.suffix}"


def assert_not_overwriting_source(input_path: Path, output_path: Path) -> None:
    from aaf_io.path_safety import require_distinct_files

    require_distinct_files(input_path, output_path)


def make_work_dir_near_input(input_aaf: Path) -> Path:
    """
    Per-run work dir next to the input AAF.
    All temporary artifacts (essence, SDK XML + streams, intermediate AAFs) go here.
    """
    import tempfile

    input_aaf = Path(input_aaf)

    # Per-run ownership supplies uniqueness; source names must not grow SDK paths.
    base = input_aaf.parent / "__aaf_tool_work"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=base))


def prune_empty_parents(start: Path, *, stop_at: Path) -> None:
    """
    Remove empty directories up from ``start`` until ``stop_at`` (exclusive).
    Used to delete leftover ``__aaf_tool_work/<stem>`` and ``__aaf_tool_work`` after a run.
    """
    from aaf_io.temp_cleanup import prune_empty_parents as _prune_empty_parents

    _prune_empty_parents(start, stop_at=stop_at)


def cleanup_work_dir(work_dir: Path, *, input_parent: Path) -> None:
    """
    Robustly delete a work dir on Windows (locks/AV/read-only).
    """
    from aaf_io.temp_cleanup import cleanup_work_dir as _cleanup_work_dir

    _cleanup_work_dir(work_dir, input_parent=input_parent)


def cleanup_aaf_tool_work_root(input_parent: Path) -> None:
    """
    Retain the shared allocation root; run cleanup owns only its child.
    """
    from aaf_io.temp_cleanup import cleanup_aaf_tool_work_root as _cleanup_aaf_tool_work_root

    _cleanup_aaf_tool_work_root(input_parent)


def run_premiere_pcm_scan_and_mend(
    *,
    work_aaf: Path,
    has_runtime_embedded_wavs: bool,
    emit: LogFn,
    cancel_event=None,
    cancel_exc_type: Optional[type[BaseException]] = None,
    media_search_roots: Optional[tuple[Path, ...]] = None,
) -> None:
    """
    Emit the Premiere ImportDescriptor PCM scan block and run mend step (best-effort).

    Behavior matches the inlined implementation in aaf_pipeline:
    - always emits the header line before scanning
    - suppresses "file not found" sample spam when embedded WAVs exist (runtime_essence_paths not empty)
    """
    from aaf_io.heal.premiere import mend_premiere_import_audio_for_sound_pipeline

    mend_premiere_import_audio_for_sound_pipeline(
        Path(work_aaf),
        log_callback=emit,
        emit_scan_summary=False,
        media_search_roots=media_search_roots,
        suppress_missing_file_samples=bool(has_runtime_embedded_wavs),
        cancel_event=cancel_event,
        cancel_exc_type=cancel_exc_type,
    )


def fail_fast_require_any_timeline_wav_for_removals(
    *,
    work_aaf: Path,
    runtime_essence_paths,
    require: bool,
    media_search_roots: Optional[tuple[Path, ...]] = None,
) -> int:
    """
    Raise RuntimeError if removals are enabled but no resolvable media file exists for timeline clips.
    Return the number of resolvable timeline media files otherwise.
    """
    if not bool(require):
        return 0
    from aaf_speech_filter.timeline_sourceclips import count_resolvable_timeline_wavs

    n_wav = count_resolvable_timeline_wavs(
        Path(work_aaf),
        runtime_essence_paths=runtime_essence_paths,
        media_search_roots=media_search_roots,
    )
    if int(n_wav) <= 0:
        raise RuntimeError(MEDIA_UNAVAILABLE_FOR_REMOVALS_ERROR)
    return int(n_wav)


def pipeline_stage_weights(experimental_lane: bool) -> tuple[float, float, float, float]:
    """
    Stage weights (sum 1): prep, clip analysis/removals, SDK, lane layout.
    Matches the previous aaf_pipeline._pipeline_stage_weights.
    """
    w_prep = 0.10
    w_clips = 0.50
    w_layout = 0.10 if experimental_lane else 0.0
    w_sdk = 0.40 - w_layout
    return (w_prep, w_clips, w_sdk, w_layout)


def pipeline_stage_boundaries(experimental_lane: bool) -> tuple[float, float, float]:
    """Return (end_prep, end_clips, end_sdk)."""
    w_prep, w_clips, w_sdk, _w_layout = pipeline_stage_weights(bool(experimental_lane))
    end_prep = w_prep
    end_clips = w_prep + w_clips
    end_sdk = end_clips + w_sdk
    return (end_prep, end_clips, end_sdk)


def prepare_work_copy_for_processing(
    *,
    input_aaf: Path,
    work_dir: Path,
    work_aaf: Path,
    emit: LogFn,
    pipeline_progress=None,
    cancel_event=None,
    cancel_exc_type: Optional[type[BaseException]] = None,
    media_search_roots: Optional[tuple[Path, ...]] = None,
) -> "AAFConverter":
    """
    Create a work-copy AAF and run the Premiere ImportDescriptor PCM scan/mend step.

    Returns an initialized AAFConverter bound to input_aaf/work_dir.
    Raises RuntimeError for user-visible preparation failures.
    """
    from aaf_io.converter import AAFConverter

    converter = AAFConverter(Path(input_aaf), work_dir=Path(work_dir))

    assert_not_overwriting_source(Path(input_aaf), Path(work_aaf))
    Path(work_aaf).unlink(missing_ok=True)

    if pipeline_progress is not None:
        pipeline_progress.set_stage("Подготовка AAF")
        pipeline_progress.set_indeterminate(True)

    ok = converter.prepare_external_media_copy(
        Path(work_aaf),
        cancel_event=cancel_event,
        cancel_exc_type=cancel_exc_type,
    )
    if not ok:
        detail = (converter.last_prepare_error or "").strip()
        msg = (
            "Не удалось подготовить AAF (копия не создана или PyAAF2 не читает её для анализа). "
            "Проверьте целостность файла."
        )
        if detail:
            msg += f" Подробности: {detail}"
        raise RuntimeError(msg)

    try:
        runtime_paths = converter.runtime_essence_paths_for_filter()
    except Exception:
        runtime_paths = {}

    run_premiere_pcm_scan_and_mend(
        work_aaf=Path(work_aaf),
        has_runtime_embedded_wavs=bool(runtime_paths),
        media_search_roots=media_search_roots,
        emit=emit,
        cancel_event=cancel_event,
        cancel_exc_type=cancel_exc_type,
    )
    try:
        from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient

        with open_aaf_lenient(Path(work_aaf), "r"):
            pass
    except Exception as exc:
        # Some Premiere AAFs can be read but are corrupted by PyAAF2's r+ save path
        # after descriptor healing. Keep the pristine work copy and let media resolve
        # through locators/runtime essence instead of failing the whole run.
        try:
            emit(
                "PCM-heal: рабочая копия не открылась после записи PyAAF2; "
                f"откат к исходной копии без PCM-heal ({exc.__class__.__name__}: {exc})."
            )
        except Exception as exc:
            if emit is not None:
                emit(
                    "WARNING: PyAAF2 post-processing could not restore CFB/OLE "
                    f"bookkeeping ({exc.__class__.__name__}: {exc})."
                )
        try:
            Path(work_aaf).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            import shutil

            shutil.copy2(Path(input_aaf), Path(work_aaf))
        except Exception:
            try:
                import shutil

                shutil.copyfile(str(input_aaf), str(work_aaf))
            except Exception as copy_exc:
                raise RuntimeError(
                    "Не удалось восстановить рабочую копию AAF после сбоя PCM-heal: "
                    f"{copy_exc}"
                ) from copy_exc

    if pipeline_progress is not None:
        pipeline_progress.set_indeterminate(False)

    return converter


def maybe_run_lane_layout_only_fastpath(
    *,
    cfg,
    work_aaf: Path,
    processed_aaf: Path,
    runtime_essence_paths,
    work_dir: Path,
    emit: LogFn,
    pipeline_progress=None,
    cancel_event=None,
    progress_callback=None,
) -> bool:
    """
    If only lane layout is requested (no removals), run the fast path and return True.
    Otherwise return False and do nothing.
    """
    plan = make_workflow_plan(
        remove_quiet_clips=bool(getattr(cfg, 'remove_quiet_clips', False)),
        remove_duplicates=bool(getattr(cfg, 'remove_duplicates', False)),
        experimental_yamnet_lane_layout=bool(getattr(cfg, 'experimental_yamnet_lane_layout', False)),
    )
    if not plan.lane_layout_only:
        return False

    import shutil
    from aaf_speech_filter.aaf_yamnet_lane_layout import apply_experimental_yamnet_lane_layout

    try:
        shutil.copy2(Path(work_aaf), Path(processed_aaf))
    except Exception:
        shutil.copyfile(str(work_aaf), str(processed_aaf))

    if pipeline_progress is not None:
        pipeline_progress.set_stage("Раскладка по дорожкам")
        pipeline_progress.set_indeterminate(False)

    apply_experimental_yamnet_lane_layout(
        Path(processed_aaf),
        cfg,
        runtime_essence_paths=runtime_essence_paths or {},
        work_dir=Path(work_dir),
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        log_callback=emit,
    )
    return True


def extract_embedded_essence_paths_for_analysis(
    *,
    input_aaf: Path,
    work_dir: Path,
    cancel_event=None,
    cancel_exc_type: Optional[type[BaseException]] = None,
) -> dict[object, Path]:
    """
    Extract embedded essence for read-only analysis when the source AAF contains it.
    """
    from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
    from aaf_io.converter import AAFConverter

    has_embedded = False
    with open_aaf_lenient(Path(input_aaf), "r") as aaf_ro:
        essence_data = getattr(getattr(aaf_ro, "content", aaf_ro), "essencedata", None)
        try:
            has_embedded = bool(essence_data is not None and len(essence_data) > 0)
        except Exception:
            try:
                has_embedded = bool(
                    essence_data is not None
                    and hasattr(essence_data, "values")
                    and len(list(essence_data.values())) > 0
                )
            except Exception:
                has_embedded = False
    if not has_embedded:
        return {}

    converter = AAFConverter(Path(input_aaf), work_dir=Path(work_dir))
    with open_aaf_lenient(Path(input_aaf), "r") as aaf_ro:
        converter.essence_map = {}
        converter._extract_all(
            aaf_ro,
            cancel_event=cancel_event,
            cancel_exc_type=cancel_exc_type,
        )
    return converter.runtime_essence_paths_for_filter() or {}


def sdk_export_xml(
    *,
    tools,
    input_aaf: Path,
    xml_out: Path,
    pipeline_progress=None,
    cancel_check=None,
) -> None:
    """
    Run aaffmtconv -xml with the same progress stage text as before.
    Raises whatever ``run_aaffmtconv_to_xml`` raises (including cancel).
    """
    from aaf_io.sdk_tools import run_aaffmtconv_to_xml

    if pipeline_progress is not None:
        pipeline_progress.set_stage("AAF SDK: экспорт XML")
        pipeline_progress.set_indeterminate(True)

    run_aaffmtconv_to_xml(tools, Path(input_aaf), Path(xml_out), cancel_check=cancel_check)

    if pipeline_progress is not None:
        pipeline_progress.set_indeterminate(False)
        pipeline_progress.set_stage("Анализ клипов (тишина)")


def sdk_cleanup_xml_side_artifacts(xml_path: Path, *, emit: Optional[LogFn] = None) -> None:
    """Remove this run's XML and stream sidecars using literal filesystem paths."""
    from aaf_io.temp_cleanup import robust_rmtree

    xml_path = Path(xml_path)
    if emit is not None:
        emit("AAF SDK XML: cleaning owned XML/stream artifacts...")
    xml_path.unlink(missing_ok=True)
    streams = xml_path.with_name(xml_path.stem + "_streams")
    if not robust_rmtree(streams):
        raise OSError(f"Could not remove owned SDK streams: {streams}")


def run_pyaaf2_filter_only(
    *,
    work_aaf: Path,
    processed_aaf: Path,
    cfg,
    runtime_essence_paths,
    work_dir: Optional[Path] = None,
    cancel_event=None,
    progress_callback=None,
    emit: Optional[LogFn] = None,
    lane_layout_result_out: Optional[dict[str, Any]] = None,
    cfb_bookkeeping_source_aaf: Optional[Path] = None,
) -> int:
    """
    Run the pure PyAAF2 pipeline (filter_aaf_speech_only).
    """
    from aaf_speech_filter.pyaaf2_filter import filter_aaf_speech_only

    def _check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelled()

    _check_cancel()
    removed = int(
        filter_aaf_speech_only(
            Path(work_aaf),
            Path(processed_aaf),
            cfg,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            runtime_essence_paths=runtime_essence_paths,
            work_dir=Path(work_dir) if work_dir is not None else None,
            log_callback=emit,
            lane_layout_result_out=lane_layout_result_out,
        )
    )
    if emit is not None:
        emit("PyAAF2: основной проход завершён; начинается постобработка результата…")
    lane_layout_sdk_safe = lane_layout_result_already_sdk_safe(lane_layout_result_out)
    if Path(processed_aaf).exists():
        if emit is not None:
            emit(
                "PyAAF2: постобработка выходного AAF — проверка пустых audio "
                "OperationGroup wrappers…"
            )
        try:
            flattened = flatten_top_level_empty_sound_operationgroups(Path(processed_aaf))
            if flattened > 0 and emit is not None:
                emit(
                    "PyAAF2 fallback: replaced empty top-level audio OperationGroup wrappers "
                    f"with Filler for strict host compatibility: {flattened}."
                )
        except (OperationCancelled, OSError):
            raise
        except Exception as exc:
            if emit is not None:
                emit(
                    "WARNING: PyAAF2 fallback could not flatten empty audio OperationGroups "
                    f"({exc.__class__.__name__}: {exc})."
                )
    elif emit is not None:
        emit("PyAAF2: постобработка пропущена — выходной AAF еще не создан.")
    if work_dir is not None and lane_layout_sdk_safe:
        norm_out = Path(processed_aaf).with_suffix(Path(processed_aaf).suffix + ".__final_sdk_ss.aaf")
        try:
            from aaf_io.sdk_tools import find_aaffmtconv, run_aaffmtconv_to_structured_storage

            if emit is not None:
                emit(
                    "PyAAF2: постобработка выходного AAF — AAF SDK "
                    "structured-storage нормализация…"
                )
            conv = find_aaffmtconv(getattr(cfg, "aaf_tools_dir", None))
            run_aaffmtconv_to_structured_storage(
                conv,
                Path(processed_aaf),
                norm_out,
                cancel_check=_check_cancel,
            )
            import shutil

            from aaf_io.heal.cfb import sync_fat_sector_count_header

            sync_fat_sector_count_header(norm_out)
            if cancel_event is not None and cancel_event.is_set():
                raise OperationCancelled()
            norm_out.replace(Path(processed_aaf))
            if lane_layout_result_out is not None:
                lane_layout_result_out["sdk_final_structured_storage_normalized"] = True
            if emit is not None:
                emit(
                    "PyAAF2 fallback: finalized post-layout cleanup with AAF SDK "
                    "structured-storage pass."
                )
        finally:
            try:
                norm_out.unlink(missing_ok=True)
            except Exception:
                pass
    if work_dir is not None and not lane_layout_sdk_safe:
        from aaf_io.heal.cfb import restore_pyAAF2_noop_bookkeeping_from_source

        if emit is not None:
            emit("PyAAF2: restoring CFB/OLE bookkeeping from the source.")
        cfb_source = (
            Path(cfb_bookkeeping_source_aaf)
            if cfb_bookkeeping_source_aaf is not None
            else Path(work_aaf)
        )
        cfb_res = restore_pyAAF2_noop_bookkeeping_from_source(
            cfb_source, Path(processed_aaf), work_dir=Path(work_dir),
        )
        if cfb_res.applied and emit is not None:
            emit(
                "CFB/OLE: restored PyAAF2 no-op bookkeeping from source "
                f"(ranges={cfb_res.restored_ranges}, bytes={cfb_res.restored_bytes}, "
                f"skipped={cfb_res.skipped_ranges}, "
                f"truncated={cfb_res.truncated_to_source_size})."
            )
    if emit is not None:
        emit("PyAAF2: постобработка завершена.")
        emit("PyAAF2: обработка завершена.")
    return removed


def sdk_apply_removals_and_build_aaf(
    *,
    tools,
    work_aaf: Path,
    xml_work: Path,
    processed_aaf: Path,
    cfg,
    runtime_essence_paths,
    cancel_check=None,
    cancel_event=None,
    emit: Optional[LogFn] = None,
    pipeline_progress=None,
    progress_callback_clips=None,
    progress_callback_sdk=None,
    cfb_bookkeeping_source_aaf: Optional[Path] = None,
) -> tuple[int, bool, int, int, bool]:
    """
    Apply removals to SDK XML and build output AAF.

    Returns (removed_clips, built_via_sdk_rebuild, replaced, removals_len, fallback_used).
    If fallback to PyAAF2 is used, built_via_sdk_rebuild=False.
    """
    from aaf_io.sdk_tools import run_aaffmtconv_to_aaf
    from aaf_io.sdk_xml import apply_removals_in_composition_xml
    from aaf_speech_filter.duplicate_filter import remove_duplicate_timeline_blocks_inplace
    from aaf_speech_filter.sdk_removals import collect_timeline_sourceclip_removals_for_sdk_xml

    removal_stats: dict[str, int] = {}
    if bool(getattr(cfg, "remove_quiet_clips", False)):
        removals = collect_timeline_sourceclip_removals_for_sdk_xml(
            Path(work_aaf), cfg,
            runtime_essence_paths=runtime_essence_paths,
            work_dir=Path(xml_work).parent,
            cancel_event=cancel_event,
            progress_callback=progress_callback_clips,
            log_callback=emit,
            removal_stats=removal_stats,
        )
    else:
        removals = set()
    removals_len = len(removals)
    replaced = 0
    removed_clips = int(removal_stats.get("removed_decisions") or removals_len)
    stage_log = StageLogger(emit=emit, progress=pipeline_progress)

    def _run_pyaaf2_fallback(reason: str) -> tuple[int, bool, int, int, bool]:
        stage_log.start(reason, stage="PyAAF2 fallback", indeterminate=False)
        fallback_removed = run_pyaaf2_filter_only(
            work_aaf=Path(work_aaf), processed_aaf=Path(processed_aaf), cfg=cfg,
            runtime_essence_paths=runtime_essence_paths, cancel_event=cancel_event,
            progress_callback=progress_callback_sdk, emit=emit,
            work_dir=Path(xml_work).parent,
            cfb_bookkeeping_source_aaf=cfb_bookkeeping_source_aaf,
        )
        return (int(fallback_removed), False, replaced, removals_len, True)

    from aaf_io.sdk_xml import RemovalPlanMismatchError

    stage_log.start("AAF SDK XML: applying the removal plan.",
                    stage=f"AAF SDK XML: removals ({removals_len})", indeterminate=False)
    try:
        replaced = apply_removals_in_composition_xml(Path(xml_work), removals)
    except RemovalPlanMismatchError as exc:
        return _run_pyaaf2_fallback(f"AAF SDK removal plan mismatch: {exc}")
    if replaced != removals_len:
        return _run_pyaaf2_fallback(
            "AAF SDK XML: removal plan was not applied exactly; switching to PyAAF2 fallback."
        )
    try:
        stage_log.start("AAF SDK XML: building output with aaffmtconv -ss.",
                        stage="AAF SDK: building AAF", indeterminate=True)
        run_aaffmtconv_to_aaf(tools, Path(xml_work), Path(processed_aaf), cancel_check=cancel_check)
    except RuntimeError as exc:
        if "aaffmtconv -ss failed" not in str(exc):
            raise
        return _run_pyaaf2_fallback(
            "AAF SDK XML: aaffmtconv -ss failed while rebuilding edited XML; "
            "switching to PyAAF2 fallback."
        )
    if pipeline_progress is not None:
        pipeline_progress.set_indeterminate(False)

    duplicate_removed = 0
    if bool(getattr(cfg, "remove_duplicates", False)):
        if emit is not None:
            try:
                emit("Дубли: проверка после SDK-сборки…")
            except Exception:
                pass
        duplicate_candidates = int(
            remove_duplicate_timeline_blocks_inplace(
                Path(processed_aaf),
                cfg=cfg,
                log_callback=None,
                cancel_event=cancel_event,
                dry_run=True,
            )
        )
        if duplicate_candidates > 0:
            if emit is not None:
                try:
                    emit("Дубли: применение после SDK-сборки…")
                except Exception:
                    pass
            duplicate_removed = int(
                remove_duplicate_timeline_blocks_inplace(
                    Path(processed_aaf),
                    cfg=cfg,
                    log_callback=emit,
                    cancel_event=cancel_event,
                )
            )
        elif emit is not None:
            try:
                emit("Дубли: не найдено.")
            except Exception:
                pass

    # Keep the optimized path: the output was already rebuilt once by SDK. Duplicate
    # cleanup is an in-place timeline edit and should not force another SDK roundtrip.
    return (
        int(removed_clips) + int(duplicate_removed),
        True,
        int(replaced),
        int(removals_len),
        False,
    )


def post_sdk_lane_layout_if_enabled(
    *,
    cfg,
    processed_aaf: Path,
    work_dir: Path,
    runtime_essence_paths,
    cancel_event=None,
    progress_callback=None,
    pipeline_progress=None,
    emit: Optional[LogFn] = None,
    lane_layout_result_out: Optional[dict[str, Any]] = None,
) -> None:
    """
    Run experimental lane layout after SDK stage, preserving the embedded-essence
    re-extraction behavior for processed_aaf.
    """
    if not bool(getattr(cfg, "experimental_yamnet_lane_layout", False)):
        return

    if pipeline_progress is not None:
        pipeline_progress.set_stage("Раскладка по дорожкам")
    if emit is not None:
        emit("YAMNet дорожки: подготовка раскладки после SDK — проверка промежуточного AAF…")

    _runtime_for_layout = runtime_essence_paths or {}
    _preflight_open_error: Optional[Exception] = None
    try:
        from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient

        with open_aaf_lenient(Path(processed_aaf), "r") as _aaf_chk:
            ed = getattr(getattr(_aaf_chk, "content", _aaf_chk), "essencedata", None)
            has_embedded = False
            try:
                has_embedded = bool(ed is not None and len(ed) > 0)
            except Exception:
                try:
                    has_embedded = bool(
                        ed is not None and hasattr(ed, "values") and len(list(ed.values())) > 0
                    )
                except Exception:
                    has_embedded = bool(ed is not None)
        if has_embedded:
            _runtime_for_layout = None
    except Exception as exc:
        _preflight_open_error = exc

    if _preflight_open_error is not None:
        if lane_layout_result_out is not None:
            try:
                lane_layout_result_out.update(
                    {
                        "moved": 0,
                        "skipped": True,
                        "skip_reason": "pyaaf2_open_failed_after_sdk",
                        "error": f"{_preflight_open_error.__class__.__name__}: {_preflight_open_error}",
                    }
                )
            except Exception:
                pass
        if emit is not None:
            try:
                emit(
                    "YAMNet дорожки: пропуск после SDK — PyAAF2 не смог открыть промежуточный AAF "
                    f"({ _preflight_open_error.__class__.__name__}: {_preflight_open_error})."
                )
            except Exception:
                pass
        if progress_callback is not None:
            try:
                progress_callback(1.0, 1.0)
            except Exception:
                pass
        return

    from aaf_speech_filter.aaf_yamnet_lane_layout import apply_experimental_yamnet_lane_layout

    if emit is not None:
        emit("YAMNet дорожки: запуск раскладки после SDK…")
    apply_experimental_yamnet_lane_layout(
        Path(processed_aaf),
        cfg,
        runtime_essence_paths=_runtime_for_layout,
        work_dir=Path(work_dir),
        cancel_event=cancel_event,
        progress_callback=progress_callback,
        log_callback=emit,
        result_out=lane_layout_result_out,
    )


def nuendo_safe_roundtrip_inplace(
    *,
    processed_aaf: Path,
    work_dir: Path,
    aaf_tools_dir: Optional[Path],
    emit: Optional[LogFn] = None,
    cancel_check=None,
) -> bool:
    """
    Best-effort: run AAF SDK roundtrip AAF→XML→AAF and replace processed_aaf with the roundtripped file.

    Returns True if replacement happened (sdk_xml_*), False otherwise.
    SDK compatibility failures are reported; cancellation and publication errors propagate.
    """
    try:
        from aaf_io.roundtrip import RoundtripMethod, default_xml_work_path, sdk_roundtrip
    except Exception:
        return False

    try:
        inp = Path(processed_aaf).resolve()
        wd = Path(work_dir).resolve()
    except Exception:
        inp = Path(processed_aaf)
        wd = Path(work_dir)

    out_aaf = wd / f"{inp.stem}.__nuendo_roundtrip{inp.suffix or '.aaf'}"
    xml_path = default_xml_work_path(inp, wd)

    try:
        if emit is not None:
            emit("AAF SDK: roundtrip финализация (Nuendo-safe) — старт…")
        res = sdk_roundtrip(
            inp,
            out_aaf,
            aaf_tools_dir=aaf_tools_dir,
            strip_this_namespace=True,
            xml_sibling=xml_path,
            on_xml_crash="binary_copy",
            cancel_check=cancel_check,
        )
    except OperationCancelled:
        raise
    except (RuntimeError, FileNotFoundError) as exc:
        if emit is not None:
            emit(f"AAF SDK finalization unavailable: {exc}")
        return False

    try:
        if not res.ok:
            return False
        # Only replace when we actually got an SDK-produced file.
        if getattr(res, "method", None) == RoundtripMethod.binary_copy:
            return False
        if not out_aaf.is_file():
            return False
        if cancel_check is not None:
            cancel_check()
        out_aaf.replace(inp)
        if emit is not None:
            try:
                emit(f"AAF SDK: roundtrip финализация (Nuendo-safe): {res.method.value}")
            except Exception:
                pass
        return True
    finally:
        try:
            out_aaf.unlink(missing_ok=True)
        except Exception:
            pass
