#!/usr/bin/env python3
"""AAF: убрать неречевые клипы → <stem>_processed.aaf (сборка через AAF SDK, см. ``sdk_bin/``).

Референсный большой AAF для проверок: локальный файл (не обязателен);
пакетный probe без аргументов сканирует дефолтный каталог — ``scripts/probe_aaf_readiness.py``.
"""

import argparse
import os
import sys
import traceback
import logging
import shutil
from pathlib import Path
from typing import Callable, Optional

# The application entry point owns source-checkout imports for both GUI and CLI.
# Register both local projects before any preflight can import their modules.
_SYS_DIR = Path(__file__).resolve().parent
if str(_SYS_DIR) not in sys.path:
    sys.path.insert(0, str(_SYS_DIR))
for _package_name in ("aaf_io", "aaf_speech_filter"):
    _package_root = _SYS_DIR / _package_name
    if (_package_root / _package_name / "__init__.py").is_file():
        if str(_package_root) not in sys.path:
            sys.path.insert(0, str(_package_root))

from aaf_io.diagnostic_log import BoundedDiagnosticLog
from aaf_io.errors import OperationCancelled
from aaf_io.converter import AAFConverter
from aaf_io.sdk_tools import find_aaffmtconv
from aaf_io.roundtrip import sdk_roundtrip as _sdk_roundtrip
from aaf_tool_config import (
    effective_aaf_tools_dir,
    resolve_media_search_roots,
)
from aaf_workflow import (
    assert_not_overwriting_source,
    cleanup_work_dir,
    compute_processed_path,
    emit_comaafinfo_header,
    extract_embedded_essence_paths_for_analysis,
    fail_fast_require_any_timeline_wav_for_removals,
    finalize_pyaaf2_lane_layout_container,
    lane_layout_result_already_sdk_safe,
    make_work_dir_near_input,
    make_workflow_plan,
    maybe_run_lane_layout_only_fastpath,
    nuendo_safe_roundtrip_inplace,
    pipeline_stage_boundaries,
    pipeline_stage_weights,
    post_sdk_lane_layout_if_enabled,
    prepare_work_copy_for_processing,
    run_premiere_pcm_scan_and_mend,
    run_pyaaf2_filter_only,
    sdk_apply_removals_and_build_aaf,
    sdk_cleanup_xml_side_artifacts,
    sdk_export_xml,
    warn_if_strict_aaf_validation_fails,
    workflow_plan_after_missing_media,
)

# Windows console часто не умеет UTF-8 (cp1252/cp866). Чтобы диагностика/GUI-логи
# можно было копировать без падений на русских строках — печатаем с заменой символов.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(level=logging.ERROR, format="%(message)s")

logger = logging.getLogger(__name__)

MSG_PIPELINE_STOPPED = "Остановлено пользователем."

def _ru_removed_clips_phrase(n: int) -> str:
    """Краткая строка для лога/диалога (сколько SourceClip заменено на filler)."""
    n = int(n)
    if n % 10 == 1 and n % 100 != 11:
        return f"Удалён {n} клип"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"Удалено {n} клипа"
    return f"Удалено {n} клипов"


def diag_aaf_sdk_roundtrip(input_path: Path, aaf_tools_dir: Optional[Path] = None) -> tuple[int, str]:
    """
    Диагностика: AAF → XML (aaffmtconv -xml) → AAF (aaffmtconv -ss) без правок таймлайна.
    Результат — проверка Nuendo на совместимость с пересборкой через AAF SDK.
    """
    input_path = Path(input_path).resolve()
    if not input_path.is_file():
        return 2, "Файл не найден: " + str(input_path)
    try:
        tools_root = effective_aaf_tools_dir(Path(aaf_tools_dir) if aaf_tools_dir else None)
        xml_out = input_path.parent / f"{input_path.stem}.__aaf_sdk_roundtrip.xml"
        aaf_out = input_path.parent / f"{input_path.stem}.__aaf_sdk_roundtrip{input_path.suffix}"
        res = _sdk_roundtrip(
            input_path,
            aaf_out,
            aaf_tools_dir=tools_root,
            strip_this_namespace=True,
            xml_sibling=xml_out,
            on_xml_crash="binary_copy",
        )
        if not res.ok:
            return 1, res.message
        return (
            0,
            "AAF roundtrip (SDK + санитизация Avid XML при необходимости; при падении -xml — копия файла):\n"
            f"  способ: {res.method.value}\n"
            f"  XML: {xml_out}\n"
            f"  AAF: {aaf_out}\n"
            f"  детали: {res.message}\n\n"
            "Откройте выходной AAF в Nuendo. При binary_copy вход и выход совпадают по байтам.",
        )
    except Exception as e:
        tb = traceback.format_exc()
        msg = f"{type(e).__name__}: {e}"
        if tb and tb.strip():
            tail = tb.strip()[-2400:]
            if len(tb.strip()) > 2400:
                tail = "…\n" + tail
            msg += "\n\n" + tail
        return 1, msg


def _load_speech_filter():
    """Load filter types without changing application import paths."""
    from aaf_speech_filter.config import FilterConfig
    from aaf_speech_filter.exceptions import SpeechFilterCancelled

    return FilterConfig, SpeechFilterCancelled


def _cleanup_speech_pipeline(
    converter: "AAFConverter",
    *,
    work_dir: Path,
    work_path: Path,
    output_aaf: Path,
) -> None:
    """Удалить частичный результат и work_dir этой попытки."""
    try:
        work_path.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        output_aaf.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        converter._unlink_mapped_essence()
    except Exception:
        pass


def _execute_aaf_pipeline(
    input_path,
    yamnet_score_threshold: float = 0.0035,
    yamnet_frame_aggregate: str = "max",
    experimental_yamnet_lane_layout: bool = False,
    lane_layout_unknown_opening_music_sec: float = 0.0,
    allow_yamnet_download: bool = True,
    remove_quiet_clips=True,
    quiet_peak_dbfs=-40.0,
    remove_duplicates: bool = False,
    pipeline_progress=None,
    cancel_event=None,
    aaf_tools_dir: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    *,
    work_dir: Path,
    processed_path: Path,
):
    """
    Результат: ``<stem>_processed.aaf`` рядом с исходником.

    Подготовка: копия AAF + при необходимости выгрузка embedded в ``essence/`` для анализа.
    Сборка результата всегда через **AAF SDK** (aaffmtconv). Временный ``.__speech_work.aaf`` удаляется после успеха.

    ``log_callback`` — опционально: строки для GUI/лога (ComAAFInfo и т.д.); без него пишется в stdout.

    ``pipeline_progress`` — опционально: объект с ``set_global_fraction(0..1)``, ``set_stage(str)``,
    ``set_indeterminate(bool)`` (см. ``aaf_speech_filter.progress.PipelineProgress``).

    Returns ``(None, output_path, None, removed_clips, None)`` при успехе; при ошибке пятый элемент ``None``.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        return None, None, "Файл не найден", None, None
    lane_layout_hint: dict[str, Any] = {}
    try:
        if input_path.stat().st_size == 0:
            return None, None, "AAF-файл пустой (0 байт).", None, None
    except OSError as e:
        return None, None, f"Не удалось прочитать размер файла: {e}", None, None

    def _emit(msg: str) -> None:
        if log_callback is not None:
            log_callback(msg)
        else:
            print(msg)

    try:
        (FilterConfig, SpeechFilterCancelled) = _load_speech_filter()
        from aaf_speech_filter.speech_yamnet import (
            YamnetConfig,
            ensure_yamnet_model_available,
            yamnet_model_available_locally,
        )
    except Exception as e:
        return (
            None,
            None,
            "Пакет aaf_speech_filter недоступен (установите PyAAF2, tensorflow, tensorflow-hub "
            "или положите папку aaf_speech_filter рядом со скриптом): "
            + str(e),
            None,
            None,
        )

    media_search_roots = resolve_media_search_roots(input_path)

    _yfa = (yamnet_frame_aggregate or "max").strip().lower()
    if _yfa not in ("max", "mean"):
        _yfa = "max"
    yamnet_cfg = YamnetConfig(
        score_threshold=float(yamnet_score_threshold),
        frame_aggregate=_yfa,
        allow_download=bool(allow_yamnet_download),
    )
    if bool(experimental_yamnet_lane_layout):
        try:
            if not yamnet_model_available_locally(yamnet_cfg):
                _emit("YAMNet: local model not found; downloading/loading from TensorFlow Hub...")
                ensure_yamnet_model_available(yamnet_cfg)
        except Exception as e:
            return None, None, f"Ошибка фильтра речи: {e}", None, None

    tools_root = effective_aaf_tools_dir(Path(aaf_tools_dir) if aaf_tools_dir else None)
    def _check_sdk_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelled()

    emit_comaafinfo_header(
        input_aaf=input_path, tools_root=tools_root, emit=_emit,
        cancel_check=_check_sdk_cancel,
    )

    # Visible "we started" marker for huge AAFs where the next step can take many minutes.
    try:
        sz_mb = float(input_path.stat().st_size) / (1024.0 * 1024.0)
        _emit(f"Старт обработки: {input_path.name} ({sz_mb:.1f} MB)")
    except Exception:
        _emit(f"Старт обработки: {input_path.name}")

    workflow_plan = make_workflow_plan(
        remove_quiet_clips=bool(remove_quiet_clips),
        remove_duplicates=bool(remove_duplicates),
        experimental_yamnet_lane_layout=bool(experimental_yamnet_lane_layout),
    )
    only_lanes = workflow_plan.lane_layout_only
    if only_lanes:
        # Minimal path: lane layout only, no analysis removals.
        # Avoid embedded essence extraction and avoid rewriting via AAF SDK.
        try:
            from aaf_speech_filter.progress import (
                pipeline_progress_to_pair_sink,
                progress_set_global,
                progress_stage_wrap,
            )
            pg = pipeline_progress_to_pair_sink(pipeline_progress)
            try:
                runtime_essence_paths = extract_embedded_essence_paths_for_analysis(
                    input_aaf=input_path,
                    work_dir=work_dir,
                    cancel_event=cancel_event,
                    cancel_exc_type=SpeechFilterCancelled,
                )
            except OperationCancelled:
                raise
            except Exception:
                runtime_essence_paths = {}

            cfg = FilterConfig(
                yamnet=yamnet_cfg,
                media_search_roots=media_search_roots,
                aaf_tools_dir=tools_root,
                experimental_yamnet_lane_layout=True,
                lane_layout_unknown_opening_music_sec=float(lane_layout_unknown_opening_music_sec),
                remove_quiet_clips=False,
                quiet_peak_dbfs=float(quiet_peak_dbfs),
                remove_duplicates=bool(remove_duplicates),
            )

            maybe_run_lane_layout_only_fastpath(
                cfg=cfg,
                work_aaf=input_path,
                processed_aaf=processed_path,
                runtime_essence_paths=runtime_essence_paths,
                work_dir=work_dir,
                emit=_emit,
                pipeline_progress=pipeline_progress,
                cancel_event=cancel_event,
                progress_callback=progress_stage_wrap(pg, 0.0, 1.0),
            )
            progress_set_global(pg, 1.0)
            return None, processed_path, None, 0, None
        except OperationCancelled:
            try:
                processed_path.unlink(missing_ok=True)
            except Exception:
                pass
            return None, None, MSG_PIPELINE_STOPPED, None, None
        except Exception as e:
            tb = traceback.format_exc()
            try:
                processed_path.unlink(missing_ok=True)
            except Exception:
                pass
            msg = f"Ошибка фильтра речи: {e}"
            if tb and tb.strip():
                tail = tb.strip()[-2400:]
                if len(tb.strip()) > 2400:
                    tail = "…\n" + tail
                msg += "\n\n" + tail
            return None, None, msg, None, None

    work_path = work_dir / "input.aaf"
    try:
        _emit("Подготовка: создание рабочей копии AAF и проверка PCM…")
        converter = prepare_work_copy_for_processing(
            input_aaf=input_path,
            work_dir=work_dir,
            work_aaf=work_path,
            media_search_roots=media_search_roots,
            emit=_emit,
            pipeline_progress=pipeline_progress,
            cancel_event=cancel_event,
            cancel_exc_type=SpeechFilterCancelled,
        )
    except OperationCancelled:
        try:
            converter  # type: ignore[name-defined]
        except Exception:
            converter = None
        if converter is not None:
            _cleanup_speech_pipeline(
                converter, work_dir=work_dir, work_path=work_path, output_aaf=processed_path
            )
        return None, None, MSG_PIPELINE_STOPPED, None, None
    except Exception as e:
        msg = str(e) or "Не удалось подготовить AAF."
        return None, None, msg, None, None

    # A valid source-only, empty or picture-only AAF has no audio cleanup work.
    # Preserve the original bytes rather than requiring a CompositionPackage.
    from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
    from aaf_speech_filter.timeline_sourceclips import iter_timeline_sourceclips
    with open_aaf_lenient(work_path, "r") as prepared_aaf:
        has_timeline_audio = next(iter_timeline_sourceclips(prepared_aaf), None) is not None
    if not has_timeline_audio:
        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelled()
        shutil.copyfile(input_path, processed_path)
        _emit("No timeline audio clips: saved an unchanged copy.")
        if pipeline_progress is not None:
            pipeline_progress.set_indeterminate(False)
            pipeline_progress.set_global_fraction(1.0)
        return None, processed_path, None, 0, None

    # Fail-fast: for **removal** modes we must be able to resolve at least one WAV for timeline clips
    # (either via embedded extraction into runtime_essence_paths or via external locators).
    #
    # For lane layout we allow "no media" runs because some Premiere AAFs have broken/missing
    # locators and we still want to support the "unknown opening music" fallback.
    workflow_plan = make_workflow_plan(
        remove_quiet_clips=bool(remove_quiet_clips),
        remove_duplicates=bool(remove_duplicates),
        experimental_yamnet_lane_layout=bool(experimental_yamnet_lane_layout),
    )

    if workflow_plan.requires_timeline_media:
        try:
            fail_fast_require_any_timeline_wav_for_removals(
                work_aaf=work_path,
                runtime_essence_paths=converter.runtime_essence_paths_for_filter(),
                require=True,
                media_search_roots=media_search_roots,
            )
        except OperationCancelled:
            raise
        except RuntimeError:
            updated_plan = workflow_plan_after_missing_media(workflow_plan)
            if updated_plan != workflow_plan:
                _emit(
                    "Анализ тишины пропущен: для клипов таймлайна не найдено доступных WAV/AIFF "
                    "или извлеченного embedded audio. Продолжаем с режимами, которым медиа не требуется."
                )
                workflow_plan = updated_plan
                remove_quiet_clips = workflow_plan.remove_quiet_clips
                remove_duplicates = workflow_plan.remove_duplicates
                experimental_yamnet_lane_layout = workflow_plan.experimental_yamnet_lane_layout

    from aaf_speech_filter.progress import (
        pipeline_progress_to_pair_sink,
        progress_set_global,
        progress_stage_wrap,
    )

    pg = pipeline_progress_to_pair_sink(pipeline_progress)

    w_prep, w_clips, w_sdk, w_layout = pipeline_stage_weights(bool(experimental_yamnet_lane_layout))
    end_prep, end_clips, end_sdk = pipeline_stage_boundaries(bool(experimental_yamnet_lane_layout))

    progress_set_global(pg, end_prep)

    cfg = FilterConfig(
        yamnet=yamnet_cfg,
        media_search_roots=media_search_roots,
        aaf_tools_dir=tools_root,
        experimental_yamnet_lane_layout=experimental_yamnet_lane_layout,
        lane_layout_unknown_opening_music_sec=float(lane_layout_unknown_opening_music_sec),
        remove_quiet_clips=remove_quiet_clips,
        quiet_peak_dbfs=quiet_peak_dbfs,
        remove_duplicates=bool(remove_duplicates),
    )
    removed_clips = 0
    try:
        tools = find_aaffmtconv(tools_root)
        from aaf_speech_filter.pyaaf2_filter import filter_aaf_speech_only
        from aaf_speech_filter.aaf_yamnet_lane_layout import (
            apply_experimental_yamnet_lane_layout,
        )

        # Cache runtime essence paths once per run: extracting embedded essence and
        # building the mapping can be expensive on huge AAFs.
        try:
            _runtime_paths_cached = converter.runtime_essence_paths_for_filter() or {}
        except Exception:
            _runtime_paths_cached = {}

        if maybe_run_lane_layout_only_fastpath(
            cfg=cfg,
            work_aaf=work_path,
            processed_aaf=processed_path,
            runtime_essence_paths=_runtime_paths_cached,
            work_dir=work_dir,
            emit=_emit,
            pipeline_progress=pipeline_progress,
            cancel_event=cancel_event,
            progress_callback=progress_stage_wrap(pg, float(end_prep), 1.0 - float(end_prep)),
        ):
            progress_set_global(pg, 1.0)
            removed_clips = 0
            return None, processed_path, None, removed_clips, None

        xml_work = work_dir / "pipeline.xml"
        try:
            xml_work.unlink(missing_ok=True)
        except Exception:
            pass
        sdk_xml_ok = False
        try:
            # Сначала AAF→XML; на очень больших/«кривых» AAF (Premiere VL и т.п.) aaffmtconv часто падает (напр. 0xC0000005).
            def _cancel_check() -> None:
                if cancel_event is not None and cancel_event.is_set():
                    raise SpeechFilterCancelled()

            sdk_export_xml(
                tools=tools,
                input_aaf=work_path,
                xml_out=xml_work,
                pipeline_progress=pipeline_progress,
                cancel_check=_cancel_check,
            )
            sdk_xml_ok = True
        except RuntimeError as _xml_e:
            if "aaffmtconv -xml failed" not in str(_xml_e):
                raise
            if pipeline_progress is not None:
                pipeline_progress.set_indeterminate(False)
                pipeline_progress.set_stage("Обработка PyAAF2")
            _emit(
                "AAF SDK: aaffmtconv -xml не выполнен. "
                "Переключаемся на режим **только PyAAF2**: те же правила тишины, YAMNet и опциональная раскладка дорожек; "
                "выходной AAF собирается без aaffmtconv -ss."
            )
            _emit("PyAAF2: подготовка перед стартом (cleanup XML, настройка прогресса)…")
            sdk_cleanup_xml_side_artifacts(xml_work, emit=_emit)

            _pyaaf_span = max(1e-6, 1.0 - float(end_prep))
            _pyaaf_prog = progress_stage_wrap(pg, float(end_prep), _pyaaf_span)
            _lane_res_pyaaf: dict[str, Any] = {}
            removed_clips = int(
                run_pyaaf2_filter_only(
                    work_aaf=work_path,
                    processed_aaf=processed_path,
                    cfg=cfg,
                    progress_callback=_pyaaf_prog,
                    cancel_event=cancel_event,
                    runtime_essence_paths=_runtime_paths_cached,
                    work_dir=work_dir,
                    emit=_emit,
                    lane_layout_result_out=_lane_res_pyaaf,
                    cfb_bookkeeping_source_aaf=input_path,
                )
            )
            lane_layout_hint = _lane_res_pyaaf
            progress_set_global(pg, 1.0)
            finalize_pyaaf2_lane_layout_container(
                processed_aaf=processed_path,
                work_dir=work_dir,
                lane_layout_result_out=_lane_res_pyaaf,
                tools_root=tools_root,
                emit=_emit,
                cancel_check=_cancel_check,
            )
            _skip_roundtrip = lane_layout_result_already_sdk_safe(_lane_res_pyaaf)
            if not _skip_roundtrip:
                nuendo_safe_roundtrip_inplace(
                    processed_aaf=processed_path,
                    work_dir=work_dir,
                    aaf_tools_dir=tools_root,
                    emit=_emit,
                    cancel_check=_cancel_check,
                )


        if sdk_xml_ok:
            _emit("AAF SDK XML: анализ таймлайна (сбор ключей удалений)…")
            removed_clips, _built_via_sdk, _replaced, _removals_len, _fallback_used = sdk_apply_removals_and_build_aaf(
                tools=tools,
                work_aaf=work_path,
                xml_work=xml_work,
                processed_aaf=processed_path,
                cfg=cfg,
                runtime_essence_paths=_runtime_paths_cached,
                cancel_check=_cancel_check,
                cancel_event=cancel_event,
                emit=_emit,
                pipeline_progress=pipeline_progress,
                progress_callback_clips=progress_stage_wrap(pg, w_prep, w_clips),
                progress_callback_sdk=progress_stage_wrap(pg, float(end_clips), float(w_sdk)),
                cfb_bookkeeping_source_aaf=input_path,
            )
            if _fallback_used:
                logger.warning(
                    "AAF SDK XML: replaced=%d при removals=%d — fallback filter_aaf_speech_only (PyAAF2).",
                    int(_replaced),
                    int(_removals_len),
                )
            progress_set_global(pg, end_sdk)
            try:
                sdk_cleanup_xml_side_artifacts(xml_work)
            except Exception:
                pass
            try:
                _lane_res: dict[str, Any] = {}
                post_sdk_lane_layout_if_enabled(
                    cfg=cfg,
                    processed_aaf=processed_path,
                    work_dir=work_dir,
                    runtime_essence_paths=_runtime_paths_cached,
                    cancel_event=cancel_event,
                    progress_callback=progress_stage_wrap(pg, end_sdk, w_layout),
                    pipeline_progress=pipeline_progress,
                    emit=_emit,
                    lane_layout_result_out=_lane_res,
                )
                lane_layout_hint = _lane_res
                progress_set_global(pg, 1.0)
            except OperationCancelled:
                raise
            except Exception:
                logger.warning("YAMNet дорожки: фатальная ошибка после SDK", exc_info=True)
                raise
            finalize_pyaaf2_lane_layout_container(
                processed_aaf=processed_path,
                work_dir=work_dir,
                lane_layout_result_out=_lane_res,
                tools_root=tools_root,
                emit=_emit,
                cancel_check=_cancel_check,
            )
            _skip_roundtrip = lane_layout_result_already_sdk_safe(_lane_res)
            if not _skip_roundtrip:
                nuendo_safe_roundtrip_inplace(
                    processed_aaf=processed_path,
                    work_dir=work_dir,
                    aaf_tools_dir=tools_root,
                    emit=_emit,
                    cancel_check=_cancel_check,
                )

    except OperationCancelled:
        _cleanup_speech_pipeline(
            converter, work_dir=work_dir, work_path=work_path, output_aaf=processed_path
        )
        return None, None, MSG_PIPELINE_STOPPED, None, None
    except Exception as e:
        tb = traceback.format_exc()
        _cleanup_speech_pipeline(
            converter, work_dir=work_dir, work_path=work_path, output_aaf=processed_path
        )
        msg = f"Ошибка фильтра речи: {e}"
        if tb and tb.strip():
            tail = tb.strip()[-2400:]
            if len(tb.strip()) > 2400:
                tail = "…\n" + tail
            msg += "\n\n" + tail
        return None, None, msg, None, None
    warn_if_strict_aaf_validation_fails(
        processed_aaf=processed_path, tools_root=tools_root, emit=_emit,
        cancel_check=_check_sdk_cancel,
    )
    return None, processed_path, None, removed_clips, (lane_layout_hint or None)


def _run_aaf_pipeline_impl(input_path, **options):
    """Own one run's workspace and publish only its completed candidate."""
    input_path = Path(input_path)
    if not input_path.is_file():
        return None, None, "Файл не найден", None, None
    destination = compute_processed_path(input_path)
    run_log = BoundedDiagnosticLog(options.get("log_callback") or print)
    options["log_callback"] = run_log
    work_dir = None
    try:
        assert_not_overwriting_source(input_path, destination)
        work_dir = make_work_dir_near_input(input_path)
        candidate = work_dir / "output.aaf"
        result = _execute_aaf_pipeline(
            input_path, work_dir=work_dir, processed_path=candidate, **options
        )
        if result[2] is not None:
            return result
        cancel_event = options.get('cancel_event')
        if cancel_event is not None and cancel_event.is_set():
            raise OperationCancelled()
        if result[1] != candidate or not candidate.is_file():
            raise RuntimeError('Pipeline did not produce its owned output candidate.')
        # Rename on the same volume is the sole publication point.
        assert_not_overwriting_source(input_path, destination)
        candidate.replace(destination)
        return result[0], destination, result[2], result[3], result[4]
    except Exception as exc:
        if isinstance(exc, OperationCancelled):
            return None, None, MSG_PIPELINE_STOPPED, None, None
        return None, None, str(exc), None, None
    finally:
        if work_dir is not None:
            try:
                cleanup_work_dir(work_dir, input_parent=input_path.parent)
            except OSError as exc:
                logger.warning('Could not clean owned workspace %s: %s', work_dir, exc)
        run_log.finish()


def run_aaf_pipeline(
    input_path,
    yamnet_score_threshold: float = 0.0035,
    yamnet_frame_aggregate: str = "max",
    experimental_yamnet_lane_layout: bool = False,
    lane_layout_unknown_opening_music_sec: float = 0.0,
    allow_yamnet_download: bool = True,
    remove_quiet_clips=True,
    quiet_peak_dbfs=-40.0,
    remove_duplicates: bool = False,
    pipeline_progress=None,
    cancel_event=None,
    aaf_tools_dir: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
):
    from aaf_speech_filter.thresholds import validate_quiet_peak_dbfs

    try:
        quiet_peak_dbfs = validate_quiet_peak_dbfs(quiet_peak_dbfs)
    except (TypeError, ValueError) as exc:
        return None, None, str(exc), None, None
    input_path = Path(input_path)
    return _run_aaf_pipeline_impl(
        input_path,
        yamnet_score_threshold=yamnet_score_threshold,
        yamnet_frame_aggregate=yamnet_frame_aggregate,
        experimental_yamnet_lane_layout=experimental_yamnet_lane_layout,
        lane_layout_unknown_opening_music_sec=lane_layout_unknown_opening_music_sec,
        allow_yamnet_download=allow_yamnet_download,
        remove_quiet_clips=remove_quiet_clips,
        quiet_peak_dbfs=quiet_peak_dbfs,
        remove_duplicates=remove_duplicates,
        pipeline_progress=pipeline_progress,
        cancel_event=cancel_event,
        aaf_tools_dir=aaf_tools_dir,
        log_callback=log_callback,
    )


def _run_gui():
    from aaf_gui import run_gui

    return run_gui()


def _cli_wants_gui():
    if len(sys.argv) == 1:
        return True
    return len(sys.argv) == 2 and sys.argv[1] in ("--gui", "-g")


if __name__ == "__main__":
    if _cli_wants_gui():
        _run_gui()
        raise SystemExit(0)

    from aaf_speech_filter.thresholds import validate_quiet_peak_dbfs

    ap = argparse.ArgumentParser(
        description="AAF: оставить только речевые клипы → <stem>_processed.aaf (рядом с входом).",
        epilog=(
            "Референс для регрессии: локальный AAF "
            "(см. scripts/probe_aaf_readiness.py — дефолтный каталог probe)."
        ),
    )
    ap.add_argument("input", type=Path, help="Входной .aaf (embedded или unembedded)")
    ap.add_argument(
        "--yamnet-score-threshold",
        type=float,
        default=0.0035,
        help=(
            "YAMNet: клип с речью, если агрегат (max/mean по кадрам) от max_j softmax(P(class j)) "
            "для j в классах речи 0..12 >= порога. При 521 классе величины обычно ~0.0025–0.02, не как «уверенность 0..1»."
        ),
    )
    ap.add_argument(
        "--yamnet-frame-aggregate",
        type=str,
        choices=("max", "mean"),
        default="max",
        help="YAMNet: max или mean по кадрам после softmax.",
    )
    ap.add_argument(
        "--experimental-yamnet-lanes",
        dest="experimental_yamnet_lane_layout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "YAMNet (речь|шум|музыка) и раскладка по звуковым дорожкам Nuendo. "
            "Выполняется после удаления тихих клипов (если включено). Сочетается с остальными опциями."
        ),
    )
    ap.add_argument(
        "--download-yamnet-model",
        dest="download_yamnet_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow TensorFlow Hub download when --experimental-yamnet-lanes is enabled "
            "and no local/cached YAMNet model is available (default: on)."
        ),
    )
    ap.add_argument(
        "--lane-unknown-opening-music-sec",
        type=float,
        default=0.0,
        help=(
            "Если WAV недоступен, YAMNet не вызывается и блоки считаются 'unknown'. "
            "Этот параметр переводит unknown→music в начале таймлайна (секунды), чтобы интро/музыка "
            "могли уехать вниз даже на AAF с битой/недоступной медиа."
        ),
    )
    ap.add_argument(
        "--remove-quiet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove clips whose peak is at or below --quiet-peak-dbfs (default: on).",
    )
    ap.add_argument(
        "--remove-duplicates",
        dest="remove_duplicates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Удалять дубли клипов на таймлайне по ключу SourceID/StartTime/Length(+SlotID) (default: on).",
    )
    ap.add_argument(
        "--quiet-peak-dbfs",
        type=validate_quiet_peak_dbfs,
        default=-40.0,
        help="Finite peak threshold <= 0 dBFS (default -40); 0 includes full-scale PCM.",
    )
    ap.add_argument(
        "--diag-aaf-sdk-roundtrip",
        action="store_true",
        help="Диагностика: aaffmtconv AAF→XML→AAF (без правок), проверка Nuendo.",
    )
    ap.add_argument(
        "--aaf-tools",
        type=Path,
        default=None,
        help="Папка с aaffmtconv.exe и ComAAFInfo.exe (или переменная окружения AAF_TOOLS).",
    )
    args = ap.parse_args()

    if args.diag_aaf_sdk_roundtrip:
        code, msg = diag_aaf_sdk_roundtrip(args.input, aaf_tools_dir=args.aaf_tools)
        print(msg)
        raise SystemExit(code)

    if (
        (not args.remove_quiet)
        and not args.experimental_yamnet_lane_layout
        and not args.remove_duplicates
    ):
        logger.error(
            "Выберите хотя бы один режим: --remove-quiet, --experimental-yamnet-lanes или --remove-duplicates."
        )
        raise SystemExit(2)

    _ue, processed_path, err, removed_n, _hint = run_aaf_pipeline(
        args.input,
        yamnet_score_threshold=args.yamnet_score_threshold,
        yamnet_frame_aggregate=args.yamnet_frame_aggregate,
        experimental_yamnet_lane_layout=args.experimental_yamnet_lane_layout,
        lane_layout_unknown_opening_music_sec=args.lane_unknown_opening_music_sec,
        allow_yamnet_download=args.download_yamnet_model,
        remove_quiet_clips=args.remove_quiet,
        quiet_peak_dbfs=args.quiet_peak_dbfs,
        remove_duplicates=args.remove_duplicates,
        aaf_tools_dir=args.aaf_tools,
        log_callback=None,
    )
    if err:
        if err == MSG_PIPELINE_STOPPED:
            logger.error(err)
            raise SystemExit(2)
        logger.error(err)
        raise SystemExit(1)
    if processed_path:
        print(processed_path)
    if removed_n is not None:
        print(_ru_removed_clips_phrase(removed_n))
