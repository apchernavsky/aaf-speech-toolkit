from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Optional


from aaf_io.errors import OperationCancelled


logger = logging.getLogger(__name__)


def write_lane_layout_with_sdk_or_pyaaf2_fallback(
    *,
    input_aaf: Path,
    output_aaf: Path,
    events: list[dict[str, Any]],
    structural_events: list[Any],
    n_lanes: int,
    cfg: Any,
    runtime_essence_paths: Optional[dict[Any, Path]],
    work_dir: Optional[Path],
    cancel_check: Callable[[], None],
    log_callback: Optional[Callable[[str], None]],
    result_out: Optional[dict[str, Any]],
    progress_callback: Optional[Callable[[float, float], None]] = None,
    build_sdk_events: Callable[[list[dict[str, Any]], list[Any]], list[Any]],
    class_zone_lane_order: Callable[[str, int], list[int]],
    pyaaf2_rebuild: Callable[..., int],
    sdk_xml_writer: Callable[..., None],
    sdk_exporter_unavailable_error: Callable[[BaseException], bool],
    immutable_lanes: Optional[set[int]] = None,
    composition_id: Optional[str] = None,
    pyaaf2_output_validator: Optional[Callable[[Path], None]] = None,
    order_tracks_by_class: bool = False,
) -> int:
    writer_details: dict[str, Any] = {}

    def _emit(msg: str) -> None:
        if log_callback is not None:
            log_callback(msg)
        else:
            logger.info("%s", msg)

    def _report_track_order() -> None:
        order = writer_details.get("lane_order")
        if order is not None and order != list(range(n_lanes)):
            _emit("YAMNet: порядок дорожек упорядочен по классам с сохранением панорамы и эффектов.")

    def _copy_input_to_output() -> None:
        input_path = Path(input_aaf)
        output_path = Path(output_aaf)
        if input_path.resolve() != output_path.resolve() and input_path.is_file():
            shutil.copy2(input_path, output_path)

    def _plan_moved_count() -> int:
        return sum(
            1
            for e in events
            if int(e.get("target_lane", e["src_lane"])) != int(e["src_lane"])
        )

    def _write_via_sdk_xml() -> None:
        sdk_xml_writer(
            input_aaf=input_aaf,
            output_aaf=output_aaf,
            events=build_sdk_events(events, structural_events),
            n_lanes=n_lanes,
            aaf_tools_dir=getattr(cfg, "aaf_tools_dir", None),
            candidate_lanes_by_kind={
                kind: class_zone_lane_order(kind, n_lanes)
                for kind in ("speech", "noise", "music", "unknown")
            },
            work_dir=(Path(work_dir) / "lane_layout_sdk_xml")
            if work_dir is not None
            else (
                Path(output_aaf).parent
                / "__aaf_tool_work"
                / f"{Path(output_aaf).stem}.__lane_layout"
            ),
            cancel_check=cancel_check,
            immutable_lanes=set(immutable_lanes or set()),
            composition_id=composition_id,
            **({"order_tracks_by_class": True, "result_out": writer_details} if order_tracks_by_class else {}),
        )

    try:
        _emit("YAMNet lanes: writing lane layout via PyAAF2.")
        _copy_input_to_output()
        moved = pyaaf2_rebuild(
            aaf_path=Path(output_aaf),
            events=events,
            structural_events=structural_events,
            n_lanes=n_lanes,
            cfg=cfg,
            runtime_essence_paths=runtime_essence_paths,
            work_dir=work_dir,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
            log_callback=log_callback,
            immutable_lanes=set(immutable_lanes or set()),
            composition_id=composition_id,
            **({"order_tracks_by_class": True, "result_out": writer_details} if order_tracks_by_class else {}),
        )
        if pyaaf2_output_validator is not None:
            pyaaf2_output_validator(Path(output_aaf))
        if result_out is not None:
            result_out.update(writer_details)
            result_out["moved"] = int(moved)
            result_out["sdk_xml_primary"] = False
            result_out["sdk_xml_fallback"] = False
            result_out["pyaaf2_lane_layout_primary"] = True
            result_out["pyaaf2_lane_layout_fallback"] = False
        _report_track_order()
        return int(moved)
    except OperationCancelled:
        raise
    except Exception as ex:
        pyaaf2_failure = ex
        _emit(
            "YAMNet lanes: PyAAF2 writer failed; falling back to AAF SDK XML "
            f"({ex.__class__.__name__}: {ex})."
        )

    try:
        _emit("YAMNet lanes: writing lane layout via AAF SDK XML fallback.")
        writer_details.clear()
        _write_via_sdk_xml()
        if result_out is not None:
            result_out.update(writer_details)
            result_out["moved"] = int(_plan_moved_count())
            result_out["sdk_xml_primary"] = False
            result_out["sdk_xml_fallback"] = True
            result_out["pyaaf2_lane_layout_primary"] = False
            result_out["pyaaf2_lane_layout_fallback"] = False
            result_out["pyaaf2_failure"] = str(pyaaf2_failure)
        _report_track_order()
        return int(_plan_moved_count())
    except OperationCancelled:
        raise
    except Exception as sdk_ex:
        if sdk_exporter_unavailable_error(sdk_ex):
            raise RuntimeError(
                "YAMNet lane layout writers failed: "
                f"PyAAF2={pyaaf2_failure}; SDK XML={sdk_ex}"
            ) from sdk_ex
        raise
