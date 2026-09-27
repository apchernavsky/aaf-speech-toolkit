"""
Рабочий цикл «открыть → править → сохранить в пригодном для обмена виде».

- **Редактирование**: всегда через :func:`aaf_io.compat.pyaaf2_lenient.open_aaf_lenient` (режим ``r+``),
  лучше на **копии** проекта (см. :func:`prepare_work_copy`).
- **«Регламентированный» выход** (AAF SDK, ``aaffmtconv``): :func:`aaf_io.roundtrip.sdk_roundtrip`
  — с санитизацией Avid XML и починкой CFB-заголовка при необходимости.
  Для части огромных embedded AAF SDK может быть недоступен; тогда остаётся
  либо байтовая копия, либо сохранение только через PyAAF2 (см. :func:`save_pyaaf2_inplace`).
- **Premiere / Nuendo-подобные предупреждения** (пустой ``ImportDescriptor``): см.
  :func:`mend_premiere_import_audio_for_sound_pipeline` (предпочтительно) или
  :func:`heal_premiere_import_audio_inplace` / ``open_work_aaf(..., heal_premiere_import_audio=True)``.
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
from aaf_io.output_transaction import output_candidate
from aaf_io.heal.cfb import copy_and_sync_fat_header
from aaf_io.heal.premiere import mend_premiere_import_audio_for_sound_pipeline
from aaf_io.roundtrip import RoundtripResult, sdk_roundtrip


def prepare_work_copy(
    source: Path,
    work_copy: Path,
    *,
    sync_cfb_fat_header: bool = True,
) -> Path:
    """
    Скопировать AAF для безопасных правок.

    По умолчанию сразу выравнивает ``fat_sector_count`` в копии (см. ``cfb_heal``),
    что улучшает совместимость с ``olefile`` и AAF SDK на «кривых» Nuendo-экспортах.
    """
    work_copy = Path(work_copy).resolve()
    with output_candidate(source, work_copy) as candidate:
        if sync_cfb_fat_header:
            copy_and_sync_fat_header(Path(source).resolve(), candidate)
        else:
            shutil.copy2(source, candidate)
    return work_copy


@contextmanager
def open_work_aaf(
    work_copy: Path,
    mode: str = "r+",
    *,
    heal_premiere_import_audio: bool = False,
    heal_fix_source_clip_lengths: bool = True,
    heal_on_progress: Optional[Callable[[int, int], None]] = None,
    heal_log_callback: Optional[Callable[[str], None]] = None,
) -> Iterator[object]:
    """
    Открыть рабочую копию с lenient CFB (рекомендуется ``r+`` для правок).

    При ``heal_premiere_import_audio=True`` **до** открытия контекста вызывается
    :func:`aaf_io.heal.premiere.mend_premiere_import_audio_for_sound_pipeline` (скан в ``r``,
    затем ``r+`` только если есть читаемые PCM-файлы по локаторам), чтобы не делать
    лишний полный ``save()`` PyAAF2 на чужих AAF.
    """
    if heal_premiere_import_audio:
        mend_premiere_import_audio_for_sound_pipeline(
            Path(work_copy),
            fix_source_clip_lengths=heal_fix_source_clip_lengths,
            on_progress=heal_on_progress,
            log_callback=heal_log_callback,
        )
    with open_aaf_lenient(Path(work_copy), mode) as aaf:
        yield aaf


def save_pyaaf2_inplace(work_copy: Path) -> None:
    """
    Сохранить файл после правок в режиме ``r+`` (сброс при ``close()``).

    Внимание: перезапись очень больших embedded-AAF через PyAAF2 меняет CFB;
    последующий ``aaffmtconv`` на результате может вести себя иначе, чем на исходнике.
    """
    with open_aaf_lenient(Path(work_copy), "r+"):
        pass


def export_regulated(
    source_aaf: Path,
    output_aaf: Path,
    *,
    aaf_tools_dir: Optional[Path] = None,
    xml_work: Optional[Path] = None,
    try_cfb_fat_header_heal: bool = True,
    strip_this_namespace: bool = True,
    on_xml_crash: str = "binary_copy",
) -> RoundtripResult:
    """
    Экспорт в AAF через официальный SDK (после XML-санитизации и т.д.).

    См. :func:`aaf_io.roundtrip.sdk_roundtrip`.
    """
    return sdk_roundtrip(
        Path(source_aaf),
        Path(output_aaf),
        aaf_tools_dir=aaf_tools_dir,
        strip_this_namespace=strip_this_namespace,
        xml_sibling=xml_work,
        on_xml_crash=on_xml_crash,
        try_cfb_fat_header_heal=try_cfb_fat_header_heal,
    )
