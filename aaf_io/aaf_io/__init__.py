"""AAF I/O: lenient PyAAF2 open, AAF SDK helpers, embedded essence extraction, roundtrip."""

from __future__ import annotations

from aaf_io.converter import AAFConverter
from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
from aaf_io.heal.cfb import (
    compute_fat_sector_count_from_chain,
    copy_and_sync_fat_header,
    read_fat_sector_count_on_disk,
    sync_fat_sector_count_header,
)
from aaf_io.roundtrip import (
    RoundtripMethod,
    RoundtripResult,
    repair_empty_essence_descriptions,
    sanitize_avid_this_extension_xml,
    sdk_roundtrip,
)
from aaf_io.heal.premiere import (
    HealStats,
    PremiereAudioMendResult,
    PremiereImportScan,
    format_heal_stats_lines,
    heal_import_descriptors_from_linked_media,
    heal_premiere_import_audio_inplace,
    mend_premiere_import_audio_for_sound_pipeline,
    scan_premiere_import_linked_audio,
)
from aaf_io.session import (
    export_regulated,
    open_work_aaf,
    prepare_work_copy,
    save_pyaaf2_inplace,
)
from aaf_io.subprocess_hidden import popen_hidden, run_hidden

__all__ = [
    "AAFConverter",
    "open_aaf_lenient",
    "RoundtripMethod",
    "RoundtripResult",
    "compute_fat_sector_count_from_chain",
    "copy_and_sync_fat_header",
    "HealStats",
    "PremiereAudioMendResult",
    "PremiereImportScan",
    "export_regulated",
    "format_heal_stats_lines",
    "heal_import_descriptors_from_linked_media",
    "heal_premiere_import_audio_inplace",
    "mend_premiere_import_audio_for_sound_pipeline",
    "scan_premiere_import_linked_audio",
    "open_work_aaf",
    "prepare_work_copy",
    "read_fat_sector_count_on_disk",
    "repair_empty_essence_descriptions",
    "sanitize_avid_this_extension_xml",
    "save_pyaaf2_inplace",
    "popen_hidden",
    "run_hidden",
    "sdk_roundtrip",
    "sync_fat_sector_count_header",
]
