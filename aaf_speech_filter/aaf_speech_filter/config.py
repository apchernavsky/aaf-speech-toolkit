from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .speech_yamnet import YamnetConfig
from .thresholds import validate_quiet_peak_dbfs


@dataclass(frozen=True)
class FilterConfig:
    yamnet: YamnetConfig = field(default_factory=YamnetConfig)
    # YAMNet speech|noise|music lane layout after removals.
    experimental_yamnet_lane_layout: bool = False
    # Treat unknown blocks near timeline start as music for lane layout.
    lane_layout_unknown_opening_music_sec: float = 0.0
    remove_quiet_clips: bool = True
    # Digital sample peak threshold, dBFS relative to full scale.
    quiet_peak_dbfs: float = -40.0
    # Remove duplicate timeline blocks by SourceClip identity.
    remove_duplicates: bool = False
    # None retains environment fallback for direct library callers; () disables it.
    media_search_roots: tuple[Path, ...] | None = None
    aaf_tools_dir: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "quiet_peak_dbfs", validate_quiet_peak_dbfs(self.quiet_peak_dbfs))
        if self.media_search_roots is not None:
            roots = tuple(Path(p) for p in self.media_search_roots)
            object.__setattr__(self, "media_search_roots", roots)
        if self.aaf_tools_dir is not None:
            object.__setattr__(self, "aaf_tools_dir", Path(self.aaf_tools_dir))
