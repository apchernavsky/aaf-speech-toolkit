from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class PipelineProgress(Protocol):
    """Public progress contract for GUI/CLI integrations."""

    def set_global_fraction(self, fraction: float) -> None:
        """Overall progress fraction, 0.0 ... 1.0."""

    def set_stage(self, label: str) -> None:
        """Short current-stage label for status displays."""

    def set_indeterminate(self, active: bool) -> None:
        """Whether the current stage has no numeric sub-progress."""


def pipeline_progress_to_pair_sink(
    progress: Optional[PipelineProgress],
) -> Optional[Callable[[float, float], None]]:
    """
    Adapt ``PipelineProgress`` to the internal ``sink(processed, total)`` convention.
    """
    if progress is None:
        return None

    def _sink(processed: float, total: float) -> None:
        if total <= 0.0:
            g = 1.0 if processed > 0.0 else 0.0
        else:
            g = min(1.0, max(0.0, float(processed) / float(total)))
        progress.set_global_fraction(g)

    return _sink


def progress_stage_wrap(
    base: Optional[Callable[[float, float], None]],
    start_frac: float,
    span_frac: float,
) -> Optional[Callable[[float, float], None]]:
    """Map local stage progress into a global progress span."""
    if base is None:
        return None
    s0 = max(0.0, min(1.0, float(start_frac)))
    sp = max(0.0, min(1.0, float(span_frac)))
    if sp <= 0.0:
        return None

    def wrapped(processed: float, total: float) -> None:
        if total <= 0.0:
            frac = 1.0 if processed > 0.0 else 0.0
        else:
            frac = min(1.0, max(0.0, float(processed) / float(total)))
        g = s0 + sp * frac
        base(min(100.0, g * 100.0), 100.0)

    return wrapped


def progress_set_global(base: Optional[Callable[[float, float], None]], frac: float) -> None:
    """Set a global progress fraction on a pair-sink callback."""
    if base is None:
        return
    f = max(0.0, min(1.0, float(frac)))
    base(min(100.0, f * 100.0), 100.0)
