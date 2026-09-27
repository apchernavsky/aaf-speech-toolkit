"""Validated digital sample-peak thresholds shared by all entry points."""
from __future__ import annotations

import math


def validate_quiet_peak_dbfs(value: float) -> float:
    """Accept finite dBFS <= 0; zero intentionally includes full-scale PCM."""
    threshold = float(value)
    if not math.isfinite(threshold) or threshold > 0:
        raise ValueError('Quiet peak threshold must be finite and <= 0 dBFS')
    return threshold
