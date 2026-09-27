from __future__ import annotations


def format_timecode_from_units(units: int, edit_rate: float) -> str:
    """
    Best-effort timeline timecode for logs.

    If edit_rate looks like fps (<= 1000), format as mm:ss:ff.
    Otherwise fall back to seconds with 3 decimals.
    """
    try:
        u = int(units)
    except Exception:
        u = 0
    er = float(edit_rate) if edit_rate else 0.0
    if er > 0.0 and er <= 1000.0:
        sec = int(u // er) if er else 0
        ff = int(u - int(sec * er))
        mm = int(sec // 60)
        ss = int(sec % 60)
        return f"{mm:02d}:{ss:02d}:{ff:02d}"
    sec = (float(u) / er) if er > 0 else 0.0
    mm = int(sec // 60.0)
    ss = sec - 60.0 * mm
    return f"{mm:02d}:{ss:06.3f}"
