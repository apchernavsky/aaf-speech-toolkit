from __future__ import annotations

from typing import Any, Optional


def datadef_for_filler_fallback(aaf: Any, sourceclip: Any) -> Optional[Any]:
    """Best-effort Sound DataDefinition for replacement fillers."""
    try:
        src_dd = sourceclip["DataDefinition"].value
        if src_dd is not None:
            return aaf.dictionary.lookup_datadef(src_dd.auid)
    except Exception:
        pass
    try:
        return aaf.dictionary.lookup_datadef("Sound")
    except Exception:
        return None


def replace_block_with_filler(
    parent_vec: Any,
    index: int,
    aaf: Any,
    length: int,
    inner_sc: Optional[Any],
) -> None:
    ln = max(0, int(length))
    filler = aaf.create.Filler(length=ln, media_kind="Sound")
    try:
        if inner_sc is not None:
            filler["DataDefinition"].value = inner_sc["DataDefinition"].value
    except Exception:
        dd = datadef_for_filler_fallback(aaf, inner_sc) if inner_sc is not None else None
        if dd is not None:
            try:
                filler["DataDefinition"].value = dd
            except Exception:
                pass
    parent_vec[index] = filler


def replace_first_sourceclip_in_block_with_filler(
    block: Any,
    aaf: Any,
    sourceclip: Any,
) -> bool:
    """
    Preserve an OperationGroup/other wrapper and mute it by replacing its inner
    SourceClip with a same-length Sound Filler.
    """

    target_id = id(sourceclip)
    stack = [block]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)
        for attr in ("components", "segments"):
            try:
                vec = getattr(cur, attr, None)
            except Exception:
                vec = None
            if vec is None:
                continue
            try:
                n = len(vec)
            except Exception:
                continue
            for i in range(n):
                try:
                    child = vec[i]
                except Exception:
                    continue
                if id(child) == target_id:
                    try:
                        length = int(child["Length"].value)
                    except Exception:
                        length = int(getattr(child, "length", 0) or 0)
                    replace_sourceclip_with_filler(vec, i, aaf, length, child)
                    return True
                stack.append(child)
    return False


def replace_sourceclip_with_filler(
    parent_container: Any,
    index: int,
    aaf: Any,
    length: int,
    sourceclip: Any,
) -> None:
    """Replace a SourceClip with a Sound Filler of the same duration."""
    ln = int(length)
    filler = aaf.create.Filler(length=ln, media_kind="Sound")
    try:
        filler["DataDefinition"].value = sourceclip["DataDefinition"].value
    except Exception:
        dd = datadef_for_filler_fallback(aaf, sourceclip)
        if dd is not None:
            try:
                filler["DataDefinition"].value = dd
            except Exception:
                try:
                    filler.media_kind = "Sound"
                except Exception:
                    pass
    parent_container[index] = filler
