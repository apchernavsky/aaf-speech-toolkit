from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _ensure_toolkit_on_path() -> None:
    here = Path(__file__).resolve()
    toolkit = here.parents[1]
    for p in (toolkit, toolkit / "aaf_io", toolkit / "aaf_speech_filter"):
        if p.is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)


_ensure_toolkit_on_path()

from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient  # noqa: E402
from aaf_io.clip_naming import resolve_clip_display_name  # noqa: E402


def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr)
    except Exception:
        return default


def _safe_prop_value(obj: Any, key: str) -> Any:
    try:
        return obj[key].value
    except Exception:
        return None


def _iter_props(obj: Any) -> Iterable[tuple[str, Any]]:
    """
    Try to list all properties as (name, value-ish). Best-effort across PyAAF2 objects.
    """
    # aaf2 objects often have .properties() returning a dict-like.
    try:
        props = obj.properties()
        if hasattr(props, "items"):
            for k, v in props.items():
                try:
                    yield str(k), getattr(v, "value", v)
                except Exception:
                    yield str(k), v
            return
    except Exception:
        pass
    # Fallback: try a few common keys.
    for k in (
        "Name",
        "MobName",
        "SourceID",
        "SourceMobSlotID",
        "Start",
        "Length",
        "DataDefinition",
        "EssenceDescription",
        "Locator",
    ):
        v = _safe_prop_value(obj, k)
        if v is not None:
            yield k, v


def _mob_name(mob: Any) -> str:
    for attr in ("name", "Name"):
        try:
            v = getattr(mob, attr)
            if v:
                return str(v)
        except Exception:
            pass
    try:
        v = mob["Name"].value
        if v:
            return str(v)
    except Exception:
        pass
    return ""


def _resolve_source_mob(aaf: Any, mob_id: Any) -> Any:
    try:
        mobs = aaf.content.mobs
        it = mobs.values() if hasattr(mobs, "values") else mobs
        for m in it:
            if getattr(m, "mob_id", None) == mob_id:
                return m
    except Exception:
        return None
    return None


def _collect_referenced_source_ids_from_mob(mob: Any) -> set[Any]:
    out: set[Any] = set()
    try:
        slots = getattr(mob, "slots", []) or []
    except Exception:
        slots = []
    for sl in slots:
        seg = getattr(sl, "segment", None)
        if seg is None:
            continue
        stack = [seg]
        seen: set[int] = set()
        while stack:
            cur = stack.pop()
            i = id(cur)
            if i in seen:
                continue
            seen.add(i)
            try:
                cn = cur.__class__.__name__
            except Exception:
                cn = ""
            if "SourceClip" in cn:
                try:
                    sid = cur["SourceID"].value
                except Exception:
                    sid = getattr(cur, "source_id", None)
                if sid is not None:
                    out.add(sid)
            for attr in ("components", "segments"):
                ch = getattr(cur, attr, None)
                if not ch:
                    continue
                try:
                    n = len(ch)
                except Exception:
                    continue
                for j in range(n):
                    try:
                        stack.append(ch[j])
                    except Exception:
                        pass
    return out


def _slot_is_soundish(slot: Any) -> bool:
    try:
        mk = getattr(slot, "media_kind", None)
        if mk and str(mk).lower() in ("sound", "audio"):
            return True
    except Exception:
        pass
    # Fallback: best-effort.
    try:
        seg = getattr(slot, "segment", None)
        if seg is None:
            return False
        nm = seg.__class__.__name__
        return ("Sequence" in nm) or ("OperationGroup" in nm)
    except Exception:
        return False


def _iter_sound_sequences(comp: Any):
    for slot in getattr(comp, "slots", []) or []:
        if not _slot_is_soundish(slot):
            continue
        seg = getattr(slot, "segment", None)
        if seg is None:
            continue
        name = seg.__class__.__name__
        if "Sequence" in name:
            yield slot, seg
        elif "OperationGroup" in name:
            try:
                inner = getattr(seg, "segments", None)
                if (
                    inner is not None
                    and len(inner) >= 1
                    and "Sequence" in inner[0].__class__.__name__
                ):
                    yield slot, inner[0]
            except Exception:
                pass


def _iter_all_sequences(comp: Any):
    """
    Iterate (slot, sequence) for any slot that contains a flat Sequence or an OperationGroup->Sequence.
    Used for diagnostics where the clip might be on picture tracks, not only sound.
    """
    for slot in getattr(comp, "slots", []) or []:
        seg = getattr(slot, "segment", None)
        if seg is None:
            continue
        name = seg.__class__.__name__
        if "Sequence" in name:
            yield slot, seg
        elif "OperationGroup" in name:
            try:
                inner = getattr(seg, "segments", None)
                if (
                    inner is not None
                    and len(inner) >= 1
                    and "Sequence" in inner[0].__class__.__name__
                ):
                    yield slot, inner[0]
            except Exception:
                pass


def _node_length(node: Any) -> int:
    try:
        return int(getattr(node, "length", 0) or 0)
    except Exception:
        pass
    try:
        return int(node["Length"].value)
    except Exception:
        return 0


def _dump(title: str, obj: Any, *, max_props: int = 80) -> None:
    print(f"\n--- {title} ---")
    if obj is None:
        print("<None>")
        return
    print("type:", obj.__class__.__name__)
    try:
        print("repr:", repr(obj)[:500])
    except Exception:
        pass
    # Common "name" candidates
    for k in ("name", "Name", "mob_name", "mob_name", "display_name"):
        v = _safe_get(obj, k, None)
        if v:
            print(f"{k}:", v)
    pv = _safe_prop_value(obj, "Name")
    if pv:
        print('["Name"].value:', pv)

    n = 0
    for k, v in _iter_props(obj):
        if n >= max_props:
            print("... (props truncated)")
            break
        if isinstance(v, (bytes, bytearray)):
            vv = f"<bytes {len(v)}>"
        else:
            vv = v
        print(f"prop {k}: {vv}")
        n += 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump one clip block with all name candidates.")
    ap.add_argument("--aaf", type=Path, required=True)
    ap.add_argument(
        "--find-mob-name",
        type=str,
        default="",
        help="If set, scan timeline blocks and dump the first one whose resolved mob name matches exactly.",
    )
    ap.add_argument("--lane", type=int, default=0)
    ap.add_argument("--index", type=int, default=-1, help="component index within lane (default: first block)")
    args = ap.parse_args()

    aaf_path = Path(args.aaf).resolve()
    if not aaf_path.is_file():
        print(f"AAF not found: {aaf_path}")
        return 2

    with open_aaf_lenient(aaf_path, "r") as aaf:
        comps = list(aaf.content.compositionmobs())
        if not comps:
            print("No composition mobs")
            return 1
        # Search all compositions and all sequences (sound+picture) to find the referenced mob.
        want_name = (args.find_mob_name or "").strip()
        want_mob_ids: set[Any] = set()
        if want_name:
            try:
                mobs = aaf.content.mobs
                it = mobs.values() if hasattr(mobs, "values") else mobs
                for m in it:
                    try:
                        nm = _mob_name(m)
                        if nm == want_name or nm.startswith(want_name):
                            mid = getattr(m, "mob_id", None)
                            if mid is not None:
                                want_mob_ids.add(mid)
                                # Timeline SourceClips may reference underlying SourceMob ids, not the MasterMob id.
                                want_mob_ids.update(_collect_referenced_source_ids_from_mob(m))
                    except Exception:
                        continue
            except Exception:
                pass
            if not want_mob_ids:
                print(f'Mob name not found in file mobs: "{want_name}"')
                return 1
        comp = comps[0]
        lanes = list(_iter_sound_sequences(comp))
        if not lanes:
            lanes = list(_iter_all_sequences(comp))
        if not lanes:
            print("No Sequence lanes found in composition")
            return 1
        chosen_lane: Optional[int] = None
        chosen_i: Optional[int] = None
        chosen_node: Any = None

        def _is_block(node: Any) -> bool:
            try:
                nm = node.__class__.__name__
            except Exception:
                return False
            return ("SourceClip" in nm) or ("OperationGroup" in nm)

        if want_name:
            # Scan all compositions/lanes/components until we find a SourceClip that references the mob id.
            for comp_idx, comp in enumerate(comps):
                lanes_any = list(_iter_all_sequences(comp))
                for lane_idx, (_slot, seq) in enumerate(lanes_any):
                    vec = seq.components
                    try:
                        n = len(vec)
                    except Exception:
                        continue
                    for i in range(n):
                        node = vec[i]
                        if not _is_block(node):
                            continue
                        inner = None
                        if "OperationGroup" in node.__class__.__name__:
                            try:
                                ins = node["InputSegments"]
                                for j in range(len(ins)):
                                    seg = ins[j]
                                    if "SourceClip" in seg.__class__.__name__:
                                        inner = seg
                                        break
                            except Exception:
                                inner = None
                        elif "SourceClip" in node.__class__.__name__:
                            inner = node
                        src_id = None
                        if inner is not None:
                            try:
                                src_id = inner["SourceID"].value
                            except Exception:
                                src_id = getattr(inner, "source_id", None)
                        if src_id in want_mob_ids:
                            chosen_lane = lane_idx
                            chosen_i = i
                            chosen_node = node
                            print(f"found_in_composition_index={comp_idx}")
                            break
                    if chosen_i is not None:
                        break
                if chosen_i is not None:
                    break
        else:
            # Pick by lane/index (old behavior).
            lane_idx = max(0, min(int(args.lane), len(lanes) - 1))
            chosen_lane = lane_idx
            _slot, seq = lanes[lane_idx]
            vec = seq.components
            n = len(vec)
            pos = 0
            if int(args.index) >= 0:
                i0 = int(args.index)
                if i0 < n:
                    chosen_i = i0
                    chosen_node = vec[i0]
            else:
                for i in range(n):
                    node = vec[i]
                    ln = max(0, _node_length(node))
                    if "Filler" in node.__class__.__name__:
                        pos += ln
                        continue
                    if _is_block(node):
                        chosen_i = i
                        chosen_node = node
                        break
                    pos += ln

        if chosen_i is None:
            print("No block found in lane")
            return 1
        assert chosen_lane is not None
        _slot, seq = lanes[int(chosen_lane)]
        vec = seq.components

        # Recompute T for chosen index
        pos = 0
        for i in range(chosen_i):
            node = vec[i]
            ln = max(0, _node_length(node))
            if "Filler" in node.__class__.__name__:
                pos += ln
            else:
                pos += ln
        T = int(pos)
        L = int(max(0, _node_length(chosen_node)))
        print("AAF:", aaf_path)
        print("lane:", lane_idx, "component_index:", chosen_i, "T:", T, "L:", L)

        _dump("BLOCK node", chosen_node)
        try:
            res = resolve_clip_display_name(aaf, chosen_node, cache={})
            print("\n--- RESOLVED display name (aaf_io.clip_naming) ---")
            print("display_name:", res.display_name)
            if res.issues:
                print("issues:", list(res.issues))
            if res.candidates:
                print("candidates:")
                for k, v in res.candidates:
                    print(f"  {k}: {v}")
        except Exception as ex:
            print("\n--- RESOLVED display name (aaf_io.clip_naming) ---")
            print("error:", type(ex).__name__, ex)

        inner = None
        if "OperationGroup" in chosen_node.__class__.__name__:
            # Try to locate first SourceClip under OperationGroup.
            try:
                ins = chosen_node["InputSegments"]
                for j in range(len(ins)):
                    seg = ins[j]
                    if "SourceClip" in seg.__class__.__name__:
                        inner = seg
                        break
            except Exception:
                inner = None
        elif "SourceClip" in chosen_node.__class__.__name__:
            inner = chosen_node

        _dump("INNER SourceClip", inner)

        src_id = None
        if inner is not None:
            try:
                src_id = inner["SourceID"].value
            except Exception:
                src_id = getattr(inner, "source_id", None)
        mob = _resolve_source_mob(aaf, src_id) if src_id is not None else None
        _dump("RESOLVED SourceMob/MasterMob", mob)
        if mob is not None:
            print("mob_name_candidate:", _mob_name(mob))
            desc = getattr(mob, "descriptor", None)
            _dump("MOB descriptor", desc)
            # Locator URLs (Premiere ImportDescriptor etc.)
            try:
                loc_vec = desc["Locator"]
                urls = []
                for k in range(len(loc_vec)):
                    try:
                        url = loc_vec[k]["URLString"].value
                    except Exception:
                        url = None
                    if url:
                        urls.append(str(url))
                if urls:
                    print("\nLocator URLs:")
                    for u in urls[:20]:
                        print(" ", u)
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
