#!/usr/bin/env python3
"""
Diagnose YAMNet timing/window + classification for specific timeline blocks in an AAF.

Example:
  python scripts/diag_yamnet_window.py --aaf "C:\\path\\input_processed.aaf" --match "example_music.wav" --match "example_dialog"
"""

from __future__ import annotations

import argparse
import audioop
import csv
import os
import sys
from pathlib import Path
from typing import Any, Optional


def _ensure_toolkit_on_path() -> None:
    here = Path(__file__).resolve()
    toolkit = here.parents[1]
    for p in (toolkit, toolkit / "aaf_io", toolkit / "aaf_speech_filter"):
        if p.is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)


_ensure_toolkit_on_path()

# Make console printing robust on Windows.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aaf", type=Path, required=True)
    ap.add_argument("--match", action="append", default=[], help="Substring to match display name.")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument(
        "--near",
        type=str,
        default="",
        help="Optional: timeline time (e.g. '144.5', 'mm:ss', 'hh:mm:ss', 'hh:mm:ss.ff') to print nearby blocks.",
    )
    ap.add_argument(
        "--span-sec",
        type=float,
        default=5.0,
        help="When using --near, include blocks starting within +/- this many seconds.",
    )
    ap.add_argument("--yamnet-threshold", type=float, default=0.0035)
    ap.add_argument("--yamnet-agg", type=str, default="max", choices=("max", "mean"))
    ap.add_argument(
        "--top-classes",
        type=int,
        default=0,
        help="Print top YAMNet class scores for the same sampled audio windows.",
    )
    ap.add_argument("--vad", action="store_true", help="Print WebRTC VAD voiced-frame ratio for the clip window.")
    ap.add_argument(
        "--silero-vad",
        action="store_true",
        help="Print Silero VAD probabilities and speech intervals for the clip window.",
    )
    ap.add_argument(
        "--whisper-model",
        type=Path,
        default=None,
        help="Optional local faster-whisper model directory for ASR diagnostics.",
    )
    ap.add_argument(
        "--frame-profile",
        action="store_true",
        help="Print frame-level YAMNet speech evidence for the clip window.",
    )
    args = ap.parse_args()

    aaf_path = Path(args.aaf).resolve()
    if not aaf_path.is_file():
        print(f"AAF not found: {aaf_path}")
        return 2

    needles = [str(x) for x in (args.match or []) if str(x).strip()]
    near_raw = str(args.near or "").strip()
    if not needles and not near_raw:
        ap.error("Provide at least one --match or --near")

    def _parse_near_seconds(s: str) -> Optional[float]:
        s = (s or "").strip()
        if not s:
            return None
        # direct float
        try:
            return float(s)
        except Exception:
            pass
        # Try formats like hh:mm:ss.ff or mm:ss.ff ('.' as frame separator)
        # Accept separators ':' and '.' in a loose way: take last 3 colon-separated parts as h/m/s-ish.
        parts = s.split(":")
        if len(parts) == 1:
            return None
        try:
            if len(parts) == 2:
                mm = float(parts[0])
                ss = float(parts[1].replace(",", "."))
                return mm * 60.0 + ss
            if len(parts) >= 3:
                hh = float(parts[-3])
                mm = float(parts[-2])
                ss = float(parts[-1].replace(",", "."))
                return hh * 3600.0 + mm * 60.0 + ss
        except Exception:
            return None
        return None

    near_sec = _parse_near_seconds(near_raw)

    from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
    from aaf_io.converter import AAFConverter

    from aaf_speech_filter.aaf_yamnet_lane_layout import (
        _classify_timeline_block,
        _event_inner_sourceclip,
        _is_operation_group,
        _is_sourceclip,
        _node_length_units,
        _og_first_sourceclip,
        _pick_main_composition,
        _resolve_wave_path_for_sourceclip,
        _lane_yamnet_cfg,
        _slot_edit_rate,
        _sound_sequence_timeline_slots,
        resolve_clip_display_name,
    )
    from aaf_speech_filter.timeline_timing import (
        choose_timebase_rate_for_sourceclip,
        essence_sample_rate_for_sourceclip,
        sourceclip_audio_timing,
        wav_sample_rate,
    )
    from aaf_speech_filter.timeline_walk import _aaf_int_length, _aaf_int_start
    from aaf_speech_filter.speech_yamnet import YamnetConfig, yamnet_clip_kind_with_scores
    from aaf_speech_filter import speech_yamnet as speech_yamnet_mod

    cfg = YamnetConfig(score_threshold=float(args.yamnet_threshold), frame_aggregate=str(args.yamnet_agg))
    # lane-layout config (may adjust thresholds)
    try:
        from aaf_speech_filter.config import FilterConfig

        lane_cfg = _lane_yamnet_cfg(FilterConfig(yamnet=cfg))
    except Exception:
        lane_cfg = cfg

    # Extract embedded essence (read-only) if needed.
    runtime_paths: Optional[dict[Any, Path]] = None
    conv: Optional[AAFConverter] = None
    try:
        conv = AAFConverter(aaf_path, work_dir=aaf_path.parent / "__aaf_tool_work" / f"{aaf_path.stem}.__diag")
        with open_aaf_lenient(aaf_path, "r") as aaf_ro:
            conv.essence_map = {}
            try:
                conv._extract_all(aaf_ro)
            except Exception:
                pass
        runtime_paths = conv.runtime_essence_paths_for_filter()
    except Exception:
        runtime_paths = None

    cache: dict[object, object] = {}
    silero_model: Any = None
    whisper_model: Any = None

    class_names: dict[int, str] = {}

    def _load_class_names() -> dict[int, str]:
        if class_names:
            return class_names
        try:
            model = speech_yamnet_mod._get_yamnet_model(cfg)
            raw_path = model.class_map_path().numpy()
            class_map = Path(raw_path.decode("utf-8") if isinstance(raw_path, bytes) else str(raw_path))
            with class_map.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        class_names[int(row["index"])] = str(row.get("display_name") or "")
                    except Exception:
                        continue
        except Exception:
            pass
        return class_names

    def _top_yamnet_classes(wav: Path, start_sec: float, dur_sec: float, limit: int) -> list[tuple[int, str, float]]:
        if limit <= 0 or dur_sec <= 0:
            return []
        try:
            import numpy as np
            import tensorflow as tf

            names = _load_class_names()
            ns = max(1, int(getattr(cfg, "clip_samples", 1) or 1))
            min_clip = float(getattr(cfg, "clip_samples_min_clip_sec", 1.2) or 1.2)
            win = float(getattr(cfg, "clip_sample_window_sec", 1.0) or 1.0)
            trim = float(getattr(cfg, "clip_edge_trim_sec", 0.1) or 0.0)
            st_eff = float(start_sec)
            du_eff = float(dur_sec)
            if trim > 0.0 and du_eff > (2.0 * trim + 0.05):
                st_eff += trim
                du_eff -= 2.0 * trim
            win = max(0.35, min(win, du_eff))
            if ns <= 1 or du_eff < min_clip or du_eff <= win + 1e-6:
                starts = [st_eff]
            else:
                span = max(0.0, du_eff - win)
                cand_n = max(ns * 3, 9)
                candidates = [st_eff + span * (i / float(cand_n - 1)) for i in range(cand_n)]
                ranked: list[tuple[float, float]] = []
                for st in candidates:
                    try:
                        pcm, _sr = speech_yamnet_mod.read_media_segment_pcm16_mono(wav, float(st), float(win))
                        peak = float(np.max(np.abs(np.frombuffer(pcm, dtype=np.int16)))) if pcm else 0.0
                    except Exception:
                        peak = 0.0
                    ranked.append((peak, float(st)))
                ranked.sort(key=lambda x: (-x[0], x[1]))
                starts = sorted(st for _peak, st in ranked[:ns])

            model = speech_yamnet_mod._get_yamnet_model(cfg)
            merged: Optional[np.ndarray] = None
            for st in starts:
                pcm, sr = speech_yamnet_mod.read_media_segment_pcm16_mono(wav, float(st), float(win))
                if not pcm or len(pcm) < 4:
                    continue
                audio = speech_yamnet_mod._pcm16_mono_to_float32_16k(pcm, sr)
                if len(audio) < 400:
                    continue
                min_samples = int(0.35 * 16000)
                if len(audio) < min_samples:
                    audio = np.pad(audio, (0, min_samples - len(audio)))
                scores, _emb, _spec = model(tf.convert_to_tensor(audio, dtype=tf.float32))
                probs = scores.numpy()
                if probs.size == 0:
                    continue
                frame_max = np.max(probs, axis=0)
                merged = frame_max if merged is None else np.maximum(merged, frame_max)
            if merged is None or merged.size == 0:
                return []
            idxs = list(np.argsort(merged)[::-1][: int(limit)])
            return [(int(i), str(names.get(int(i), "")), float(merged[int(i)])) for i in idxs]
        except Exception as exc:
            return [(-1, f"ERR:{type(exc).__name__}:{exc}", 0.0)]

    def _vad_ratio(wav: Path, start_sec: float, dur_sec: float) -> float:
        if not bool(args.vad) or dur_sec <= 0:
            return -1.0
        try:
            import webrtcvad

            pcm, sr = speech_yamnet_mod.read_media_segment_pcm16_mono(wav, float(start_sec), float(dur_sec))
            if not pcm or len(pcm) < 2:
                return 0.0
            if int(sr) != 16000:
                pcm, _state = audioop.ratecv(pcm, 2, 1, int(sr), 16000, None)
                sr = 16000
            vad = webrtcvad.Vad(3)
            frame_ms = 30
            frame_bytes = int(sr * frame_ms / 1000) * 2
            total = 0
            voiced = 0
            for off in range(0, max(0, len(pcm) - frame_bytes + 1), frame_bytes):
                frame = pcm[off : off + frame_bytes]
                if len(frame) != frame_bytes:
                    continue
                total += 1
                if vad.is_speech(frame, int(sr)):
                    voiced += 1
            return float(voiced) / float(total) if total > 0 else 0.0
        except Exception:
            return -1.0

    def _silero_vad_profile(wav: Path, start_sec: float, dur_sec: float) -> dict[str, Any]:
        nonlocal silero_model
        if not bool(args.silero_vad) or dur_sec <= 0:
            return {}
        try:
            import numpy as np
            import torch
            from silero_vad import get_speech_timestamps, load_silero_vad

            pcm, sr = speech_yamnet_mod.read_media_segment_pcm16_mono(wav, float(start_sec), float(dur_sec))
            if not pcm or len(pcm) < 2:
                return {}
            audio = speech_yamnet_mod._pcm16_mono_to_float32_16k(pcm, sr)
            if len(audio) < 1:
                return {}
            tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))
            if silero_model is None:
                silero_model = load_silero_vad(onnx=True)
            probs = silero_model.audio_forward(tensor, 16000).numpy().reshape(-1)
            speech = get_speech_timestamps(
                tensor,
                silero_model,
                sampling_rate=16000,
                min_speech_duration_ms=120,
                min_silence_duration_ms=80,
                speech_pad_ms=0,
            )
            speech_samples = sum(max(0, int(x.get("end", 0)) - int(x.get("start", 0))) for x in speech)
            return {
                "max": float(np.max(probs)) if probs.size else 0.0,
                "mean": float(np.mean(probs)) if probs.size else 0.0,
                "support_050": float(np.mean(probs >= 0.50)) if probs.size else 0.0,
                "speech_ratio": float(speech_samples) / float(len(audio)),
                "speech": speech,
            }
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _whisper_profile(wav: Path, start_sec: float, dur_sec: float) -> dict[str, Any]:
        nonlocal whisper_model
        if args.whisper_model is None or dur_sec <= 0:
            return {}
        try:
            import numpy as np
            from faster_whisper import WhisperModel

            pcm, sr = speech_yamnet_mod.read_media_segment_pcm16_mono(wav, float(start_sec), float(dur_sec))
            if not pcm or len(pcm) < 2:
                return {}
            audio = speech_yamnet_mod._pcm16_mono_to_float32_16k(pcm, sr)
            if len(audio) < 1:
                return {}
            if whisper_model is None:
                whisper_model = WhisperModel(
                    str(Path(args.whisper_model).resolve()),
                    device="cpu",
                    compute_type="int8",
                    local_files_only=True,
                )
            segments_iter, info = whisper_model.transcribe(
                np.asarray(audio, dtype=np.float32),
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            segments = []
            for segment in segments_iter:
                segments.append(
                    {
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": str(segment.text or "").strip(),
                        "avg_logprob": float(segment.avg_logprob),
                        "no_speech_prob": float(segment.no_speech_prob),
                    }
                )
            return {
                "language": str(getattr(info, "language", "") or ""),
                "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
                "segments": segments,
            }
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _yamnet_frame_profile(wav: Path, start_sec: float, dur_sec: float) -> dict[str, Any]:
        if not bool(args.frame_profile) or dur_sec <= 0:
            return {}
        try:
            import numpy as np
            import tensorflow as tf

            trim = float(getattr(cfg, "clip_edge_trim_sec", 0.1) or 0.0)
            st_eff = float(start_sec)
            du_eff = float(dur_sec)
            if trim > 0.0 and du_eff > (2.0 * trim + 0.05):
                st_eff += trim
                du_eff -= 2.0 * trim
            pcm, sr = speech_yamnet_mod.read_media_segment_pcm16_mono(wav, st_eff, du_eff)
            if not pcm or len(pcm) < 4:
                return {}
            audio = speech_yamnet_mod._pcm16_mono_to_float32_16k(pcm, sr)
            if len(audio) < 400:
                return {}
            min_samples = int(0.35 * 16000)
            if len(audio) < min_samples:
                audio = np.pad(audio, (0, min_samples - len(audio)))
            model = speech_yamnet_mod._get_yamnet_model(cfg)
            scores, _emb, _spec = model(tf.convert_to_tensor(audio, dtype=tf.float32))
            probs = scores.numpy()
            if probs.size == 0:
                return {}
            stop = max(1, min(int(cfg.speech_class_stop), probs.shape[-1]))
            generic = probs[:, 0]
            concrete = np.max(probs[:, 1:stop], axis=-1) if stop > 1 else np.zeros_like(generic)
            linguistic_indices = [i for i in (1, 2, 3, 5, 12) if i < stop]
            linguistic = (
                np.max(probs[:, linguistic_indices], axis=-1)
                if linguistic_indices
                else np.zeros_like(generic)
            )
            return {
                "generic": [float(v) for v in generic],
                "concrete": [float(v) for v in concrete],
                "linguistic": [float(v) for v in linguistic],
                "linguistic_support_001": float(np.mean(linguistic >= 0.01)),
                "linguistic_support_002": float(np.mean(linguistic >= 0.02)),
            }
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def clip_name(aaf: Any, node: Any) -> str:
        try:
            r = resolve_clip_display_name(aaf, node, cache=cache)
            return (r.display_name or "").strip()
        except Exception:
            return ""

    printed = 0
    collected: list[dict[str, Any]] = []
    with open_aaf_lenient(aaf_path, "r") as aaf:
        comps = list(aaf.content.compositionmobs())
        comp = _pick_main_composition(comps) if comps else None
        if comp is None:
            print("No composition mobs.")
            return 1
        lanes, skipped = _sound_sequence_timeline_slots(comp, cancel_check=lambda: None)
        if skipped:
            print(f"Skipped non-Sequence sound slots: {skipped}")

        for lane_idx, (slot, seq) in enumerate(lanes):
            top = seq.components
            try:
                tn = len(top)
            except Exception:
                continue
            pos = 0
            for ti in range(tn):
                try:
                    node = top[ti]
                except Exception:
                    break
                ln = int(_node_length_units(node))
                if _is_operation_group(node) or _is_sourceclip(node):
                    nm = clip_name(aaf, node)
                    # Optional "near" filter: select by timeline start position in seconds.
                    er_slot = _safe_float(_slot_edit_rate(slot), 48000.0) or 48000.0
                    t_sec = (float(pos) / float(er_slot)) if er_slot > 0 else 0.0
                    near_ok = True
                    if near_sec is not None:
                        near_ok = abs(t_sec - float(near_sec)) <= float(args.span_sec)

                    match_ok = bool(nm and any(nd in nm for nd in needles)) if needles else False
                    if match_ok or (near_sec is not None and near_ok):
                        inner_sc = _og_first_sourceclip(node) if _is_operation_group(node) else node
                        wav = None
                        if inner_sc is not None:
                            try:
                                wav = _resolve_wave_path_for_sourceclip(aaf, inner_sc, runtime_paths)
                            except Exception:
                                wav = None
                        wav_ok = int(bool(wav is not None and Path(wav).is_file()))
                        wav_sr = 0.0
                        wav_frames = 0
                        if wav_ok:
                            try:
                                wav_sr = float(wav_sample_rate(Path(wav)))
                            except Exception:
                                wav_sr = 0.0
                            try:
                                import wave

                                with wave.open(str(wav), "rb") as wf:
                                    wav_frames = int(wf.getnframes())
                            except Exception:
                                wav_frames = 0

                        st_units = int(_aaf_int_start(inner_sc)) if inner_sc is not None else 0
                        ln_units = int(_aaf_int_length(inner_sc)) if inner_sc is not None else 0
                        ess_sr = _safe_float(essence_sample_rate_for_sourceclip(aaf, inner_sc, runtime_paths), 0.0) if inner_sc is not None else 0.0
                        chosen_rate = _safe_float(
                            choose_timebase_rate_for_sourceclip(
                                er_slot,
                                length_units=ln_units,
                                start_units=st_units,
                                wav_path=Path(wav) if wav is not None else None,
                                essence_sr=ess_sr,
                            ),
                            er_slot,
                        )
                        start_sec, dur_sec = (0.0, 0.0)
                        if inner_sc is not None:
                            start_sec, dur_sec = sourceclip_audio_timing(
                                aaf,
                                er_slot,
                                inner_sc,
                                Path(wav) if wav is not None else None,
                                runtime_paths,
                                visible_duration_sec=(float(ln) / float(er_slot)) if er_slot > 0 else 0.0,
                            )

                        kind = "unknown"
                        kind_lane = "unknown"
                        s = m = n = 0.0
                        if wav_ok and dur_sec > 0:
                            try:
                                kind, s, m, n = yamnet_clip_kind_with_scores(
                                    Path(wav),
                                    float(start_sec),
                                    float(dur_sec),
                                    cfg,
                                    cancel_check=lambda: None,
                                )
                            except Exception as e:
                                kind = f"ERR:{type(e).__name__}"
                        try:
                            kind_lane = _classify_timeline_block(
                                aaf,
                                node,
                                runtime_paths,
                                lane_cfg,
                                er_slot,
                                cancel_check=lambda: None,
                            )
                        except Exception as e:
                            kind_lane = f"ERR:{type(e).__name__}"

                        rec = {
                            "lane": int(lane_idx),
                            "top": int(ti),
                            "name": str(nm),
                            "T": int(pos),
                            "T_sec": float(t_sec),
                            "L": int(ln),
                            "slot_er": float(er_slot),
                            "chosen_rate": float(chosen_rate),
                            "start_units": int(st_units),
                            "len_units": int(ln_units),
                            "start_sec": float(start_sec),
                            "dur_sec": float(dur_sec),
                            "wav_ok": int(wav_ok),
                            "wav": str(wav) if wav is not None else "",
                            "wav_sr": float(wav_sr),
                            "wav_frames": int(wav_frames),
                            "ess_sr": float(ess_sr),
                            "yamnet_kind": str(kind),
                            "s": float(_safe_float(s)),
                            "m": float(_safe_float(m)),
                            "n": float(_safe_float(n)),
                            "lane_kind": str(kind_lane),
                            "top_classes": _top_yamnet_classes(Path(wav), float(start_sec), float(dur_sec), int(args.top_classes)) if wav_ok else [],
                            "vad_ratio": _vad_ratio(Path(wav), float(start_sec), float(dur_sec)) if wav_ok else -1.0,
                            "silero_profile": _silero_vad_profile(Path(wav), float(start_sec), float(dur_sec)) if wav_ok else {},
                            "whisper_profile": _whisper_profile(Path(wav), float(start_sec), float(dur_sec)) if wav_ok else {},
                            "frame_profile": _yamnet_frame_profile(Path(wav), float(start_sec), float(dur_sec)) if wav_ok else {},
                            "node": node,
                        }
                        collected.append(rec)
                        printed += 1
                        if printed >= int(args.limit):
                            break
                pos += ln
            if printed >= int(args.limit):
                break

    if printed == 0:
        print("No matching clips found by display name.")
        return 0

    for r in collected:
        print(
            f"[lane={r['lane']} top={r['top']}] name='{r['name']}' "
            f"T={r['T']} (t={r.get('T_sec', 0.0):.3f}s) L={r['L']} slot_er={r['slot_er']:g} chosen_rate={r['chosen_rate']:g} "
            f"start_units={r['start_units']} len_units={r['len_units']} "
            f"start_sec={r['start_sec']:.3f} dur_sec={r['dur_sec']:.3f} "
            f"wav_ok={r['wav_ok']} wav='{r['wav']}' wav_sr={r['wav_sr']:g} wav_frames={r['wav_frames']} ess_sr={r['ess_sr']:g} "
            f"yamnet(kind={r['yamnet_kind']} s={r['s']:.4f} m={r['m']:.4f} n={r['n']:.4f}) "
            f"lane_kind={r['lane_kind']}"
            + (f" vad={r['vad_ratio']:.3f}" if float(r.get("vad_ratio", -1.0)) >= 0.0 else "")
        )
        if r.get("top_classes"):
            for idx, label, score in r["top_classes"]:
                print(f"    class[{idx}] {label}: {score:.6f}")
        if r.get("frame_profile"):
            profile = r["frame_profile"]
            if profile.get("error"):
                print(f"    frame_profile error: {profile['error']}")
            else:
                generic = ", ".join(f"{v:.4f}" for v in profile.get("generic", []))
                concrete = ", ".join(f"{v:.4f}" for v in profile.get("concrete", []))
                linguistic = ", ".join(f"{v:.4f}" for v in profile.get("linguistic", []))
                print(f"    frame generic_speech: [{generic}]")
                print(f"    frame concrete_speech: [{concrete}]")
                print(f"    frame linguistic_speech: [{linguistic}]")
                print(
                    "    frame linguistic_support: "
                    f">=.01 {float(profile.get('linguistic_support_001', 0.0)):.3f}, "
                    f">=.02 {float(profile.get('linguistic_support_002', 0.0)):.3f}"
                )
        if r.get("silero_profile"):
            profile = r["silero_profile"]
            if profile.get("error"):
                print(f"    silero error: {profile['error']}")
            else:
                print(
                    "    silero: "
                    f"max={float(profile.get('max', 0.0)):.3f} "
                    f"mean={float(profile.get('mean', 0.0)):.3f} "
                    f"support>=.50={float(profile.get('support_050', 0.0)):.3f} "
                    f"speech_ratio={float(profile.get('speech_ratio', 0.0)):.3f} "
                    f"speech={profile.get('speech', [])}"
                )
        if r.get("whisper_profile"):
            profile = r["whisper_profile"]
            if profile.get("error"):
                print(f"    whisper error: {profile['error']}")
            else:
                print(
                    "    whisper: "
                    f"language={profile.get('language', '')!r} "
                    f"prob={float(profile.get('language_probability', 0.0)):.3f} "
                    f"segments={profile.get('segments', [])}"
                )

    try:
        if conv is not None:
            conv._unlink_mapped_essence()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
