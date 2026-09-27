"""Embedded essence extraction and AAF preparation (read-only on source)."""

from __future__ import annotations

import math
import os
import shutil
import struct
import traceback
import wave
import audioop
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

import aaf2

from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
from aaf_io.errors import OperationCancelled
from aaf_io.output_transaction import output_candidate
from aaf_io.path_safety import require_distinct_files

def _safe_int(v, d=0):
    try: return int(v)
    except: return d

def _safe_float(v, d=0.0):
    try: return float(v)
    except: return d

def _prop_value(obj, key: str):
    try:
        v = obj[key]
        try:
            return v.value
        except Exception:
            return v
    except Exception:
        return None

def _bytes_from_summary(summary) -> bytes:
    if summary is None:
        return b""
    if isinstance(summary, (bytes, bytearray)):
        return bytes(summary)
    try:
        return bytes(int(x) & 0xFF for x in summary)
    except Exception:
        return b""

def _parse_wav_summary_params(summary) -> dict:
    data = _bytes_from_summary(summary)
    if len(data) < 28 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return {}
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
        chunk_start = pos + 8
        if chunk_id == b"fmt " and chunk_size >= 16 and chunk_start + 16 <= len(data):
            _fmt, channels, sample_rate, _byte_rate, _align, bits = struct.unpack_from(
                "<HHIIHH", data, chunk_start
            )
            return {
                "sample_rate_float": float(sample_rate),
                "channels": int(channels),
                "quantization_bits": int(bits),
            }
        pos += 8 + chunk_size
        if chunk_size % 2:
            pos += 1
    return {}

def _parse_aifc_summary_params(summary) -> dict:
    data = _bytes_from_summary(summary)
    if len(data) < 38 or data[:4] != b"FORM" or data[8:12] not in (b"AIFC", b"AIFF"):
        return {}
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        chunk_size = struct.unpack_from(">I", data, pos + 4)[0]
        chunk_start = pos + 8
        if chunk_id == b"COMM" and chunk_size >= 18 and chunk_start + 18 <= len(data):
            channels, _frames, bits = struct.unpack_from(">hIh", data, chunk_start)
            compression = b"NONE"
            if data[8:12] == b"AIFC":
                if chunk_size < 22 or chunk_start + 22 > len(data):
                    raise ValueError("AIFC COMM is missing compression metadata")
                compression = data[chunk_start + 18:chunk_start + 22]
            params = {
                "channels": int(channels),
                "quantization_bits": int(bits),
                "compression": compression,
            }
            exponent, mantissa = struct.unpack_from(">HQ", data, chunk_start + 8)
            if mantissa and not (exponent & 0x8000) and exponent < 0x7fff:
                params["sample_rate_float"] = math.ldexp(mantissa, exponent - 16383 - 63)
            return params
        pos += 8 + chunk_size
        if chunk_size % 2:
            pos += 1
    return {}

def _parse_aifc_pcm(data: bytes) -> bytes:
    if data[:4] != b"FORM" or data[8:12] not in (b"AIFC", b"AIFF"):
        return data
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack_from(">I", data, pos + 4)[0]
        end = pos + 8 + chunk_size
        if end > len(data):
            raise ValueError("Truncated AIFF/AIFC essence chunk")
        if chunk_id == b"SSND":
            if chunk_size < 8:
                raise ValueError("Truncated AIFF/AIFC SSND header")
            offset = struct.unpack_from(">I", data, pos + 8)[0]
            start = pos + 16 + offset
            if start > end:
                raise ValueError("Invalid AIFF/AIFC sound offset")
            return data[start:end]
        pos = end + chunk_size % 2
    raise ValueError("AIFF/AIFC essence is missing SSND")


def _parse_riff_wav_pcm(data: bytes) -> bytes:
    """If *data* is a complete RIFF/WAVE file, return only the raw PCM bytes
    from its 'data' chunk.  Otherwise return *data* unchanged."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return data
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
        if chunk_id == b"data":
            return data[pos + 8 : pos + 8 + chunk_size]
        pos += 8 + chunk_size
        if chunk_size % 2:
            pos += 1
    return data

def _write_wav(path: Path, raw: bytes, sr: float, ch: int, bps: int):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(max(1, bps // 8))
        wf.setframerate(int(sr))
        wf.writeframes(raw)


class AAFConverter:
    """Read-only source preparation with an exclusively owned extraction child.

    Successful preparation retains extracted media for analysis. The caller must
    release it with _unlink_mapped_essence(), or remove the caller-owned work_dir
    after all readers close. Failed preparation releases this converter's child.
    No allocation happens in the constructor or before output identity checks.
    """
    def __init__(self, input_path, *, work_dir: Optional[Path] = None):
        self.input_path = Path(input_path)
        # Extraction is session-owned even when the caller supplies a shared parent.
        self.output_dir = Path(work_dir) if work_dir is not None else self.input_path.parent
        self.essence_dir = self.output_dir / ("essence-" + uuid4().hex)
        self._owns_essence_dir = False
        self.essence_map = {}
        self.last_prepare_error: Optional[str] = None

    def runtime_essence_paths_for_filter(self) -> Optional[dict]:
        """
        MobID → путь к выгруженному WAV (для YAMNet / VAD), если embedded выгружали
        без правки AAF. None — если выгрузки не было.
        """
        if not self.essence_map:
            return None
        return {
            mid: self.essence_dir / fname
            for mid, (fname, _params) in self.essence_map.items()
        }

    def _ensure_essence_dir(self) -> None:
        if not self._owns_essence_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.essence_dir.mkdir(exist_ok=False)
            self._owns_essence_dir = True

    def _unlink_mapped_essence(self) -> None:
        """Release only this converter's extraction child; failures are observable."""
        if self._owns_essence_dir:
            shutil.rmtree(self.essence_dir)
            self._owns_essence_dir = False
        self.essence_map.clear()

    def _get_dict_obj(self, aaf): return getattr(aaf, 'content', aaf)

    def _get_aaf_dict(self, aaf, possible_keys):
        c = self._get_dict_obj(aaf)
        for key in possible_keys:
            try:
                val = c[key]
                if val is not None: return val
            except Exception: pass
            try:
                val = getattr(c, key, None)
                if val is not None: return val
            except Exception: pass
        return None

    def _get_mobs_dict(self, aaf): return self._get_aaf_dict(aaf, ["Mobs", "mobs"])
    def _get_essence_dict(self, aaf): return self._get_aaf_dict(aaf, ["EssenceData", "essencedata", "EssenceDatas"])

    def _get_audio_params(self, desc):
        summary = _prop_value(desc, "Summary")
        summary_params = _parse_wav_summary_params(summary)
        if not summary_params:
            summary_params = _parse_aifc_summary_params(summary)

        sr_raw = getattr(desc, "sample_rate", None) or _prop_value(desc, "SampleRate")
        ch_raw = getattr(desc, "channels", None) or _prop_value(desc, "Channels")
        bits_raw = (
            getattr(desc, "quantization_bits", None)
            or _prop_value(desc, "QuantizationBits")
        )
        return {
            "sample_rate": sr_raw,
            "sample_rate_float": _safe_float(
                sr_raw, summary_params.get("sample_rate_float", 48000.0)
            ),
            "channels": _safe_int(ch_raw, summary_params.get("channels", 1)),
            "quantization_bits": _safe_int(
                bits_raw, summary_params.get("quantization_bits", 16)
            ),
        }

    def _is_embedded(self, desc):
        if not desc: return False
        fp = getattr(desc, "file_path", None)
        return not (fp and str(fp).strip())

    def _get_essence_bytes(self, edata, aaf_ref):
        # The stream borrows its storage handle from the caller's open AAF.
        # Data.value is a stream name, not the payload or a file offset.
        return bytes(edata.open("r").read())

    def _extract_audio(self, edata, out_path, desc, aaf_ref):
        raw = self._get_essence_bytes(edata, aaf_ref)
        if not raw:
            return False
        drive, tail = os.path.splitdrive(str(out_path))
        out_path = Path(drive + tail.replace(":", "_").replace("?", "_"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        p = self._get_audio_params(desc)
        if "AIFC" in desc.__class__.__name__:
            encoding = _parse_aifc_summary_params(raw)
            if not encoding:
                encoding = _parse_aifc_summary_params(_prop_value(desc, "Summary"))
            if not encoding:
                raise ValueError("AIFC essence is missing explicit encoding metadata")
            for field in ("channels", "quantization_bits", "sample_rate_float"):
                if field in encoding and encoding[field] != p[field]:
                    raise ValueError(f"AIFC descriptor/payload format conflict: {field}")
            summary = _parse_aifc_summary_params(_prop_value(desc, "Summary"))
            if summary and summary["compression"] != encoding["compression"]:
                raise ValueError("AIFC descriptor/payload compression conflict")
            compression = encoding["compression"]
            if compression not in (b"NONE", b"twos", b"sowt"):
                raise ValueError(f"Unsupported AIFC compression: {compression!r}")
            bits = encoding["quantization_bits"]
            if bits not in (8, 16, 24, 32):
                raise ValueError(f"Unsupported AIFC PCM bit depth: {bits}")
            raw = _parse_aifc_pcm(raw)
            sw = bits // 8
            if len(raw) % (sw * encoding["channels"]):
                raise ValueError("Incomplete AIFF/AIFC PCM frame")
            if sw == 1:
                raw = audioop.bias(raw, 1, 128)
            elif compression != b"sowt":
                raw = audioop.byteswap(raw, sw)
        else:
            raw = _parse_riff_wav_pcm(raw)
        _write_wav(out_path, raw, p["sample_rate_float"], p["channels"], p["quantization_bits"])
        return True

    def _extract_video(self, edata, out_path, desc, aaf_ref):
        raw = self._get_essence_bytes(edata, aaf_ref)
        if not raw:
            return False
        drive, tail = os.path.splitdrive(str(out_path))
        out_path = Path(drive + tail.replace(":", "_").replace("?", "_"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(raw)
        return True

    def _extract_all(self, aaf, cancel_event=None, cancel_exc_type=None):
        self._ensure_essence_dir()
        def _check_cancel() -> None:
            if (
                cancel_event is not None
                and cancel_event.is_set()
            ):
                raise (cancel_exc_type or OperationCancelled)()

        ed_by_mob = {}
        ed_dict = self._get_essence_dict(aaf)

        if ed_dict:
            try:
                ed_items = ed_dict.items()
            except TypeError:
                ed_items = enumerate(ed_dict)

            for key, edata in ed_items:
                _check_cancel()
                mid = None
                try: mid = edata.mob_id
                except Exception: pass
                if not mid:
                    try: mid = getattr(edata, 'mob_id', None)
                    except Exception: pass
                if not mid:
                    try: mid = getattr(edata, 'MobID', None)
                    except Exception: pass
                if not mid:
                    try: mid = edata["MobID"]
                    except Exception: pass
                if not mid:
                    try: mid = UUID(str(key))
                    except: pass
                if mid: ed_by_mob[mid] = edata

        if not ed_by_mob and ed_dict:
            mobs_dict = self._get_mobs_dict(aaf)
            if mobs_dict:
                mv = mobs_dict.values() if hasattr(mobs_dict, "values") else mobs_dict
                ev = ed_dict.values() if hasattr(ed_dict, "values") else ed_dict
                for mob, edata in zip(mv, ev):
                    try:
                        if mob.mob_id:
                            ed_by_mob[mob.mob_id] = edata
                    except Exception:
                        pass

        if not ed_by_mob: return 0
        mobs_dict = self._get_mobs_dict(aaf)
        if not mobs_dict: return 0

        count = 0
        comp_mod = getattr(aaf2, 'components', None)
        mob_iter = mobs_dict.values() if hasattr(mobs_dict, 'values') else mobs_dict

        for mob in mob_iter:
            _check_cancel()
            if not hasattr(mob, 'descriptor') or not mob.descriptor: continue
            if not self._is_embedded(mob.descriptor): continue
            try: mid = mob.mob_id
            except: continue
            if mid not in ed_by_mob: continue

            desc = mob.descriptor
            cname = desc.__class__.__name__
            edata = ed_by_mob[mid]
            count += 1

            is_audio = ('PCM' in cname or 'WAVE' in cname or 'AIFC' in cname)
            is_video = ('CDCI' in cname or 'RGBA' in cname)
            if comp_mod:
                if hasattr(comp_mod, 'PCMDescriptor') and isinstance(desc, comp_mod.PCMDescriptor): is_audio = True
                if hasattr(comp_mod, 'WAVEDescriptor') and isinstance(desc, comp_mod.WAVEDescriptor): is_audio = True
                if hasattr(comp_mod, 'AIFCDescriptor') and isinstance(desc, comp_mod.AIFCDescriptor): is_audio = True
                if hasattr(comp_mod, 'CDCIDescriptor') and isinstance(desc, comp_mod.CDCIDescriptor): is_video = True
                if hasattr(comp_mod, 'RGBADescriptor') and isinstance(desc, comp_mod.RGBADescriptor): is_video = True

            if is_audio:
                out_path = self.essence_dir / f"audio_{count:03d}.wav"
                if self._extract_audio(edata, out_path, desc, aaf):
                    self.essence_map[mid] = (out_path.name, self._get_audio_params(desc))
            elif is_video:
                out_path = self.essence_dir / f"video_{count:03d}.raw"
                if self._extract_video(edata, out_path, desc, aaf):
                    self.essence_map[mid] = (out_path.name, {})
        return count

    def _try_inplace_unembed(
        self,
        out_path: Path,
        cancel_event=None,
        cancel_exc_type=None,
    ) -> bool:
        """
        Копия AAF + выгрузка встроенного audio в ``essence/*.wav`` только для анализа (YAMNet / VAD).
        Сам контейнер AAF **не** перезаписываем в ``r+`` (PyAAF2 портит большие embedded Nuendo-AAF);
        фильтр получает пути через ``runtime_essence_paths_for_filter``.
        """
        def _check_cancel() -> None:
            if (
                cancel_event is not None
                and cancel_event.is_set()
            ):
                raise (cancel_exc_type or OperationCancelled)()

        # Identity rejection happens before allocation, source open, or cleanup.
        require_distinct_files(self.input_path, out_path)
        self.last_prepare_error = None
        _check_cancel()
        try:
            with output_candidate(self.input_path, out_path, cancel_check=_check_cancel) as candidate:
                self._ensure_essence_dir()
                with open_aaf_lenient(self.input_path, "r") as aaf:
                    self.essence_map = {}
                    self._extract_all(aaf, cancel_event=cancel_event, cancel_exc_type=cancel_exc_type)
                _check_cancel()
                shutil.copyfile(str(self.input_path), str(candidate))
                with open_aaf_lenient(candidate, "r") as checked:
                    _ = checked.content
            return True
        except Exception as exc:
            self.last_prepare_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1800:]}"
            self._unlink_mapped_essence()
            if isinstance(exc, OperationCancelled) or (
                cancel_exc_type is not None and isinstance(exc, cancel_exc_type)
            ):
                raise
            return False

    def prepare_external_media_copy(
        self,
        out_path: Path,
        cancel_event=None,
        cancel_exc_type=None,
    ) -> bool:
        """
        Копия AAF в ``out_path`` и выгрузка встроенного essence в собственный каталог essence (только для анализа
        в фильтре). Локаторы в AAF не меняются — результат остаётся embedded, как в Nuendo.
        """
        return self._try_inplace_unembed(
            out_path, cancel_event=cancel_event, cancel_exc_type=cancel_exc_type
        )
