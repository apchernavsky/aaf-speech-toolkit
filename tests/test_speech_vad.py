from __future__ import annotations

import math
from fractions import Fraction
import tempfile
import unittest
import warnings
import wave
from pathlib import Path
from unittest import mock

from aaf_speech_filter.speech_vad import (
    is_quiet_clip,
    quiet_clip_removal_decision,
    read_media_segment_pcm16_mono,
)
from aaf_speech_filter.media_resolve import UnsupportedMediaMapping
from aaf_speech_filter.timeline_timing import (
    choose_timebase_rate_for_sourceclip,
    sourceclip_audio_timing,
)


class SpeechVadTests(unittest.TestCase):
    def test_quiet_decision_uses_timeline_segment_not_whole_take(self) -> None:
        sr = 48000
        samples: list[int] = []
        samples.extend([0] * sr)
        for i in range(sr):
            samples.append(int(28000 * math.sin(2.0 * math.pi * 440.0 * i / sr)))

        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "take.wav"
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(b"".join(int(s).to_bytes(2, "little", signed=True) for s in samples))

            self.assertTrue(is_quiet_clip(wav_path, 0.0, 1.0, -40.0))
            self.assertFalse(is_quiet_clip(wav_path, 1.0, 1.0, -40.0))

    def test_aifc_none_bytes_compression_marker_reads_as_uncompressed_pcm(self) -> None:
        import aifc

        sr = 8000
        samples = [1000, -1000, 2000, -2000]

        with tempfile.TemporaryDirectory() as td:
            aifc_path = Path(td) / "take.aifc"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                wf = aifc.open(str(aifc_path), "wb")
            try:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.setcomptype(b"NONE", b"not compressed")
                wf.writeframes(b"".join(int(s).to_bytes(2, "big", signed=True) for s in samples))
            finally:
                wf.close()

            pcm, read_sr = read_media_segment_pcm16_mono(aifc_path, Fraction(0), Fraction(len(samples), sr))

        decoded = [
            int.from_bytes(pcm[i : i + 2], "little", signed=True)
            for i in range(0, len(samples) * 2, 2)
        ]
        self.assertEqual(read_sr, sr)
        self.assertEqual(decoded, samples)

    def test_quiet_removal_decision_treats_zero_gain_as_mute(self) -> None:
        decision, reason = quiet_clip_removal_decision(
            Path("missing.wav"),
            0.0,
            1.0,
            remove_quiet_clips=True,
            quiet_peak_dbfs=-40.0,
            gain_multiplier=0.0,
        )

        self.assertTrue(decision)
        self.assertEqual(reason, "quiet")

    def test_quiet_removal_decision_respects_disabled_mode(self) -> None:
        decision, reason = quiet_clip_removal_decision(
            Path("missing.wav"),
            0.0,
            1.0,
            remove_quiet_clips=False,
            quiet_peak_dbfs=-40.0,
            gain_multiplier=0.0,
        )

        self.assertFalse(decision)
        self.assertEqual(reason, "")

    def test_short_sourceclip_uses_declared_frame_clock(self) -> None:
        sr = 48000
        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "take.wav"
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(b"\x00\x00" * int(sr * 4.0))

            rate = choose_timebase_rate_for_sourceclip(
                25.0,
                length_units=8,
                start_units=50,
                wav_path=wav_path,
                essence_sr=sr,
            )
            self.assertEqual(rate, 25.0)

    def test_sourceclip_audio_timing_clamps_to_visible_timeline_duration(self) -> None:
        class SourceClip:
            start = 50
            length = 250

        with mock.patch(
            "aaf_speech_filter.timeline_timing.essence_sample_rate_for_sourceclip",
            return_value=48000.0,
        ), mock.patch(
            "aaf_speech_filter.timeline_timing.choose_timebase_rate_for_sourceclip",
            return_value=25.0,
        ), mock.patch(
            "aaf_speech_filter.timeline_timing.resolve_sourceclip_window",
            side_effect=lambda aaf, node, start, duration: (None, start, duration),
        ):
            start_sec, dur_sec = sourceclip_audio_timing(
                object(),
                25.0,
                SourceClip(),
                None,
                visible_duration_sec=2.0,
            )

        self.assertEqual(start_sec, 2.0)
        self.assertEqual(dur_sec, 2.0)


if __name__ == "__main__":
    unittest.main()
