from __future__ import annotations

import struct
import unittest
from pathlib import Path
from unittest.mock import patch

from aaf_speech_filter.speech_vad import read_media_segment_pcm16_mono


class PcmChannelFrameTests(unittest.TestCase):
    def test_downmix_preserves_frames_for_every_channel_count(self):
        for channels in (1, 2, 3, 4, 6, 8):
            with self.subTest(channels=channels):
                frames = [tuple((i + 1) * 1000 - c * 321 for c in range(channels)) for i in range(7)]
                raw = struct.pack('<' + 'h' * (7 * channels), *(s for frame in frames for s in frame))
                with patch('aaf_speech_filter.speech_vad._read_media_segment_raw_16le', return_value=(raw, channels, 48000)):
                    mono, rate = read_media_segment_pcm16_mono(Path('media.wav'), 0, 1)
                self.assertEqual(rate, 48000)
                self.assertEqual(struct.unpack('<' + 'h' * 7, mono), tuple(sum(f) // channels for f in frames))

    def test_incomplete_frame_is_rejected(self):
        with patch('aaf_speech_filter.speech_vad._read_media_segment_raw_16le', return_value=(b'\x00\x00' * 5, 3, 48000)):
            with self.assertRaises(ValueError):
                read_media_segment_pcm16_mono(Path('media.wav'), 0, 1)

    def test_nonpositive_channel_count_is_rejected(self):
        for channels in (0, -1):
            with patch('aaf_speech_filter.speech_vad._read_media_segment_raw_16le', return_value=(b'', channels, 48000)):
                with self.assertRaises(ValueError):
                    read_media_segment_pcm16_mono(Path('media.wav'), 0, 1)

    def test_container_signature_overrides_filename(self):
        import aifc
        import tempfile
        import wave
        from aaf_speech_filter.timeline_timing import wav_sample_rate
        with tempfile.TemporaryDirectory() as directory:
            for kind, suffix, opener, byte_order in (('AIFF', '.wav', aifc.open, 'big'), ('WAV', '.aifc', wave.open, 'little')):
                with self.subTest(kind=kind):
                    path = Path(directory) / ('media' + suffix)
                    with opener(str(path), 'wb') as stream:
                        stream.setnchannels(1)
                        stream.setsampwidth(2)
                        stream.setframerate(44100)
                        stream.writeframes((16000).to_bytes(2, byte_order, signed=True) * 441)
                    mono, rate = read_media_segment_pcm16_mono(path, 0, .01)
                    self.assertEqual(rate, 44100)
                    self.assertEqual(mono, struct.pack('<h', 16000) * 441)
                    self.assertEqual(wav_sample_rate(path), 44100)

    def test_diagnostic_peak_uses_all_channels_per_frame(self):
        from aaf_speech_filter.speech_vad import measure_segment_peak_dbfs, peak_dbfs_16le_mono
        raw = struct.pack('<hhhh', 16000, 16000, 0, 0) * 480
        with patch('aaf_speech_filter.speech_vad._read_media_segment_raw_16le', return_value=(raw, 4, 48000)):
            self.assertEqual(measure_segment_peak_dbfs(Path('media.wav'), 0, .01), peak_dbfs_16le_mono(struct.pack('<h', 8000)))
