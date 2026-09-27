from __future__ import annotations

import hashlib
import struct
import sys
import tempfile
import unittest
import warnings
import wave
from fractions import Fraction
from pathlib import Path

import aaf2
from aaf_speech_filter.config import FilterConfig
from aaf_speech_filter.media_resolve import UnsupportedMediaMapping
from aaf_speech_filter.pyaaf2_filter import filter_aaf_speech_only
from aaf_speech_filter.sdk_removals import collect_timeline_sourceclip_removals_for_sdk_xml
from aaf_speech_filter.speech_vad import read_media_segment_pcm16_mono, is_quiet_clip
from aaf_speech_filter.timeline_sourceclips import iter_timeline_sourceclips
from aaf_speech_filter.timeline_timing import sourceclip_audio_timing, choose_timebase_rate_for_sourceclip
from tests.test_audit_audio import add_comp, add_source

ROOT = Path(__file__).resolve().parents[1]


class ReauditAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.media = self.folder / 'media.bin'
        self.aaf = self.folder / 'input.aaf'
        self.output = self.folder / 'output.aaf'

    def write_pcm(self, samples, *, rate=48000, width=2, kind='wav', compression=b'NONE'):
        if kind == 'wav':
            with wave.open(str(self.media), 'wb') as stream:
                stream.setparams((1, width, rate, 0, 'NONE', 'not compressed'))
                stream.writeframes(samples)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                import aifc
            with aifc.open(str(self.media), 'wb') as stream:
                stream.aifc()
                stream.setparams((1, width, rate, 0, b'NONE', b'PCM'))
                stream.writeframes(samples)
            if compression == b'sowt':
                data = bytearray(self.media.read_bytes())
                comm = data.index(b'COMM') + 8
                data[comm + 18:comm + 22] = b'sowt'
                payload = data.index(b'SSND') + 16
                data[payload:payload + len(samples)] = b''.join(samples[i:i + width][::-1] for i in range(0, len(samples), width))
                self.media.write_bytes(data)

    def make_aaf(self, length, *, start=0, edit_rate=48000, kind='wav', media_rate=48000):
        with aaf2.open(str(self.aaf), 'w') as aaf:
            if kind == 'wav':
                source = add_source(aaf, self.media)
            else:
                source = aaf.create.SourceMob()
                aaf.content.mobs.append(source)
                desc = aaf.create.AIFCDescriptor()
                desc['SampleRate'].value = media_rate
                desc['Length'].value = length
                data = self.media.read_bytes()
                desc['Summary'].value = data[:data.index(b'SSND')]
                locator = aaf.create.NetworkLocator()
                locator['URLString'].value = self.media.as_uri()
                desc['Locator'].append(locator)
                source.descriptor = desc
                source.create_timeline_slot(media_rate).segment = aaf.create.SourceClip(media_kind='sound', length=length)
            comp = add_comp(aaf)
            comp.create_timeline_slot(edit_rate).segment = source.create_source_clip(1, start=start, length=length, media_kind='sound')

    def assert_backends(self, removed, *, diagnostic=None):
        before = hashlib.sha256(self.aaf.read_bytes()).digest()
        logs = []
        keys = collect_timeline_sourceclip_removals_for_sdk_xml(self.aaf, FilterConfig(), log_callback=logs.append)
        with self.subTest(backend='SDK'):
            self.assertEqual(len(keys), removed)
        with self.subTest(backend='PyAAF2'):
            self.assertEqual(filter_aaf_speech_only(self.aaf, self.output, FilterConfig()), removed)
            with aaf2.open(str(self.output), 'r') as aaf:
                self.assertEqual(len(list(iter_timeline_sourceclips(aaf))), 1 - removed)
        self.assertEqual(hashlib.sha256(self.aaf.read_bytes()).digest(), before)
        if diagnostic:
            self.assertTrue(any(diagnostic in message for message in logs), logs)

    def test_boundary_impulse_is_retained_by_both_backends(self):
        self.write_pcm(b'\0\0' * 3002 + struct.pack('<h', 16000))
        self.make_aaf(3003)
        self.assert_backends(0)

    def test_exact_nonzero_window_does_not_include_adjacent_impulses(self):
        samples = struct.pack('<h', 16000) * 3003 + b'\0\0' * 3003 + struct.pack('<h', 16000)
        self.write_pcm(samples)
        raw, _ = read_media_segment_pcm16_mono(self.media, Fraction(3003, 48000), Fraction(3003, 48000))
        self.assertEqual(raw, b'\0\0' * 3003)
        self.make_aaf(3003, start=3003)
        self.assert_backends(1)

    def test_float_windows_round_to_nearest_source_frame(self):
        self.write_pcm(b'\0\0' * 3002 + struct.pack('<h', 16000))
        raw, _ = read_media_segment_pcm16_mono(self.media, 0.0, 3003 / 48000)
        self.assertEqual(len(raw), 6006)
        self.assertFalse(is_quiet_clip(self.media, 0, 3003 / 48000, -40))

    def test_fractional_frame_window_covers_intersecting_frames(self):
        self.write_pcm(struct.pack('<hhh', 1000, 2000, 3000))
        raw, _ = read_media_segment_pcm16_mono(self.media, Fraction(1, 96000), Fraction(1, 48000))
        self.assertEqual(raw, struct.pack('<hh', 1000, 2000))

    def test_truncated_payload_is_retained_by_both_backends(self):
        self.write_pcm(b'\0\0' * 48000)
        self.make_aaf(48000)
        self.media.write_bytes(self.media.read_bytes()[:44 + 4800 * 2])
        self.assert_backends(0, diagnostic='incomplete')

    def test_window_past_eof_is_retained_by_both_backends(self):
        self.write_pcm(b'\0\0' * 4800)
        self.make_aaf(48000)
        self.assert_backends(0, diagnostic='incomplete')

    def test_complete_short_window_at_eof_can_be_removed(self):
        self.write_pcm(struct.pack('<h', 16000) * 7 + b'\0\0')
        self.make_aaf(1, start=7)
        self.assert_backends(1)

    def test_negative_and_empty_windows_are_not_quiet_evidence(self):
        self.write_pcm(b'\0\0' * 4800)
        for start, duration in ((-1, 1), (0, 0), (0, -1)):
            with self.subTest(start=start, duration=duration):
                self.assertFalse(is_quiet_clip(self.media, start, duration, -40))

    def test_aifc_decoded_byte_order_for_signed_integer_widths(self):
        for width in (1, 2, 3, 4):
            for compression in (b'NONE', b'sowt'):
                with self.subTest(width=width, compression=compression):
                    scale = 1 << (width * 8 - 1)
                    values = (0, scale // 2, -scale // 2, -scale, scale - 1)
                    self.write_pcm(b''.join(x.to_bytes(width, 'big', signed=True) for x in values), width=width, kind='aiff', compression=compression)
                    # CPython 3.10 aifc rejects sowt before reading PCM.
                    if compression == b'sowt' and sys.version_info < (3, 11):
                        with self.assertRaisesRegex(ValueError, 'unsupported compression type'):
                            read_media_segment_pcm16_mono(self.media, Fraction(0), Fraction(5, 48000))
                        continue
                    if compression == b'sowt' and width != 2:
                        with self.assertRaises(ValueError):
                            read_media_segment_pcm16_mono(self.media, Fraction(0), Fraction(5, 48000))
                        continue
                    raw, _ = read_media_segment_pcm16_mono(self.media, Fraction(0), Fraction(5, 48000))
                    expected = tuple(x * 256 if width == 1 else x >> (8 * (width - 2)) for x in values)
                    self.assertEqual(struct.unpack('<hhhhh', raw), expected)

    def test_aifc_sowt_loud_clip_is_retained_by_both_backends(self):
        self.write_pcm(struct.pack('>h', 16384) * 4800, kind='aiff', compression=b'sowt')
        data = self.media.read_bytes()
        self.assertEqual(struct.unpack('<h', data[data.index(b'SSND') + 16:][:2])[0], 16384)
        self.make_aaf(4800, kind='aiff')
        self.assert_backends(0)

    def test_declared_long_window_includes_audible_content(self):
        for rate, length, seconds in ((8000, 3000, 125), (48000, 18000, 750)):
            with self.subTest(rate=rate):
                self.write_pcm(b'\0\0' * rate + struct.pack('<h', 16000) * rate + b'\0\0' * (rate * (seconds - 2)), rate=rate)
                self.make_aaf(length, edit_rate=25, media_rate=rate)
                self.assert_backends(0)
                with aaf2.open(str(self.aaf), 'r') as aaf:
                    node = next(aaf.content.compositionmobs()).slots[0].segment
                    self.assertEqual(sourceclip_audio_timing(aaf, 25, node, self.media),
                                     (Fraction(0), Fraction(length, 25)))

    def test_declared_frame_window_silence_is_removed(self):
        self.write_pcm(b'\0\0' * (48000 * 2))
        self.make_aaf(25, edit_rate=25)
        self.assert_backends(1)

    def test_sample_unit_fallback_requires_independent_unit_evidence(self):
        self.write_pcm(b'\0\0' * 48000)
        self.make_aaf(48000, edit_rate=25)
        self.assert_backends(0, diagnostic='unresolved')

    def test_unresolvable_low_clock_window_is_explicitly_retained(self):
        self.write_pcm(b'\0\0' * 4800)
        self.make_aaf(48000, edit_rate=25)
        self.assert_backends(0, diagnostic='unresolved')
        with self.assertRaises(UnsupportedMediaMapping):
            choose_timebase_rate_for_sourceclip(25, length_units=48000, start_units=0, wav_path=self.media)

    def test_exact_fractional_edit_rate_is_preserved(self):
        self.write_pcm(b'\0\0' * 48048)
        self.make_aaf(48000, edit_rate='48000000/1001')
        with aaf2.open(str(self.aaf), 'r') as aaf:
            slot = next(aaf.content.compositionmobs()).slots[0]
            start, duration = sourceclip_audio_timing(aaf, slot.edit_rate, slot.segment, self.media)
            self.assertIsInstance(duration, Fraction)
            self.assertEqual(duration, Fraction(1001, 1000))
        self.assert_backends(1)

    def test_aiff_boundary_impulse_is_retained(self):
        self.write_pcm(b'\0\0' * 3002 + struct.pack('>h', 16000), kind='aiff')
        self.make_aaf(3003, kind='aiff')
        self.assert_backends(0)

    def test_aiff_truncated_payload_is_retained(self):
        self.write_pcm(b'\0\0' * 48000, kind='aiff')
        self.make_aaf(48000, kind='aiff')
        data = self.media.read_bytes()
        self.media.write_bytes(data[:data.index(b'SSND') + 16 + 9600])
        self.assert_backends(0, diagnostic='incomplete')

    def test_unrelated_loud_sample_interpretation_does_not_override_declared_window(self):
        self.write_pcm(b'\0\0' * 1000 + struct.pack('<h', 16000) * 3000 + b'\0\0' * (8000 * 160 - 4000), rate=8000)
        self.make_aaf(3000, start=1000, edit_rate=25)
        self.assert_backends(1)

    def test_lane_classification_uses_valid_declared_frame_clock(self):
        from unittest.mock import patch
        from aaf_speech_filter.aaf_yamnet_lane_layout import _classify_timeline_block_with_scores
        from aaf_speech_filter.speech_yamnet import YamnetConfig
        self.write_pcm(b'\0\0' * 48000)
        self.make_aaf(25, edit_rate=25)
        logs = []
        with aaf2.open(str(self.aaf), 'r') as aaf:
            node = next(aaf.content.compositionmobs()).slots[0].segment
            with patch('aaf_speech_filter.aaf_yamnet_lane_layout.yamnet_clip_kind_with_scores', return_value=('noise', 0.0, 0.0, 1.0)) as model:
                result = _classify_timeline_block_with_scores(aaf, node, None, YamnetConfig(), 25, lambda: None, log_callback=logs.append)
            self.assertEqual(model.call_args.args[1:3], (Fraction(0), Fraction(1)))
        self.assertEqual(result, ('noise', 0.0, 0.0, 1.0))
        self.assertEqual(logs, [])

    def test_invalid_owning_rate_is_not_defaulted_for_analysis(self):
        self.write_pcm(b'\0\0' * 4800)
        self.make_aaf(4800)
        with aaf2.open(str(self.aaf), 'r') as aaf:
            node = next(aaf.content.compositionmobs()).slots[0].segment
            with self.assertRaises(UnsupportedMediaMapping):
                sourceclip_audio_timing(aaf, 0, node, self.media)

    def test_float_half_frame_boundaries_use_ties_to_even(self):
        self.write_pcm(struct.pack('<hhhh', 1000, 2000, 3000, 4000), rate=8000)
        raw, _ = read_media_segment_pcm16_mono(self.media, 0.5 / 8000, 2.0 / 8000)
        self.assertEqual(raw, struct.pack('<hh', 1000, 2000))

    def test_unsupported_sowt_widths_are_retained_in_both_backends(self):
        for width in (1, 3, 4):
            with self.subTest(width=width):
                self.write_pcm(bytes(width * 4800), width=width, kind='aiff', compression=b'sowt')
                self.make_aaf(4800, kind='aiff')
                diagnostic = 'unsupported compression type' if sys.version_info < (3, 11) else '16-bit'
                self.assert_backends(0, diagnostic=diagnostic)

    def test_missing_media_is_explicitly_retained(self):
        self.write_pcm(b'\0\0' * 4800)
        self.make_aaf(4800)
        self.media.unlink()
        self.assert_backends(0)

    def test_partial_channel_frame_is_not_quiet_evidence(self):
        with wave.open(str(self.media), 'wb') as stream:
            stream.setparams((2, 2, 48000, 0, 'NONE', 'not compressed'))
            stream.writeframes(bytes(4800 * 4))
        self.make_aaf(4800)
        self.media.write_bytes(self.media.read_bytes()[:-1])
        self.assert_backends(0, diagnostic='incomplete')

    def test_insufficient_fps_media_does_not_prove_sample_units(self):
        self.write_pcm(b'\0\0' * 25 + struct.pack('<h', 16000) * (4800 - 25))
        self.make_aaf(25, edit_rate=25)
        self.assert_backends(0, diagnostic='unresolved')

    def test_lane_classification_skips_unproven_sample_unit_fallback(self):
        from unittest.mock import patch
        from aaf_speech_filter.aaf_yamnet_lane_layout import _classify_timeline_block_with_scores
        from aaf_speech_filter.speech_yamnet import YamnetConfig
        self.write_pcm(b'\0\0' * 25 + struct.pack('<h', 16000) * (4800 - 25))
        self.make_aaf(25, edit_rate=25)
        logs = []
        with aaf2.open(str(self.aaf), 'r') as aaf:
            node = next(aaf.content.compositionmobs()).slots[0].segment
            with patch('aaf_speech_filter.aaf_yamnet_lane_layout.yamnet_clip_kind_with_scores', return_value=('noise', 0.0, 0.0, 1.0)) as model:
                result = _classify_timeline_block_with_scores(aaf, node, None, YamnetConfig(), 25, lambda: None, log_callback=logs.append)
            self.assertEqual(result, ('unknown', 0.0, 0.0, 0.0))
            model.assert_not_called()
        self.assertTrue(any('unresolved' in message for message in logs))

    def test_stored_zero_edit_rate_is_retained_by_both_backends(self):
        self.write_pcm(b'\0\0' * 25 + struct.pack('<h', 16000) * (4800 - 25))
        self.make_aaf(25, edit_rate=25)
        with aaf2.open(str(self.aaf), 'r+') as aaf:
            next(aaf.content.compositionmobs()).slots[0].edit_rate = 0
        with aaf2.open(str(self.aaf), 'r') as aaf:
            self.assertEqual(next(aaf.content.compositionmobs()).slots[0].edit_rate, 0)
        self.assert_backends(0, diagnostic='unresolved')

    def test_missing_edit_rate_is_not_invented_during_traversal(self):
        from aaf_speech_filter.timeline_sourceclips import iter_slot_sourceclip_occurrences
        self.write_pcm(b'\0\0' * 4800)
        self.make_aaf(25, edit_rate=25)
        with aaf2.open(str(self.aaf), 'r+') as aaf:
            comp = next(aaf.content.compositionmobs())
            slot = comp.slots[0]
            original_rate = slot.edit_rate
            del slot['EditRate']
            try:
                occurrence = next(iter_slot_sourceclip_occurrences(comp, slot))
                self.assertIsNone(occurrence.edit_rate)
                with self.assertRaises(UnsupportedMediaMapping):
                    sourceclip_audio_timing(aaf, occurrence.edit_rate, occurrence.sourceclip, self.media)
            finally:
                slot.edit_rate = original_rate

    def test_silent_sowt_requires_decoder_support_before_removal(self):
        self.write_pcm(b'\0\0' * 4800, kind='aiff', compression=b'sowt')
        self.make_aaf(4800, kind='aiff')
        if sys.version_info < (3, 11):
            self.assert_backends(0, diagnostic='unsupported compression type')
        else:
            self.assert_backends(1)

if __name__ == '__main__':
    unittest.main()
