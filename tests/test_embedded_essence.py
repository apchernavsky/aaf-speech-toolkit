"""Real PyAAF2 stream extraction, independent of host/file names."""
import hashlib
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import aaf2
from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
from aaf_io.converter import AAFConverter
from aaf_speech_filter.media_resolve import _resolve_wave_path_for_sourceclip
from aaf_speech_filter.timeline_sourceclips import iter_timeline_sourceclips

ROOT = Path(__file__).resolve().parents[1]


class EmbeddedEssenceTests(unittest.TestCase):
    def create_embedded(self, folder, *, container, frames):
        wav = folder / 'audio.wav'
        pcm = bytes((i * 17) % 256 for i in range(frames * 3))
        with wave.open(str(wav), 'wb') as stream:
            stream.setparams((1, 3, 48000, 0, 'NONE', 'not compressed'))
            stream.writeframes(pcm)
        payload = wav.read_bytes() if container else pcm
        path = folder / 'input.aaf'
        with aaf2.open(str(path), 'w') as aaf:
            source = aaf.create.SourceMob()
            aaf.content.mobs.append(source)
            source.import_audio_essence(str(wav))
            if container:
                descriptor = aaf.create.WAVEDescriptor()
                descriptor['SampleRate'].value = 48000
                descriptor['Length'].value = frames
                descriptor['Summary'].value = list(payload[:44])
                source.descriptor = descriptor
                essence = aaf.content.essencedata[source.mob_id]
                essence.open('w').write(payload)
            comp = aaf.create.CompositionMob()
            aaf.content.mobs.append(comp)
            slot = comp.create_timeline_slot(48000)
            slot.segment = source.create_source_clip(1, start=0, length=frames, media_kind='sound')
        return path, payload, pcm

    def test_extracts_real_pcm_and_wave_streams_without_modifying_source(self):
        for container in (False, True):
            for frames in (8, 4096):
                with self.subTest(container=container, frames=frames), tempfile.TemporaryDirectory(dir=ROOT) as td:
                    folder = Path(td)
                    path, payload, pcm = self.create_embedded(folder, container=container, frames=frames)
                    before = hashlib.sha256(path.read_bytes()).digest()
                    converter = AAFConverter(path, work_dir=folder / 'work')
                    with open_aaf_lenient(path, 'r') as aaf:
                        essence = next(iter(aaf.content.essencedata))
                        self.assertEqual(converter._get_essence_bytes(essence, aaf), payload)
                        converter._extract_all(aaf)
                        mapping = converter.runtime_essence_paths_for_filter()
                        self.assertEqual(len(mapping or {}), 1)
                        _, _, clip = next(iter_timeline_sourceclips(aaf))
                        resolved = _resolve_wave_path_for_sourceclip(aaf, clip, mapping, media_search_roots=())
                        with wave.open(str(resolved), 'rb') as decoded:
                            self.assertEqual(decoded.getparams()[:3], (1, 3, 48000))
                            self.assertEqual(decoded.readframes(frames + 1), pcm)
                    self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), before)

    def test_stream_failure_propagates_instead_of_becoming_missing_media(self):
        essence = mock.Mock()
        essence.open.side_effect = OSError('embedded stream read failed')
        with self.assertRaisesRegex(OSError, 'embedded stream read failed'):
            AAFConverter('input.aaf')._get_essence_bytes(essence, None)


if __name__ == '__main__':
    unittest.main()
