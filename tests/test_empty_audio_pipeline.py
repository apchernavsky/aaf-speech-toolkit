"""AAF without timeline audio is a valid no-op input."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import aaf_pipeline
import aaf2

ROOT = Path(__file__).resolve().parents[1]


class EmptyAudioPipelineTests(unittest.TestCase):
    def test_valid_empty_and_picture_only_inputs_are_preserved(self):
        for picture in (False, True):
            with self.subTest(picture=picture), tempfile.TemporaryDirectory(dir=ROOT) as tmp:
                source = Path(tmp)/'source.aaf'
                with aaf2.open(str(source), 'w') as aaf:
                    if picture:
                        comp = aaf.create.CompositionMob()
                        aaf.content.mobs.append(comp)
                        slot = comp.create_timeline_slot(edit_rate=25)
                        slot.segment = aaf.create.Filler(media_kind='picture', length=50)
                before = source.read_bytes()
                with patch('aaf_pipeline.sdk_export_xml', side_effect=AssertionError('No audio to convert')):
                    result = aaf_pipeline.run_aaf_pipeline(source, remove_quiet_clips=True,
                        remove_duplicates=True, allow_yamnet_download=False, log_callback=lambda m: None)
                self.assertIsNone(result[2], result[2])
                self.assertEqual(result[3], 0)
                self.assertNotEqual(Path(result[1]), source)
                self.assertEqual(Path(result[1]).read_bytes(), before)
                self.assertEqual(source.read_bytes(), before)

    def test_no_audio_completes_with_gui_progress_object(self):
        from types import SimpleNamespace
        fractions, indeterminate = [], []
        progress = SimpleNamespace(set_global_fraction=fractions.append,
            set_stage=lambda label: None, set_indeterminate=indeterminate.append)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            source = Path(tmp)/'source.aaf'
            with aaf2.open(str(source), 'w'):
                pass
            result = aaf_pipeline.run_aaf_pipeline(source, remove_duplicates=True,
                pipeline_progress=progress, allow_yamnet_download=False, log_callback=lambda m: None)
            self.assertIsNone(result[2], result[2])
            self.assertEqual(fractions[-1], 1.0)
            self.assertFalse(indeterminate[-1])

    def test_invalid_container_is_not_accepted_as_empty(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            source = Path(tmp)/'source.aaf'; source.write_bytes(b'not an AAF')
            result = aaf_pipeline.run_aaf_pipeline(source, allow_yamnet_download=False,
                remove_duplicates=True, log_callback=lambda m: None)
            self.assertIsNotNone(result[2])
            self.assertFalse(source.with_name('source_processed.aaf').exists())
