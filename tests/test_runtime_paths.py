from __future__ import annotations

import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import aaf2
import aaf_tool_config
from aaf_io.heal.premiere import resolve_import_sidecar_media_path
from aaf_speech_filter.config import FilterConfig
from aaf_speech_filter.sdk_removals import collect_timeline_sourceclip_removals_for_sdk_xml
from tests.test_audit_audio import add_source, add_comp, clip, write_wave

ROOT = Path(__file__).resolve().parents[1]


class RuntimePathsTests(unittest.TestCase):
    def test_explicit_roots_isolate_same_basename_and_empty_disables_environment(self):
        self.assertIn('media_search_roots', inspect.signature(resolve_import_sidecar_media_path).parameters)
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            folder = Path(td)
            a, b = folder / 'a', folder / 'b'
            a.mkdir(); b.mkdir()
            for item in (a, b):
                (item / 'audio.wav').write_bytes(b'audio')
            missing = folder / 'missing' / 'audio.wav'
            with patch.dict(os.environ, {'AAF_MEDIA_ROOTS': str(b)}):
                for roots, expected in (((a,), a / 'audio.wav'), ((b,), b / 'audio.wav'), ((), None), ((a,), a / 'audio.wav')):
                    self.assertEqual(resolve_import_sidecar_media_path(missing, media_search_roots=roots), expected)
                self.assertEqual(resolve_import_sidecar_media_path(missing), b / 'audio.wav')
                self.assertEqual(os.environ['AAF_MEDIA_ROOTS'], str(b))

    def test_config_copies_mutable_root_input(self):
        roots = [Path('first')]
        cfg = FilterConfig(media_search_roots=roots)
        roots.append(Path('second'))
        self.assertEqual(cfg.media_search_roots, (Path('first'),))

    def test_configuration_resolution_is_pure_snapshot(self):
        self.assertTrue(hasattr(aaf_tool_config, 'resolve_media_search_roots'))
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            folder = Path(td)
            a, b = folder / 'a', folder / 'b'
            a.mkdir(); b.mkdir()
            with patch.dict(os.environ, {'AAF_MEDIA_ROOTS': str(b)}), patch.object(aaf_tool_config, 'load_media_search_roots_from_config', return_value=[a]):
                before = dict(os.environ)
                roots = aaf_tool_config.resolve_media_search_roots(folder / 'input.aaf')
                self.assertIsInstance(roots, tuple)
                self.assertEqual(roots[0], a)
                self.assertIn(b, roots)
                self.assertEqual(dict(os.environ), before)

    def test_pipeline_passes_snapshot_to_lane_path_without_environment_writes(self):
        import aaf_pipeline
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            folder = Path(td)
            source = folder / 'input.aaf'
            source.write_bytes(b'input')
            tools = folder / 'sdk'
            tools.mkdir()
            (tools / 'aaffmtconv.exe').write_bytes(b'tool')
            with patch.dict(os.environ, {'AAF_MEDIA_ROOTS': 'sentinel-roots', 'AAF_TOOLS': 'sentinel-sdk'}):
                before = dict(os.environ)
                with patch.object(aaf_pipeline, 'effective_aaf_tools_dir', return_value=tools), patch.object(aaf_pipeline, 'emit_comaafinfo_header'), patch.object(aaf_pipeline, 'extract_embedded_essence_paths_for_analysis', return_value={}), patch.object(aaf_pipeline, 'maybe_run_lane_layout_only_fastpath') as lane, patch('aaf_speech_filter.speech_yamnet.yamnet_model_available_locally', return_value=True):
                    result = aaf_pipeline._execute_aaf_pipeline(source, remove_quiet_clips=False, experimental_yamnet_lane_layout=True, work_dir=folder, processed_path=folder / 'out.aaf', log_callback=lambda _: None)
                self.assertIsNone(result[2])
                self.assertEqual(dict(os.environ), before)
                cfg = lane.call_args.kwargs['cfg']
                self.assertIn(folder, cfg.media_search_roots)
                self.assertEqual(cfg.aaf_tools_dir, tools)

    def test_premiere_scan_and_heal_use_same_explicit_roots(self):
        from aaf_io.heal.premiere import mend_premiere_import_audio_for_sound_pipeline
        from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            folder = Path(td)
            media = folder / 'media'
            media.mkdir()
            write_wave(media / 'audio.wav', b'\0\0' * 9600)
            source_aaf = folder / 'input.aaf'
            with aaf2.open(str(source_aaf), 'w') as aaf:
                source = aaf.create.SourceMob()
                aaf.content.mobs.append(source)
                source.descriptor = aaf.create.ImportDescriptor()
                locator = aaf.create.NetworkLocator()
                locator['URLString'].value = (folder / 'missing' / 'audio.wav').as_uri()
                source.descriptor['Locator'].append(locator)
            with patch.dict(os.environ, {'AAF_MEDIA_ROOTS': ''}):
                result = mend_premiere_import_audio_for_sound_pipeline(source_aaf, media_search_roots=(media,))
            self.assertEqual(result.scan.healable_pcm_jobs, 1)
            self.assertEqual(result.heal.pcm_replaced, 1)
            with open_aaf_lenient(source_aaf, 'r') as aaf:
                source = next(iter(aaf.content.sourcemobs()))
                self.assertEqual(source.descriptor.length, 9600)

    def test_sdk_writer_fallback_receives_run_tool_directory(self):
        from unittest.mock import Mock
        from aaf_speech_filter.lane_layout_writer import write_lane_layout_with_sdk_or_pyaaf2_fallback as write
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            folder = Path(td)
            source = folder / 'input.aaf'
            source.write_bytes(b'input')
            cfg = FilterConfig(aaf_tools_dir=folder / 'sdk')
            sdk = Mock()
            write(input_aaf=source, output_aaf=folder / 'out.aaf', events=[], structural_events=[], n_lanes=0, cfg=cfg, runtime_essence_paths={}, work_dir=folder, cancel_check=lambda: None, log_callback=lambda _: None, result_out={}, build_sdk_events=lambda *_: [], class_zone_lane_order=lambda *_: [], pyaaf2_rebuild=Mock(side_effect=RuntimeError('force SDK')), sdk_xml_writer=sdk, sdk_exporter_unavailable_error=lambda _: False)
            self.assertEqual(sdk.call_args.kwargs['aaf_tools_dir'], cfg.aaf_tools_dir)

    def test_sdk_validator_uses_explicit_tools_for_both_programs(self):
        from aaf_speech_filter.aaf_yamnet_lane_layout import _validate_pyaaf2_lane_layout_output_for_sdk_open as validate
        self.assertIn('aaf_tools_dir', inspect.signature(validate).parameters)
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            folder = Path(td)
            tools = folder / 'sdk'
            with patch('aaf_io.sdk_tools.find_comaafinfo', return_value=tools / 'ComAAFInfo.exe') as info, patch('aaf_io.sdk_tools.run_comaafinfo', return_value=(0, '')), patch('aaf_io.sdk_tools.find_aaffmtconv', return_value=tools / 'aaffmtconv.exe') as conv, patch('aaf_io.sdk_tools.run_aaffmtconv_to_structured_storage'):
                validate(folder / 'input.aaf', work_dir=folder, aaf_tools_dir=tools)
            info.assert_called_once_with(tools)
            conv.assert_called_once_with(tools)

    def test_quiet_analysis_uses_configured_roots_through_source_resolution(self):
        self.assertIn('media_search_roots', FilterConfig.__dataclass_fields__)
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            folder = Path(td)
            silent, loud = folder / 'silent', folder / 'loud'
            silent.mkdir(); loud.mkdir()
            wave_path = folder / 'audio.wav'
            write_wave(wave_path, b'\0\0' * 4800)
            write_wave(silent / 'audio.wav', b'\0\0' * 4800)
            write_wave(loud / 'audio.wav', b'\xff\x7f' * 4800)
            aaf_path = folder / 'input.aaf'
            with aaf2.open(str(aaf_path), 'w') as aaf:
                source = add_source(aaf, wave_path)
                comp = add_comp(aaf)
                slot = comp.create_timeline_slot(48000)
                slot.segment = clip(source)
            wave_path.unlink()
            with patch.dict(os.environ, {'AAF_MEDIA_ROOTS': str(loud)}):
                silent_cfg = FilterConfig(media_search_roots=(silent,))
                loud_cfg = FilterConfig(media_search_roots=(loud,))
                for cfg, expected in ((silent_cfg, 1), (loud_cfg, 0), (silent_cfg, 1)):
                    self.assertEqual(len(collect_timeline_sourceclip_removals_for_sdk_xml(aaf_path, cfg)), expected)
                self.assertEqual(os.environ['AAF_MEDIA_ROOTS'], str(loud))


if __name__ == '__main__':
    unittest.main()
