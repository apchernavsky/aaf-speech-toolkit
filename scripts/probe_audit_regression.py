"""Exercise SDK occurrence edits and the pipeline on an owned copy of a supplied AAF."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
from aaf_io.sdk_tools import find_aaffmtconv, run_aaffmtconv_to_xml, run_aaffmtconv_to_aaf
from aaf_io.sdk_xml import apply_removals_in_composition_xml
from aaf_speech_filter.sdk_removals import sourceclip_occurrence_key
from aaf_speech_filter.timeline_sourceclips import iter_timeline_sourceclip_occurrences
from aaf_pipeline import run_aaf_pipeline

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def occurrences(path):
    with open_aaf_lenient(path, 'r') as aaf:
        return {sourceclip_occurrence_key(item) for item in iter_timeline_sourceclip_occurrences(aaf)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    before = digest(args.input)
    report = {}
    try:
        with tempfile.TemporaryDirectory(prefix='audit_manual_', dir=ROOT) as directory:
            work = Path(directory)
            source = work / 'input.aaf'
            shutil.copyfile(args.input, source)
            expected = occurrences(source)
            if not expected:
                raise RuntimeError('Probe requires a sound SourceClip occurrence.')
            selected = min(expected, key=repr)
            xml = work / 'roundtrip.xml'
            output = work / 'edited.aaf'
            tool = find_aaffmtconv(ROOT / 'sdk_bin')
            run_aaffmtconv_to_xml(tool, source, xml)
            assert apply_removals_in_composition_xml(xml, {selected}) == 1
            run_aaffmtconv_to_aaf(tool, xml, output)
            actual = occurrences(output)
            assert actual == expected - {selected}, (len(expected), len(actual))
            report['sdk_occurrences_before'] = len(expected)
            report['sdk_occurrences_after'] = len(actual)
            result = run_aaf_pipeline(source, remove_quiet_clips=True, remove_duplicates=True,
                experimental_yamnet_lane_layout=False, allow_yamnet_download=False,
                aaf_tools_dir=ROOT / 'sdk_bin', log_callback=lambda _message: None)
            assert result[2] is None, result[2]
            assert result[1].is_file()
            report['pipeline_removed'] = result[3]
            report['pipeline_remaining'] = len(occurrences(result[1]))
            assert digest(source) == before
            assert not (work / '__aaf_tool_work').exists()
    finally:
        assert digest(args.input) == before, 'Original source was changed'
    report['source_sha256_unchanged'] = before
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
