"""Run all processing modes on isolated copies, retaining evidence and outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Use the application's own package bootstrap, as a normal source launch does.
sys.path.insert(0, str(ROOT))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect_aaf(path):
    from aaf_io.compat.pyaaf2_lenient import open_aaf_lenient
    from aaf_speech_filter.timeline_sourceclips import total_timeline_sourceclips
    with open_aaf_lenient(path, 'r') as aaf:
        return dict(mobs=len(aaf.content.mobs), essence=len(aaf.content.essencedata),
                    sound_clips=total_timeline_sourceclips(aaf),
                    compositions=len(list(aaf.content.compositionmobs())))


def worker(source, case):
    from aaf_pipeline import run_aaf_pipeline
    from aaf_tool_config import resolve_media_search_roots
    # This worker process owns its environment. Preserve original sidecar roots
    # when the AAF itself is copied into an isolated input directory.
    os.environ['AAF_MEDIA_ROOTS'] = ';'.join(map(str, resolve_media_search_roots(source)))
    report = dict(source=str(source), bytes=source.stat().st_size, passed=False)
    started = time.monotonic()
    before = digest(source)
    report['source_sha256'] = before
    try:
        with tempfile.TemporaryDirectory(prefix='input-', dir=case) as directory:
            copied = Path(directory) / source.name
            shutil.copyfile(source, copied)
            if digest(copied) != before:
                raise RuntimeError('Input copy does not match the original snapshot')
            report['input'] = inspect_aaf(copied)
            print('INPUT_READ', report['input'], flush=True)
            result = run_aaf_pipeline(copied, remove_quiet_clips=True, remove_duplicates=True,
                experimental_yamnet_lane_layout=True, allow_yamnet_download=False,
                aaf_tools_dir=ROOT / 'sdk_bin', log_callback=lambda msg: print(msg, flush=True))
            if result[2] is not None:
                raise RuntimeError(result[2])
            output = Path(result[1])
            report['output'] = inspect_aaf(output)
            report['removed'] = result[3]
            report['layout'] = result[4]
            report['output_bytes'] = output.stat().st_size
            if digest(copied) != before:
                raise RuntimeError('Pipeline changed its input copy')
            destination = case / 'processed.aaf'
            output.replace(destination)
            report['output_path'] = str(destination)
            report['passed'] = True
    except Exception:
        report['error'] = traceback.format_exc()
        print(report['error'], flush=True)
    finally:
        report['original_unchanged'] = digest(source) == before
        if not report['original_unchanged']:
            report['passed'] = False
        report['seconds'] = round(time.monotonic() - started, 2)
        (case / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if report['passed'] else 1


def main():
    # Redirected Windows consoles may otherwise use a legacy code page.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--only', action='append', default=[], help='Relative input path to recheck')
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args.input.resolve(), args.report_dir.resolve())
    source_root = args.input.resolve()
    jobs = sorted(source_root.rglob('*.aaf'), key=lambda path: str(path).lower())
    if args.only:
        wanted = set(args.only)
        jobs = [path for path in jobs if path.relative_to(source_root).as_posix() in wanted]
    records = []
    (args.report_dir / 'manifest.json').write_text(json.dumps([str(path) for path in jobs], ensure_ascii=False, indent=2), encoding='utf-8')
    for index, source in enumerate(jobs, 1):
        case = args.report_dir / f'case-{index:03d}'
        case.mkdir(exist_ok=False)
        print(f'START {index}/{len(jobs)} {source.relative_to(source_root)}', flush=True)
        with (case / 'run.log').open('w', encoding='utf-8') as log:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(source),
                '--report-dir', str(case.resolve()), '--worker'], stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'})
        result_file = case / 'result.json'
        if result_file.is_file():
            record = json.loads(result_file.read_text(encoding='utf-8'))
        else:
            record = dict(source=str(source), passed=False, error=f'Worker exited {process.returncode} without a result')
        records.append(record)
        (args.report_dir / 'results.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"{'PASS' if record['passed'] else 'FAIL'} {index}/{len(jobs)} {source.name} {record.get('seconds', '?')}s", flush=True)
    print(f"TOTAL {sum(item['passed'] for item in records)}/{len(records)} passed", flush=True)
    return 0 if all(item['passed'] for item in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
