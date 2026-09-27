"""Process every local AAF on owned copies, with bounded isolated workers."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_aaf_corpus import run_bounded, sha256, snapshot, assert_cleanup_invariants, fixture_media_roots



def source_fingerprint():
    files = [ROOT / name for name in ("aaf_pipeline.py", "aaf_gui.py", "aaf_workflow.py", "aaf_tool_config.py")]
    for package in (ROOT / "aaf_io/aaf_io", ROOT / "aaf_speech_filter/aaf_speech_filter"):
        files.extend(package.rglob("*.py"))
    files.extend(ROOT / "scripts" / name for name in (
        "verify_local_aafs.py", "check_aaf_corpus.py", "check_aaf_horizontal_positions.py", "check_aaf_sequence_lengths.py", "aaf_removal_oracle.py"))
    digests = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in sorted(set(files))}
    digest = hashlib.sha256(json.dumps(digests, sort_keys=True).encode("utf8")).hexdigest()
    return dict(sha256=digest, files=digests)


@dataclass
class _ClassifiedPlanCapture:
    composition_id: str | None = None
    events: tuple | None = None

    def plan(self):
        from scripts.check_aaf_horizontal_positions import ClassifiedLayoutPlan
        if self.events is None:
            return None
        if self.composition_id is None:
            raise RuntimeError("Classified plan has no selected composition identity")
        return ClassifiedLayoutPlan(self.composition_id, self.events)


@contextmanager
def capture_classified_layout():
    """Diagnostic-only observation of this run's actual selected classified plan."""
    from unittest.mock import patch
    from aaf_speech_filter import aaf_yamnet_lane_layout as layout
    from scripts.check_aaf_horizontal_positions import ClassifiedOccurrence
    capture = _ClassifiedPlanCapture()
    select = layout._pick_main_composition
    assign = layout._assign_target_lanes

    def selected(compositions):
        composition = select(compositions)
        identity = str(composition.mob_id)
        if capture.composition_id not in (None, identity):
            raise RuntimeError("Run selected multiple composition identities")
        capture.composition_id = identity
        return composition

    def assigned(events, *args, **kwargs):
        result = assign(events, *args, **kwargs)
        capture.events = tuple(ClassifiedOccurrence.from_event(event) for event in events)
        return result

    with patch.object(layout, "_pick_main_composition", selected), patch.object(layout, "_assign_target_lanes", assigned):
        yield capture


class DiagnosticProgress:
    def __init__(self):
        self.stage = "starting"
        self.fraction = 0.0
        self.last_fraction = -1.0
        self.last_emit = 0.0

    def _emit(self, force=False):
        now = time.monotonic()
        if force or now - self.last_emit >= 15 or self.fraction - self.last_fraction >= .05:
            print(f"PROGRESS {self.fraction:.1%}: {self.stage}", flush=True)
            self.last_emit, self.last_fraction = now, self.fraction

    def set_stage(self, stage):
        changed = str(stage) != self.stage
        self.stage = str(stage)
        self._emit(force=changed)

    def set_global_fraction(self, fraction):
        self.fraction = float(fraction)
        self._emit()

    def set_indeterminate(self, value):
        self._emit()


def worker(source, work, report):
    import aaf_pipeline
    from aaf_tool_config import resolve_media_search_roots
    from aaf_io.sdk_tools import find_comaafinfo, run_comaafinfo
    from scripts.check_aaf_horizontal_positions import collect_clip_positions, compare_collected_positions
    from scripts.check_aaf_sequence_lengths import compare_sequence_lengths
    started = time.monotonic()
    result = dict(source=str(source), passed=False)
    code_manifest = source_fingerprint()
    result["code_sha256"] = code_manifest["sha256"]
    (report / "source-manifest.json").write_text(json.dumps(code_manifest, indent=2), encoding="utf8")
    original = sha256(source)
    try:
        copied = work / 'input.aaf'
        shutil.copyfile(source, copied)
        before = snapshot(copied)
        media_roots = resolve_media_search_roots(source)
        with fixture_media_roots(media_roots), capture_classified_layout() as classified:
            outcome = aaf_pipeline.run_aaf_pipeline(copied, remove_quiet_clips=True,
                remove_duplicates=True, experimental_yamnet_lane_layout=True,
                allow_yamnet_download=False, aaf_tools_dir=ROOT / 'sdk_bin', pipeline_progress=DiagnosticProgress(),
                log_callback=lambda message: print(message, flush=True))
        if outcome[2] is not None:
            raise AssertionError(str(outcome[2]))
        output = Path(outcome[1])
        if output.resolve() == copied.resolve() or not output.is_file():
            raise AssertionError('No separate processed output')
        after = snapshot(output)
        from scripts.aaf_removal_oracle import validate_cleanup
        proof = validate_cleanup(copied, output, work_dir=work/'oracle',
                                 media_roots=media_roots)
        (report/'removal-oracle.json').write_text(json.dumps(proof, indent=2), encoding='utf8')
        assert_cleanup_invariants(before, after, allowed_removals=Counter(before['clips'])-Counter(after['clips']))
        if sha256(copied) != original:
            raise AssertionError('Pipeline changed its input copy')
        classified_plan = classified.plan()
        if classified_plan is not None:
            (report / "classified-plan.json").write_text(json.dumps(asdict(classified_plan), indent=2), encoding="utf8")
        selection = {} if classified_plan is None else {"composition_id": classified_plan.composition_id}
        source_positions = collect_clip_positions(copied, **selection)
        output_positions = collect_clip_positions(output, **selection)
        for label, records in (("source", source_positions), ("output", output_positions)):
            (report / f"positions-{label}.json").write_text(
                json.dumps([asdict(item) for item in records], indent=2), encoding="utf8")
        position_code, positions = compare_collected_positions(
            source_positions, output_positions, max_report=10, classified_plan=classified_plan,
            src_label=copied, dst_label=output,
        )
        length_code, lengths = compare_sequence_lengths(output, max_report=10)
        sdk_code, sdk_message = run_comaafinfo(find_comaafinfo(ROOT / 'sdk_bin'), output, timeout_sec=120)
        result.update(input_clips=len(before['clips']), output_clips=len(after['clips']),
            removed=outcome[3], essence_streams=len(before['essence']),
            positions=positions, sequence_lengths=lengths, sdk_exit=sdk_code, sdk_message=sdk_message[:2000])
        if position_code or length_code or sdk_code:
            raise AssertionError(f'Output validation failed: positions={position_code}, lengths={length_code}, SDK={sdk_code}')
        shared = work / '__aaf_tool_work'
        if shared.exists() and any(shared.iterdir()):
            raise AssertionError('Owned pipeline work children remain')
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    finally:
        result['source_sha256'] = original
        result['original_unchanged'] = sha256(source) == original
        result['source_code_changed_during_run'] = source_fingerprint()["sha256"] != result["code_sha256"]
        result['passed'] = result['passed'] and result['original_unchanged'] and not result['source_code_changed_during_run']
        result['seconds'] = round(time.monotonic() - started, 3)
        (report / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    return int(not result['passed'])


def run_case(source, index, report, timeout):
    case_dir = report / f'case-{index:03}'
    case_dir.mkdir()
    started = time.monotonic()
    print(f'START {index}: {source}', flush=True)
    try:
        # Keep native SDK paths short independently of the report directory depth.
        work_parent = ROOT / '__manual_checks'
        work_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='aaf-', dir=work_parent) as owned:
            command = [sys.executable, str(Path(__file__).resolve()), str(source), '--worker',
                '--work-dir', owned, '--report-dir', str(case_dir)]
            environment = dict(os.environ, PYTHONIOENCODING='utf8', PYTHONUNBUFFERED='1')
            with (case_dir / 'run.log').open('w', encoding='utf8') as output:
                code = run_bounded(command, stdout=output, env=environment, timeout=timeout)
            result = json.loads((case_dir / 'result.json').read_text(encoding='utf8'))
            if code:
                result['passed'] = False
    except Exception:
        result = dict(source=str(source), passed=False, error=traceback.format_exc(), seconds=time.monotonic()-started)
        (case_dir / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(f"{'PASS' if result['passed'] else 'FAIL'} {index}: {source.name} ({result['seconds']:.1f}s)", flush=True)
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=5400)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--work-dir', type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.root, args.work_dir, args.report_dir)
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    sources = [args.root.resolve()] if args.root.is_file() else sorted(p.resolve() for p in args.root.rglob('*') if p.is_file() and p.suffix.lower() == '.aaf' and '__aaf_tool_work' not in p.parts)
    if not sources:
        parser.error('No AAF files found')
    args.report_dir.mkdir(parents=True, exist_ok=False)
    (args.report_dir / 'source-manifest.json').write_text(json.dumps(source_fingerprint(), indent=2), encoding='utf8')
    (args.report_dir / 'manifest.json').write_text(json.dumps([dict(path=str(p), bytes=p.stat().st_size) for p in sources], indent=2, ensure_ascii=False), encoding='utf8')
    indexed = list(enumerate(sources, 1))
    results = []
    def record(result):
        results.append(result)
        (args.report_dir / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf8')
    # Large embedded essence can occupy several gigabytes while decoding.
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run_case, p, i, args.report_dir, args.timeout) for i, p in indexed if p.stat().st_size < 2 * 1024**3]
        for future in as_completed(futures):
            record(future.result())
    for index, source in indexed:
        if source.stat().st_size >= 2 * 1024**3:
            record(run_case(source, index, args.report_dir, args.timeout))
    print(f"Completed: {sum(r['passed'] for r in results)}/{len(sources)}", flush=True)
    return int(any(not r['passed'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
