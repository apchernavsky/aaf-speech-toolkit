"""Check human-reviewed audio labels and output ordering in corpus reports.

This is an external regression harness. Reviewed media identities never enter
production classification or placement. Missing reviewed sources fail explicitly.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def check_case(expectation, classified_events, positions):
    errors = []
    lanes = {}
    for reviewed in expectation['clips']:
        identity = tuple(reviewed[key] for key in ('source_id', 'source_track', 'source_start', 'source_length'))
        def matches(item):
            return tuple(item[key] for key in ('source_id', 'source_track', 'source_start', 'source_length')) == identity
        labels = [event for event in classified_events if matches(event)
                  and event['edit_start'] == reviewed['edit_start']
                  and event['timeline_length'] == reviewed['timeline_length']]
        output = [item for item in positions if matches(item)]
        name = reviewed['id']
        if not labels or any(event['class_kind'] != reviewed['kind'] for event in labels):
            errors.append(f'{name}: reviewed class mismatch or absent classified occurrence')
        if len(output) != 1:
            errors.append(f'{name}: expected one output occurrence, got {len(output)}')
            continue
        item = output[0]
        if item['edit_start'] != reviewed['edit_start'] or item['timeline_length'] != reviewed['timeline_length']:
            errors.append(f'{name}: visible edit position or duration changed')
        lanes[name] = item['lane']
    for above, below in expectation.get('above', []):
        if above not in lanes or below not in lanes or lanes[above] >= lanes[below]:
            errors.append(f'{above} must be above {below}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report_dir', type=Path)
    parser.add_argument('--cases', type=Path, required=True, help='Explicit path to a local reviewed-case manifest.')
    args = parser.parse_args()
    expectations = json.loads(args.cases.read_text(encoding='utf8'))
    reports = {}
    for result_path in args.report_dir.glob('case-*/result.json'):
        result = json.loads(result_path.read_text(encoding='utf8'))
        reports[result.get('source_sha256')] = result_path.parent
    errors = []
    for case in expectations:
        folder = reports.get(case['source_sha256'])
        if folder is None:
            errors.append(case['id'] + ': reviewed input absent from corpus run')
            continue
        plan = json.loads((folder/'classified-plan.json').read_text(encoding='utf8'))
        positions = json.loads((folder/'positions-output.json').read_text(encoding='utf8'))
        errors.extend(case['id'] + ': ' + error for error in check_case(case, plan['events'], positions))
    print(json.dumps(dict(passed=not errors, reviewed_cases=len(expectations), errors=errors), indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
