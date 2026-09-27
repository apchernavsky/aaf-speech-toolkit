"""Fetch pinned public regression fixtures, or verify an offline cache."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.public_aaf_corpus import DEFAULT_CACHE, MANIFEST, fetch_asset, load_manifest, verify_asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 16:
        parser.error('--jobs must be between 1 and 16')
    manifest = load_manifest(args.manifest)
    operation = verify_asset if args.verify_only else fetch_asset
    def execute(asset):
        try:
            operation(asset, args.cache)
            return asset['path'], None
        except Exception as exc:
            # Batch boundary: retain every independent failure and exit nonzero.
            return asset['path'], str(exc)
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for index, (path, error) in enumerate(pool.map(execute, manifest['assets']), 1):
            if error:
                failures.append(path)
                print('FAIL', path, error, flush=True)
            elif index % 25 == 0:
                print('VERIFIED', index, '/', len(manifest['assets']), flush=True)
    print(f"Assets: {len(manifest['assets']) - len(failures)}/{len(manifest['assets'])}; unique AAF cases: {len(manifest['cases'])}")
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
