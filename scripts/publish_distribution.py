"""Promote a staged build, retaining the previous distribution for rollback."""
from __future__ import annotations

import argparse
from pathlib import Path
from uuid import uuid4


def publish_distribution(workspace: Path, staging: Path, destination: Path) -> Path | None:
    workspace, staging, destination = map(lambda p: Path(p).resolve(),
                                           (workspace, staging, destination))
    if (staging.parent != workspace or destination.parent != workspace
            or not staging.name.startswith('dist_pyinstaller.stage-')
            or destination.name != 'dist_pyinstaller'):
        raise ValueError('Distribution paths must be owned children of the workspace.')
    if not staging.is_dir():
        raise FileNotFoundError(staging)
    backup = None
    if destination.exists():
        backup = workspace / f'dist_pyinstaller.previous-{uuid4().hex}'
        destination.rename(backup)
    try:
        staging.rename(destination)
    except OSError:
        if backup is not None:
            backup.rename(destination)
        raise
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('staging', type=Path)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    backup = publish_distribution(workspace, args.staging, workspace / 'dist_pyinstaller')
    if backup is not None:
        print(f'Previous distribution retained: {backup}')


if __name__ == '__main__':
    main()
