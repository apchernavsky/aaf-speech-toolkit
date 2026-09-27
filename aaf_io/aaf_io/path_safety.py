"""File identity checks used before any destructive output operation."""
from pathlib import Path


def require_distinct_files(source: Path, destination: Path) -> None:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    same = source == destination
    if not same and source.exists() and destination.exists():
        same = source.samefile(destination)
    if same:
        raise RuntimeError('Output must not overwrite the input/source file.')
