"""Private output candidates with one atomic publication boundary."""
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile

from aaf_io.path_safety import require_distinct_files


@contextmanager
def output_candidate(source, destination, *, cancel_check=None):
    """Publish a closed, validated sibling file; errors preserve the destination.

    The caller must close every writer and validator before leaving this context.
    No directory cleanup or other fallible work follows the atomic replacement.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    require_distinct_files(source, destination)
    if cancel_check is not None:
        cancel_check()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, owned = tempfile.mkstemp(prefix='.aaf-candidate-', suffix='.aaf', dir=destination.parent)
    candidate = Path(owned)
    published = False
    try:
        os.close(descriptor)
        yield candidate
        if cancel_check is not None:
            cancel_check()
        require_distinct_files(source, destination)
        candidate.replace(destination)
        published = True
    finally:
        if not published:
            candidate.unlink(missing_ok=True)
