"""Control-flow errors shared by I/O and processing layers."""

class OperationCancelled(Exception):
    """The caller cancelled an operation; never a compatibility fallback."""
