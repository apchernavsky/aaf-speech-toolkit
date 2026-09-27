from aaf_io.errors import OperationCancelled


class SpeechFilterCancelled(OperationCancelled):
    """User requested cancellation of speech-filter processing."""
