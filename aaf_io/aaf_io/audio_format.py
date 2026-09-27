"""PCM container identification from content, independent of locator names."""
from pathlib import Path


def pcm_container_kind(path: Path) -> str:
    with Path(path).open("rb") as stream:
        header = stream.read(12)
    if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "wave"
    if header[:4] == b"FORM" and header[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    raise ValueError("Unsupported PCM container header")
