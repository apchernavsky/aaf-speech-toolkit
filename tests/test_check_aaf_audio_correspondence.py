from __future__ import annotations

import unittest

from scripts.check_aaf_audio_correspondence import ClipAudioFingerprint, _covering


def _clip(*, raw: int, edit: int, length: int) -> ClipAudioFingerprint:
    return ClipAudioFingerprint(
        name="clip",
        source_id="source",
        source_track=1,
        source_start=96000,
        source_length=length,
        timeline_start=raw,
        edit_start=edit,
        timeline_length=length,
        lane=0,
        top_index=0,
        sample_rate=48000,
        pcm_bytes=length * 2,
        pcm_sha256="0" * 64,
    )


class AafAudioCorrespondenceTests(unittest.TestCase):
    def test_timecode_covering_uses_daw_visible_edit_start(self) -> None:
        item = _clip(raw=97228800, edit=97213440, length=1534080)

        self.assertEqual(_covering([item], 97213440, visible=True), [item])
        self.assertEqual(_covering([item], 97213440, visible=False), [])


if __name__ == "__main__":
    unittest.main()
