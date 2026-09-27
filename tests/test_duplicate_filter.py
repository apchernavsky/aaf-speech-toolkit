from __future__ import annotations

import unittest

from aaf_speech_filter.duplicate_filter import duplicate_timeline_key


class DuplicateTimelineKeyTests(unittest.TestCase):
    def test_same_content_at_different_timeline_positions_is_not_duplicate(self) -> None:
        source_key = ("mob-a", 48000, 0, 1)

        first = duplicate_timeline_key(source_key, timeline_pos_units=48000, slot_edit_rate=48000)
        later = duplicate_timeline_key(source_key, timeline_pos_units=96000, slot_edit_rate=48000)

        self.assertNotEqual(first, later)

    def test_same_content_at_same_timeline_position_is_duplicate_candidate(self) -> None:
        source_key = ("mob-a", 48000, 0, 1)

        first = duplicate_timeline_key(source_key, timeline_pos_units=48000, slot_edit_rate=48000)
        stacked = duplicate_timeline_key(source_key, timeline_pos_units=48000, slot_edit_rate=48000)

        self.assertEqual(first, stacked)


if __name__ == "__main__":
    unittest.main()
