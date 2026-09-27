from __future__ import annotations

import unittest

from aaf_speech_filter.timeline_walk import _vectors_on_container


class TimelineWalkTests(unittest.TestCase):
    def test_vectors_on_container_deduplicates_aliases(self) -> None:
        shared = [object()]

        class OperationGroupLike:
            segments = shared

            def __getitem__(self, key: str):
                if key == "InputSegments":
                    return shared
                raise KeyError(key)

        vectors = list(_vectors_on_container(OperationGroupLike()))

        self.assertEqual(vectors, [shared])


if __name__ == "__main__":
    unittest.main()
