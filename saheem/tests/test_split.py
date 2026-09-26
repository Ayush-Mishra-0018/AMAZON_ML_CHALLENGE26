from __future__ import annotations

import unittest

from entity_resolution.split import SplitFractions, assign_split, match_count_bin


class SplitTests(unittest.TestCase):
    def test_assignment_is_deterministic(self) -> None:
        first = assign_split("S1-123", seed=2026)
        second = assign_split("S1-123", seed=2026)
        self.assertEqual(first, second)

    def test_group_key_keeps_near_duplicates_together(self) -> None:
        first = assign_split("S1-1", group_key="same-business")
        second = assign_split("S1-2", group_key="same-business")
        self.assertEqual(first, second)

    def test_invalid_fractions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SplitFractions(train=0.7, calibration=0.2, holdout=0.2)

    def test_match_count_bins(self) -> None:
        self.assertEqual(
            [match_count_bin(value) for value in (0, 1, 2, 3, 4, 5, 6, 11)],
            ["0", "1", "2-3", "2-3", "4-5", "4-5", "6+", "6+"],
        )


if __name__ == "__main__":
    unittest.main()
