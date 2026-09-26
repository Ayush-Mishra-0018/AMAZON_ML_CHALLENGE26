from __future__ import annotations

import unittest

from entity_resolution.calibration import fbeta_from_counts


class CalibrationMetricTests(unittest.TestCase):
    def test_singleton_convention(self) -> None:
        self.assertEqual(fbeta_from_counts(0, 0, 0), 1.0)
        self.assertEqual(fbeta_from_counts(0, 1, 0), 0.0)

    def test_partial_match(self) -> None:
        self.assertAlmostEqual(fbeta_from_counts(2, 3, 2), 5 / 7)

    def test_empty_prediction_for_match_is_zero(self) -> None:
        self.assertEqual(fbeta_from_counts(2, 0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
