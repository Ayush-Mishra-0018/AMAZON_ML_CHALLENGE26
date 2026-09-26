from __future__ import annotations

import unittest

from entity_resolution.blocking import EXACT_NAME, NAME_LSH, character_ngrams, lsh_bands
from entity_resolution.contracts import SourceRecord
from entity_resolution.features import FEATURE_NAMES, heuristic_score, pair_features
from entity_resolution.records import PreparedRecord


class BlockingTests(unittest.TestCase):
    def test_character_ngrams_and_lsh_are_deterministic(self) -> None:
        self.assertIn("ape", character_ngrams("apex"))
        self.assertEqual(lsh_bands("apex industries"), lsh_bands("apex industries"))
        self.assertEqual(len(lsh_bands("apex industries")), 4)


class FeatureTests(unittest.TestCase):
    def _record(self, entity_id: str, name: str, address: str) -> PreparedRecord:
        return PreparedRecord.from_source(
            SourceRecord(entity_id, name, address, "US")
        )

    def test_exact_pair_has_strong_features(self) -> None:
        query = self._record("S1-1", "Apex Industries LLC", "100 Main St")
        target = self._record("S2-1", "APEX INDUSTRIES", "100 Main Street")
        features = pair_features(query, target, EXACT_NAME | NAME_LSH)
        values = dict(zip(FEATURE_NAMES, features, strict=True))
        self.assertEqual(len(features), len(FEATURE_NAMES))
        self.assertEqual(values["name_core_exact"], 1.0)
        self.assertEqual(values["address_digit_exact"], 1.0)
        self.assertGreater(heuristic_score(features), 0.8)

    def test_conflicting_numbers_reduce_rank(self) -> None:
        query = self._record("S1-1", "Little Auto Body", "803 Lafayette Road")
        good = self._record("S2-1", "Little Auto Body", "803 Lafayette Rd")
        bad = self._record("S2-2", "Little Auto Body", "618 Concord Road")
        good_score = heuristic_score(pair_features(query, good, EXACT_NAME))
        bad_score = heuristic_score(pair_features(query, bad, EXACT_NAME))
        self.assertGreater(good_score, bad_score)


if __name__ == "__main__":
    unittest.main()
