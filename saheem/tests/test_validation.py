from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from entity_resolution.io import DataContractError
from entity_resolution.validation import validate_generated_submission


class SubmissionValidationTests(unittest.TestCase):
    def test_validates_aligned_subset_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tsv"
            matching = root / "matching.tsv"
            candidates = root / "candidates.tsv"
            source.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-1\tAlpha\t1 Road\tUS\nS1-2\tBeta\t2 Road\tIndia\n",
                encoding="utf-8",
            )
            matching.write_text(
                "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\t\n",
                encoding="utf-8",
            )
            candidates.write_text(
                "source1_entity_id\tcandidate_entity_ids\n"
                "S1-1\tS2-1,S3-1\nS1-2\tS2-2\n",
                encoding="utf-8",
            )
            report = validate_generated_submission(source, matching, candidates)
            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["matched_links"], 1)

    def test_rejects_match_outside_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tsv"
            matching = root / "matching.tsv"
            candidates = root / "candidates.tsv"
            source.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-1\tAlpha\t1 Road\tUS\n",
                encoding="utf-8",
            )
            matching.write_text(
                "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\n",
                encoding="utf-8",
            )
            candidates.write_text(
                "source1_entity_id\tcandidate_entity_ids\nS1-1\tS3-1\n",
                encoding="utf-8",
            )
            with self.assertRaises(DataContractError):
                validate_generated_submission(source, matching, candidates)


if __name__ == "__main__":
    unittest.main()
