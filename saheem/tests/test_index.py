from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from entity_resolution.contracts import SourceRecord
from entity_resolution.index import TargetIndex, prepare_index
from entity_resolution.records import PreparedRecord


class PreparedIndexTests(unittest.TestCase):
    def _write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_prepares_truth_split_and_retrieves_exact_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            self._write(
                dataset / "train/train_source1.tsv",
                header + "S1-1\tApex Industries LLC\t100 Main St\tUS\n",
            )
            self._write(
                dataset / "train/train_source2.tsv",
                header + "S2-1\tAPEX INDUSTRIES\t100 Main Street\tUS\n",
            )
            self._write(
                dataset / "train/train_source3.tsv",
                header + "S3-1\tDifferent Company\t900 Other Road\tUS\n",
            )
            self._write(
                dataset / "train/train_ground_truth.tsv",
                "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\n",
            )
            output = root / "prepared"
            prepare_index(dataset, output, mode="train")

            query = PreparedRecord.from_source(
                SourceRecord("S1-1", "Apex Industries LLC", "100 Main St", "US")
            )
            with TargetIndex(output / "train_index.sqlite3") as index:
                retrieved = index.retrieve(query)
                truth, split, truth_count = index.truth_and_split("S1-1")

            self.assertIn("S2-1", {item.record.entity_id for item in retrieved})
            self.assertEqual(truth, frozenset({"S2-1"}))
            self.assertIn(split, {"train", "calibration", "holdout"})
            self.assertEqual(truth_count, 1)


if __name__ == "__main__":
    unittest.main()
