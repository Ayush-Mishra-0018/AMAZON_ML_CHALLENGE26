from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from entity_resolution.io import DataContractError
from entity_resolution.metrics import (
    entity_fbeta,
    evaluate_aligned_files,
    evaluate_keyed_files,
)


class EntityFbetaTests(unittest.TestCase):
    def test_exact_non_singleton(self) -> None:
        self.assertEqual(entity_fbeta({"S2-1"}, {"S2-1"}), 1.0)

    def test_correct_singleton(self) -> None:
        self.assertEqual(entity_fbeta(set(), set()), 1.0)

    def test_false_merge_on_singleton(self) -> None:
        self.assertEqual(entity_fbeta(set(), {"S2-1"}), 0.0)

    def test_missed_non_singleton(self) -> None:
        self.assertEqual(entity_fbeta({"S2-1"}, set()), 0.0)

    def test_problem_statement_example(self) -> None:
        truth = {"S2-00047", "S3-00812"}
        prediction = {"S2-00047", "S2-00193", "S3-00812"}
        self.assertAlmostEqual(entity_fbeta(truth, prediction), 5 / 7)


class AlignedEvaluationTests(unittest.TestCase):
    def _write(self, path: Path, header: tuple[str, str], rows: list[tuple[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)

    def test_macro_is_entity_level(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = root / "truth.tsv"
            predictions = root / "predictions.tsv"
            self._write(
                truth,
                ("source1_entity_id", "matched_entity_ids"),
                [("S1-1", ""), ("S1-2", "S2-1,S3-1")],
            )
            self._write(
                predictions,
                ("source1_entity_id", "matched_entity_ids"),
                [("S1-1", ""), ("S1-2", "S2-1")],
            )
            report = evaluate_aligned_files(truth, predictions)
            self.assertAlmostEqual(report.overall.to_dict()["macro_f0_5"], 11 / 12)

    def test_alignment_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = root / "truth.tsv"
            predictions = root / "predictions.tsv"
            self._write(
                truth,
                ("source1_entity_id", "matched_entity_ids"),
                [("S1-1", "")],
            )
            self._write(
                predictions,
                ("source1_entity_id", "matched_entity_ids"),
                [("S1-2", "")],
            )
            with self.assertRaises(DataContractError):
                evaluate_aligned_files(truth, predictions)

    def test_keyed_evaluation_accepts_different_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = root / "truth.tsv"
            predictions = root / "predictions.tsv"
            self._write(
                truth,
                ("source1_entity_id", "matched_entity_ids"),
                [("S1-1", ""), ("S1-2", "S2-1,S3-1")],
            )
            self._write(
                predictions,
                ("source1_entity_id", "matched_entity_ids"),
                [("S1-2", "S2-1"), ("S1-1", "")],
            )
            report = evaluate_keyed_files(truth, predictions)
            self.assertAlmostEqual(report.overall.to_dict()["macro_f0_5"], 11 / 12)


if __name__ == "__main__":
    unittest.main()
