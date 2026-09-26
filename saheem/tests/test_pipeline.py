from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from entity_resolution.pipeline import run_complete_pipeline


class PipelineResumeTests(unittest.TestCase):
    def test_reuses_final_outputs_and_reclaims_intermediates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_root = root / "dataset"
            work_root = root / "work"
            (dataset_root / "test").mkdir(parents=True)
            prediction_root = work_root / "07_submission"
            prediction_root.mkdir(parents=True)

            (dataset_root / "test/test_source1.tsv").write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-1\tExample Ltd\t1 Main Road\tGBR\n",
                encoding="utf-8",
            )
            (prediction_root / "matching_results.tsv").write_text(
                "source1_entity_id\tmatched_entity_ids\nS1-1\t\n",
                encoding="utf-8",
            )
            (prediction_root / "candidate_pairs.tsv").write_text(
                "source1_entity_id\tcandidate_entity_ids\nS1-1\t\n",
                encoding="utf-8",
            )
            (prediction_root / "prediction_report.json").write_text(
                json.dumps({"stage": "prediction"}),
                encoding="utf-8",
            )

            generated_paths = (
                work_root / "01_prepared/train_index.sqlite3",
                work_root / "02_candidates/train_pairs.parquet",
                work_root / "01_prepared/test_index.sqlite3",
                work_root / "02_candidates/test_pairs.parquet",
            )
            for path in generated_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"temporary")

            report = run_complete_pipeline(
                dataset_root,
                work_root,
                reclaim_disk=True,
            )

            self.assertIn("prediction", report["reused"])
            self.assertEqual(report["output_validation"]["rows"], 1)
            self.assertTrue(
                (prediction_root / "matching_results.tsv").exists()
            )
            self.assertTrue(
                (prediction_root / "candidate_pairs.tsv").exists()
            )
            for path in generated_paths:
                self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
