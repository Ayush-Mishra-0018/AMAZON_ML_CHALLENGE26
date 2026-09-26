from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from entity_resolution.submission import write_empty_submission


class EmptySubmissionTests(unittest.TestCase):
    def test_writes_both_required_files_in_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source1 = root / "source1.tsv"
            with source1.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(
                    ("entity_id", "business_name", "business_address", "country")
                )
                writer.writerow(("S1-2", "Beta", "2 Road", "US"))
                writer.writerow(("S1-1", "Alpha", "1 Road", "India"))

            matching, candidates, rows = write_empty_submission(source1, root / "out")

            self.assertEqual(rows, 2)
            self.assertEqual(
                matching.read_text(encoding="utf-8"),
                "source1_entity_id\tmatched_entity_ids\nS1-2\t\nS1-1\t\n",
            )
            self.assertEqual(
                candidates.read_text(encoding="utf-8"),
                "source1_entity_id\tcandidate_entity_ids\nS1-2\t\nS1-1\t\n",
            )

    def test_refuses_implicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source1 = root / "source1.tsv"
            source1.write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S1-1\tAlpha\t1 Road\tUS\n",
                encoding="utf-8",
            )
            write_empty_submission(source1, root / "out")
            with self.assertRaises(FileExistsError):
                write_empty_submission(source1, root / "out")


if __name__ == "__main__":
    unittest.main()
