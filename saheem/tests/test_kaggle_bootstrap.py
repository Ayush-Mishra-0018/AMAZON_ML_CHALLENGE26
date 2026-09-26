from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kaggle.scripts.bootstrap import (  # noqa: E402
    REQUIRED_DATA_FILES,
    STAGE_DIRECTORIES,
    create_run_context,
    discover_data_root,
)


class KaggleBootstrapTests(unittest.TestCase):
    def _make_data_root(self, parent: Path) -> Path:
        root = parent / "uploaded-data" / "dataset"
        for relative in REQUIRED_DATA_FILES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return root

    def test_discovers_nested_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_root = Path(directory)
            expected = self._make_data_root(input_root)
            self.assertEqual(discover_data_root(input_root), expected.resolve())

    def test_creates_reproducible_stage_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = self._make_data_root(root)
            work_root = root / "working"
            context = create_run_context(data_root, PROJECT_ROOT, work_root)

            self.assertTrue((work_root / "run_context.json").is_file())
            self.assertEqual(context["source_root"], str(PROJECT_ROOT / "code/business_entity_resolution/src"))
            for stage in STAGE_DIRECTORIES:
                self.assertTrue((work_root / stage).is_dir())


if __name__ == "__main__":
    unittest.main()
