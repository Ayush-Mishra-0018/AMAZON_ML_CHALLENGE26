"""Deterministic, atomic writers for competition output files."""

from __future__ import annotations

import csv
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from .contracts import CANDIDATE_COLUMNS, MATCHING_COLUMNS
from .io import DataContractError, iter_source_records


@contextmanager
def _atomic_text_writer(path: Path, *, overwrite: bool) -> TextIO:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_empty_submission(
    source1_path: str | Path,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path, int]:
    """Write a correct-format all-singleton baseline in Source 1 row order."""

    output_directory = Path(output_directory)
    matching_path = output_directory / "matching_results.tsv"
    candidate_path = output_directory / "candidate_pairs.tsv"
    existing = [path for path in (matching_path, candidate_path) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing file(s): {joined}")

    with (
        _atomic_text_writer(matching_path, overwrite=overwrite) as matching_handle,
        _atomic_text_writer(candidate_path, overwrite=overwrite) as candidate_handle,
    ):
        matching_writer = csv.writer(
            matching_handle,
            delimiter="\t",
            lineterminator="\n",
        )
        candidate_writer = csv.writer(
            candidate_handle,
            delimiter="\t",
            lineterminator="\n",
        )
        matching_writer.writerow(MATCHING_COLUMNS)
        candidate_writer.writerow(CANDIDATE_COLUMNS)

        rows = 0
        for record in iter_source_records(source1_path):
            if not record.entity_id.startswith("S1-"):
                raise DataContractError(
                    f"Source 1 file contains invalid ID {record.entity_id!r}"
                )
            matching_writer.writerow((record.entity_id, ""))
            candidate_writer.writerow((record.entity_id, ""))
            rows += 1

    return matching_path, candidate_path, rows
