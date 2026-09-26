"""Memory-bounded structural validation for generated submission files."""

from __future__ import annotations

import csv
from itertools import zip_longest
from pathlib import Path

from .contracts import CANDIDATE_COLUMNS, MATCHING_COLUMNS
from .io import DataContractError, iter_source_records


def _iter_result_rows(path: Path, expected_header: tuple[str, str]):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = tuple(next(reader, ()))
        if header != expected_header:
            raise DataContractError(
                f"{path}: expected header {expected_header!r}, found {header!r}"
            )
        for line_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise DataContractError(
                    f"{path}:{line_number}: expected exactly two tab-separated columns"
                )
            entity_ids = row[1].split(",") if row[1] else []
            if len(entity_ids) != len(set(entity_ids)):
                raise DataContractError(
                    f"{path}:{line_number}: duplicate ID within comma-separated list"
                )
            invalid = [
                entity_id
                for entity_id in entity_ids
                if not entity_id.startswith(("S2-", "S3-"))
            ]
            if invalid:
                raise DataContractError(
                    f"{path}:{line_number}: invalid target ID prefix {invalid[0]!r}"
                )
            yield row[0], frozenset(entity_ids)


def validate_generated_submission(
    source1_path: str | Path,
    matching_path: str | Path,
    candidate_path: str | Path,
) -> dict[str, int]:
    """Validate generated files in Source 1 order with constant auxiliary memory."""

    source_rows = iter_source_records(source1_path)
    matching_rows = _iter_result_rows(Path(matching_path), MATCHING_COLUMNS)
    candidate_rows = _iter_result_rows(Path(candidate_path), CANDIDATE_COLUMNS)
    sentinel = object()
    rows = empty_matches = matched_links = candidate_links = 0
    for line_number, triple in enumerate(
        zip_longest(source_rows, matching_rows, candidate_rows, fillvalue=sentinel),
        start=2,
    ):
        source, matching, candidates = triple
        if sentinel in triple:
            raise DataContractError(
                "Source 1, matching, and candidate files contain different row counts"
            )
        source_id = source.entity_id
        if matching[0] != source_id or candidates[0] != source_id:
            raise DataContractError(
                f"row {line_number}: output IDs do not align with Source 1 {source_id!r}"
            )
        if not matching[1].issubset(candidates[1]):
            raise DataContractError(
                f"row {line_number}: final matches are not a subset of candidates"
            )
        rows += 1
        empty_matches += int(not matching[1])
        matched_links += len(matching[1])
        candidate_links += len(candidates[1])
    return {
        "rows": rows,
        "empty_matches": empty_matches,
        "matched_links": matched_links,
        "candidate_links": candidate_links,
    }
