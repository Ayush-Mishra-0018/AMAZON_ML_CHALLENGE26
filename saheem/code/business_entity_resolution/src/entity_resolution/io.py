"""Strict streaming readers for the competition's UTF-8 TSV files."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import GROUND_TRUTH_COLUMNS, SOURCE_COLUMNS, IdListRow, SourceRecord


class DataContractError(ValueError):
    """Raised when a source or prediction file violates its declared schema."""


@contextmanager
def _validated_reader(
    path: Path,
    expected_columns: Sequence[str],
) -> Iterator[csv.DictReader]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        actual = tuple(reader.fieldnames or ())
        if actual != tuple(expected_columns):
            raise DataContractError(
                f"{path}: expected tab-separated header {tuple(expected_columns)!r}, "
                f"found {actual!r}"
            )
        yield reader


def _reject_extra_columns(path: Path, line_number: int, row: dict[str, str]) -> None:
    if None in row:
        raise DataContractError(
            f"{path}:{line_number}: row has more tab-separated fields than the header"
        )


def iter_source_records(path: str | Path) -> Iterator[SourceRecord]:
    """Yield source records one at a time without retaining the table in memory."""

    path = Path(path)
    with _validated_reader(path, SOURCE_COLUMNS) as reader:
        for line_number, row in enumerate(reader, start=2):
            _reject_extra_columns(path, line_number, row)
            yield SourceRecord(
                entity_id=row["entity_id"],
                business_name=row["business_name"],
                business_address=row["business_address"],
                country=row["country"],
            )


def parse_id_list(value: str, *, path: Path, line_number: int) -> frozenset[str]:
    if not value.strip():
        return frozenset()
    values = [part.strip() for part in value.split(",")]
    if any(not item for item in values):
        raise DataContractError(f"{path}:{line_number}: malformed comma-separated ID list")
    if len(values) != len(set(values)):
        raise DataContractError(f"{path}:{line_number}: duplicate ID within ID list")
    return frozenset(values)


def iter_id_list_rows(
    path: str | Path,
    *,
    expected_columns: Sequence[str] = GROUND_TRUTH_COLUMNS,
    check_unique: bool = False,
) -> Iterator[IdListRow]:
    """Yield ground-truth, prediction, or candidate rows under a declared schema.

    ``check_unique`` retains all Source 1 IDs and is therefore intended for bounded
    development files. Full scoring instead verifies every row against the aligned,
    deduplicated Source 1 file without building a large Python set.
    """

    path = Path(path)
    id_column = expected_columns[1]
    seen: set[str] | None = set() if check_unique else None
    with _validated_reader(path, expected_columns) as reader:
        for line_number, row in enumerate(reader, start=2):
            _reject_extra_columns(path, line_number, row)
            source1_id = row[expected_columns[0]].strip()
            if not source1_id:
                raise DataContractError(f"{path}:{line_number}: blank Source 1 ID")
            if seen is not None and source1_id in seen:
                raise DataContractError(
                    f"{path}:{line_number}: duplicate Source 1 row {source1_id!r}"
                )
            if seen is not None:
                seen.add(source1_id)
            yield IdListRow(
                source1_entity_id=source1_id,
                entity_ids=parse_id_list(row[id_column], path=path, line_number=line_number),
            )


@dataclass(slots=True)
class SourceProfile:
    path: str
    rows: int
    countries: dict[str, int]
    missing: dict[str, int]
    prefix_errors: int
    stopped_early: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def profile_source(
    path: str | Path,
    *,
    expected_prefix: str,
    max_rows: int | None = None,
) -> SourceProfile:
    """Profile schema-critical fields with bounded memory."""

    counts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    prefix_errors = 0
    rows = 0
    stopped_early = False

    for record in iter_source_records(path):
        rows += 1
        counts[record.country or "<EMPTY>"] += 1
        for column, value in (
            ("entity_id", record.entity_id),
            ("business_name", record.business_name),
            ("business_address", record.business_address),
            ("country", record.country),
        ):
            if not value.strip():
                missing[column] += 1
        if not record.entity_id.startswith(expected_prefix):
            prefix_errors += 1
        if max_rows is not None and rows >= max_rows:
            stopped_early = True
            break

    return SourceProfile(
        path=str(path),
        rows=rows,
        countries=dict(sorted(counts.items())),
        missing={column: missing[column] for column in SOURCE_COLUMNS},
        prefix_errors=prefix_errors,
        stopped_early=stopped_early,
    )
