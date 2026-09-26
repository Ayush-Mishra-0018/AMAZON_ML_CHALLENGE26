"""Disk-backed preparation and multi-channel target retrieval."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .blocking import (
    ADDRESS_LAST_NUMBER,
    ADDRESS_LSH,
    ADDRESS_NUMBER_TAIL,
    EXACT_ADDRESS,
    EXACT_CORE_NAME,
    EXACT_FOLDED_NAME,
    EXACT_NAME,
    NAME_LSH,
)
from .io import DataContractError, iter_id_list_rows, iter_source_records
from .records import PreparedRecord
from .split import assign_split
from .tracking import stage_timer, write_json_atomic

_RECORD_COLUMNS = (
    "entity_id",
    "source",
    "country",
    "name_norm",
    "name_fold",
    "name_core",
    "address_norm",
    "address_fold",
    "address_canon",
    "name_digits",
    "address_digits",
    "scripts",
    "name_lsh0",
    "name_lsh1",
    "name_lsh2",
    "name_lsh3",
    "address_lsh0",
    "address_lsh1",
    "address_lsh2",
    "address_lsh3",
    "address_first_number",
    "address_last_number",
    "address_tail",
)

_CREATE_RECORDS = """
CREATE TABLE records (
    entity_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    country TEXT NOT NULL,
    name_norm TEXT NOT NULL,
    name_fold TEXT NOT NULL,
    name_core TEXT NOT NULL,
    address_norm TEXT NOT NULL,
    address_fold TEXT NOT NULL,
    address_canon TEXT NOT NULL,
    name_digits TEXT NOT NULL,
    address_digits TEXT NOT NULL,
    scripts TEXT NOT NULL,
    name_lsh0 INTEGER NOT NULL,
    name_lsh1 INTEGER NOT NULL,
    name_lsh2 INTEGER NOT NULL,
    name_lsh3 INTEGER NOT NULL,
    address_lsh0 INTEGER NOT NULL,
    address_lsh1 INTEGER NOT NULL,
    address_lsh2 INTEGER NOT NULL,
    address_lsh3 INTEGER NOT NULL,
    address_first_number TEXT NOT NULL,
    address_last_number TEXT NOT NULL,
    address_tail TEXT NOT NULL
)
"""

_INSERT_RECORD = (
    f"INSERT INTO records ({', '.join(_RECORD_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in _RECORD_COLUMNS)})"
)

_INDEX_FIELDS = (
    "name_norm",
    "name_fold",
    "name_core",
    "address_canon",
    "name_lsh0",
    "name_lsh1",
    "name_lsh2",
    "name_lsh3",
    "address_lsh0",
    "address_lsh1",
    "address_lsh2",
    "address_lsh3",
)


def _configure_write_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute("PRAGMA cache_size = -262144")
    connection.execute("PRAGMA locking_mode = EXCLUSIVE")


def _insert_prepared_records(
    connection: sqlite3.Connection,
    paths: Iterable[Path],
    *,
    max_records_per_source: int | None,
    batch_size: int = 25_000,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        batch: list[tuple[object, ...]] = []
        count = 0
        connection.execute("BEGIN")
        try:
            for source_record in iter_source_records(path):
                prepared = PreparedRecord.from_source(source_record)
                batch.append(prepared.sqlite_values())
                count += 1
                if len(batch) >= batch_size:
                    connection.executemany(_INSERT_RECORD, batch)
                    batch.clear()
                if max_records_per_source and count >= max_records_per_source:
                    break
            if batch:
                connection.executemany(_INSERT_RECORD, batch)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        counts[path.name] = count
        print(f"prepared {count:,} records from {path.name}", flush=True)
    return counts


def _create_retrieval_indices(connection: sqlite3.Connection) -> None:
    for field in _INDEX_FIELDS:
        print(f"creating index for {field}", flush=True)
        connection.execute(
            f"CREATE INDEX ix_records_{field} ON records (country, {field})"
        )
    connection.execute(
        "CREATE INDEX ix_records_address_number_tail "
        "ON records (country, address_first_number, address_tail)"
    )
    connection.execute(
        "CREATE INDEX ix_records_address_last_number "
        "ON records (country, address_last_number, address_tail)"
    )
    connection.execute("ANALYZE")
    connection.commit()


def _load_truth(connection: sqlite3.Connection, ground_truth_path: Path) -> int:
    connection.execute(
        """
        CREATE TABLE truth (
            source1_entity_id TEXT PRIMARY KEY,
            matched_entity_ids TEXT NOT NULL,
            truth_count INTEGER NOT NULL
        )
        """
    )
    batch: list[tuple[str, str, int]] = []
    count = 0
    connection.execute("BEGIN")
    try:
        for row in iter_id_list_rows(ground_truth_path):
            ids = ",".join(sorted(row.entity_ids))
            batch.append((row.source1_entity_id, ids, len(row.entity_ids)))
            count += 1
            if len(batch) >= 50_000:
                connection.executemany("INSERT INTO truth VALUES (?, ?, ?)", batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO truth VALUES (?, ?, ?)", batch)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    return count


def _build_query_manifest(
    connection: sqlite3.Connection,
    source1_path: Path,
    manifest_path: Path,
    *,
    mode: str,
    seed: int,
    max_queries: int | None,
) -> int:
    connection.execute(
        """
        CREATE TABLE query_manifest (
            source1_entity_id TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            country TEXT NOT NULL,
            truth_count INTEGER NOT NULL
        )
        """
    )
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    batch: list[tuple[str, str, str, int]] = []
    connection.execute("BEGIN")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("source1_entity_id", "split", "country", "truth_count"))
            for source_record in iter_source_records(source1_path):
                prepared = PreparedRecord.from_source(source_record)
                if mode == "train":
                    truth_row = connection.execute(
                        "SELECT truth_count FROM truth WHERE source1_entity_id = ?",
                        (source_record.entity_id,),
                    ).fetchone()
                    if truth_row is None:
                        raise DataContractError(
                            f"ground truth is missing {source_record.entity_id}"
                        )
                    truth_count = int(truth_row[0])
                    group_key = "\x1f".join(
                        (prepared.country, prepared.name_core, prepared.address_canon)
                    )
                    split = assign_split(
                        source_record.entity_id,
                        seed=seed,
                        group_key=group_key,
                    )
                else:
                    truth_count = 0
                    split = "test"
                row = (source_record.entity_id, split, source_record.country, truth_count)
                writer.writerow(row)
                batch.append(row)
                rows += 1
                if len(batch) >= 50_000:
                    connection.executemany(
                        "INSERT INTO query_manifest VALUES (?, ?, ?, ?)",
                        batch,
                    )
                    batch.clear()
                if max_queries and rows >= max_queries:
                    break
            if batch:
                connection.executemany(
                    "INSERT INTO query_manifest VALUES (?, ?, ?, ?)",
                    batch,
                )
            handle.flush()
            os.fsync(handle.fileno())
        connection.commit()
        temporary.replace(manifest_path)
    except BaseException:
        connection.rollback()
        temporary.unlink(missing_ok=True)
        raise
    return rows


def prepare_index(
    dataset_root: str | Path,
    output_directory: str | Path,
    *,
    mode: str,
    seed: int = 2026,
    overwrite: bool = False,
    max_records_per_source: int | None = None,
    max_queries: int | None = None,
) -> dict[str, object]:
    """Prepare a train or test target index and its Source 1 manifest."""

    if mode not in {"train", "test"}:
        raise ValueError("mode must be 'train' or 'test'")
    dataset_root = Path(dataset_root)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    database_path = output_directory / f"{mode}_index.sqlite3"
    manifest_path = output_directory / f"{mode}_query_manifest.tsv"
    report_path = output_directory / f"{mode}_prepare_report.json"
    existing = [path for path in (database_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite prepared artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    temporary_database = database_path.with_name(f".{database_path.name}.tmp")
    temporary_database.unlink(missing_ok=True)
    if overwrite:
        manifest_path.unlink(missing_ok=True)
    report: dict[str, object] = {
        "stage": "prepare",
        "mode": mode,
        "seed": seed,
        "dataset_root": str(dataset_root.resolve()),
        "max_records_per_source": max_records_per_source,
        "max_queries": max_queries,
    }
    with stage_timer(report):
        connection = sqlite3.connect(temporary_database)
        try:
            _configure_write_connection(connection)
            connection.execute(_CREATE_RECORDS)
            source_paths = (
                dataset_root / mode / f"{mode}_source2.tsv",
                dataset_root / mode / f"{mode}_source3.tsv",
            )
            report["target_rows"] = _insert_prepared_records(
                connection,
                source_paths,
                max_records_per_source=max_records_per_source,
            )
            _create_retrieval_indices(connection)
            if mode == "train":
                report["truth_rows"] = _load_truth(
                    connection,
                    dataset_root / "train/train_ground_truth.tsv",
                )
            report["query_rows"] = _build_query_manifest(
                connection,
                dataset_root / mode / f"{mode}_source1.tsv",
                manifest_path,
                mode=mode,
                seed=seed,
                max_queries=max_queries,
            )
            connection.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            for key, value in (
                ("mode", mode),
                ("seed", str(seed)),
                ("report", json.dumps(report, sort_keys=True)),
            ):
                connection.execute("INSERT INTO metadata VALUES (?, ?)", (key, value))
            connection.commit()
        finally:
            connection.close()
        temporary_database.replace(database_path)
    report["database_path"] = str(database_path)
    report["manifest_path"] = str(manifest_path)
    write_json_atomic(report_path, report)
    return report


@dataclass(slots=True)
class RetrievedCandidate:
    record: PreparedRecord
    channel_mask: int


class TargetIndex:
    """Read-only target lookup with exact and approximate blocking channels."""

    def __init__(self, database_path: str | Path) -> None:
        path = Path(database_path).resolve()
        self.connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA query_only = ON")
        self.connection.execute("PRAGMA cache_size = -262144")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "TargetIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def truth_and_split(self, source1_entity_id: str) -> tuple[frozenset[str], str, int]:
        manifest = self.connection.execute(
            "SELECT split, truth_count FROM query_manifest WHERE source1_entity_id = ?",
            (source1_entity_id,),
        ).fetchone()
        if manifest is None:
            raise DataContractError(f"query manifest is missing {source1_entity_id}")
        try:
            truth = self.connection.execute(
                "SELECT matched_entity_ids FROM truth WHERE source1_entity_id = ?",
                (source1_entity_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            truth = None
        ids = frozenset(truth[0].split(",")) if truth and truth[0] else frozenset()
        return ids, manifest["split"], int(manifest["truth_count"])

    @staticmethod
    def _mask(query: PreparedRecord, target: PreparedRecord) -> int:
        mask = 0
        if query.name_norm and query.name_norm == target.name_norm:
            mask |= EXACT_NAME
        if query.name_fold and query.name_fold == target.name_fold:
            mask |= EXACT_FOLDED_NAME
        if query.name_core and query.name_core == target.name_core:
            mask |= EXACT_CORE_NAME
        if query.address_canon and query.address_canon == target.address_canon:
            mask |= EXACT_ADDRESS
        if any(left == right for left, right in zip(query.name_bands, target.name_bands)):
            mask |= NAME_LSH
        if query.address_canon and any(
            left == right for left, right in zip(query.address_bands, target.address_bands)
        ):
            mask |= ADDRESS_LSH
        if (
            query.address_first_number
            and query.address_tail
            and query.address_first_number == target.address_first_number
            and query.address_tail == target.address_tail
        ):
            mask |= ADDRESS_NUMBER_TAIL
        if (
            query.address_last_number
            and query.address_tail
            and query.address_last_number == target.address_last_number
            and query.address_tail == target.address_tail
        ):
            mask |= ADDRESS_LAST_NUMBER
        return mask

    def retrieve(
        self,
        query: PreparedRecord,
        *,
        exact_pool_limit: int = 2_000,
        approximate_pool_limit: int = 800,
    ) -> list[RetrievedCandidate]:
        exact_sql = """
            SELECT * FROM records
            WHERE country = ? AND (
                name_norm = ? OR name_fold = ? OR name_core = ? OR address_canon = ?
            )
            LIMIT ?
        """
        exact_rows = self.connection.execute(
            exact_sql,
            (
                query.country,
                query.name_norm,
                query.name_fold,
                query.name_core,
                query.address_canon or "<NO-ADDRESS>",
                exact_pool_limit,
            ),
        )
        approximate_sql = """
            SELECT * FROM records
            WHERE country = ? AND (
                name_lsh0 = ? OR name_lsh1 = ? OR name_lsh2 = ? OR name_lsh3 = ? OR
                address_lsh0 = ? OR address_lsh1 = ? OR
                address_lsh2 = ? OR address_lsh3 = ? OR
                (address_first_number = ? AND address_tail = ?) OR
                (address_last_number = ? AND address_tail = ?)
            )
            LIMIT ?
        """
        approximate_rows = self.connection.execute(
            approximate_sql,
            (
                query.country,
                *query.name_bands,
                *query.address_bands,
                query.address_first_number or "<NO-NUMBER>",
                query.address_tail or "<NO-TAIL>",
                query.address_last_number or "<NO-NUMBER>",
                query.address_tail or "<NO-TAIL>",
                approximate_pool_limit,
            ),
        )
        candidates: dict[str, RetrievedCandidate] = {}
        for row in (*list(exact_rows), *list(approximate_rows)):
            target = PreparedRecord.from_sqlite_row(row)
            mask = self._mask(query, target)
            existing = candidates.get(target.entity_id)
            if existing is None:
                candidates[target.entity_id] = RetrievedCandidate(target, mask)
            else:
                existing.channel_mask |= mask
        return list(candidates.values())

    def fetch_records(self, entity_ids: Iterable[str]) -> dict[str, PreparedRecord]:
        ids = tuple(dict.fromkeys(entity_ids))
        if not ids:
            return {}
        placeholders = ", ".join("?" for _ in ids)
        rows = self.connection.execute(
            f"SELECT * FROM records WHERE entity_id IN ({placeholders})",
            ids,
        )
        return {
            row["entity_id"]: PreparedRecord.from_sqlite_row(row)
            for row in rows
        }
