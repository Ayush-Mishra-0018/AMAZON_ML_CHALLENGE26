"""Official entity-level macro F-beta evaluation and diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import zip_longest
from pathlib import Path
import sqlite3
import tempfile

from .contracts import MATCHING_COLUMNS
from .io import DataContractError, iter_id_list_rows, iter_source_records


def entity_fbeta(
    truth: set[str] | frozenset[str],
    prediction: set[str] | frozenset[str],
    *,
    beta: float = 0.5,
) -> float:
    """Score one Source 1 entity using the competition's singleton convention."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    if not truth and not prediction:
        return 1.0
    if not truth or not prediction:
        return 0.0
    true_positives = len(truth & prediction)
    if true_positives == 0:
        return 0.0
    precision = true_positives / len(prediction)
    recall = true_positives / len(truth)
    beta_squared = beta * beta
    return (1.0 + beta_squared) * precision * recall / (
        beta_squared * precision + recall
    )


@dataclass(slots=True)
class MetricAccumulator:
    entities: int = 0
    macro_score_sum: float = 0.0
    non_singleton_entities: int = 0
    non_singleton_score_sum: float = 0.0
    exact_sets: int = 0
    true_links: int = 0
    predicted_links: int = 0
    true_positive_links: int = 0
    singleton_entities: int = 0
    correct_singletons: int = 0

    def update(self, truth: frozenset[str], prediction: frozenset[str]) -> None:
        score = entity_fbeta(truth, prediction)
        self.entities += 1
        self.macro_score_sum += score
        self.exact_sets += int(truth == prediction)
        self.true_links += len(truth)
        self.predicted_links += len(prediction)
        self.true_positive_links += len(truth & prediction)
        if truth:
            self.non_singleton_entities += 1
            self.non_singleton_score_sum += score
        else:
            self.singleton_entities += 1
            self.correct_singletons += int(not prediction)

    def to_dict(self) -> dict[str, int | float | None]:
        link_precision = (
            self.true_positive_links / self.predicted_links
            if self.predicted_links
            else None
        )
        link_recall = (
            self.true_positive_links / self.true_links if self.true_links else None
        )
        return {
            "entities": self.entities,
            "macro_f0_5": self.macro_score_sum / self.entities if self.entities else None,
            "non_singleton_macro_f0_5": (
                self.non_singleton_score_sum / self.non_singleton_entities
                if self.non_singleton_entities
                else None
            ),
            "exact_set_accuracy": self.exact_sets / self.entities if self.entities else None,
            "singleton_entities": self.singleton_entities,
            "singleton_accuracy": (
                self.correct_singletons / self.singleton_entities
                if self.singleton_entities
                else None
            ),
            "singleton_false_positive_rate": (
                1.0 - self.correct_singletons / self.singleton_entities
                if self.singleton_entities
                else None
            ),
            "true_links": self.true_links,
            "predicted_links": self.predicted_links,
            "true_positive_links": self.true_positive_links,
            "link_precision_diagnostic": link_precision,
            "link_recall_diagnostic": link_recall,
        }


@dataclass(slots=True)
class EvaluationReport:
    overall: MetricAccumulator = field(default_factory=MetricAccumulator)
    by_country: dict[str, MetricAccumulator] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "overall": self.overall.to_dict(),
            "by_country": {
                country: accumulator.to_dict()
                for country, accumulator in sorted(self.by_country.items())
            },
        }


def evaluate_aligned_files(
    truth_path: str | Path,
    prediction_path: str | Path,
    *,
    source1_path: str | Path | None = None,
) -> EvaluationReport:
    """Evaluate same-order files in a memory-bounded pass.

    Output generation preserves Source 1 order, so strict alignment gives us an exact
    join with constant memory. A mismatch raises instead of scoring the wrong entity.
    """

    truth_rows = iter_id_list_rows(truth_path)
    prediction_rows = iter_id_list_rows(
        prediction_path,
        expected_columns=MATCHING_COLUMNS,
    )
    source_rows = iter_source_records(source1_path) if source1_path else None
    sentinel = object()
    report = EvaluationReport()

    if source_rows is None:
        rows = zip_longest(truth_rows, prediction_rows, fillvalue=sentinel)
        for row_number, (truth, prediction) in enumerate(rows, start=2):
            if truth is sentinel or prediction is sentinel:
                raise DataContractError(
                    "truth and prediction files contain different numbers of rows"
                )
            if truth.source1_entity_id != prediction.source1_entity_id:
                raise DataContractError(
                    f"row {row_number}: truth ID {truth.source1_entity_id!r} does not "
                    f"match prediction ID {prediction.source1_entity_id!r}"
                )
            report.overall.update(truth.entity_ids, prediction.entity_ids)
        return report

    rows = zip_longest(truth_rows, prediction_rows, source_rows, fillvalue=sentinel)
    for row_number, (truth, prediction, source) in enumerate(rows, start=2):
        if truth is sentinel or prediction is sentinel or source is sentinel:
            raise DataContractError(
                "truth, prediction, and Source 1 files contain different numbers of rows"
            )
        ids = (
            truth.source1_entity_id,
            prediction.source1_entity_id,
            source.entity_id,
        )
        if len(set(ids)) != 1:
            raise DataContractError(
                f"row {row_number}: truth, prediction, and Source 1 IDs are not "
                f"aligned: {ids!r}"
            )
        report.overall.update(truth.entity_ids, prediction.entity_ids)
        country_accumulator = report.by_country.setdefault(
            source.country,
            MetricAccumulator(),
        )
        country_accumulator.update(truth.entity_ids, prediction.entity_ids)
    return report


def _serialized_ids(entity_ids: frozenset[str]) -> str:
    return ",".join(sorted(entity_ids))


def _deserialized_ids(value: str) -> frozenset[str]:
    return frozenset(value.split(",")) if value else frozenset()


def _load_id_list_table(
    connection: sqlite3.Connection,
    *,
    table: str,
    path: str | Path,
    expected_columns: tuple[str, str],
    batch_size: int = 50_000,
) -> None:
    connection.execute(
        f"CREATE TABLE {table} (source1_entity_id TEXT PRIMARY KEY, entity_ids TEXT NOT NULL)"
    )
    batch: list[tuple[str, str]] = []
    try:
        connection.execute("BEGIN")
        for row in iter_id_list_rows(path, expected_columns=expected_columns):
            batch.append((row.source1_entity_id, _serialized_ids(row.entity_ids)))
            if len(batch) >= batch_size:
                connection.executemany(f"INSERT INTO {table} VALUES (?, ?)", batch)
                batch.clear()
        if batch:
            connection.executemany(f"INSERT INTO {table} VALUES (?, ?)", batch)
        connection.commit()
    except sqlite3.IntegrityError as error:
        connection.rollback()
        raise DataContractError(f"{path}: duplicate Source 1 ID") from error


def _load_country_table(
    connection: sqlite3.Connection,
    source1_path: str | Path,
    *,
    batch_size: int = 50_000,
) -> None:
    connection.execute(
        "CREATE TABLE countries (source1_entity_id TEXT PRIMARY KEY, country TEXT NOT NULL)"
    )
    batch: list[tuple[str, str]] = []
    try:
        connection.execute("BEGIN")
        for record in iter_source_records(source1_path):
            batch.append((record.entity_id, record.country))
            if len(batch) >= batch_size:
                connection.executemany("INSERT INTO countries VALUES (?, ?)", batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO countries VALUES (?, ?)", batch)
        connection.commit()
    except sqlite3.IntegrityError as error:
        connection.rollback()
        raise DataContractError(f"{source1_path}: duplicate Source 1 ID") from error


def _first_missing_id(
    connection: sqlite3.Connection,
    *,
    required_table: str,
    observed_table: str,
) -> str | None:
    row = connection.execute(
        f"""
        SELECT required.source1_entity_id
        FROM {required_table} AS required
        LEFT JOIN {observed_table} AS observed USING (source1_entity_id)
        WHERE observed.source1_entity_id IS NULL
        LIMIT 1
        """
    ).fetchone()
    return row[0] if row else None


def evaluate_keyed_files(
    truth_path: str | Path,
    prediction_path: str | Path,
    *,
    source1_path: str | Path | None = None,
    temporary_directory: str | Path | None = None,
) -> EvaluationReport:
    """Evaluate arbitrarily ordered TSVs through a disk-backed exact ID join."""

    temp_parent = str(temporary_directory) if temporary_directory else None
    with tempfile.TemporaryDirectory(prefix="amazon-er-score-", dir=temp_parent) as directory:
        database_path = Path(directory) / "evaluation.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute("PRAGMA cache_size = -131072")
            _load_id_list_table(
                connection,
                table="truth",
                path=truth_path,
                expected_columns=("source1_entity_id", "matched_entity_ids"),
            )
            _load_id_list_table(
                connection,
                table="predictions",
                path=prediction_path,
                expected_columns=MATCHING_COLUMNS,
            )

            missing_prediction = _first_missing_id(
                connection,
                required_table="truth",
                observed_table="predictions",
            )
            if missing_prediction:
                raise DataContractError(
                    f"prediction file is missing Source 1 ID {missing_prediction!r}"
                )
            extra_prediction = _first_missing_id(
                connection,
                required_table="predictions",
                observed_table="truth",
            )
            if extra_prediction:
                raise DataContractError(
                    f"prediction file contains unknown Source 1 ID {extra_prediction!r}"
                )

            report = EvaluationReport()
            if source1_path is None:
                query = """
                    SELECT truth.entity_ids, predictions.entity_ids
                    FROM truth JOIN predictions USING (source1_entity_id)
                """
                for truth_ids, prediction_ids in connection.execute(query):
                    report.overall.update(
                        _deserialized_ids(truth_ids),
                        _deserialized_ids(prediction_ids),
                    )
                return report

            _load_country_table(connection, source1_path)
            missing_source = _first_missing_id(
                connection,
                required_table="truth",
                observed_table="countries",
            )
            if missing_source:
                raise DataContractError(
                    f"Source 1 file is missing ground-truth ID {missing_source!r}"
                )
            extra_source = _first_missing_id(
                connection,
                required_table="countries",
                observed_table="truth",
            )
            if extra_source:
                raise DataContractError(
                    f"Source 1 file contains ID absent from ground truth {extra_source!r}"
                )

            query = """
                SELECT truth.entity_ids, predictions.entity_ids, countries.country
                FROM truth
                JOIN predictions USING (source1_entity_id)
                JOIN countries USING (source1_entity_id)
            """
            for truth_ids, prediction_ids, country in connection.execute(query):
                truth_set = _deserialized_ids(truth_ids)
                prediction_set = _deserialized_ids(prediction_ids)
                report.overall.update(truth_set, prediction_set)
                report.by_country.setdefault(country, MetricAccumulator()).update(
                    truth_set,
                    prediction_set,
                )
            return report
        finally:
            connection.close()
