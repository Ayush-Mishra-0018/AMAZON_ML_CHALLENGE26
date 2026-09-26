"""Candidate generation, hard-negative selection, and Parquet pair features."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .blocking import POSITIVE_INJECTION
from .features import FEATURE_NAMES, heuristic_score, pair_features
from .index import RetrievedCandidate, TargetIndex
from .io import iter_source_records
from .metrics import entity_fbeta
from .records import PreparedRecord
from .tracking import stage_timer, write_json_atomic


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "candidate generation requires pyarrow; install the Kaggle dependencies first"
        ) from error
    return pa, pq


@dataclass(slots=True)
class CandidateStatistics:
    entities: int = 0
    entities_with_candidates: int = 0
    candidate_links: int = 0
    true_links: int = 0
    recalled_links: int = 0
    all_true_recalled_entities: int = 0
    singleton_entities: int = 0
    oracle_score_sum: float = 0.0

    def update(self, truth: frozenset[str], candidates: set[str]) -> None:
        self.entities += 1
        self.entities_with_candidates += int(bool(candidates))
        self.candidate_links += len(candidates)
        self.true_links += len(truth)
        recalled = truth & candidates
        self.recalled_links += len(recalled)
        self.all_true_recalled_entities += int(recalled == truth)
        self.singleton_entities += int(not truth)
        self.oracle_score_sum += entity_fbeta(truth, recalled)

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "entities": self.entities,
            "entities_with_candidates": self.entities_with_candidates,
            "mean_candidates": self.candidate_links / self.entities if self.entities else None,
            "candidate_links": self.candidate_links,
            "true_links": self.true_links,
            "recalled_links": self.recalled_links,
            "link_recall": self.recalled_links / self.true_links if self.true_links else None,
            "all_true_recalled_rate": (
                self.all_true_recalled_entities / self.entities if self.entities else None
            ),
            "singleton_entities": self.singleton_entities,
            "oracle_macro_f0_5": (
                self.oracle_score_sum / self.entities if self.entities else None
            ),
        }


class PairParquetWriter:
    def __init__(self, path: Path, *, batch_rows: int = 100_000) -> None:
        pa, pq = _require_pyarrow()
        self.pa = pa
        self.path = path
        self.temporary = path.with_name(f".{path.name}.tmp")
        self.temporary.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.unlink(missing_ok=True)
        fields = [
            pa.field("source1_entity_id", pa.string()),
            pa.field("candidate_entity_id", pa.string()),
            pa.field("split", pa.string()),
            pa.field("country", pa.string()),
            pa.field("target_source", pa.string()),
            pa.field("truth_count", pa.int16()),
            pa.field("label", pa.int8()),
            pa.field("is_candidate", pa.bool_()),
            pa.field("retrieved", pa.bool_()),
            pa.field("channel_mask", pa.int16()),
            pa.field("heuristic_score", pa.float32()),
        ]
        fields.extend(pa.field(name, pa.float32()) for name in FEATURE_NAMES)
        self.schema = pa.schema(fields)
        self.writer = pq.ParquetWriter(
            self.temporary,
            self.schema,
            compression="zstd",
            use_dictionary=["split", "country", "target_source"],
            write_statistics=True,
        )
        self.batch_rows = batch_rows
        self.columns: dict[str, list[object]] = {
            name: [] for name in self.schema.names
        }
        self.rows = 0

    def append(
        self,
        *,
        source1_entity_id: str,
        candidate_entity_id: str,
        split: str,
        country: str,
        target_source: str,
        truth_count: int,
        label: int,
        is_candidate: bool,
        retrieved: bool,
        channel_mask: int,
        score: float,
        features: tuple[float, ...],
    ) -> None:
        values: tuple[object, ...] = (
            source1_entity_id,
            candidate_entity_id,
            split,
            country,
            target_source,
            truth_count,
            label,
            is_candidate,
            retrieved,
            channel_mask,
            score,
            *features,
        )
        if len(values) != len(self.schema.names):
            raise ValueError("pair row does not match the declared feature schema")
        for name, value in zip(self.schema.names, values, strict=True):
            self.columns[name].append(value)
        self.rows += 1
        if len(self.columns["source1_entity_id"]) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self.columns["source1_entity_id"]:
            return
        table = self.pa.Table.from_pydict(self.columns, schema=self.schema)
        self.writer.write_table(table, row_group_size=self.batch_rows)
        for values in self.columns.values():
            values.clear()

    def close(self, *, success: bool) -> None:
        try:
            self.flush()
        finally:
            self.writer.close()
        if success:
            self.temporary.replace(self.path)
        else:
            self.temporary.unlink(missing_ok=True)


@dataclass(slots=True)
class _ScoredCandidate:
    record: PreparedRecord
    channel_mask: int
    features: tuple[float, ...]
    score: float
    retrieved: bool = True


def _score_pool(
    query: PreparedRecord,
    pool: list[RetrievedCandidate],
    *,
    max_candidates: int,
) -> list[_ScoredCandidate]:
    scored: list[_ScoredCandidate] = []
    for retrieved in pool:
        features = pair_features(query, retrieved.record, retrieved.channel_mask)
        scored.append(
            _ScoredCandidate(
                record=retrieved.record,
                channel_mask=retrieved.channel_mask,
                features=features,
                score=heuristic_score(features),
            )
        )
    scored.sort(key=lambda item: (-item.score, item.record.entity_id))
    return scored[:max_candidates]


def _inject_training_positives(
    index: TargetIndex,
    query: PreparedRecord,
    truth: frozenset[str],
    selected: list[_ScoredCandidate],
) -> list[_ScoredCandidate]:
    present = {candidate.record.entity_id for candidate in selected}
    missing = sorted(truth - present)
    targets = index.fetch_records(missing)
    for entity_id in missing:
        target = targets.get(entity_id)
        if target is None:
            raise ValueError(f"ground-truth target is absent from target index: {entity_id}")
        features = pair_features(query, target, POSITIVE_INJECTION)
        selected.append(
            _ScoredCandidate(
                record=target,
                channel_mask=POSITIVE_INJECTION,
                features=features,
                score=heuristic_score(features),
                retrieved=False,
            )
        )
    return selected


def generate_pair_features(
    source1_path: str | Path,
    index_path: str | Path,
    output_directory: str | Path,
    *,
    mode: str,
    max_candidates: int = 40,
    negative_candidates_per_train_query: int = 12,
    exact_pool_limit: int = 2_000,
    approximate_pool_limit: int = 800,
    max_queries: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Generate final model candidates and materialize their similarity features."""

    if mode not in {"train", "test"}:
        raise ValueError("mode must be 'train' or 'test'")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    pairs_path = output_directory / f"{mode}_pairs.parquet"
    report_path = output_directory / f"{mode}_candidate_report.json"
    if pairs_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {pairs_path}")

    report: dict[str, object] = {
        "stage": "candidates",
        "mode": mode,
        "source1_path": str(Path(source1_path).resolve()),
        "index_path": str(Path(index_path).resolve()),
        "max_candidates": max_candidates,
        "negative_candidates_per_train_query": negative_candidates_per_train_query,
        "exact_pool_limit": exact_pool_limit,
        "approximate_pool_limit": approximate_pool_limit,
        "max_queries": max_queries,
        "feature_names": list(FEATURE_NAMES),
    }
    statistics: dict[str, CandidateStatistics] = {}
    writer = PairParquetWriter(pairs_path)
    success = False
    query_rows = 0
    with stage_timer(report):
        try:
            with TargetIndex(index_path) as index:
                for source_record in iter_source_records(source1_path):
                    query = PreparedRecord.from_source(source_record)
                    truth, split, truth_count = index.truth_and_split(query.entity_id)
                    pool = index.retrieve(
                        query,
                        exact_pool_limit=exact_pool_limit,
                        approximate_pool_limit=approximate_pool_limit,
                    )
                    selected = _score_pool(
                        query,
                        pool,
                        max_candidates=max_candidates,
                    )
                    retrieved_ids = {item.record.entity_id for item in selected}
                    statistics.setdefault(split, CandidateStatistics()).update(
                        truth,
                        retrieved_ids,
                    )
                    statistics.setdefault(
                        f"{split}:{query.country}",
                        CandidateStatistics(),
                    ).update(truth, retrieved_ids)

                    model_rows = list(selected)
                    if mode == "train" and split == "train":
                        model_rows = _inject_training_positives(
                            index,
                            query,
                            truth,
                            model_rows,
                        )
                        positives = [
                            item for item in model_rows if item.record.entity_id in truth
                        ]
                        negatives = [
                            item for item in model_rows if item.record.entity_id not in truth
                        ][:negative_candidates_per_train_query]
                        model_rows = positives + negatives
                        model_rows.sort(
                            key=lambda item: (-item.score, item.record.entity_id)
                        )

                    if not model_rows:
                        writer.append(
                            source1_entity_id=query.entity_id,
                            candidate_entity_id="",
                            split=split,
                            country=query.country,
                            target_source="",
                            truth_count=truth_count,
                            label=0,
                            is_candidate=False,
                            retrieved=False,
                            channel_mask=0,
                            score=0.0,
                            features=tuple(0.0 for _ in FEATURE_NAMES),
                        )
                    else:
                        for item in model_rows:
                            writer.append(
                                source1_entity_id=query.entity_id,
                                candidate_entity_id=item.record.entity_id,
                                split=split,
                                country=query.country,
                                target_source=item.record.source,
                                truth_count=truth_count,
                                label=int(item.record.entity_id in truth),
                                is_candidate=True,
                                retrieved=item.retrieved,
                                channel_mask=item.channel_mask,
                                score=item.score,
                                features=item.features,
                            )
                    query_rows += 1
                    if query_rows % 10_000 == 0:
                        print(f"generated candidates for {query_rows:,} queries", flush=True)
                    if max_queries and query_rows >= max_queries:
                        break
            success = True
        finally:
            writer.close(success=success)

    report["query_rows"] = query_rows
    report["pair_rows"] = writer.rows
    report["pairs_path"] = str(pairs_path)
    report["statistics"] = {
        key: value.to_dict() for key, value in sorted(statistics.items())
    }
    write_json_atomic(report_path, report)
    return report
