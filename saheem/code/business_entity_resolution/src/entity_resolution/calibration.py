"""Entity-level model selection, threshold tuning, and holdout evaluation."""

from __future__ import annotations

import shutil
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .modeling import MODEL_FEATURE_NAMES
from .tracking import stage_timer, write_json_atomic


def fbeta_from_counts(true_count: int, predicted_count: int, true_positive_count: int) -> float:
    if true_count == 0 and predicted_count == 0:
        return 1.0
    if true_count == 0 or predicted_count == 0 or true_positive_count == 0:
        return 0.0
    precision = true_positive_count / predicted_count
    recall = true_positive_count / true_count
    return 1.25 * precision * recall / (0.25 * precision + recall)


def default_thresholds() -> tuple[float, ...]:
    values = [index / 100 for index in range(5, 96, 2)]
    values.extend((0.97, 0.98, 0.99, 0.995))
    return tuple(sorted(set(values)))


def _require_scoring_dependencies():
    try:
        import joblib
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "calibration requires numpy, pyarrow, and joblib; run the Kaggle dependency cell first"
        ) from error
    return joblib, np, pq


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    entity_id: str
    score: float
    label: int


@dataclass(frozen=True, slots=True)
class ScoredEntity:
    source1_entity_id: str
    country: str
    truth_count: int
    candidates: tuple[ScoredCandidate, ...]


def iter_scored_entities(
    pairs_path: str | Path,
    bundle: dict[str, object],
    *,
    split_name: str,
    batch_size: int = 250_000,
) -> Iterator[ScoredEntity]:
    """Score candidate rows while preserving Source 1 entity boundaries."""

    _, np, pq = _require_scoring_dependencies()
    feature_names = tuple(bundle["feature_names"])
    if feature_names != MODEL_FEATURE_NAMES:
        raise ValueError("model feature schema does not match this pipeline version")
    model = bundle["model"]
    columns = (
        "source1_entity_id",
        "candidate_entity_id",
        "split",
        "country",
        "truth_count",
        "label",
        "is_candidate",
        *feature_names,
    )
    parquet_file = pq.ParquetFile(pairs_path)
    current_id: str | None = None
    current_country = ""
    current_truth_count = 0
    current_candidates: list[ScoredCandidate] = []
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        splits = batch.column("split").to_pylist()
        selected_indices = [
            index for index, split in enumerate(splits) if split == split_name
        ]
        if not selected_indices:
            continue
        is_candidate = batch.column("is_candidate").to_numpy(zero_copy_only=False)
        candidate_indices = [index for index in selected_indices if is_candidate[index]]
        score_by_index: dict[int, float] = {}
        if candidate_indices:
            matrix = np.column_stack(
                [
                    batch.column(name).to_numpy(zero_copy_only=False).astype(np.float32)[
                        candidate_indices
                    ]
                    for name in feature_names
                ]
            )
            probabilities = model.predict_proba(matrix)[:, 1]
            score_by_index = {
                index: float(score)
                for index, score in zip(candidate_indices, probabilities, strict=True)
            }
        source1_ids = batch.column("source1_entity_id").to_pylist()
        candidate_ids = batch.column("candidate_entity_id").to_pylist()
        countries = batch.column("country").to_pylist()
        truth_counts = batch.column("truth_count").to_numpy(zero_copy_only=False)
        labels = batch.column("label").to_numpy(zero_copy_only=False)
        for index in selected_indices:
            source1_id = source1_ids[index]
            if current_id is not None and source1_id != current_id:
                yield ScoredEntity(
                    current_id,
                    current_country,
                    current_truth_count,
                    tuple(current_candidates),
                )
                current_candidates.clear()
            if current_id != source1_id:
                current_id = source1_id
                current_country = countries[index]
                current_truth_count = int(truth_counts[index])
            if is_candidate[index]:
                current_candidates.append(
                    ScoredCandidate(
                        candidate_ids[index],
                        score_by_index[index],
                        int(labels[index]),
                    )
                )
    if current_id is not None:
        yield ScoredEntity(
            current_id,
            current_country,
            current_truth_count,
            tuple(current_candidates),
        )


def threshold_curve(
    entities: Iterator[ScoredEntity],
    thresholds: tuple[float, ...],
) -> list[dict[str, float | int]]:
    score_sums = [0.0] * len(thresholds)
    exact_counts = [0] * len(thresholds)
    predicted_links = [0] * len(thresholds)
    true_positive_links = [0] * len(thresholds)
    entity_count = 0
    true_links = 0
    singleton_count = 0
    singleton_correct = [0] * len(thresholds)
    for entity in entities:
        entity_count += 1
        true_links += entity.truth_count
        singleton_count += int(entity.truth_count == 0)
        ordered = sorted(entity.candidates, key=lambda candidate: -candidate.score)
        negative_scores = [-candidate.score for candidate in ordered]
        true_positive_prefix = [0]
        for candidate in ordered:
            true_positive_prefix.append(true_positive_prefix[-1] + candidate.label)
        for threshold_index, threshold in enumerate(thresholds):
            predicted_count = bisect_right(negative_scores, -threshold)
            true_positive_count = true_positive_prefix[predicted_count]
            score_sums[threshold_index] += fbeta_from_counts(
                entity.truth_count,
                predicted_count,
                true_positive_count,
            )
            exact_counts[threshold_index] += int(
                predicted_count == entity.truth_count
                and true_positive_count == entity.truth_count
            )
            predicted_links[threshold_index] += predicted_count
            true_positive_links[threshold_index] += true_positive_count
            singleton_correct[threshold_index] += int(
                entity.truth_count == 0 and predicted_count == 0
            )
    results = []
    for index, threshold in enumerate(thresholds):
        predicted = predicted_links[index]
        true_positive = true_positive_links[index]
        results.append(
            {
                "threshold": threshold,
                "entities": entity_count,
                "macro_f0_5": score_sums[index] / entity_count if entity_count else 0.0,
                "exact_set_accuracy": exact_counts[index] / entity_count if entity_count else 0.0,
                "link_precision": true_positive / predicted if predicted else 0.0,
                "link_recall": true_positive / true_links if true_links else 0.0,
                "predicted_links": predicted,
                "true_positive_links": true_positive,
                "singleton_accuracy": (
                    singleton_correct[index] / singleton_count if singleton_count else 0.0
                ),
            }
        )
    return results


@dataclass(slots=True)
class _SingleThresholdAccumulator:
    entities: int = 0
    score_sum: float = 0.0
    exact_sets: int = 0
    true_links: int = 0
    predicted_links: int = 0
    true_positive_links: int = 0
    singleton_entities: int = 0
    singleton_correct: int = 0

    def update(self, entity: ScoredEntity, threshold: float) -> None:
        predicted = [
            candidate for candidate in entity.candidates if candidate.score >= threshold
        ]
        predicted_count = len(predicted)
        true_positive_count = sum(candidate.label for candidate in predicted)
        self.entities += 1
        self.score_sum += fbeta_from_counts(
            entity.truth_count,
            predicted_count,
            true_positive_count,
        )
        self.exact_sets += int(
            predicted_count == entity.truth_count
            and true_positive_count == entity.truth_count
        )
        self.true_links += entity.truth_count
        self.predicted_links += predicted_count
        self.true_positive_links += true_positive_count
        if entity.truth_count == 0:
            self.singleton_entities += 1
            self.singleton_correct += int(predicted_count == 0)

    def to_dict(self, threshold: float) -> dict[str, float | int]:
        return {
            "threshold": threshold,
            "entities": self.entities,
            "macro_f0_5": self.score_sum / self.entities if self.entities else 0.0,
            "exact_set_accuracy": self.exact_sets / self.entities if self.entities else 0.0,
            "link_precision": (
                self.true_positive_links / self.predicted_links
                if self.predicted_links
                else 0.0
            ),
            "link_recall": (
                self.true_positive_links / self.true_links if self.true_links else 0.0
            ),
            "predicted_links": self.predicted_links,
            "true_positive_links": self.true_positive_links,
            "singleton_accuracy": (
                self.singleton_correct / self.singleton_entities
                if self.singleton_entities
                else 0.0
            ),
        }


def evaluate_selected_threshold(
    entities: Iterator[ScoredEntity],
    threshold: float,
) -> dict[str, object]:
    overall = _SingleThresholdAccumulator()
    by_country: dict[str, _SingleThresholdAccumulator] = {}
    for entity in entities:
        overall.update(entity, threshold)
        by_country.setdefault(entity.country, _SingleThresholdAccumulator()).update(
            entity,
            threshold,
        )
    return {
        "overall": overall.to_dict(threshold),
        "by_country": {
            country: accumulator.to_dict(threshold)
            for country, accumulator in sorted(by_country.items())
        },
    }


def calibrate_models(
    pairs_path: str | Path,
    model_directory: str | Path,
    output_directory: str | Path,
    *,
    thresholds: tuple[float, ...] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Choose the model and threshold on calibration, then score holdout once."""

    joblib, _, _ = _require_scoring_dependencies()
    pairs_path = Path(pairs_path)
    model_directory = Path(model_directory)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    selection_path = output_directory / "selection.json"
    selected_model_path = output_directory / "selected_model.joblib"
    if (selection_path.exists() or selected_model_path.exists()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite calibration output in {output_directory}")
    thresholds = thresholds or default_thresholds()
    model_paths = sorted(model_directory.glob("*.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"no trained .joblib models found in {model_directory}")

    report: dict[str, object] = {
        "stage": "calibrate",
        "pairs_path": str(pairs_path.resolve()),
        "thresholds": list(thresholds),
    }
    with stage_timer(report):
        model_results: dict[str, object] = {}
        best: tuple[float, float, Path, dict[str, object]] | None = None
        for model_path in model_paths:
            bundle = joblib.load(model_path)
            curve = threshold_curve(
                iter_scored_entities(pairs_path, bundle, split_name="calibration"),
                thresholds,
            )
            selected = max(
                curve,
                key=lambda item: (item["macro_f0_5"], item["threshold"]),
            )
            model_results[bundle["name"]] = {
                "path": str(model_path),
                "selected": selected,
                "curve": curve,
            }
            candidate = (
                float(selected["macro_f0_5"]),
                float(selected["threshold"]),
                model_path,
                selected,
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        assert best is not None
        _, threshold, best_model_path, calibration_metrics = best
        shutil.copy2(best_model_path, selected_model_path)
        selected_bundle = joblib.load(selected_model_path)
        holdout = evaluate_selected_threshold(
            iter_scored_entities(pairs_path, selected_bundle, split_name="holdout"),
            threshold,
        )
        selection = {
            "model_name": selected_bundle["name"],
            "model_file": selected_model_path.name,
            "threshold": threshold,
            "feature_names": list(selected_bundle["feature_names"]),
            "calibration": calibration_metrics,
            "holdout": holdout,
        }
        report["models"] = model_results
        report["selection"] = selection
    write_json_atomic(selection_path, selection)
    write_json_atomic(output_directory / "calibration_report.json", report)
    return report
