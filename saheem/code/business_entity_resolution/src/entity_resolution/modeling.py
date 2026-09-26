"""Bounded-memory sampling and classical pair-model training."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

from .features import FEATURE_NAMES
from .tracking import stage_timer, write_json_atomic

MODEL_FEATURE_NAMES = ("heuristic_score", *FEATURE_NAMES)


def _require_training_dependencies():
    try:
        import joblib
        import numpy as np
        import pyarrow.parquet as pq
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import fbeta_score, precision_score, recall_score
    except ImportError as error:
        raise RuntimeError(
            "training requires numpy, pyarrow, joblib, and scikit-learn; "
            "run the Kaggle dependency cell first"
        ) from error
    return (
        joblib,
        np,
        pq,
        HistGradientBoostingClassifier,
        LogisticRegression,
        fbeta_score,
        precision_score,
        recall_score,
    )


def _training_class_counts(parquet_file, np) -> tuple[int, int]:
    negatives = positives = 0
    for batch in parquet_file.iter_batches(
        batch_size=250_000,
        columns=("split", "is_candidate", "label"),
    ):
        split = batch.column("split").to_numpy(zero_copy_only=False)
        is_candidate = batch.column("is_candidate").to_numpy(zero_copy_only=False)
        labels = batch.column("label").to_numpy(zero_copy_only=False)
        mask = (split == "train") & is_candidate
        selected = labels[mask]
        positives += int(np.count_nonzero(selected == 1))
        negatives += int(np.count_nonzero(selected == 0))
    return negatives, positives


def _sample_training_rows(
    pairs_path: Path,
    *,
    max_rows: int,
    seed: int,
):
    _, np, pq, *_ = _require_training_dependencies()
    parquet_file = pq.ParquetFile(pairs_path)
    negative_count, positive_count = _training_class_counts(parquet_file, np)
    if not positive_count or not negative_count:
        raise ValueError(
            f"training data needs both classes; found {positive_count} positive and "
            f"{negative_count} negative pairs"
        )

    positive_cap = min(positive_count, max(1, int(max_rows * 0.40)))
    negative_cap = min(negative_count, max_rows - positive_cap)
    if positive_cap + negative_cap < max_rows:
        spare = max_rows - positive_cap - negative_cap
        extra_positive = min(spare, positive_count - positive_cap)
        positive_cap += extra_positive
        spare -= extra_positive
        negative_cap += min(spare, negative_count - negative_cap)

    positive_rate = min(1.0, positive_cap * 1.10 / positive_count)
    negative_rate = min(1.0, negative_cap * 1.10 / negative_count)
    rng = np.random.default_rng(seed)
    feature_chunks = []
    label_chunks = []
    columns = ("split", "is_candidate", "label", *MODEL_FEATURE_NAMES)
    for batch in parquet_file.iter_batches(batch_size=250_000, columns=columns):
        split = batch.column("split").to_numpy(zero_copy_only=False)
        is_candidate = batch.column("is_candidate").to_numpy(zero_copy_only=False)
        labels = batch.column("label").to_numpy(zero_copy_only=False).astype(np.int8)
        base_mask = (split == "train") & is_candidate
        random_values = rng.random(batch.num_rows)
        sample_mask = base_mask & (
            ((labels == 1) & (random_values < positive_rate))
            | ((labels == 0) & (random_values < negative_rate))
        )
        if not np.any(sample_mask):
            continue
        matrix = np.column_stack(
            [
                batch.column(name).to_numpy(zero_copy_only=False).astype(np.float32)
                for name in MODEL_FEATURE_NAMES
            ]
        )
        feature_chunks.append(matrix[sample_mask])
        label_chunks.append(labels[sample_mask])

    features = np.concatenate(feature_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    final_indices = []
    for label, cap in ((1, positive_cap), (0, negative_cap)):
        indices = np.flatnonzero(labels == label)
        if len(indices) > cap:
            indices = rng.choice(indices, size=cap, replace=False)
        final_indices.append(indices)
    selected_indices = np.concatenate(final_indices)
    rng.shuffle(selected_indices)
    return (
        features[selected_indices],
        labels[selected_indices],
        {
            "available_positive_rows": positive_count,
            "available_negative_rows": negative_count,
            "sampled_positive_rows": int(np.count_nonzero(labels[selected_indices] == 1)),
            "sampled_negative_rows": int(np.count_nonzero(labels[selected_indices] == 0)),
        },
    )


def train_models(
    pairs_path: str | Path,
    output_directory: str | Path,
    *,
    max_training_rows: int = 2_000_000,
    seed: int = 2026,
    overwrite: bool = False,
) -> dict[str, object]:
    """Train transparent linear and nonlinear classical matching models."""

    (
        joblib,
        np,
        _,
        HistGradientBoostingClassifier,
        LogisticRegression,
        fbeta_score,
        precision_score,
        recall_score,
    ) = _require_training_dependencies()
    pairs_path = Path(pairs_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    model_paths = {
        "logistic": output_directory / "logistic.joblib",
        "hist_gradient_boosting": output_directory / "hist_gradient_boosting.joblib",
    }
    existing = [path for path in model_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite trained models: "
            + ", ".join(str(path) for path in existing)
        )

    report: dict[str, object] = {
        "stage": "train",
        "pairs_path": str(pairs_path.resolve()),
        "max_training_rows": max_training_rows,
        "seed": seed,
        "feature_names": list(MODEL_FEATURE_NAMES),
    }
    with stage_timer(report):
        features, labels, sample_report = _sample_training_rows(
            pairs_path,
            max_rows=max_training_rows,
            seed=seed,
        )
        report["sample"] = sample_report
        models = {
            "logistic": LogisticRegression(
                max_iter=300,
                solver="lbfgs",
                class_weight="balanced",
                random_state=seed,
            ),
            "hist_gradient_boosting": HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=250,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.1,
                class_weight="balanced",
                random_state=seed,
            ),
        }
        model_reports: dict[str, object] = {}
        for name, model in models.items():
            print(f"training {name} on {len(labels):,} sampled pairs", flush=True)
            model.fit(features, labels)
            probabilities = model.predict_proba(features)[:, 1]
            predictions = probabilities >= 0.5
            model_report = {
                "training_precision_at_0_5": float(
                    precision_score(labels, predictions, zero_division=0)
                ),
                "training_recall_at_0_5": float(
                    recall_score(labels, predictions, zero_division=0)
                ),
                "training_pair_f0_5_at_0_5": float(
                    fbeta_score(labels, predictions, beta=0.5, zero_division=0)
                ),
            }
            bundle = {
                "name": name,
                "model": model,
                "feature_names": MODEL_FEATURE_NAMES,
                "seed": seed,
                "training_sample": sample_report,
            }
            temporary = model_paths[name].with_name(f".{model_paths[name].name}.tmp")
            joblib.dump(bundle, temporary, compress=3)
            temporary.replace(model_paths[name])
            model_report["path"] = str(model_paths[name])
            model_reports[name] = model_report
        report["models"] = model_reports
        report["versions"] = {
            distribution: importlib.metadata.version(distribution)
            for distribution in ("numpy", "pyarrow", "scikit-learn", "joblib")
        }
        report["sample_matrix_shape"] = list(features.shape)
        report["sample_matrix_mib"] = float(features.nbytes / 2**20)
    write_json_atomic(output_directory / "training_report.json", report)
    return report
