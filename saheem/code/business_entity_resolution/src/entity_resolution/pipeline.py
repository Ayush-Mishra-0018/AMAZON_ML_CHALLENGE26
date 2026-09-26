"""Resumable end-to-end orchestration for a single Kaggle working directory."""

from __future__ import annotations

import json
from pathlib import Path

from .calibration import calibrate_models
from .candidates import generate_pair_features
from .index import prepare_index
from .inference import predict_submission
from .modeling import train_models
from .tracking import stage_timer, write_json_atomic
from .validation import validate_generated_submission


def _load_report(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_complete_pipeline(
    dataset_root: str | Path,
    work_root: str | Path,
    *,
    seed: int = 2026,
    max_candidates: int = 40,
    negative_candidates_per_train_query: int = 12,
    max_training_rows: int = 2_000_000,
    reclaim_disk: bool = False,
) -> dict[str, object]:
    """Run every classical stage, resuming atomic completed artifacts when present."""

    dataset_root = Path(dataset_root)
    work_root = Path(work_root)
    prepared_root = work_root / "01_prepared"
    candidate_root = work_root / "02_candidates"
    model_root = work_root / "04_models"
    validation_root = work_root / "05_validation"
    prediction_root = work_root / "07_submission"
    for directory in (
        prepared_root,
        candidate_root,
        model_root,
        validation_root,
        prediction_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "stage": "run_all",
        "dataset_root": str(dataset_root.resolve()),
        "work_root": str(work_root.resolve()),
        "seed": seed,
        "max_candidates": max_candidates,
        "negative_candidates_per_train_query": negative_candidates_per_train_query,
        "max_training_rows": max_training_rows,
        "reclaim_disk": reclaim_disk,
        "reused": [],
        "removed_for_disk_reclamation": [],
    }
    reused: list[str] = report["reused"]  # type: ignore[assignment]
    removed: list[str] = report["removed_for_disk_reclamation"]  # type: ignore[assignment]
    with stage_timer(report):
        matching_path = prediction_root / "matching_results.tsv"
        candidate_path = prediction_root / "candidate_pairs.tsv"
        prediction_report_path = prediction_root / "prediction_report.json"
        prediction_ready = all(
            path.exists()
            for path in (matching_path, candidate_path, prediction_report_path)
        )

        train_index = prepared_root / "train_index.sqlite3"
        train_prepare_report = prepared_root / "train_prepare_report.json"
        train_pairs = candidate_root / "train_pairs.parquet"
        train_candidate_report = candidate_root / "train_candidate_report.json"
        training_report_path = model_root / "training_report.json"
        calibration_report_path = validation_root / "calibration_report.json"
        calibration_ready = all(
            path.exists()
            for path in (
                calibration_report_path,
                validation_root / "selected_model.joblib",
                validation_root / "selection.json",
            )
        )

        if prediction_ready:
            report["train_pipeline"] = "skipped: final prediction artifacts exist"
        elif calibration_ready:
            report["calibration"] = _load_report(calibration_report_path)
            reused.append("calibration")
        else:
            if train_index.exists() and train_prepare_report.exists():
                report["train_prepare"] = _load_report(train_prepare_report)
                reused.append("train_prepare")
            else:
                report["train_prepare"] = prepare_index(
                    dataset_root,
                    prepared_root,
                    mode="train",
                    seed=seed,
                )

            if train_pairs.exists() and train_candidate_report.exists():
                report["train_candidates"] = _load_report(train_candidate_report)
                reused.append("train_candidates")
            else:
                report["train_candidates"] = generate_pair_features(
                    dataset_root / "train/train_source1.tsv",
                    train_index,
                    candidate_root,
                    mode="train",
                    max_candidates=max_candidates,
                    negative_candidates_per_train_query=negative_candidates_per_train_query,
                )

            if training_report_path.exists() and all(
                (model_root / name).exists()
                for name in ("logistic.joblib", "hist_gradient_boosting.joblib")
            ):
                report["training"] = _load_report(training_report_path)
                reused.append("training")
            else:
                report["training"] = train_models(
                    train_pairs,
                    model_root,
                    max_training_rows=max_training_rows,
                    seed=seed,
                )

            report["calibration"] = calibrate_models(
                train_pairs,
                model_root,
                validation_root,
            )

        if reclaim_disk:
            for generated_path in (train_index, train_pairs):
                if generated_path.exists():
                    generated_path.unlink()
                    removed.append(str(generated_path))

        test_index = prepared_root / "test_index.sqlite3"
        test_prepare_report = prepared_root / "test_prepare_report.json"
        test_pairs = candidate_root / "test_pairs.parquet"
        test_candidate_report = candidate_root / "test_candidate_report.json"
        if prediction_ready:
            report["prediction"] = _load_report(prediction_report_path)
            reused.append("prediction")
        else:
            if test_index.exists() and test_prepare_report.exists():
                report["test_prepare"] = _load_report(test_prepare_report)
                reused.append("test_prepare")
            else:
                report["test_prepare"] = prepare_index(
                    dataset_root,
                    prepared_root,
                    mode="test",
                    seed=seed,
                )

            if test_pairs.exists() and test_candidate_report.exists():
                report["test_candidates"] = _load_report(test_candidate_report)
                reused.append("test_candidates")
            else:
                report["test_candidates"] = generate_pair_features(
                    dataset_root / "test/test_source1.tsv",
                    test_index,
                    candidate_root,
                    mode="test",
                    max_candidates=max_candidates,
                    negative_candidates_per_train_query=negative_candidates_per_train_query,
                )

            report["prediction"] = predict_submission(
                test_pairs,
                validation_root / "selection.json",
                prediction_root,
            )
        report["output_validation"] = validate_generated_submission(
            dataset_root / "test/test_source1.tsv",
            matching_path,
            candidate_path,
        )
        if reclaim_disk:
            for generated_path in (test_index, test_pairs):
                if generated_path.exists():
                    generated_path.unlink()
                    removed.append(str(generated_path))
    write_json_atomic(work_root / "run_all_report.json", report)
    return report
