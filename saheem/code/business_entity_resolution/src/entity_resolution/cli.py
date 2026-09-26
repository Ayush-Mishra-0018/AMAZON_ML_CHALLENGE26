"""Command-line entry points for data checks and local evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import calibrate_models
from .candidates import generate_pair_features
from .index import prepare_index
from .inference import predict_submission
from .io import DataContractError, profile_source
from .metrics import evaluate_aligned_files, evaluate_keyed_files
from .modeling import train_models
from .packaging import create_submission_archive
from .pipeline import run_complete_pipeline
from .submission import write_empty_submission
from .validation import validate_generated_submission

_SOURCE_FILES = (
    ("train/train_source1.tsv", "S1-"),
    ("train/train_source2.tsv", "S2-"),
    ("train/train_source3.tsv", "S3-"),
    ("test/test_source1.tsv", "S1-"),
    ("test/test_source2.tsv", "S2-"),
    ("test/test_source3.tsv", "S3-"),
)


def _inspect(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root)
    profiles = []
    for relative_path, prefix in _SOURCE_FILES:
        path = root / relative_path
        profiles.append(
            profile_source(path, expected_prefix=prefix, max_rows=args.max_rows).to_dict()
        )
    print(json.dumps({"source_files": profiles}, indent=2, ensure_ascii=False))
    return 0


def _score(args: argparse.Namespace) -> int:
    if args.aligned:
        report = evaluate_aligned_files(
            args.truth,
            args.predictions,
            source1_path=args.source1,
        )
    else:
        report = evaluate_keyed_files(
            args.truth,
            args.predictions,
            source1_path=args.source1,
            temporary_directory=args.temporary_directory,
        )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


def _empty_baseline(args: argparse.Namespace) -> int:
    matching, candidates, rows = write_empty_submission(
        args.source1,
        args.output_directory,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "rows": rows,
                "matching_results": str(matching),
                "candidate_pairs": str(candidates),
            },
            indent=2,
        )
    )
    return 0


def _print_report(report: dict[str, object]) -> int:
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def _prepare(args: argparse.Namespace) -> int:
    return _print_report(
        prepare_index(
            args.dataset_root,
            args.output_directory,
            mode=args.mode,
            seed=args.seed,
            overwrite=args.overwrite,
            max_records_per_source=args.max_records_per_source,
            max_queries=args.max_queries,
        )
    )


def _candidates(args: argparse.Namespace) -> int:
    return _print_report(
        generate_pair_features(
            args.source1,
            args.index,
            args.output_directory,
            mode=args.mode,
            max_candidates=args.max_candidates,
            negative_candidates_per_train_query=args.train_negatives,
            exact_pool_limit=args.exact_pool_limit,
            approximate_pool_limit=args.approximate_pool_limit,
            max_queries=args.max_queries,
            overwrite=args.overwrite,
        )
    )


def _train(args: argparse.Namespace) -> int:
    return _print_report(
        train_models(
            args.pairs,
            args.output_directory,
            max_training_rows=args.max_training_rows,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    )


def _calibrate(args: argparse.Namespace) -> int:
    return _print_report(
        calibrate_models(
            args.pairs,
            args.model_directory,
            args.output_directory,
            overwrite=args.overwrite,
        )
    )


def _predict(args: argparse.Namespace) -> int:
    return _print_report(
        predict_submission(
            args.pairs,
            args.selection,
            args.output_directory,
            overwrite=args.overwrite,
        )
    )


def _validate_outputs(args: argparse.Namespace) -> int:
    return _print_report(
        validate_generated_submission(
            args.source1,
            args.matching,
            args.candidates,
        )
    )


def _package(args: argparse.Namespace) -> int:
    return _print_report(
        create_submission_archive(
            args.project_root,
            args.prediction_directory,
            args.documentation,
            args.archive,
            overwrite=args.overwrite,
        )
    )


def _run_all(args: argparse.Namespace) -> int:
    return _print_report(
        run_complete_pipeline(
            args.dataset_root,
            args.work_root,
            seed=args.seed,
            max_candidates=args.max_candidates,
            negative_candidates_per_train_query=args.train_negatives,
            max_training_rows=args.max_training_rows,
            reclaim_disk=args.reclaim_disk,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="validate source headers and report countries/missing fields",
    )
    inspect_parser.add_argument("--dataset-root", required=True)
    inspect_parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="optional per-file row limit for a fast development check",
    )
    inspect_parser.set_defaults(function=_inspect)

    score_parser = subparsers.add_parser(
        "score",
        help="compute official macro F0.5 for same-order truth and predictions",
    )
    score_parser.add_argument("--truth", required=True)
    score_parser.add_argument("--predictions", required=True)
    score_parser.add_argument(
        "--source1",
        default=None,
        help="optional Source 1 file for country-level metrics",
    )
    score_parser.add_argument(
        "--temporary-directory",
        default=None,
        help="parent directory for the temporary disk-backed evaluation database",
    )
    score_parser.add_argument(
        "--aligned",
        action="store_true",
        help="use the faster constant-memory scorer when all three files share row order",
    )
    score_parser.set_defaults(function=_score)

    empty_parser = subparsers.add_parser(
        "empty-baseline",
        help="write valid all-singleton matching and candidate TSV files",
    )
    empty_parser.add_argument("--source1", required=True)
    empty_parser.add_argument("--output-directory", required=True)
    empty_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output files explicitly",
    )
    empty_parser.set_defaults(function=_empty_baseline)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="normalize Source 2/3 into an indexed train or test SQLite artifact",
    )
    prepare_parser.add_argument("--dataset-root", required=True)
    prepare_parser.add_argument("--output-directory", required=True)
    prepare_parser.add_argument("--mode", choices=("train", "test"), required=True)
    prepare_parser.add_argument("--seed", type=int, default=2026)
    prepare_parser.add_argument("--max-records-per-source", type=int, default=None)
    prepare_parser.add_argument("--max-queries", type=int, default=None)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.set_defaults(function=_prepare)

    candidate_parser = subparsers.add_parser(
        "candidates",
        help="retrieve candidates, evaluate blocking, and write pair features",
    )
    candidate_parser.add_argument("--source1", required=True)
    candidate_parser.add_argument("--index", required=True)
    candidate_parser.add_argument("--output-directory", required=True)
    candidate_parser.add_argument("--mode", choices=("train", "test"), required=True)
    candidate_parser.add_argument("--max-candidates", type=int, default=40)
    candidate_parser.add_argument("--train-negatives", type=int, default=12)
    candidate_parser.add_argument("--exact-pool-limit", type=int, default=2000)
    candidate_parser.add_argument("--approximate-pool-limit", type=int, default=800)
    candidate_parser.add_argument("--max-queries", type=int, default=None)
    candidate_parser.add_argument("--overwrite", action="store_true")
    candidate_parser.set_defaults(function=_candidates)

    train_parser = subparsers.add_parser(
        "train",
        help="train logistic and histogram-gradient-boosted pair models",
    )
    train_parser.add_argument("--pairs", required=True)
    train_parser.add_argument("--output-directory", required=True)
    train_parser.add_argument("--max-training-rows", type=int, default=2_000_000)
    train_parser.add_argument("--seed", type=int, default=2026)
    train_parser.add_argument("--overwrite", action="store_true")
    train_parser.set_defaults(function=_train)

    calibration_parser = subparsers.add_parser(
        "calibrate",
        help="select the model/threshold on calibration and score holdout once",
    )
    calibration_parser.add_argument("--pairs", required=True)
    calibration_parser.add_argument("--model-directory", required=True)
    calibration_parser.add_argument("--output-directory", required=True)
    calibration_parser.add_argument("--overwrite", action="store_true")
    calibration_parser.set_defaults(function=_calibrate)

    prediction_parser = subparsers.add_parser(
        "predict",
        help="score test pairs and write both required output TSV files",
    )
    prediction_parser.add_argument("--pairs", required=True)
    prediction_parser.add_argument("--selection", required=True)
    prediction_parser.add_argument("--output-directory", required=True)
    prediction_parser.add_argument("--overwrite", action="store_true")
    prediction_parser.set_defaults(function=_predict)

    validate_parser = subparsers.add_parser(
        "validate-outputs",
        help="verify row alignment, ID lists, and final-match subset rules",
    )
    validate_parser.add_argument("--source1", required=True)
    validate_parser.add_argument("--matching", required=True)
    validate_parser.add_argument("--candidates", required=True)
    validate_parser.set_defaults(function=_validate_outputs)

    package_parser = subparsers.add_parser(
        "package",
        help="create the final code/output/documentation submission archive",
    )
    package_parser.add_argument("--project-root", required=True)
    package_parser.add_argument("--prediction-directory", required=True)
    package_parser.add_argument("--documentation", required=True)
    package_parser.add_argument("--archive", required=True)
    package_parser.add_argument("--overwrite", action="store_true")
    package_parser.set_defaults(function=_package)

    all_parser = subparsers.add_parser(
        "run-all",
        help="run the entire resumable classical pipeline",
    )
    all_parser.add_argument("--dataset-root", required=True)
    all_parser.add_argument("--work-root", required=True)
    all_parser.add_argument("--seed", type=int, default=2026)
    all_parser.add_argument("--max-candidates", type=int, default=40)
    all_parser.add_argument("--train-negatives", type=int, default=12)
    all_parser.add_argument("--max-training-rows", type=int, default=2_000_000)
    all_parser.add_argument(
        "--reclaim-disk",
        action="store_true",
        help="delete generated train/test index and pair files when no longer needed",
    )
    all_parser.set_defaults(function=_run_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (
        DataContractError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    return 2
