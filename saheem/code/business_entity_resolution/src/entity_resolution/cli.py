"""Command-line entry points for data checks and local evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import DataContractError, profile_source
from .metrics import evaluate_aligned_files, evaluate_keyed_files
from .submission import write_empty_submission

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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (DataContractError, FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))
    return 2
