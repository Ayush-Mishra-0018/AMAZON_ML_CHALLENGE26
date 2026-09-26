#!/usr/bin/env python3
"""Resolve Kaggle mounts and create a reproducible writable stage tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REQUIRED_DATA_FILES = (
    "train/train_source1.tsv",
    "train/train_source2.tsv",
    "train/train_source3.tsv",
    "train/train_ground_truth.tsv",
    "test/test_source1.tsv",
    "test/test_source2.tsv",
    "test/test_source3.tsv",
)

STAGE_DIRECTORIES = (
    "00_environment",
    "01_prepared",
    "02_candidates",
    "03_features",
    "04_models",
    "05_validation",
    "06_predictions",
    "07_submission",
    "logs",
)


class BootstrapError(RuntimeError):
    """Raised when Kaggle inputs cannot be resolved unambiguously."""


def is_data_root(path: Path) -> bool:
    return path.is_dir() and all((path / relative).is_file() for relative in REQUIRED_DATA_FILES)


def discover_data_root(input_root: Path) -> Path:
    """Find the unique directory containing all seven competition data files."""

    candidates: set[Path] = set()
    if is_data_root(input_root):
        candidates.add(input_root.resolve())
    if input_root.is_dir():
        for source1_file in input_root.rglob("train_source1.tsv"):
            candidate = source1_file.parent.parent
            if is_data_root(candidate):
                candidates.add(candidate.resolve())
    if not candidates:
        raise BootstrapError(
            f"no complete challenge dataset found below {input_root}; "
            "pass --data-root explicitly"
        )
    if len(candidates) > 1:
        choices = ", ".join(str(path) for path in sorted(candidates))
        raise BootstrapError(
            f"multiple challenge datasets found: {choices}; pass --data-root explicitly"
        )
    return candidates.pop()


def is_repo_root(path: Path) -> bool:
    package = path / "code/business_entity_resolution/src/entity_resolution"
    return package.is_dir() and (package / "__init__.py").is_file()


def discover_repo_root(input_root: Path) -> Path:
    """Find the unique uploaded project root containing the shared package."""

    local_root = Path(__file__).resolve().parents[2]
    candidates: set[Path] = set()
    if is_repo_root(local_root):
        candidates.add(local_root)
    if input_root.is_dir():
        for package in input_root.rglob("entity_resolution"):
            if package.is_dir() and package.parent.name == "src":
                candidate = package.parents[3]
                if is_repo_root(candidate):
                    candidates.add(candidate.resolve())
    if not candidates:
        raise BootstrapError(
            f"no project checkout found below {input_root}; pass --repo-root explicitly"
        )
    if len(candidates) > 1:
        choices = ", ".join(str(path) for path in sorted(candidates))
        raise BootstrapError(
            f"multiple project checkouts found: {choices}; pass --repo-root explicitly"
        )
    return candidates.pop()


def resolve_explicit_path(value: str | None, *, label: str) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise BootstrapError(f"{label} does not exist: {path}")
    return path


def create_run_context(data_root: Path, repo_root: Path, work_root: Path) -> dict[str, object]:
    if not is_data_root(data_root):
        missing = [relative for relative in REQUIRED_DATA_FILES if not (data_root / relative).is_file()]
        raise BootstrapError(f"data root is incomplete; missing: {', '.join(missing)}")
    if not is_repo_root(repo_root):
        raise BootstrapError(f"repository root does not contain the shared package: {repo_root}")

    work_root.mkdir(parents=True, exist_ok=True)
    stages: dict[str, str] = {}
    for stage in STAGE_DIRECTORIES:
        stage_path = work_root / stage
        stage_path.mkdir(exist_ok=True)
        stages[stage] = str(stage_path)

    context: dict[str, object] = {
        "data_root": str(data_root),
        "repo_root": str(repo_root),
        "source_root": str(repo_root / "code/business_entity_resolution/src"),
        "work_root": str(work_root),
        "stages": stages,
    }
    manifest = work_root / "run_context.json"
    temporary = work_root / ".run_context.json.tmp"
    temporary.write_text(
        json.dumps(context, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest)
    context["manifest"] = str(manifest)
    return context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        default=os.environ.get("KAGGLE_INPUT_ROOT", "/kaggle/input"),
        help="root containing attached Kaggle datasets",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("AMAZON_ER_DATA_ROOT"),
        help="directory containing train/ and test/; auto-discovered when omitted",
    )
    parser.add_argument(
        "--repo-root",
        default=os.environ.get("AMAZON_ER_REPO_ROOT"),
        help="uploaded repository root; auto-discovered when omitted",
    )
    parser.add_argument(
        "--work-root",
        default=os.environ.get("AMAZON_ER_WORK_ROOT", "/kaggle/working/amazon-er"),
        help="writable directory for stage outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        input_root = Path(args.input_root).expanduser().resolve()
        data_root = resolve_explicit_path(args.data_root, label="data root")
        repo_root = resolve_explicit_path(args.repo_root, label="repository root")
        data_root = data_root or discover_data_root(input_root)
        repo_root = repo_root or discover_repo_root(input_root)
        work_root = Path(args.work_root).expanduser().resolve()
        context = create_run_context(data_root, repo_root, work_root)
    except (BootstrapError, OSError) as error:
        raise SystemExit(f"bootstrap failed: {error}") from error
    print(json.dumps(context, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
