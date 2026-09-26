"""Create the final reproducibility archive from verified pipeline artifacts."""

from __future__ import annotations

import importlib.metadata
import shutil
import tempfile
import zipfile
from pathlib import Path

from .tracking import write_json_atomic

_RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "pyarrow",
    "scikit-learn",
    "rapidfuzz",
    "joblib",
    "scipy",
)


def _runtime_requirements() -> str:
    lines = []
    for distribution in _RUNTIME_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"cannot package before required distribution is installed: {distribution}"
            ) from error
        lines.append(f"{distribution}=={version}")
    return "\n".join(lines) + "\n"


def create_submission_archive(
    project_root: str | Path,
    prediction_directory: str | Path,
    documentation_path: str | Path,
    archive_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    project_root = Path(project_root)
    prediction_directory = Path(prediction_directory)
    documentation_path = Path(documentation_path)
    archive_path = Path(archive_path)
    matching = prediction_directory / "matching_results.tsv"
    candidates = prediction_directory / "candidate_pairs.tsv"
    required = (
        project_root / "code/business_entity_resolution/src",
        matching,
        candidates,
        documentation_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("cannot package missing artifact(s): " + ", ".join(missing))
    if archive_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_name(f".{archive_path.name}.tmp")
    temporary_archive.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="amazon-er-package-",
        dir=archive_path.parent,
    ) as directory:
        package_root = Path(directory) / "submission"
        output_root = package_root / "output"
        code_root = package_root / "code/business_entity_resolution"
        output_root.mkdir(parents=True)
        shutil.copy2(matching, output_root / matching.name)
        shutil.copy2(candidates, output_root / candidates.name)
        shutil.copytree(
            project_root / "code/business_entity_resolution/src",
            code_root / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(project_root / "code/business_entity_resolution/README.md", code_root / "README.md")
        (code_root / "requirements.txt").write_text(
            _runtime_requirements(),
            encoding="utf-8",
        )
        shutil.copy2(documentation_path, package_root / "Documentation_template.md")
        with zipfile.ZipFile(
            temporary_archive,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in sorted(package_root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(package_root))
    temporary_archive.replace(archive_path)
    report = {
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "matching_path": str(matching),
        "candidate_path": str(candidates),
        "documentation_path": str(documentation_path),
    }
    write_json_atomic(archive_path.with_suffix(".report.json"), report)
    return report
