# Kaggle cells: complete classical pipeline

Attach the private raw challenge dataset first. Enable internet only to clone this
code repository and install code dependencies; never use it for entity lookup or data
augmentation.

## Cell 1 — clone a reproducible revision

```python
from pathlib import Path
import os
import subprocess
import sys

REPO_URL = "https://github.com/Ayush-Mishra-0018/AMAZON_ML_CHALLENGE26.git"
BRANCH = "main"
PINNED_COMMIT = ""  # Set to the pushed commit SHA used for the final run.
REPO_DIR = Path("/kaggle/working/AMAZON_ML_CHALLENGE26")

if not REPO_DIR.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO_DIR)],
        check=True,
    )
elif not (REPO_DIR / ".git").is_dir():
    raise RuntimeError(f"{REPO_DIR} exists but is not a Git repository")

if PINNED_COMMIT:
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", PINNED_COMMIT],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "checkout", "--detach", "FETCH_HEAD"],
        check=True,
    )

PROJECT_ROOT = REPO_DIR / "saheem"
COMMIT_SHA = subprocess.check_output(
    ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
).strip()
print("commit:", COMMIT_SHA)
```

## Cell 2 — install code dependencies

```python
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
        "-r",
        str(PROJECT_ROOT / "code/business_entity_resolution/requirements.txt"),
    ],
    check=True,
)
```

Restart the kernel after this cell only if Kaggle reports that an already-imported
package was replaced.

## Cell 3 — discover inputs and initialize writable stages

```python
import json

bootstrap = subprocess.run(
    [
        sys.executable,
        str(PROJECT_ROOT / "kaggle/scripts/bootstrap.py"),
        "--input-root", "/kaggle/input",
        "--repo-root", str(PROJECT_ROOT),
        "--work-root", "/kaggle/working/amazon-er",
    ],
    check=True,
    capture_output=True,
    text=True,
)
context = json.loads(bootstrap.stdout)
SOURCE_ROOT = Path(context["source_root"])
DATA_ROOT = Path(context["data_root"])
WORK_ROOT = Path(context["work_root"])
RUN_ENV = os.environ.copy()
RUN_ENV["PYTHONPATH"] = str(SOURCE_ROOT) + os.pathsep + RUN_ENV.get("PYTHONPATH", "")
print(json.dumps(context, indent=2))
```

## Cell 4 — verify code and data contract

```python
subprocess.run(
    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
    cwd=PROJECT_ROOT,
    env=RUN_ENV,
    check=True,
)
subprocess.run(
    [
        sys.executable, "-m", "entity_resolution", "inspect",
        "--dataset-root", str(DATA_ROOT),
        "--max-rows", "10000",
    ],
    cwd=PROJECT_ROOT,
    env=RUN_ENV,
    check=True,
)
```

## Cell 5 — run every training and inference stage

```python
subprocess.run(
    [
        sys.executable, "-m", "entity_resolution", "run-all",
        "--dataset-root", str(DATA_ROOT),
        "--work-root", str(WORK_ROOT),
        "--max-candidates", "40",
        "--train-negatives", "12",
        "--max-training-rows", "2000000",
        "--reclaim-disk",
    ],
    cwd=PROJECT_ROOT,
    env=RUN_ENV,
    check=True,
)
```

This cell is resumable. Rerunning it reuses atomic completed stages. `--reclaim-disk`
removes generated training intermediates after calibration and test intermediates
after output validation, leaving the models, reports, and final submission files.

## Cell 6 — inspect final metrics and validate outputs

```python
report = json.loads((WORK_ROOT / "run_all_report.json").read_text())
selection = json.loads((WORK_ROOT / "05_validation/selection.json").read_text())
print(json.dumps(selection, indent=2))
print(json.dumps(report["output_validation"], indent=2))

subprocess.run(
    [
        sys.executable, "-m", "entity_resolution", "validate-outputs",
        "--source1", str(DATA_ROOT / "test/test_source1.tsv"),
        "--matching", str(WORK_ROOT / "07_submission/matching_results.tsv"),
        "--candidates", str(WORK_ROOT / "07_submission/candidate_pairs.tsv"),
    ],
    cwd=PROJECT_ROOT,
    env=RUN_ENV,
    check=True,
)
```

## Cell 7 — create the final archive

Fill in team details and final report metrics in `Documentation_template.md` before the
final archive run.

```python
subprocess.run(
    [
        sys.executable, "-m", "entity_resolution", "package",
        "--project-root", str(PROJECT_ROOT),
        "--prediction-directory", str(WORK_ROOT / "07_submission"),
        "--documentation", str(PROJECT_ROOT / "Documentation_template.md"),
        "--archive", str(WORK_ROOT / "amazon_ml_submission.zip"),
    ],
    cwd=PROJECT_ROOT,
    env=RUN_ENV,
    check=True,
)
print(WORK_ROOT / "amazon_ml_submission.zip")
```
