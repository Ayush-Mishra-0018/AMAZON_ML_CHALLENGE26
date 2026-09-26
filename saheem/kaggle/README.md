# Kaggle training layout

Kaggle mounts attached datasets below `/kaggle/input` as read-only files. Code can
read from those mounts, but caches, models, and predictions must be written below
`/kaggle/working`. Notebook sessions are temporary, so each expensive stage should be
saved as a notebook version and attached as a private dataset to the next stage.

## Repository layout

```text
kaggle/
├── configs/
│   ├── cpu_baseline.json
│   └── gpu_optional.json
├── datasets/
│   └── README.md
├── notebooks/
│   └── README.md
├── scripts/
│   └── bootstrap.py
├── working/                 # local/Kaggle generated stage products; ignored
└── output/                  # copied final submission files; ignored
```

The shared Python package remains at
`code/business_entity_resolution/src/entity_resolution`. Do not copy its logic into
notebooks; notebooks should import the package and orchestrate stages.

## Recommended Kaggle inputs

Attach two private datasets to each notebook:

1. A raw-data dataset containing the supplied `dataset/train` and `dataset/test`
   directories.
2. A code dataset containing this repository snapshot.

Do not attach external entity, address, geocoding, or business datasets. Disable
internet access for the notebook after any allowed environment setup. The matching
pipeline must use only the challenge data.

## Bootstrap cell

From the uploaded repository root, run:

```python
import json
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "kaggle/scripts/bootstrap.py"],
    check=True,
    capture_output=True,
    text=True,
)
context = json.loads(result.stdout)
sys.path.insert(0, context["source_root"])
context
```

If discovery finds multiple copies of the raw data or repository, pass explicit
paths:

```bash
python kaggle/scripts/bootstrap.py \
  --data-root /kaggle/input/<raw-dataset>/dataset \
  --repo-root /kaggle/input/<code-dataset>/saheem \
  --work-root /kaggle/working/amazon-er
```

The bootstrap creates this writable runtime tree:

```text
/kaggle/working/amazon-er/
├── 00_environment/
├── 01_prepared/
├── 02_candidates/
├── 03_features/
├── 04_models/
├── 05_validation/
├── 06_predictions/
├── 07_submission/
├── logs/
└── run_context.json
```

`run_context.json` records resolved paths. Each later stage should also write its
configuration, seed, metrics, runtime, and input artifact version to its own folder.

## Accelerator choice

Use `configs/cpu_baseline.json` for preprocessing, blocking, similarity features,
logistic regression, and tree models. These stages are CPU- and memory-oriented.

Use `configs/gpu_optional.json` only for a later neural reranker. A GPU does not make
Polars parsing, sparse blocking, or classical feature computation automatically
faster, so requesting a GPU for every notebook wastes the limited accelerator quota.
