# Amazon ML Challenge 2026 — entity resolution

This repository contains a reproducible pipeline for matching each Source 1 business
to zero or more records from Sources 2 and 3. It uses only the competition data; it
does not call entity-lookup services, geocoders, external databases, or web sources.

## Implemented pipeline

The repository implements:

- strict, streaming TSV readers;
- Unicode-safe multi-view name and address normalization;
- the official entity-level macro F0.5 metric, including singleton handling;
- deterministic entity-level split assignment;
- disk-backed exact, numeric, and SimHash/LSH candidate generation;
- candidate-recall/oracle-F0.5 evaluation and hard-negative selection;
- interpretable name, address, number, script, and retrieval-channel features;
- logistic and histogram-gradient-boosted matching models;
- entity-level threshold calibration and untouched holdout evaluation;
- streaming test inference, output validation, and final archive creation;
- resumable Kaggle orchestration and stage reports.

## Run the checks

The current checkpoint uses only Python's standard library. From this directory:

```bash
PYTHONPATH=code/business_entity_resolution/src \
  python -m unittest discover -s tests -v
```

Inspect a bounded sample of the supplied dataset:

```bash
PYTHONPATH=code/business_entity_resolution/src \
  python -m entity_resolution inspect \
  --dataset-root ../../6ab10eb3b23ba_student_resource/student_resource/dataset \
  --max-rows 10000
```

Score predictions by Source 1 ID, regardless of row order:

```bash
PYTHONPATH=code/business_entity_resolution/src \
  python -m entity_resolution score \
  --truth ../../6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv \
  --predictions path/to/validation_predictions.tsv \
  --source1 ../../6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source1.tsv
```

The scorer uses a temporary SQLite database for an exact disk-backed join, keeping
memory bounded on the full dataset. Add `--aligned` only when truth, predictions, and
Source 1 deliberately share row order.

Create a format-correct all-singleton baseline:

```bash
PYTHONPATH=code/business_entity_resolution/src \
  python -m entity_resolution empty-baseline \
  --source1 ../../6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source1.tsv \
  --output-directory artifacts/train_empty
```

## Kaggle execution

Kaggle-specific configuration, runtime setup, and staged notebook conventions are in
[`kaggle/`](kaggle/README.md). The core implementation is shared with local runs; the
Kaggle layer only resolves mounted input paths and writable working directories.

The copy-paste notebook sequence is in
[`kaggle/notebooks/RUN_ALL_CELLS.md`](kaggle/notebooks/RUN_ALL_CELLS.md).

Run the local bootstrap smoke test with:

```bash
python kaggle/scripts/bootstrap.py \
  --data-root ../../6ab10eb3b23ba_student_resource/student_resource/dataset \
  --repo-root . \
  --work-root kaggle/working/amazon-er
```

## Data policy

Raw data, cached tables, candidate pairs, models, and predictions are generated
locally and excluded from Git. The final package will place reproducible code under
`code/business_entity_resolution/` and the two required TSV files under `output/`.
