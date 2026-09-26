# Kaggle datasets

Create private Kaggle datasets rather than placing data in this repository.

## Raw challenge dataset

Preserve this relative structure when uploading the supplied files:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Keep the dataset private. It contains competition-provided data and should not be
published or mixed with any external lookup data.

## Code dataset

Upload the `saheem/` project directory as a second private dataset. A notebook can
then import the same tested package used locally. Refresh this dataset only at known
Git commits so every Kaggle run can record the exact commit or snapshot it used.

## Stage outputs

After a costly notebook stage finishes, save a notebook version and use its output as
the next notebook's input. Treat the files as immutable. This avoids rebuilding large
Parquet caches or candidate tables after a session timeout.
