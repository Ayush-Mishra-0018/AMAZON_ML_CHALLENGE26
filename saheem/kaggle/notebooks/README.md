# Notebook stages

Keep notebooks thin: resolve paths, load a checked configuration, call package code,
and save metrics. Put reusable algorithms and tests in the shared Python package.

Recommended notebook sequence:

1. `00_environment_check.ipynb`
   - run `bootstrap.py`;
   - print Python/package versions, CPU count, RAM, disk, and accelerator visibility;
   - validate the seven raw input files.
2. `01_prepare_data.ipynb`
   - normalize records in chunks;
   - create the permanent entity-level split manifest;
   - write partitioned Parquet files to `01_prepared/`.
3. `02_generate_candidates.ipynb`
   - build country/source-partitioned retrieval indices;
   - generate candidate shards;
   - report link recall, all-match coverage, and candidate-count statistics.
4. `03_build_features_and_train.ipynb`
   - compute pair features;
   - sample hard negatives;
   - train deterministic, logistic, and boosted-tree baselines;
   - write models and experiment records.
5. `04_calibrate_and_validate.ipynb`
   - tune thresholds and the singleton gate on calibration entities;
   - evaluate once on the untouched holdout;
   - save error-analysis samples and the selected configuration.
6. `05_full_train_and_infer.ipynb`
   - retrain with the selected design;
   - run chunked test inference;
   - write `matching_results.tsv` and `candidate_pairs.tsv`.
7. `06_validate_and_package.ipynb`
   - run the official validator;
   - verify row counts, IDs, and final-match subset rules;
   - copy final files to `07_submission/` and create the submission archive.

Splitting stages protects progress from notebook time limits. Do not use the holdout
notebook repeatedly to select features or thresholds; that would turn the holdout into
another training set.
