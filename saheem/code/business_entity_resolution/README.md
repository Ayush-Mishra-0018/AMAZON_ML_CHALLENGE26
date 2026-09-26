# Business entity resolution pipeline

The package under `src/entity_resolution` is the self-contained implementation used
to reproduce data preparation, candidate generation, model training, calibration,
test inference, validation, and final packaging.

The implementation uses only the supplied challenge data. It does not call business
lookup services, geocoders, external databases, or internet data sources.

## Installation

Python 3.11 or newer is required:

```bash
python -m pip install -r code/business_entity_resolution/requirements.txt
export PYTHONPATH="$PWD/code/business_entity_resolution/src"
```

## Complete run

```bash
python -m entity_resolution run-all \
  --dataset-root /path/to/dataset \
  --work-root /path/to/writable/amazon-er \
  --max-candidates 40 \
  --train-negatives 12 \
  --max-training-rows 2000000
```

The runner is resumable: it reuses atomic completed artifacts and reruns missing
stages. Add `--reclaim-disk` on Kaggle to remove the generated training index and
training pair table after model calibration.

Final output is written under:

```text
<work-root>/07_submission/matching_results.tsv
<work-root>/07_submission/candidate_pairs.tsv
```

## Individual stages

```bash
# Train preparation and blocking index
python -m entity_resolution prepare \
  --dataset-root /path/to/dataset \
  --mode train \
  --output-directory work/01_prepared

# Candidate generation, blocker evaluation, features, and hard negatives
python -m entity_resolution candidates \
  --source1 /path/to/dataset/train/train_source1.tsv \
  --index work/01_prepared/train_index.sqlite3 \
  --mode train \
  --output-directory work/02_candidates

# Logistic and histogram-gradient-boosting models
python -m entity_resolution train \
  --pairs work/02_candidates/train_pairs.parquet \
  --output-directory work/04_models

# Macro-F0.5 threshold calibration and untouched holdout evaluation
python -m entity_resolution calibrate \
  --pairs work/02_candidates/train_pairs.parquet \
  --model-directory work/04_models \
  --output-directory work/05_validation
```

Repeat `prepare` and `candidates` with `--mode test`, then run:

```bash
python -m entity_resolution predict \
  --pairs work/02_candidates/test_pairs.parquet \
  --selection work/05_validation/selection.json \
  --output-directory work/07_submission
```

Every stage writes a JSON report containing its configuration, metrics, paths, and
runtime. Candidate reports include link recall, all-true-match coverage, mean
candidate count, and oracle macro F0.5. Calibration reports include the full threshold
curve and holdout metrics overall and by country.

## Final validation and archive

```bash
python -m entity_resolution validate-outputs \
  --source1 /path/to/dataset/test/test_source1.tsv \
  --matching work/07_submission/matching_results.tsv \
  --candidates work/07_submission/candidate_pairs.tsv

python -m entity_resolution package \
  --project-root . \
  --prediction-directory work/07_submission \
  --documentation Documentation_template.md \
  --archive work/team_submission.zip
```

The packager records the exact installed versions of every runtime dependency inside
the archive.

## Tests

```bash
PYTHONPATH=code/business_entity_resolution/src \
  python -m unittest discover -s tests -v
```
