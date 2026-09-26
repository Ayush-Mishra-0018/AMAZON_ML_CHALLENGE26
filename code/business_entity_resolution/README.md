# Business Entity Resolution: reproduction guide

Pipeline: normalise -> record-centric conjunctive blocking -> LightGBM pair scorer -> S2<->S3 consensus re-scorer -> precision-biased threshold -> `output/*.tsv`.

## Layout
```
code/business_entity_resolution/
  business_entity_resolution.ipynb   # narrative + evidence + the driver cells (run this)
  build_notebook.py                  # regenerates the notebook
  src/er/                            # all source code
    config.py     paths + every hyper-parameter (incl. the accuracy gate)
    textnorm.py   normalisation, phonetic skeleton, tokenisers
    keys.py       hashed conjunctive blocking keys
    blocking.py   record-centric candidate generation (memory-mapped S1 index)
    features.py   vectorised pair features
    relational.py stage-2 S2<->S3 consensus features
    decision.py   threshold selection
    evaluate.py   macro F0.5
    pipeline.py   train / infer / write_outputs
    __main__.py   CLI
```

## Environment
Python 3.12. `pip install -r requirements.txt`. Tested versions are pinned; no GPU is required.
Only the provided data is used; no external lookups. Model: LightGBM (MIT).

## Data location
Set `ER_DATA_DIR` to the folder that contains `train/` and `test/` (the official `dataset/` folder). If unset, the code searches the repository for it. Intermediate files go to `ER_WORK_DIR` (default `<repo>/work`), outputs to `ER_OUT_DIR` (default `<repo>/output`).

## Run
Option A: notebook (recommended): open `business_entity_resolution.ipynb` from this folder and run all cells.

Option B: command line, from `code/business_entity_resolution/src`:
```
python -m er train      # build universes, validate, fit, save models (work/models)
python -m er infer      # test -> output/matching_results.tsv + output/candidate_pairs.tsv
python -m er all
```

## Accuracy gate
`train` computes grouped out-of-fold macro F0.5 per country. `infer` **refuses to write outputs** unless every country is at least `Params.min_val_f05` (default 0.90). Adjust in `config.py`.

## Validate the files
```
python <student_resource>/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir <dataset>/test
```

## Notes
* Leaderboard upload: only `matching_results.tsv`. Suggested local file naming for archives: `SNO_matching_results_ModelUsed_AccuracyAchievedOnLeaderBoard_YourName.tsv` and `SNO_candidate_pairs_...tsv`.
* Memory: 16 GB machine. Reduce `Params.chunk_records` if RAM is tight.
* Runtime for the full test run has not been measured; expect hours.
* Set `Params.seed` for reproducibility (LightGBM and universe sampling use it).
