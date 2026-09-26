# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [fill before submission]
**Team Members:** [fill before submission]
**Submission Date:** [fill before submission]

## 1. Executive Summary

The solution uses multi-channel blocking followed by an interpretable classical pair
classifier. It retrieves candidates through exact normalized fields, address-number
keys, and character SimHash bands, then selects final match sets using a threshold
tuned directly for entity-level macro F0.5.

## 2. Methodology

### 2.1 Problem Analysis

Source 1 is deduplicated, while Sources 2 and 3 contain multiple noisy observations.
The training set contains 2,206,821 Source 1 entities and 7,638,365 positive links;
123,247 Source 1 entities are singletons. Important noise includes punctuation and
spacing, legal-suffix changes, accents, Indic scripts, transliteration, aliases,
missing addresses, reordered components, abbreviations, typos, and conflicting
street or unit numbers.

An all-pairs comparison is infeasible, and macro F0.5 places more value on avoiding
false merges than on recovering every possible link. Candidate recall and conservative
entity-level thresholding are therefore measured separately.

### 2.2 Solution Strategy

**Approach Type:** Multi-channel blocking plus classical pair classifier
**Core Innovation:** A disk-backed, open-country pipeline combining Unicode-preserving
normalization, exact/numeric/LSH retrieval, explicit retrieval-channel features, hard
negatives, and direct macro-F0.5 threshold selection.

The split is deterministic by Source 1 entity. Near-identical Source 1 records sharing
country, normalized core name, and normalized address receive the same split. The
fractions are 80% training, 10% calibration, and 10% final holdout.

## 3. Candidate Generation

Targets are stored in source-agnostic, country-partitioned SQLite indices. Country is
treated as an open string rather than a fixed one-hot set, allowing France at test
time. Retrieval unions:

- exact normalized, Latin-folded, and suffix-stripped names;
- exact canonicalized addresses;
- four 16-bit SimHash bands for names and addresses;
- first/last address-number plus trailing locality token.

The broad pool is ranked by name, address, channel, and number agreement. At most 40
candidates per Source 1 entity reach the model by default. Candidate reports record
link recall, all-true-match coverage, candidate counts, and oracle macro F0.5. Missed
positives are injected only into training rows; calibration, holdout, and test remain
fully realistic.

## 4. Matching Model

Features include exact normalized views, edit similarity, token-sort and token-set
similarity, Jaccard/containment, length ratios, address missingness, numeric equality,
numeric overlap/conflict, script overlap, source identity, retrieval channels, and
channel count.

Two models are trained on all sampled positives and high-ranked hard negatives:

- balanced logistic regression for a transparent linear baseline;
- histogram gradient boosting for nonlinear interactions and missing-data behavior.

The calibration split selects both the model and probability threshold by exact macro
F0.5. Ties prefer the higher threshold to preserve precision. The holdout is evaluated
once after selection, overall and by country.

## 5. Results and Error Analysis

Copy the final values from:

- `02_candidates/train_candidate_report.json` for blocker recall and oracle F0.5;
- `05_validation/calibration_report.json` for calibration and holdout metrics;
- `07_submission/prediction_report.json` for prediction counts.

**Holdout macro F0.5:** [fill from calibration report]
**Candidate link recall:** [fill from candidate report]
**Singleton accuracy:** [fill from calibration report]

Expected remaining false positives are generic-name collisions and records with strong
name agreement but corrupted address numbers. Expected false negatives include severe
aliases, cross-script names with weak addresses, and matches that share no blocking
key. These categories should be confirmed from saved holdout errors before submission.

## 6. Conclusion

The pipeline is designed for high precision, bounded memory, and exact reproducibility
on Kaggle. It uses no external entity data and produces both the scored result file and
the exact candidates evaluated by the model.

## Appendix A. Code Artefacts

All implementation is under `code/business_entity_resolution/src/entity_resolution`.
The primary entry point is:

```bash
python -m entity_resolution run-all --dataset-root DATASET --work-root WORK
```

Every stage is also independently runnable and writes an atomic JSON report. The final
archive command copies code, outputs, this methodology, and exact runtime dependency
versions into the required competition structure.
