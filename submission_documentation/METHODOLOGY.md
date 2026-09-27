# Detailed methodology

## 1. Task formulation

For every Source 1 entity, the pipeline predicts a set of matching Source 2 and Source
3 record identifiers. This is a one-to-many set prediction problem rather than a
single-label classification problem. A Source 1 entity may legitimately have an empty
set.

The full Cartesian product is computationally infeasible. Candidate generation first
reduces the comparison space; a supervised pair classifier then estimates whether
each retained pair is a true link. Predictions are aggregated back into one set per
Source 1 entity and thresholded for macro F0.5.

## 2. Data handling

- TSV files are read using `sep="\t"`, string dtypes, disabled default NA conversion,
  and no CSV quoting assumptions.
- Source is derived from the input file and entity-ID prefix.
- Source 2 and Source 3 are concatenated into a pool with an explicit source flag.
- Parsed tables are cached as Parquet under the working directory. Raw challenge
  files are never modified.
- Country is used as a hard blocking partition because labelled true pairs agree on
  country. Country values are discovered from the data rather than hard-coded.

## 3. Text normalization

Names and addresses are lowercased, transliterated to ASCII and reduced to normalized
alphanumeric tokens. The pipeline retains several complementary views:

- ordinary normalized tokens;
- a joined-name prefix to connect domain/handle forms with spaced business names;
- phonetic skeleton tokens that collapse selected transliteration and spelling
  variations;
- normalized numeric tokens with leading zeros removed;
- two-digit numeric suffix tokens for common house-number corruption patterns.

The normalizer is deterministic and performs no network or external entity lookup.

## 4. Candidate generation

### 4.1 Record-centric orientation

Source 1 is the deduplicated reference and is indexed once. Source 2 and Source 3
records are streamed against that index. This orientation avoids issuing millions of
independent database queries and permits bounded-memory processing of the larger pool.

### 4.2 Conjunctive keys

Three key families are generated:

- `A`: unordered pairs of address tokens;
- `X`: a name token combined with an address token;
- `N`: unordered pairs of name tokens.

Each typed key is mapped to a deterministic 64-bit hash. The Source 1 hashes are
sorted and stored in NumPy memory-mapped arrays, allowing worker processes to share
the index without copying it into every process.

Single tokens such as “services”, “road” or a city name have poor discriminative
power. Conjunctions provide stronger identity evidence while still tolerating missing
or reordered fields. Keys whose Source 1 document frequency exceeds a configured cap
are ignored.

### 4.3 Ranking and candidate budget

For a pool record and Source 1 candidate, matching keys contribute an
inverse-document-frequency-style score. The blocker records the contribution and
count for each key family, the rarest shared key, the number of possible Source 1
candidates, candidate rank, score share and the gap between the strongest candidates.
Only the highest-ranked candidates per pool record are retained.

The candidate budget is evaluated using recall-at-K on labelled data. The final match
set is always selected from the exact candidate set written to `candidate_pairs.tsv`.

## 5. Pairwise feature engineering

### 5.1 Name features

- character ratio;
- token-set ratio;
- partial ratio;
- token-sort ratio;
- phonetic-skeleton ratio and token-set ratio;
- normalized name length;
- non-Latin-source flag;
- domain/handle-like name flag.

### 5.2 Address features

- token-set, character and partial ratios;
- normalized address length and empty-address flag;
- Jaccard similarity of numeric component sets;
- exact equality of non-empty numeric component sets.

### 5.3 Retrieval features

- score and shared-key count for `A`, `X` and `N` keys;
- total key score;
- candidate rank and candidate count for the pool record;
- score share and best-versus-second-best score gap;
- key-frequency measures relative to the blocking cap;
- Source 2/Source 3 indicator.

## 6. Two-stage matching model

### 6.1 Stage 1

A LightGBM binary classifier scores every retained candidate using the pairwise and
retrieval features. Very low-scoring pairs are removed before relational processing.
This reduces memory and computation while preserving nearly all plausible matches.

### 6.2 Stage 2 consensus

The second LightGBM model augments the Stage 1 score and features with relational
evidence among records proposed for the same Source 1 entity:

- support from other confident records;
- support specifically crossing Source 2 and Source 3;
- strongest name and address agreement with another confident member;
- numeric agreement with another member;
- number of other confident members.

This stage targets decoy records that resemble Source 1 directly but disagree with
the independently noisy Source 2/Source 3 cluster.

## 7. Validation and decision rule

Training produces grouped out-of-fold predictions. The metric implementation computes
F0.5 separately for each Source 1 entity and then averages, including entities with no
true links. A singleton scores 1 only when no match is predicted.

The validation report includes:

- candidate link recall;
- perfect-scorer candidate ceiling;
- macro F0.5 by threshold;
- per-country F0.5;
- singleton and non-singleton F0.5;
- micro precision and recall as diagnostics.

The selected threshold is the highest threshold within a small tolerance of the best
validation F0.5. This chooses the precision-favoring edge of a statistically flat
region, consistent with beta = 0.5.

## 8. Inference and outputs

Inference repeats the same normalization, blocking and feature transformations. It
processes every country label found in test data, including unseen France. Candidate
and final match lists are aggregated in original Source 1 order. Empty strings encode
singletons.

Before submission, both TSV files must be checked for exact headers, row count and
order, duplicate identifiers, valid S2/S3 membership, and the invariant that every
final match is also a candidate.

## 9. Reproducibility and fair play

- Random seeds and model parameters are centralized in configuration.
- Package versions are pinned in `code/business_entity_resolution/requirements.txt`.
- The modular entry point supports separate train and inference runs.
- Training reports and inference diagnostics are written to the working directory.
- Only challenge-provided records and labels are used.
- No business registry, entity-resolution API, geocoder, external database or
  internet-derived business information is queried.
- The declared final model is LightGBM under the MIT license.
- Optional downloaded foundation-model code is excluded from the declared final
  method; use `USE_FM = False` for the final run.

## 10. Known limitations

- France has no labelled validation set.
- Extremely corrupted pairs with no reliable shared token conjunction can be missed
  during blocking.
- Generic names combined with empty or fragmentary addresses can be inherently
  ambiguous.
- Validation and leaderboard distributions may differ in orphan prevalence and the
  density of similar Source 1 businesses.
