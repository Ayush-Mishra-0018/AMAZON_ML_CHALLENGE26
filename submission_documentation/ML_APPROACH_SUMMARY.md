# Business Entity Resolution - ML approach summary

**Team:** [FILL BEFORE SUBMISSION]
**Team members:** [FILL BEFORE SUBMISSION]
**Final code commit:** [FILL BEFORE SUBMISSION]
**Leaderboard submission identifier:** [FILL BEFORE SUBMISSION]

## Problem and design objective

The task is to associate every deduplicated Source 1 business with zero or more
records from Source 2 and Source 3. Names and addresses contain abbreviations,
transliterations, omissions, reordered components, typographical errors, domain or
trade names, and misleading near-duplicates. Test data also introduces France,
which is absent from the labelled training countries.

Evaluation uses macro F0.5 per Source 1 entity. Precision receives twice the weight
of recall, and a true singleton receives full credit only when the prediction is
empty. The system is therefore designed to retain high candidate recall while making
the final decision conservatively enough to avoid false merges.

## Approach

The solution is a country-partitioned, record-centric entity-resolution pipeline:

1. Read the official TSV files with an explicit tab separator and cache them as
   Parquet for repeatable, faster subsequent runs.
2. Normalize names and addresses using lowercase transliteration, punctuation and
   whitespace normalization, number normalization, domain-aware name tokens, and
   lightweight phonetic skeletons.
3. Index Source 1 using deterministic 64-bit conjunctive blocking keys. Keys combine
   pairs of address tokens, name-address token pairs, and pairs of name tokens.
4. Stream Source 2 and Source 3 records through the index. Very common keys are
   ignored, and each record retains its strongest Source 1 candidates according to
   inverse-frequency-weighted blocking evidence.
5. Compute pairwise name, address, number-consistency, source, and retrieval features.
6. Score candidate pairs with a first LightGBM classifier.
7. Re-score plausible pairs with a second LightGBM model using Source 2/Source 3
   consensus features. A genuine member often agrees with other confident members of
   the same Source 1 entity, whereas a near-copy decoy frequently does not.
8. Select a precision-biased probability threshold using grouped out-of-fold macro
   F0.5, including singleton Source 1 entities.
9. Write one aligned row for every test Source 1 entity to both required TSV files,
   with final matches guaranteed to be a subset of the candidate set.

All countries are treated as open string labels. The code partitions by the observed
country value but never restricts the allowed values to US and India, so France is
processed by the same generic path.

## Models and features

Both stages use LightGBM gradient-boosted decision trees. This model family captures
nonlinear rules such as “high name similarity is trustworthy only when address
numbers do not conflict” while remaining practical for millions of sparse candidate
pairs. LightGBM is MIT licensed and is far below the competition's 8-billion-parameter
limit.

The first-stage features include multiple RapidFuzz name and address similarities,
phonetic-skeleton similarities, name/address lengths, missing-address and non-Latin
flags, exact and partial number agreement, Source 2 versus Source 3 identity, blocking
key counts and rarity, candidate rank, and the gap between the best and next blocking
scores. The second stage adds support from other confident records, cross-source
support, best name/address agreement with another member, number agreement, and the
number of other confident members.

## Validation and experiments

Folds are grouped by normalized Source 1 name so exact-name chains do not cross
training and validation folds. Macro F0.5 is computed per Source 1 entity using all
true links, including links missed during blocking. The pipeline separately reports
candidate-pair recall and a perfect-scorer ceiling, preventing classifier performance
from hiding a weak blocker.

Prototype ablations on labelled US and India data showed that name-only and
address-only features were insufficient. Combining both fields produced the largest
gain; retrieval evidence and Source 2/Source 3 consensus improved the precision-heavy
score further. One-owner assignment and purely listwise features produced negligible
improvements and were not enabled. Exact prototype figures and spaces for the final
reproducible run are recorded in `EXPERIMENTS_AND_RESULTS.md`.

## Conclusion

The main contribution is a scalable blocker that uses conjunctions rather than
ambiguous single tokens, followed by a two-stage classifier designed around the
challenge's dominant false-positive pattern: near-copy orphan records. The final
threshold deliberately favors precision and singleton protection. Remaining risks
are candidate misses for severely corrupted records, distribution shift in the
unlabelled France partition, and uncertainty in the test orphan rate. No external
business lookup, geocoding, external database, or internet data augmentation is used.
