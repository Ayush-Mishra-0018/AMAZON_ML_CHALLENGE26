# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** DataForge
**Team Members:** Ayush Mishra (Team Leader), Syed Naveed Mohammed, Md Mudassir Ali, Saheem Showkat Reshi
**Submission Date:** 27 September 2026

> Status of numbers: values below marked *(prototype)* were measured on the provided training data with prototype scripts. Fill the final validation score from `work/validation_report.json` after running the notebook; do not submit a score that the pipeline's accuracy gate did not produce.

---

## 1. Executive Summary
A country-partitioned, two-stage system: record-centric conjunctive-key blocking (hashed token-pair keys) feeds a LightGBM pair scorer; a second LightGBM stage re-scores using agreement between an S1 entity's other confident S2/S3 members; a precision-biased threshold decides, with empty output for singletons. Only the provided data and LightGBM (MIT) are used.

---

## 2. Methodology

### 2.1 Problem Analysis
* Scale: 2.2M S1 / 5.0M S2 / 5.3M S3 (train); 1.7M / 4.9M / 5.1M (test, includes France).
* Country label agrees on all 7.6M true train pairs; every matched S2/S3 id has one owning S1 (observed in labels).
* ~26.5% of S2/S3 records are orphans; 5.6% of S1 are singletons; mean 3.46 matches per S1.
* 39% of S1 names collide with another S1 at a different address, so name alone is insufficient.
* India: 18% of S2/S3 names are in Indic scripts (S1 is ASCII); some names are handles/domains or unrelated trade names; addresses can be fragments ("11, Calcutta, WB").
* Orphans are near-copy decoys of S1 records; the dominant false-positive type is a decoy with a perturbed house number/suffix. True pairs also contain number typos, so some ambiguity is irreducible.
* Test differs from train: France (unlabeled), a smaller S1 sample, and more pool records per S1 (cause unknown).

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage classifier (hybrid, no foundation model).  
**Core Innovation:** (a) conjunctive token-pair blocking keys with phonetic/number/domain fixes; (b) S2<->S3 consensus features to reject decoys; (c) an accuracy gate and precision-biased threshold plateau rule.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** hashed pairs of address tokens (A), name-token x address-token (X), name-token pairs (N); numbers lose leading zeros and add a 2-digit-suffix token; a joined-name prefix token (domains/handles); phonetic-skeleton tokens (transliterated names).
- **Design:** S1 (small, deduplicated) is indexed once; every S2/S3 record is streamed against it; keys shared by more than `cap` S1 records are ignored; each record keeps its top-K S1 candidates by key score. Country is a hard partition (open-set labels).
- **Candidate pairs generated:** see `output/candidate_pairs.tsv` and `work/test_diagnostics.json` (measured by the final run).
- **Recall evidence (prototype, S1-centric, full US pool):** single-token keys <=34% recall; conjunctive keys 98.2% at ~308 candidates/S1 (cap 100), 99.3% at ~1586 (cap 500). India: 95.0% at 515 candidates. v2 keys raised US recall at cap 100 from 98.24% to 98.90%.
- **How true matches are protected:** union of key types, per-country partition, miss analysis by mechanism (domain names, number corruption, typos, empty addresses); generic-name/empty-address misses are intentionally left unresolved (inherently ambiguous, precision-protective).

---

## 4. Matching Model

**Features used:**
- Name features: ratio, token-set, partial, token-sort on normalised names; phonetic-skeleton ratios (transliteration AUC 0.994 vs random names *(prototype)*); length; non-Latin and domain/handle flags.
- Address features: token-set, ratio, partial, number Jaccard and exact number-set equality, empty flag, length.
- Other: retrieval evidence (key-type scores/counts, best-key rarity, share/gap vs the record's best S1, candidate count), source (S2/S3). Stage 2: S2<->S3 consensus features. Absolute S1-name-frequency features deliberately excluded (distribution shift).

**Model type:** LightGBM (two stages).  
**Threshold selection method:** grouped out-of-fold macro F0.5 (singletons included); choose the highest threshold within 0.001 of the best.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** *(prototype, grouped 3-fold, US / India, includes singletons)* name only 0.639 / 0.625; address only 0.732 / 0.701; name+address 0.928 / 0.895; +retrieval keys 0.941 / 0.908; +consensus stage 0.949 / 0.935. **Final validation value: fill from `validation_report.json`.**
- **Common false positives (wrong merges):** ~70-75% are orphan decoys (near-copies with a perturbed house number, suffix change or truncated address).
- **Common false negatives (missed matches):** generic names with empty/degraded addresses, house-number corruption, name typos/handles; some are irreducibly ambiguous.
- **What did not help (measured):** one-owner assignment (+~0.001), listwise-only features (~0). A foundation model was not used; it was unproven necessary.
- **Unknowns:** France accuracy (no labels); test orphan prevalence.

---

## 6. Conclusion
Blocking on conjunctions of tokens gets near-complete candidate recall cheaply, and decoy rejection through consensus among an entity's own S2/S3 records beat both global assignment and heavier models in our tests. The remaining risk is France (unlabeled) and the unknown test orphan rate; the threshold is set conservatively for that reason.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `business_entity_resolution.ipynb` (driver + evidence), `src/er/` (all source), `README.md` (exact run steps), `requirements.txt`. Entry points: run the notebook, or `python -m er train` then `python -m er infer` from `src/`. Outputs: `output/matching_results.tsv`, `output/candidate_pairs.tsv`.

### B. Additional Results
`work/validation_report.json` (validation), `work/test_diagnostics.json` (per-country candidate counts, predicted singleton rate).
