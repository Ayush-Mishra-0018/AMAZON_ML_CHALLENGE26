# Experiments and results record

This document separates historical prototype measurements from the final reproducible
run. Values labelled “prototype” must not be reported as final validation results.

## 1. Prototype feature ablation

The following grouped three-fold macro F0.5 values are recorded in the existing
methodology notes. They are supporting development evidence, not a substitute for the
final `validation_report.json`.

| Experiment | US F0.5 | India F0.5 | Interpretation |
|---|---:|---:|---|
| Name similarities only | 0.639 | 0.625 | Names alone collide heavily |
| Address similarities only | 0.732 | 0.701 | Addresses are stronger but incomplete/noisy |
| Name + address | 0.928 | 0.895 | Complementary fields provide the largest gain |
| + blocking/retrieval evidence | 0.941 | 0.908 | Key rarity and rank help reject confounders |
| + S2/S3 consensus stage | 0.949 | 0.935 | Relational agreement improves decoy rejection |

## 2. Prototype blocking observations

| Configuration | Observation |
|---|---|
| Single-token blocking | At most approximately 34% recall in the recorded US prototype |
| Conjunctive keys, cap 100 | Approximately 98.2% recall at approximately 308 candidates/S1 in the recorded US prototype |
| Conjunctive keys, cap 500 | Approximately 99.3% recall at approximately 1,586 candidates/S1 in the recorded US prototype |
| India prototype | Approximately 95.0% recall at approximately 515 candidates/S1 |

These figures must be regenerated or explicitly retained as prototype-only evidence in
the final write-up.

## 3. Negative results retained for auditability

- Enforcing one Source 1 owner per pool record improved the recorded validation score
  by only approximately 0.001 and is disabled by default.
- Listwise-only features produced approximately zero material improvement.
- Absolute Source 1 name-frequency features helped one validation slice but were
  excluded because train and test Source 1 density differed.
- External foundation-model features are not part of the declared final method.

## 4. Final reproducible run

Fill this section directly from the artifacts produced by the final code commit.

| Item | Final value | Source artifact |
|---|---|---|
| Git commit | [FILL BEFORE SUBMISSION] | `git rev-parse HEAD` |
| Python version | [FILL BEFORE SUBMISSION] | runtime log |
| Training seed | [FILL BEFORE SUBMISSION] | configuration |
| Candidate top-K | [FILL BEFORE SUBMISSION] | validation report/configuration |
| Candidate link recall | [FILL BEFORE SUBMISSION] | `work/validation_report.json` |
| Candidate ceiling macro F0.5 | [FILL BEFORE SUBMISSION] | `work/validation_report.json` |
| Overall validation macro F0.5 | [FILL BEFORE SUBMISSION] | `work/validation_report.json` |
| US validation macro F0.5 | [FILL BEFORE SUBMISSION] | `work/validation_report.json` |
| India validation macro F0.5 | [FILL BEFORE SUBMISSION] | `work/validation_report.json` |
| Singleton validation F0.5 | [FILL BEFORE SUBMISSION] | `work/validation_report.json` |
| Non-singleton validation F0.5 | [FILL BEFORE SUBMISSION] | `work/validation_report.json` |
| Selected threshold | [FILL BEFORE SUBMISSION] | `work/models/meta.json` |
| Public leaderboard F0.5 | [FILL BEFORE SUBMISSION] | competition portal |
| Private leaderboard F0.5 | [FILL WHEN AVAILABLE] | competition portal |
| Training runtime | [FILL BEFORE SUBMISSION] | `work/run.log` |
| Inference runtime | [FILL BEFORE SUBMISSION] | `work/run.log` |
| Peak memory | [FILL BEFORE SUBMISSION] | run log/platform metrics |

## 5. Test-set diagnostics

These are not accuracy measurements because test labels are unavailable. Record them to
detect obvious distribution or pipeline failures.

| Country | S1 rows | Candidate pairs | Mean candidates/S1 | Predicted singleton rate | Mean predicted matches/S1 |
|---|---:|---:|---:|---:|---:|
| US | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| India | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |
| France | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] |

## 6. Final conclusion

[FILL BEFORE SUBMISSION: summarize which configuration was selected, why it was
selected, its validation/leaderboard behavior, and the remaining failure modes. Do not
claim that France performance was measured without labels.]
