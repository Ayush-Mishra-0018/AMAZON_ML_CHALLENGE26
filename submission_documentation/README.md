# Final submission documentation

This directory contains the supporting documentation for the team's Amazon ML
Challenge 2026 final submission. It complements the required filled methodology
file at [`../Documentation_template.md`](../Documentation_template.md).

## Documents

| File | Purpose | Official requirement covered |
|---|---|---|
| `ML_APPROACH_SUMMARY.md` | Concise 1-2 page-equivalent summary | ML approach, models, experiments, conclusion |
| `METHODOLOGY.md` | Detailed technical design | Methodology, blocking, architecture, feature engineering |
| `EXPERIMENTS_AND_RESULTS.md` | Auditable experiment record | Experiments, validation evidence, final metrics |
| `FINAL_SUBMISSION_CHECKLIST.md` | Compliance and packaging audit | Fair play, licensing, reproducibility, output validation |

## Authoritative implementation

The final implementation is under `../code/business_entity_resolution/`:

- `src/er/` is the modular training and inference package.
- `business_entity_resolution.ipynb` is the primary narrative notebook.
- `README.md` contains end-to-end reproduction instructions.
- `requirements.txt` pins the runtime dependencies.

The final submitted method is the two-stage LightGBM pipeline described in these
documents. It uses only the supplied challenge data. Optional foundation-model
experiments in `kaggle_end_to_end.ipynb` are not part of the declared final method;
keep `USE_FM = False` for a compliant final run unless the organisers explicitly
confirm that downloaded pretrained weights are permitted.

## Before packaging

Replace every marker of the form `[FILL BEFORE SUBMISSION]` with the value from the
final reproducible run. Do not replace a missing result with a prototype result.
Prototype measurements are explicitly labelled as such throughout these documents.

The required ZIP layout is:

```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
├── Documentation_template.md
└── submission_documentation/       # supporting documentation from this directory
```

Extra documentation is included for auditability; the four artifacts explicitly
required by the problem statement remain `output/`, `code/business_entity_resolution/`,
and the root `Documentation_template.md`.
