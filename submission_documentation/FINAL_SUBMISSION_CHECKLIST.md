# Final submission, reproducibility and compliance checklist

Complete every unchecked item before creating the final archive.

## 1. Identity and experiment provenance

- [ ] Team name and members are filled in `Documentation_template.md` and
      `ML_APPROACH_SUMMARY.md`.
- [ ] The final Git commit is recorded.
- [ ] The final configuration and random seed are recorded.
- [ ] Prototype values are still labelled “prototype”.
- [ ] Final validation values come from the final commit's generated reports.
- [ ] Public leaderboard score and submission identifier are recorded.

## 2. Fair-play declaration

- [ ] Only the provided train/test TSV files were used as entity data.
- [ ] No external business or entity-resolution API was used.
- [ ] No government/company registry was queried.
- [ ] No geocoding or address-enrichment API was used.
- [ ] No internet data was used to augment business names, addresses or identities.
- [ ] `USE_FM = False` for the declared final run unless organisers explicitly approved
      downloaded pretrained weights.
- [ ] Internet access, if used, was limited to obtaining code dependencies permitted by
      the rules and was not used for entity lookup.

## 3. Model and dependency audit

- [ ] Final classifier is LightGBM.
- [ ] LightGBM's MIT license is documented.
- [ ] No model exceeds the 8-billion-parameter limit.
- [ ] `code/business_entity_resolution/requirements.txt` contains pinned versions.
- [ ] A clean environment can execute the README commands.

## 4. Output validation

- [ ] `output/matching_results.tsv` exists.
- [ ] `output/candidate_pairs.tsv` exists.
- [ ] Both files are tab-separated and have the exact official headers.
- [ ] Every test Source 1 ID occurs exactly once and in source order.
- [ ] Empty predictions are represented by an empty second field.
- [ ] Every listed ID exists in test Source 2 or Source 3.
- [ ] No ID list contains duplicates.
- [ ] Every final matched ID is present in the corresponding candidate list.
- [ ] The official validator exits with status 0 and prints `PASS`.

Run from the challenge resource directory:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## 5. Required archive contents

- [ ] `output/matching_results.tsv`
- [ ] `output/candidate_pairs.tsv`
- [ ] `code/business_entity_resolution/src/`
- [ ] `code/business_entity_resolution/README.md`
- [ ] `code/business_entity_resolution/requirements.txt`
- [ ] Filled `Documentation_template.md` at the ZIP root
- [ ] `submission_documentation/ML_APPROACH_SUMMARY.md`
- [ ] Other supporting documents in `submission_documentation/`

Inspect the archive before upload:

```bash
unzip -l <team_name>_submission.zip
```

The archive root must not contain an unintended extra parent directory.

## 6. Reproduction test

- [ ] Start from a clean environment.
- [ ] Install only `requirements.txt`.
- [ ] Point `ER_DATA_DIR` to the official dataset.
- [ ] Run training and inference exactly as documented.
- [ ] Compare generated report/configuration hashes with the final run.
- [ ] Re-run the official validator.
- [ ] Confirm generated row counts and country diagnostics are plausible.

## 7. Final sign-off

**Prepared by:** DataForge
**Reviewed by:** Ayush Mishra (Team Leader)
**Date:** 27 September 2026
**Final commit:** [FILL BEFORE SUBMISSION]
**Validator result:** [FILL BEFORE SUBMISSION]
**Archive SHA-256:** [FILL BEFORE SUBMISSION]
