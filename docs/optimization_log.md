# Optimization log

Each step is validated, committed separately, and pushed to the configured GitHub repository.

## Step 1 — Correct ranking metrics

- Correct single-positive Precision@k to HR@k / k.
- Report both full MRR and explicitly truncated MRR@k.
- Resolve equal scores by the reproducibly shuffled candidate order, without using relevance labels.
- Reject non-finite scores and malformed relevance labels.
- Add regression tests for metric denominators, rank-11 positives, and all-equal scores.

Existing result tables and presentation numbers predate these fixes; they will be replaced after the model and data corrections in the final experiment step.

## Step 2 — Metadata and dependencies

Remove duplicate year features, handle constant ages, retain complete sequences and declare reportlab. Validate unique feature names and finite standardized features.

## Step 3 — LightFM core model

Complete metadata feature biases, refresh reproducible negative samples per epoch, and reject users without available negatives. Six mechanism and training tests pass. Old checkpoints require retraining; BPR+Adam is a disclosed ranking adaptation.


## Step 4 — Restore ICSRec core objectives

Implement sliding subsequences, same-target CICL pairs with false-negative masking, refreshed KMeans-centroid FICL, and full-catalog CE. Retain all training history before per-prefix windowing. GRU/CPU KMeans and explicit 30000-prefix/10000-clustering CPU caps are disclosed adaptations; no pretrained checkpoint. Six mechanism tests and the complete 25-test suite pass.

## Step 5 — Robust recommendation interface

Preserve columns for empty candidates, explain fewer-than-five results, use actual genre labels, check all required inputs, and show corrected metric definitions. Full-history exclusion is retained.

## Step 6 — Report evidence and reproducibility

Correct metric and algorithm descriptions; show the same user Top5 from all models; wrap PDF cells; add column evidence and full-catalog unseen examples. Declare verified-TLS requests download. Add empty-filter and complete-history exclusion regression coverage.

## Step 7 — Validate the real application and runtime

Pin the CPU NumPy/PyTorch/PyArrow compatibility range. Replace unsupported Streamlit width strings. Preserve training duration and record test scoring duration separately. Fix PDF header contrast and metadata-statistics wording. Actual Streamlit AppTest loads all tabs and switches models; 26 unit/regression tests pass.

## Step 8 — Re-run MovieLens and refresh deliverables

Rebuilt official MovieLens-1M statistics (SHA256 a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20), trained all configurations and selected by validation NDCG@10. Test NDCG@10: ItemKNN 0.369578, LightFM-BPR 0.331063, ICSRec 0.301354. Re-exported 600 full-catalog examples; all 60 user/model groups contain 10 unseen items, verified against complete rating history. Updated report, column-evidence appendix, figures and six-slide deck. Deck package/geometry validators passed with zero findings; the wrapper could not spawn its child process, so both validators were executed directly. PDF pages and slide previews visually checked. GitHub push is pending repository write permission for the logged-in account.

## Step 9 — Preserve binary deliverables in Git

Declare PDF, PPTX and image assets binary so Windows autocrlf cannot alter their bytes or PDF offsets. Verify committed PDF bytes match the rendered working file.

## Step 10 — Correct report wording and metadata generation

Escape Markdown pipes/newlines, correct duration locations, remove unsupported cold-start/drift claims and distinguish training positives from low-rated negative candidates. Make selected training durations part of regenerated metadata. Add a Markdown column regression test.

## Step 11 — Unify product examples with the interface

Use one full-history filter, deterministic movie-ID tie rule and sampled user set in training, re-export and UI. Exported profiles and score fields now have the same schema. Add ranking/export parity and user-selection regression checks.

## Step 12 — Add reproduction, submission and presentation documentation

Add core-method/difference mapping, a runbook, course checklist, five-minute script and a page-specific review of the supplied 24-page PDF. Reproduce saved metrics and example rankings with a reusable verification script. Keep absent lecture provenance and administrative records explicit rather than inventing evidence.

## Step 13 — Independently validate a clean environment

Install all direct dependencies in a new Windows/Python 3.11 venv without system packages. pip check passes; 30 tests pass. Recompute saved-model metrics/examples, run live AppTest including empty filters, and train all three models on isolated synthetic data without replacing MovieLens results. Record all resolved packages and validation evidence. Full clean-environment optimization was not rerun.

## Step 14 — Respect custom preprocessing paths

Downloaded archives now extract the expected dataset files directly into --raw-dir. Reprocessing retains the cached archive checksum. Required members are checked before writes and unrelated ZIP paths are ignored. Add custom-path/rerun and incomplete-archive regression tests.
