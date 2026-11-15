# Optimization log

Each step is validated, committed separately, and pushed to the configured GitHub repository.

## Step 1 — Correct ranking metrics

- Correct single-positive Precision@k to HR@k / k.
- Report both full MRR and explicitly truncated MRR@k.
- Resolve equal scores by the reproducibly shuffled candidate order, without using relevance labels.
- Reject non-finite scores and malformed relevance labels.
- Add regression tests for metric denominators, rank-11 positives, and all-equal scores.

Existing result tables and presentation numbers predate these fixes; they will be replaced after the model and data corrections in the final experiment step.
