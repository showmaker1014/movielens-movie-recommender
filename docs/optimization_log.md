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
