# Algorithm reproduction and evaluation notes

This document distinguishes implemented mechanisms, engineering adaptations, measured results and unverified claims. Parameter variants are configurations of the same three algorithms, not additional algorithms.

| Role | Method | Implemented core | Disclosed differences |
|---|---|---|---|
| Lecture | Item-based CF / ItemKNN | Binary positive matrix, normalized item cosine, top-k neighborhoods, similarity sum | Shrinkage and a normalized log-popularity prior; confirm the lecture reference before submission |
| Reference | LightFM-BPR | Weighted sums of identity/metadata vectors and biases; pairwise BPR ranking | PyTorch mini-batch Adam instead of the original paper's logistic/Adagrad experiment and official Cython optimizer |
| Recent | ICSRec (GRU), WSDM 2024 | Sliding prefixes with same-target partners, CICL false-negative masking, refreshed KMeans centers for FICL, full-catalog CE | Supported GRU backbone, CPU scikit-learn KMeans, explicit sampled-prefix budget, singleton targets skip CICL |

LightFM sources: [paper §2.2](https://arxiv.org/abs/1507.08439), [official feature model and supported BPR loss](https://making.lyst.com/lightfm/docs/lightfm.html). Implementation: `src/models/lightfm_adapted.py`.

ICSRec sources: [paper §§2.3–2.7](https://arxiv.org/html/2310.14318v3), [official repository](https://github.com/QinHsiu/ICSRec). Implementation: `src/models/icsrec.py`. The arXiv posting predates 2024; the selected conference method is WSDM 2024, as listed in the coursework brief.

No pretrained checkpoint is used. Models are initialized from seed 42 independently for each configuration. Model files created by this project are local outputs, not external pretrained weights. The project reproduces core algorithm mechanisms, not the papers' original benchmark tables or complete training environments.

## Training and model selection

ItemKNN tests k=30/60/120. LightFM tests 32/64 dimensions for 10 epochs. ICSRec tests (dimension, intents)=(32,8)/(64,16) for 6 epochs. Epoch counts are fixed; no early stopping is used. Each algorithm's configuration is selected by validation NDCG@10, with HR@10 as the tie breaker. Final test scores are not used to choose features or parameters.

ICSRec uses every training transition as an eligible prefix/target pair before windowing each input to 50 items. Each epoch samples up to 30000 training prefixes and 10000 clustering prefixes from that pool. Both limits are exposed in `ICSRecTrainingConfig`; `None` removes the corresponding cap. Each history row records actual prefix and partner counts. These are CPU budget choices, not full-data paper-training claims.

LightFM refreshes negatives each epoch and excludes only known training positives. A training rating below 4 can therefore be sampled as a negative. Unobserved and low-rated items are not separated by a calibrated dislike model. Future validation/test interactions are never read to build the training negative pool.

## Static metadata and temporal information

Each user's ratings are sorted by timestamp and then movie ID, split 80/10/10. The movie-ID rule makes equal timestamps reproducible; it does not recover the true order of events within a timestamp. This is a per-user split, not a global calendar cutoff.

Interaction matrices, similarities, popularity and training histories use train ratings only. Static age/year scaling and metadata vocabularies use the known user/item catalog. The latter is a transductive catalog assumption and does not include held-out rating labels.

## Two different candidate protocols

**Offline quality:** latest relevant item in the held-out partition plus 99 items never rated by that user in any period. Complete history defines the offline negative pool but does not fit a model. All algorithms receive identical users, items and shuffled order. This predicts a held-out liked movie, not necessarily the immediately next interaction. Users without a relevant item in that partition are excluded; there are 5802 validation users and 5730 test users.

**Product examples and app:** score the full catalog and exclude every observed rating for the user, including low and held-out ratings. Shared code in `src/recommendation.py` applies filtering and uses movie ID to break score ties. A filter with fewer than five candidates produces a message asking for broader filters; it cannot manufacture five matching movies.

## Metrics and interpretation

With one positive, Recall@10=HR@10 and Precision@10=HR@10/10. NDCG@10 is zero outside the first ten; MRR@10 is similarly truncated, while MRR uses the whole 100-item ranking. Evaluation ties use the fixed shuffled candidate order and never inspect the positive label. Scores are ranking utilities, not predicted star ratings or calibrated probabilities.

Only the shared sampled-candidate comparison has been measured. Cold-start advantages, subgroup fairness, popularity calibration and interest-drift benefits remain hypotheses requiring dedicated experiments. Validation-to-test drops alone do not establish drift as their cause.

## Lecture provenance

ItemKNN is implemented and classified as item-based collaborative filtering. The provided course project brief requires a lecture method, but the relevant lecture slides were not available in the supplied project/course files. The exact lecture/chapter has therefore not been asserted. Confirm that provenance with the course materials before submission; do not invent a page number.
