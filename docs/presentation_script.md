# Five-minute presentation plan

Use `presentation/MovieLens_Recommender_5min.pptx`, not the outdated 24-page PDF. Total planned time: 300 seconds. Rehearse and shorten wording if needed; a time plan is not proof of actual delivery time.

| Slide | Time | What to explain |
|---|---:|---|
| Task | 0:00–0:30 | Given a user, recommend at least five unrated films, with names and genres. Compare exactly three algorithms. |
| Data/protocol | 0:30–1:20 | 6040 users, 3883 catalog movies and 1000209 ratings. Per-user 80/10/10 split; rating≥4 is relevant. One held-out positive plus 99 shared negatives. Only training ratings fit models; static metadata uses the known catalog. |
| Data evidence | 1:20–1:50 | Ratings favor 4–5 stars; no source nulls. User/item activity is uneven. Explain the relevance threshold and that the interaction matrix remains sparse. |
| Algorithms | 1:50–3:05 | ItemKNN uses co-liked cosine neighbors. LightFM-BPR adds identity and metadata vectors/biases. ICSRec pairs prefixes with the same target, masks false negatives for CICL, aligns to refreshed centroids for FICL and uses full-catalog CE. State the GRU/CPU budget differences and no pretrained checkpoint. |
| Evaluation | 3:05–4:05 | Validation selects configurations. Test NDCG@10: 0.3696 / 0.3311 / 0.3014. Explain HR versus rank-sensitive NDCG. Precision equals HR/10 in this single-positive protocol; these are sampled-candidate results. |
| App/findings | 4:05–5:00 | Show user 518's model-specific picks, or a brief live app action if time permits. Product examples use the full catalog and exclude all rated films. ItemKNN leads here; cold-start/drift explanations require further tests. Mention full-catalog evaluation and graded feedback as future work. |

## Useful answers for questions

**Why is ICSRec's arXiv posting from 2023?** The coursework selects ICSRec as WSDM 2024, its conference publication.

**Is this a complete paper reproduction?** Core mechanisms are independently implemented. BPR/Adam, CPU KMeans, prefix sampling and the supported GRU backbone are disclosed differences. The project does not reproduce the original benchmark setup.

**Why not claim a 64% recommendation accuracy?** HR@10=0.6414 is the fraction of 100-candidate lists whose one positive is in the first ten. It does not measure full-catalog accuracy or user satisfaction.

**Does metadata improve cold start?** It provides shared features, but this project has not measured a dedicated cold-start subgroup. Treat the claim as motivation, not an established experimental result.

**Why can filters return fewer than five films?** The app obeys genre/year and complete-history exclusions. It explains how to broaden filters when fewer than five matching unseen films exist.
