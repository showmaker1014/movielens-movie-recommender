# MovieLens Hybrid Movie Recommender

**Technical report - Group Project, Project 1**

Date: 15 November 2026

## Abstract

This project builds and evaluates a movie recommendation system on the MovieLens 1M dataset. The system uses three distinct models: item-based collaborative filtering, a LightFM-style hybrid metadata factorizer, and an ICSRec (GRU) sequential recommender with latent intents. All models share a per-user chronological split and identical sampled candidates. The strongest held-out test model by NDCG@10 is **ItemKNN** at 0.3696, with HR@10 of 0.6414.

The project also includes a Streamlit interface that accepts an anonymized user and displays unseen movie titles and genres.

## 1. Task and success criteria

Given a user, the recommender must rank at least five movies that the user has not rated. The deliverable must cover dataset evidence, preprocessing, exactly three algorithms, comparative evaluation, and an interactive application.

The operational success criteria are:

- Reproducible data split and random seed.
- No held-out ratings in training features or learned popularity counts.
- A shared evaluation protocol for all algorithms.
- An interface that accepts a user and shows titles and genres.

## 2. Dataset evidence

The official archive provides three relational tables.

| table | rows | columns | null cells | null columns |
| --- | --- | --- | --- | --- |
| users | 6040 | 5 | 0 | 0 |
| movies | 3883 | 3 | 0 | 0 |
| ratings | 1000209 | 4 | 0 | 0 |

Ratings range from 1 to 5 with a mean of 3.582. The tables contain 0 duplicate user-movie pairs. The source files have no missing cells, so no imputation was required. Metadata preprocessing still includes release-year extraction, genre multi-hot encoding, age standardization, and gender/occupation one-hot encoding.

The requested inspection evidence is reproduced by `scripts/prepare_data.py` in `data/processed/dataset_summary.json` and `dataset_comparison.csv`. It includes shapes, null totals, null columns, unique-value counts, dtypes, and a table comparison.

![Rating distribution](../figures/rating_distribution.png)

![Activity distributions](../figures/activity_distributions.png)

## 3. Preprocessing and split

The processed split contains 802,553 training, 100,273 validation, and 97,383 test ratings. Interactions are ordered by timestamp inside each user, then divided into 80/10/10 partitions. This prevents a future interaction from becoming a training feature. Item popularity, sequence histories and item-item similarities use training ratings only. Static metadata statistics use the known user/item catalog.

A rating of 4 or 5 is converted to a positive relevance label. For evaluation, each user contributes the latest positive rating from the relevant split plus 99 movies that the user never rated in any period. A fixed seed shuffles the 100 candidates. Hiding the future label from model fitting prevents leakage, while using the complete observed set for candidate exclusion matches the product requirement that recommendations be unseen.

![Temporal split](../figures/temporal_split.png)

## 4. Algorithms

### 4.1 Item-based collaborative filtering (lecture algorithm)

A binary user-item matrix is built from training ratings of 4 or higher. Each movie column is L2-normalized and multiplied by the transpose to obtain cosine item-item similarity. A user's score for an unseen movie is the sum of similarities from movies the user liked, plus a small normalized log-popularity prior. Neighbor counts of 30, 60, and 120 are compared on validation NDCG@10.

### 4.2 LightFM-style metadata embeddings (reference algorithm)

The LightFM paper represents users and items as additive sums of identity and metadata embeddings. This implementation follows that decomposition:

`score(u,i) = b_u + b_i + F_u c_u + F_i c_i + <e_u + F_u w_u, e_i + F_i w_i>`

User metadata includes standardized age plus gender and occupation indicators. Item metadata includes genre indicators and standardized release year. The model is trained with Bayesian Personalized Ranking (BPR), which increases the margin between an observed positive and a uniformly sampled unobserved negative. Embedding dimensions 32 and 64 are compared on validation NDCG@10.

### 4.3 ICSRec (GRU) latent-intent sequence model (recent algorithm)

ICSRec (WSDM 2024) is reimplemented with sliding training subsequences and a supported GRU backbone. Prefixes sharing the next target form CICL pairs; same-target false negatives are masked. FICL queries actual refreshed KMeans centers, while full-catalog softmax CE trains recommendation. Windows contain at most 50 items. CPU adaptations: scikit-learn KMeans, 30000 sampled training prefixes and 10000 clustering prefixes per epoch; singleton target groups skip CICL. No pretrained checkpoint is used. This reproduces the core objectives, not the original benchmark setup.

![Genre distribution](../figures/genre_distribution.png)

## 5. Evaluation protocol

Every algorithm receives the same users, positive items, 99 negatives, and random order. Metrics are:

- **HR@10:** whether the positive appears in the top 10.
- **NDCG@10:** position-sensitive ranking quality.
- **MRR@10:** reciprocal rank, zero outside the first 10 candidates.
- **Precision@10 / Recall@10:** Recall@10 equals HR@10; Precision@10 equals HR@10 / 10.

Hyperparameters are selected using validation data only. The selected configuration is evaluated once on the test partition. No test metric is used for model selection.

## 6. Results

| algorithm | HR@10 | NDCG@10 | MRR@10 | test_users |
| --- | --- | --- | --- | --- |
| ItemKNN | 0.6414 | 0.3696 | 0.2863 | 5730 |
| LightFM-BPR | 0.5993 | 0.3311 | 0.2492 | 5730 |
| ICSRec (GRU) | 0.5375 | 0.3014 | 0.2291 | 5730 |

![Model comparison](../figures/model_comparison.png)

Validation selection table:

| algorithm | config | HR@10 | NDCG@10 | MRR@10 |
| --- | --- | --- | --- | --- |
| ItemKNN | {"neighbors": 30, "popularity_weight": 0.02} | 0.6999 | 0.4204 | 0.3344 |
| ItemKNN | {"neighbors": 60, "popularity_weight": 0.02} | 0.7132 | 0.4300 | 0.3428 |
| ItemKNN | {"neighbors": 120, "popularity_weight": 0.02} | 0.7106 | 0.4313 | 0.3454 |
| LightFM-BPR | {"batch_size": 4096, "embedding_dim": 32, "epochs": 10, "learning_rate": 0.01, "loss": "BPR"} | 0.6541 | 0.3764 | 0.2914 |
| LightFM-BPR | {"batch_size": 4096, "embedding_dim": 64, "epochs": 10, "learning_rate": 0.01, "loss": "BPR"} | 0.6553 | 0.3750 | 0.2891 |
| ICSRec (GRU) | {"batch_size": 256, "cluster_sample_size": 10000, "embedding_dim": 32, "epochs": 6, "learning_rate": 0.003, "loss": "CE + CICL + FICL", "max_examples_per_epoch": 30000, "max_length": 50, "n_intents": 8, "pretrained": false, "seed": 42} | 0.5893 | 0.3360 | 0.2588 |
| ICSRec (GRU) | {"batch_size": 256, "cluster_sample_size": 10000, "embedding_dim": 64, "epochs": 6, "learning_rate": 0.003, "loss": "CE + CICL + FICL", "max_examples_per_epoch": 30000, "max_length": 50, "n_intents": 16, "pretrained": false, "seed": 42} | 0.6293 | 0.3794 | 0.3025 |

## 7. Interpretation and error analysis

The best test NDCG@10 is 0.3696. The result should be read as performance on a controlled 100-candidate task rather than full-catalog quality. A score below one is expected because the positive is one item among 99 plausible alternatives and many users have broad tastes.

ItemKNN is transparent and learns local co-consumption patterns, but its dense movie-to-movie neighborhoods can over-reward mainstream films with many co-raters. The LightFM-style model uses genre and user metadata, which helps generalize when direct interaction data is sparse. The intent model conditions on sequence order; it can capture changing preferences, but its small CPU budget and short history limit temporal depth.

Concrete error modes are:

- **Popularity bias:** common genres and frequently rated movies have more neighbors and stronger prior support.
- **Sparse users:** users with few positives provide weak collaborative and sequence evidence.
- **Cold items:** movies with few training ratings have unstable item-item neighbors but can still receive metadata scores.
- **Taste drift:** one static user representation cannot fully separate genres watched in different periods.
- **Ratings below 4:** the current implicit formulation discards future low ratings during fitting. A future version should model graded negative feedback.

## 8. Recommendation examples

The following is a sample from `results/sample_user_recommendations.csv`. The full file contains 20 users and top-10 recommendations from all three algorithms.

| user_id | algorithm | rank | title | genres | release_year |
| --- | --- | --- | --- | --- | --- |
| 518 | ItemKNN | 1 | One Flew Over the Cuckoo's Nest (1975) | Drama | 1975 |
| 518 | ItemKNN | 2 | Emma (1996) | Comedy|Drama|Romance | 1996 |
| 518 | ItemKNN | 3 | Wizard of Oz, The (1939) | Adventure|Children's|Drama|Musical | 1939 |
| 518 | ItemKNN | 4 | Dead Again (1991) | Mystery|Romance|Thriller | 1991 |
| 518 | ItemKNN | 5 | Sleepless in Seattle (1993) | Comedy|Romance | 1993 |
| 518 | LightFM-BPR | 1 | Out of Africa (1985) | Drama|Romance | 1985 |
| 518 | LightFM-BPR | 2 | Witness (1985) | Drama|Romance|Thriller | 1985 |
| 518 | LightFM-BPR | 3 | Godfather, The (1972) | Action|Crime|Drama | 1972 |
| 518 | LightFM-BPR | 4 | Braveheart (1995) | Action|Drama|War | 1995 |
| 518 | LightFM-BPR | 5 | African Queen, The (1951) | Action|Adventure|Romance|War | 1951 |
| 518 | ICSRec (GRU) | 1 | Bug's Life, A (1998) | Animation|Children's|Comedy | 1998 |
| 518 | ICSRec (GRU) | 2 | Clerks (1994) | Comedy | 1994 |
| 518 | ICSRec (GRU) | 3 | American Pie (1999) | Comedy | 1999 |
| 518 | ICSRec (GRU) | 4 | Toy Story (1995) | Animation|Children's|Comedy | 1995 |
| 518 | ICSRec (GRU) | 5 | Lion King, The (1994) | Animation|Children's|Musical | 1994 |

## 9. Application

The Streamlit application accepts a user, algorithm, genre, release-year filter, and result count. It excludes every movie observed for that user, displays the top titles and genres, compares the three algorithms side by side, and exposes validation/test metrics and dataset evidence. Users can download their result table as CSV.

## 10. Limitations and future work

The main limitation is sampled evaluation. A production system should rank the full unseen catalog and measure recall, NDCG, catalog coverage, and novelty. The project does not model repeated viewing, time decay, user text reviews, or item content beyond genre/year. The LightFM and ICSRec methods are transparent adaptations rather than exact paper reproductions. Future work should add graded feedback, popularity calibration, cold-start stress tests, and online A/B tests.

## 11. Reproducibility

Random seed: `42`. Positive threshold: `4`. Selected configurations: `{"ICSRec (GRU)": {"batch_size": 256, "cluster_sample_size": 10000, "embedding_dim": 64, "epochs": 6, "learning_rate": 0.003, "loss": "CE + CICL + FICL", "max_examples_per_epoch": 30000, "max_length": 50, "n_intents": 16, "pretrained": false, "seed": 42}, "ItemKNN": {"neighbors": 120, "popularity_weight": 0.02}, "LightFM-BPR": {"batch_size": 4096, "embedding_dim": 32, "epochs": 10, "learning_rate": 0.01, "loss": "BPR"}}`.

Reference environment: Python 3.11, pandas 1.5.1, NumPy 1.23.4, PyTorch, Streamlit, matplotlib, SciPy, and PyArrow. Run instructions, dependencies, seeds, data citation, and output locations are in `README.md`.

Data files are intentionally excluded from Git. Rebuild with:

```bash
python scripts/prepare_data.py --download
python scripts/train_models.py
python scripts/build_report.py
streamlit run app.py
```

## References

1. F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM TiiS 5(4), Article 19. https://doi.org/10.1145/2827872
2. Maciej Kula. 2015. Metadata Embeddings for User and Item Cold-start Recommendations (LightFM). arXiv:1507.08439.
3. Xiuyuan Qin et al. 2024. Intent Contrastive Learning with Cross Subsequences for Sequential Recommendation (ICSRec). WSDM 2024. https://doi.org/10.1145/3616855.3635773


## Appendix: column inspection

| table | column | dtype | unique_values | null_values |
| --- | --- | --- | --- | --- |
| users | user_id | int64 | 6040 | 0 |
| users | gender | object | 2 | 0 |
| users | age | int64 | 7 | 0 |
| users | occupation | int64 | 21 | 0 |
| users | zip | object | 3439 | 0 |
| movies | movie_id | int64 | 3883 | 0 |
| movies | title | object | 3883 | 0 |
| movies | genres | object | 301 | 0 |
| ratings | user_id | int64 | 6040 | 0 |
| ratings | movie_id | int64 | 3706 | 0 |
| ratings | rating | int64 | 5 | 0 |
| ratings | timestamp | int64 | 458455 | 0 |
