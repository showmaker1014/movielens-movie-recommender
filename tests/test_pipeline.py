from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from src.data import make_candidates, prepare_movies, temporal_user_split
from src.metrics import ranking_metrics
from src.models.icsrec import ICSRecInspired, candidate_scores as ics_candidate_scores
from src.models.item_knn import ItemKNNRecommender
from src.models.lightfm_adapted import LightFMAdapted


def test_release_year_and_genres() -> None:
    movies = pd.DataFrame(
        {
            "movie_id": [1, 2],
            "title": ["Toy Story (1995)", "No Year"],
            "genres": ["Animation|Children's|Comedy", "Drama"],
        }
    )
    prepared, genres = prepare_movies(movies)
    assert prepared.loc[0, "release_year"] == 1995
    assert {"Animation", "Children's", "Comedy", "Drama"}.issubset(set(genres))
    assert prepared["Comedy"].dtype == np.int8


def test_temporal_split_is_per_user_and_ordered() -> None:
    rows = []
    for user in [1, 2]:
        for index in range(10):
            rows.append(
                {
                    "user_id": user,
                    "movie_id": index + 1,
                    "rating": 5 if index >= 7 else 3,
                    "timestamp": index + 1,
                }
            )
    split = temporal_user_split(pd.DataFrame(rows))
    counts = split.groupby(["user_id", "split"]).size().unstack(fill_value=0)
    assert counts.loc[1, "train"] == 8
    assert counts.loc[1, "validation"] == 1
    assert counts.loc[1, "test"] == 1
    assert split.loc[(split["user_id"] == 1) & (split["split"] == "test"), "timestamp"].item() == 10


def test_candidates_share_exactly_one_positive() -> None:
    split = temporal_user_split(
        pd.DataFrame(
            {
                "user_id": np.repeat(np.arange(1, 6), 10),
                "movie_id": np.tile(np.arange(1, 11), 5),
                "rating": np.tile([3, 3, 3, 3, 3, 3, 3, 4, 4, 5], 5),
                "timestamp": np.tile(np.arange(1, 11), 5),
            }
        )
    )
    data = make_candidates(
        split,
        np.arange(1, 6),
        np.arange(1, 16),
        {i: i - 1 for i in range(1, 6)},
        {i: i - 1 for i in range(1, 16)},
        "test",
        negatives_per_user=4,
    )
    assert np.all(data["labels"].sum(axis=1) == 1)
    for row, uid in enumerate(data["user_ids"]):
        seen = set(split.loc[split["user_id"] == uid, "movie_id"])
        negatives = data["candidate_items"][row][data["labels"][row] == 0]
        assert not set(negatives) & seen


def test_ranking_metrics_handles_position() -> None:
    scores = np.asarray([[0.2, 0.9, 0.4], [0.8, 0.1, 0.2]], dtype=np.float32)
    labels = np.asarray([[0, 1, 0], [1, 0, 0]], dtype=np.int8)
    result = ranking_metrics(scores, labels, k=2)
    assert result["HR@2"] == 1.0
    assert result["mean_rank"] == 1.0
    assert np.isclose(result["NDCG@2"], 1.0)


def test_item_knn_scores_candidates_without_using_seen_rows() -> None:
    train = np.asarray(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
        ],
        dtype=np.float32,
    )
    model = ItemKNNRecommender(neighbor_count=3).fit(train)
    scores = model.score_candidates(
        np.asarray([0]),
        np.asarray([[0, 3, 4]], dtype=np.int32),
    )
    assert scores.shape == (1, 3)
    assert np.isfinite(scores).all()


def test_three_models_return_candidate_scores() -> None:
    rng = np.random.default_rng(7)
    user_features = rng.normal(size=(4, 3)).astype(np.float32)
    item_features = rng.normal(size=(8, 2)).astype(np.float32)
    lightfm = LightFMAdapted(4, 8, 3, 2, embedding_dim=8)
    users = np.asarray([0, 1])
    items = np.asarray([[1, 2, 3], [4, 5, 6]])

    import torch

    with torch.inference_mode():
        all_scores = lightfm.all_item_scores(
            torch.tensor(users),
            torch.as_tensor(user_features),
            torch.as_tensor(item_features),
        )
    assert all_scores.shape == (2, 8)

    ics = ICSRecInspired(8, embedding_dim=8, hidden_dim=8, n_intents=2)
    sequences = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5]), np.array([6, 7])]
    output = ics_candidate_scores(ics, sequences, users, items)
    assert output.shape == (2, 3)

