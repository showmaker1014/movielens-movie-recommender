"""Shared, label-independent product ranking and reproducible example export."""
from __future__ import annotations
import numpy as np
import pandas as pd


def example_user_ids(user_ids: np.ndarray, seed: int = 42) -> np.ndarray:
    ids = np.sort(np.asarray(user_ids))
    return np.sort(np.random.default_rng(seed).choice(ids, min(20, len(ids)), replace=False))


def rank_unseen(movies, scores, observed, top_n=10, genre="All genres", minimum_year=0):
    scores = np.asarray(scores)
    if scores.shape != (len(movies),):
        raise ValueError("One score is required per catalog movie")
    if top_n < 1:
        raise ValueError("top_n must be positive")
    mask = ~movies["movie_id"].isin(observed).to_numpy()
    mask &= movies["release_year"].to_numpy() >= minimum_year
    if genre != "All genres":
        mask &= movies["genres"].fillna("").str.split("|", regex=False).map(lambda labels: genre in labels).to_numpy()
    candidates = np.flatnonzero(mask)
    # ItemKNN intentionally marks training items -inf; those are already
    # excluded. Require finite utilities only for actual eligible candidates.
    if not np.isfinite(scores[candidates]).all():
        raise ValueError("Eligible candidate scores must be finite")
    ids = movies.iloc[candidates]["movie_id"].to_numpy()
    order = candidates[np.lexsort((ids, -scores[candidates]))[:top_n]]
    result = movies.iloc[order][["movie_id", "title", "genres", "release_year"]].copy()
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result["model_score"] = scores[order]
    return result.reset_index(drop=True)


def recommendation_examples(users, movies, user_ids, scores, observed, top_k=10):
    profiles = users.set_index("user_id")
    frames = []
    for row, uid in enumerate(user_ids):
        for algorithm, matrix in scores.items():
            rec = rank_unseen(movies, matrix[row], observed.get(int(uid), set()), top_k)
            rec.insert(0, "user_id", int(uid))
            rec.insert(1, "gender", profiles.loc[uid, "gender"])
            rec.insert(2, "age", int(profiles.loc[uid, "age"]))
            rec.insert(3, "algorithm", algorithm)
            frames.append(rec)
    return pd.concat(frames, ignore_index=True)
