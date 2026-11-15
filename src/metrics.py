"""Ranking metrics for one-positive evaluation with shared candidates."""

from __future__ import annotations

import numpy as np


def rank_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return the 1-based rank of the single relevant item for each row."""
    if scores.ndim != 2 or scores.size == 0:
        raise ValueError("Scores must be a non-empty user-by-candidate matrix")
    if scores.shape != labels.shape:
        raise ValueError(f"Shape mismatch: {scores.shape} != {labels.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("Candidate scores must be finite")
    if not np.isin(labels, [0, 1]).all() or not np.all(labels.sum(axis=1) == 1):
        raise ValueError("Each candidate row must contain exactly one relevant item")
    # Candidate order is shuffled once with a fixed seed in make_candidates.
    # Stable sorting breaks ties by that order, never by the relevance label.
    order = np.argsort(-scores, axis=1, kind="stable")
    ordered_labels = np.take_along_axis(labels, order, axis=1)
    return (ordered_labels.argmax(axis=1) + 1).astype(np.int64)


def ranking_metrics(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> dict[str, float]:
    ranks = rank_scores(scores, labels)
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= scores.shape[1]:
        raise ValueError("k must be between 1 and the number of candidates")
    hit = ranks <= k
    return {
        f"HR@{k}": float(hit.mean()),
        f"Recall@{k}": float(hit.mean()),
        f"Precision@{k}": float(hit.mean() / k),
        f"NDCG@{k}": float(np.where(hit, 1.0 / np.log2(ranks + 1), 0.0).mean()),
        "MRR": float((1.0 / ranks).mean()),
        f"MRR@{k}": float(np.where(hit, 1.0 / ranks, 0.0).mean()),
        "mean_rank": float(ranks.mean()),
        "test_users": int(len(ranks)),
    }


def score_candidate_matrix(
    all_item_scores: np.ndarray,
    candidate_items: np.ndarray,
) -> np.ndarray:
    """Gather candidate scores from a user x item score matrix."""
    users = np.arange(candidate_items.shape[0])[:, None]
    return all_item_scores[users, candidate_items]

