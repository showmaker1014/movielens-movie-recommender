"""Ranking metrics for one-positive evaluation with shared candidates."""

from __future__ import annotations

import numpy as np


def rank_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return the 1-based rank of the single relevant item for each row."""
    if scores.shape != labels.shape:
        raise ValueError(f"Shape mismatch: {scores.shape} != {labels.shape}")
    if not np.all(labels.sum(axis=1) == 1):
        raise ValueError("Each candidate row must contain exactly one relevant item")
    positive_scores = scores[labels == 1]
    ranks = 1 + (scores > positive_scores[:, None]).sum(axis=1)
    return ranks.astype(np.int64)


def ranking_metrics(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> dict[str, float]:
    ranks = rank_scores(scores, labels)
    hit = ranks <= k
    return {
        f"HR@{k}": float(hit.mean()),
        f"Recall@{k}": float(hit.mean()),
        f"Precision@{k}": float(hit.mean()),
        f"NDCG@{k}": float(np.where(hit, 1.0 / np.log2(ranks + 1), 0.0).mean()),
        "MRR": float(np.where(hit, 1.0 / ranks, 0.0).mean()),
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

