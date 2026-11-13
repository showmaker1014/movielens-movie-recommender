"""Lecture algorithm: item-based collaborative filtering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class ItemKNNResult:
    neighbor_count: int
    similarity: sparse.csr_matrix
    train_matrix: sparse.csr_matrix
    popularity: np.ndarray


class ItemKNNRecommender:
    """Binary co-rating item KNN with cosine similarity.

    The item-user matrix is L2-normalized before multiplication, so each
    nonzero similarity is a cosine value. A small log-popularity prior breaks
    ties and supplies scores for items without co-rating neighbors.
    """

    def __init__(
        self,
        neighbor_count: int = 100,
        popularity_weight: float = 0.02,
        shrink: float = 5.0,
    ) -> None:
        self.neighbor_count = neighbor_count
        self.popularity_weight = popularity_weight
        self.shrink = shrink
        self.result: ItemKNNResult | None = None

    def fit(self, train_matrix: np.ndarray) -> "ItemKNNRecommender":
        user_item = sparse.csr_matrix(train_matrix, dtype=np.float32)
        item_user = user_item.T.tocsr()
        norm = np.sqrt(np.asarray(item_user.multiply(item_user).sum(axis=1)).ravel())
        inv = np.divide(1.0, norm, out=np.zeros_like(norm), where=norm > 0)
        normalized = sparse.diags(inv) @ item_user
        similarity = (normalized @ normalized.T).tocsr()
        similarity.setdiag(0)
        similarity.eliminate_zeros()

        # Keep each item's strongest neighbors and add co-rating shrinkage.
        similarity = similarity.tolil()
        for row in range(similarity.shape[0]):
            values = np.asarray(similarity.data[row])
            if len(values) <= self.neighbor_count:
                continue
            indices = similarity.rows[row]
            keep = np.argpartition(values, -self.neighbor_count)[-self.neighbor_count :]
            keep_indices = [indices[i] for i in keep]
            kept_values = values[keep]
            similarity.rows[row] = keep_indices
            similarity.data[row] = (kept_values / (1.0 + self.shrink / 100.0)).tolist()
        similarity = similarity.tocsr()
        popularity = np.asarray(user_item.sum(axis=0)).ravel().astype(np.float32)
        self.result = ItemKNNResult(
            neighbor_count=self.neighbor_count,
            similarity=similarity,
            train_matrix=user_item,
            popularity=popularity,
        )
        return self

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("Call fit before scoring")
        neighbor_scores = self.result.train_matrix[user_indices] @ self.result.similarity.T
        neighbor_scores = neighbor_scores.toarray().astype(np.float32, copy=False)
        prior = np.log1p(self.result.popularity)
        if prior.max() > 0:
            prior = prior / prior.max()
        scores = neighbor_scores + self.popularity_weight * prior[None, :]
        seen = self.result.train_matrix[user_indices].toarray() > 0
        scores[seen] = -np.inf
        return scores

    def score_candidates(
        self,
        user_indices: np.ndarray,
        candidate_items: np.ndarray,
    ) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("Call fit before scoring")
        profiles = self.result.train_matrix[user_indices]
        output = np.empty(candidate_items.shape, dtype=np.float32)
        for start in range(0, len(user_indices), 256):
            stop = min(start + 256, len(user_indices))
            subset = profiles[start:stop]
            batched = (subset @ self.result.similarity.T).toarray().astype(
                np.float32,
                copy=False,
            )
            prior = np.log1p(self.result.popularity)
            if prior.max() > 0:
                prior = prior / prior.max()
            batched += self.popularity_weight * prior[None, :]
            rows = np.arange(stop - start)[:, None]
            output[start:stop] = batched[rows, candidate_items[start:stop]]
        return output

