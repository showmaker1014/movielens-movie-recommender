"""Reference algorithm: LightFM-style metadata embeddings with BPR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class InteractionDataset(Dataset):
    def __init__(
        self,
        users: np.ndarray,
        positives: np.ndarray,
        n_items: int,
        observed: list[set[int]],
        seed: int,
    ) -> None:
        self.users = users.astype(np.int64)
        self.positives = positives.astype(np.int64)
        self.n_items = n_items
        self.observed = observed
        self.seed = seed

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int) -> tuple[int, int, int]:
        user = int(self.users[index])
        positive = int(self.positives[index])
        rng = np.random.default_rng(self.seed + index)
        negative = int(rng.integers(0, self.n_items))
        while negative in self.observed[user]:
            negative = int(rng.integers(0, self.n_items))
        return user, positive, negative


class LightFMAdapted(nn.Module):
    """Hybrid factorizer with LightFM's additive metadata-embedding score.

    score(u, i) = b_u + b_i + <e_u + E_gender/age/occupation(u),
                         e_i + E_genre/year(i)>
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_user_features: int,
        n_item_features: int,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_user_features = n_user_features
        self.n_item_features = n_item_features
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_feature_embedding = nn.Embedding(n_user_features, embedding_dim)
        self.item_feature_embedding = nn.Embedding(n_item_features, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.normal_(self.user_feature_embedding.weight, std=0.02)
        nn.init.normal_(self.item_feature_embedding.weight, std=0.02)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def user_vectors(self, user_ids: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        direct = self.user_embedding(user_ids)
        side = user_features[user_ids] @ self.user_feature_embedding.weight
        return direct + side

    def item_vectors(self, item_ids: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        direct = self.item_embedding(item_ids)
        side = item_features[item_ids] @ self.item_feature_embedding.weight
        return direct + side

    def score(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        uv = self.user_vectors(user_ids, user_features)
        iv = self.item_vectors(item_ids, item_features)
        return (
            (uv * iv).sum(dim=1)
            + self.user_bias(user_ids).squeeze(1)
            + self.item_bias(item_ids).squeeze(1)
        )

    def all_item_scores(
        self,
        user_ids: torch.Tensor,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        uv = self.user_vectors(user_ids, user_features)
        all_items = torch.arange(self.n_items, device=user_ids.device)
        iv = self.item_vectors(all_items, item_features)
        return (
            uv @ iv.T
            + self.item_bias(all_items).squeeze(1)[None, :]
            + self.user_bias(user_ids)
        )


@dataclass
class LightFMTrainingConfig:
    embedding_dim: int = 64
    epochs: int = 8
    batch_size: int = 4096
    learning_rate: float = 0.01
    weight_decay: float = 1e-5
    seed: int = 42


def train_lightfm(
    model: LightFMAdapted,
    train_users: np.ndarray,
    train_items: np.ndarray,
    user_features: np.ndarray,
    item_features: np.ndarray,
    config: LightFMTrainingConfig,
) -> list[float]:
    """Train with BPR and uniformly sampled unobserved negatives."""
    torch.manual_seed(config.seed)
    observed_sets: list[set[int]] = [set() for _ in range(model.n_users)]
    for user, item in zip(train_users, train_items):
        observed_sets[int(user)].add(int(item))
    dataset = InteractionDataset(
        train_users,
        train_items,
        model.n_items,
        observed_sets,
        config.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    user_feature_tensor = torch.as_tensor(user_features, dtype=torch.float32)
    item_feature_tensor = torch.as_tensor(item_features, dtype=torch.float32)
    losses: list[float] = []
    model.train()
    for epoch in range(config.epochs):
        total = 0.0
        count = 0
        for users_t, positives_t, negatives_t in loader:
            users_t = users_t.long()
            positives_t = positives_t.long()
            negatives_t = negatives_t.long()
            pos_scores = model.score(users_t, positives_t, user_feature_tensor, item_feature_tensor)
            neg_scores = model.score(users_t, negatives_t, user_feature_tensor, item_feature_tensor)
            loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(loss.item()) * len(users_t)
            count += len(users_t)
        losses.append(total / max(count, 1))
        print(f"LightFM epoch {epoch + 1:02d}/{config.epochs}: BPR loss={losses[-1]:.5f}")
    model.eval()
    return losses


def candidate_scores(
    model: LightFMAdapted,
    user_indices: np.ndarray,
    candidate_items: np.ndarray,
    user_features: np.ndarray,
    item_features: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    user_feature_tensor = torch.as_tensor(user_features, dtype=torch.float32)
    item_feature_tensor = torch.as_tensor(item_features, dtype=torch.float32)
    output = np.empty(candidate_items.shape, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(user_indices), batch_size):
            stop = min(start + batch_size, len(user_indices))
            users_t = torch.as_tensor(user_indices[start:stop], dtype=torch.long)
            scores = model.all_item_scores(users_t, user_feature_tensor, item_feature_tensor)
            rows = torch.arange(stop - start)
            items_t = torch.as_tensor(candidate_items[start:stop], dtype=torch.long)
            output[start:stop] = scores[rows[:, None], items_t].cpu().numpy()
    return output

