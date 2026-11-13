"""Recent algorithm: a compact ICSRec-inspired sequential recommender."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class ICSRecInspired(nn.Module):
    """GRU sequence encoder with learnable latent-intent prototypes.

    The architecture preserves the central ICSRec idea: sequential user
    representations are clustered into latent intents, and an InfoNCE-style
    objective aligns each sequence with its intent prototype. A BPR next-item
    objective supplies the recommendation signal.
    """

    def __init__(
        self,
        n_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        n_intents: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_intents = n_intents
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.position_embedding = nn.Embedding(64, embedding_dim)
        self.encoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.intent_prototypes = nn.Parameter(torch.randn(n_intents, embedding_dim) * 0.05)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def encode(self, padded_sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if padded_sequences.shape[1] == 0:
            return torch.zeros(
                (padded_sequences.shape[0], self.embedding_dim),
                device=padded_sequences.device,
            )
        positions = torch.arange(padded_sequences.shape[1], device=padded_sequences.device)
        x = self.item_embedding(padded_sequences) + self.position_embedding(positions)[None, :, :]
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        return self.output(hidden[-1])


@dataclass
class ICSRecTrainingConfig:
    embedding_dim: int = 64
    hidden_dim: int = 64
    n_intents: int = 16
    epochs: int = 6
    batch_size: int = 256
    learning_rate: float = 0.003
    weight_decay: float = 1e-5
    intent_loss_weight: float = 0.10
    temperature: float = 0.20
    seed: int = 42


def pad_prefixes(
    sequences: list[np.ndarray],
    user_indices: np.ndarray,
    cutoffs: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefixes: list[np.ndarray] = []
    for index, user in enumerate(user_indices):
        sequence = sequences[int(user)]
        cutoff = len(sequence) if cutoffs is None else int(cutoffs[index])
        prefixes.append(sequence[:cutoff])
    lengths = np.array([max(len(prefix), 1) for prefix in prefixes], dtype=np.int64)
    width = max(int(v) for v in lengths)
    padded = np.zeros((len(prefixes), width), dtype=np.int64)
    for row, prefix in enumerate(prefixes):
        if len(prefix):
            padded[row, : len(prefix)] = prefix
    return torch.from_numpy(padded), torch.from_numpy(lengths)


def _kmeans_labels(vectors: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """Small deterministic KMeans used to refresh latent-intent assignments."""
    rng = np.random.default_rng(seed)
    if len(vectors) < n_clusters:
        return np.arange(len(vectors)) % max(len(vectors), 1)
    centers = vectors[rng.choice(len(vectors), size=n_clusters, replace=False)].copy()
    labels = np.zeros(len(vectors), dtype=np.int64)
    for _ in range(20):
        distances = (
            (vectors * vectors).sum(axis=1)[:, None]
            - 2 * vectors @ centers.T
            + (centers * centers).sum(axis=1)[None, :]
        )
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(n_clusters):
            members = vectors[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return labels


def train_icsrec(
    model: ICSRecInspired,
    sequences: list[np.ndarray],
    observed_sets: list[set[int]],
    config: ICSRecTrainingConfig,
) -> list[dict[str, float]]:
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    eligible_users = np.array(
        [index for index, sequence in enumerate(sequences) if len(sequence) >= 2],
        dtype=np.int64,
    )
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(config.epochs):
        # Refresh intent assignments from the current sequence encoder.
        model.eval()
        with torch.inference_mode():
            padded, lengths = pad_prefixes(sequences, eligible_users)
            vectors = model.encode(padded, lengths).cpu().numpy()
        labels = _kmeans_labels(vectors, config.n_intents, config.seed + epoch)
        model.train()
        order = rng.permutation(len(eligible_users))
        total_bpr = 0.0
        total_intent = 0.0
        batches = 0
        for start in range(0, len(order), config.batch_size):
            selected = order[start : start + config.batch_size]
            users = eligible_users[selected]
            cutoffs = np.array(
                [rng.integers(1, len(sequences[int(user)])) for user in users],
                dtype=np.int64,
            )
            targets = np.array(
                [sequences[int(user)][int(cutoff)] for user, cutoff in zip(users, cutoffs)],
                dtype=np.int64,
            )
            negatives = np.empty(len(users), dtype=np.int64)
            for row, user in enumerate(users):
                negative = int(rng.integers(0, model.n_items))
                while negative in observed_sets[int(user)]:
                    negative = int(rng.integers(0, model.n_items))
                negatives[row] = negative

            padded, lengths = pad_prefixes(sequences, users, cutoffs)
            encoded = model.encode(padded, lengths)
            target_t = torch.from_numpy(targets)
            negative_t = torch.from_numpy(negatives)
            target_vec = model.item_embedding(target_t)
            negative_vec = model.item_embedding(negative_t)
            pos_scores = (encoded * target_vec).sum(dim=1)
            neg_scores = (encoded * negative_vec).sum(dim=1)
            bpr_loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()

            normalized_encoded = torch.nn.functional.normalize(encoded, dim=1)
            normalized_prototypes = torch.nn.functional.normalize(
                model.intent_prototypes,
                dim=1,
            )
            intent_logits = (
                normalized_encoded @ normalized_prototypes.T
            ) / config.temperature
            intent_labels = torch.from_numpy(labels[selected])
            intent_loss = torch.nn.functional.cross_entropy(intent_logits, intent_labels)
            loss = bpr_loss + config.intent_loss_weight * intent_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_bpr += float(bpr_loss.item())
            total_intent += float(intent_loss.item())
            batches += 1
        row = {
            "epoch": float(epoch + 1),
            "bpr_loss": total_bpr / max(batches, 1),
            "intent_loss": total_intent / max(batches, 1),
        }
        history.append(row)
        print(
            f"ICSRec epoch {epoch + 1:02d}/{config.epochs}: "
            f"BPR={row['bpr_loss']:.5f}, intent={row['intent_loss']:.5f}"
        )
    model.eval()
    return history


def candidate_scores(
    model: ICSRecInspired,
    sequences: list[np.ndarray],
    user_indices: np.ndarray,
    candidate_items: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    output = np.empty(candidate_items.shape, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(user_indices), batch_size):
            stop = min(start + batch_size, len(user_indices))
            users = user_indices[start:stop]
            padded, lengths = pad_prefixes(sequences, users)
            encoded = model.encode(padded, lengths)
            candidates = torch.as_tensor(candidate_items[start:stop], dtype=torch.long)
            vectors = model.item_embedding(candidates)
            output[start:stop] = (encoded[:, None, :] * vectors).sum(dim=2).numpy()
    return output


def all_item_scores(
    model: ICSRecInspired,
    sequences: list[np.ndarray],
    user_indices: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    output = np.empty((len(user_indices), model.n_items), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(user_indices), batch_size):
            stop = min(start + batch_size, len(user_indices))
            users = user_indices[start:stop]
            padded, lengths = pad_prefixes(sequences, users)
            encoded = model.encode(padded, lengths)
            output[start:stop] = (encoded @ model.item_embedding.weight.T).numpy()
    return output

