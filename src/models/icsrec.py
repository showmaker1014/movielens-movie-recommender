"""ICSRec with a CPU GRU encoder and the paper's three training objectives.

Independent implementation of Qin et al., WSDM 2024, sections 2.3--2.7:
https://arxiv.org/html/2310.14318v3
Reference behavior: https://github.com/QinHsiu/ICSRec (datasets.py/trainers.py).

Each training transition supplies a sliding-window subsequence. Other
subsequences sharing its next-item target provide CICL pairs. FICL uses actual
KMeans centroids of prefix representations, refreshed each epoch, with the
same-label false-negative mask in both contrastive objectives. Recommendation
uses full-catalog softmax cross entropy, not the previous BPR approximation.

Disclosed choices: GRU is an officially supported backbone; this version uses
packed sequences and scikit-learn CPU KMeans instead of FAISS/CUDA. Optional,
explicit example/cluster sample caps support CPU runs. Singleton target groups
have no distinct CICL partner and skip that objective. These choices do not
reproduce the paper's exact benchmark training setup. No pretrained weights
or validation/test interactions are used by training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F


class ICSRec(nn.Module):
    """GRU variant with separate padding and refreshed, nonlearned centroids."""

    def __init__(
        self,
        n_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        n_intents: int = 16,
        dropout: float = 0.1,
        max_length: int = 50,
    ) -> None:
        super().__init__()
        if min(n_items, embedding_dim, hidden_dim, n_intents, max_length) < 1:
            raise ValueError("Model dimensions, intent count, and window must be positive")
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_intents = n_intents
        self.max_length = max_length
        self.padding_idx = n_items
        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=n_items)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, embedding_dim)
        # Cluster IDs and the vectors they name are replaced together. They
        # are not optimizer parameters, so label permutations cannot drift.
        self.register_buffer("intent_centers", torch.zeros(n_intents, embedding_dim))
        self.register_buffer("active_intents", torch.tensor(0, dtype=torch.long))
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        with torch.no_grad():
            self.item_embedding.weight[self.padding_idx].zero_()

    def encode(self, padded_sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if padded_sequences.shape[0] == 0:
            return self.item_embedding.weight.new_empty((0, self.embedding_dim))
        if padded_sequences.shape[1] == 0:
            return self.item_embedding.weight.new_zeros((len(lengths), self.embedding_dim))
        embedded = self.embedding_dropout(self.item_embedding(padded_sequences))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False
        )
        _, state = self.encoder(packed)
        result = self.output(state[-1])
        # An empty history must not masquerade as a rating of item index zero.
        return result.masked_fill((lengths == 0).to(result.device)[:, None], 0.0)

    def item_scores(self, encoded: torch.Tensor) -> torch.Tensor:
        return encoded @ self.item_embedding.weight[: self.n_items].T


# Backward-compatible import only; old inspired-model checkpoints must be rebuilt.
ICSRecInspired = ICSRec


@dataclass
class ICSRecTrainingConfig:
    embedding_dim: int = 64
    hidden_dim: int = 64
    n_intents: int = 16
    epochs: int = 6
    batch_size: int = 256
    learning_rate: float = 0.003
    weight_decay: float = 1e-5
    intent_loss_weight: float = 0.10  # CICL lambda; retained API name
    fine_intent_loss_weight: float = 0.10
    temperature: float = 0.20
    seed: int = 42
    max_length: int = 50
    max_examples_per_epoch: int | None = None
    cluster_sample_size: int | None = None
    cluster_batch_size: int = 1024


@dataclass
class SubsequenceIndex:
    """Compact transition index: inputs exclude the next-item target itself."""

    user_indices: np.ndarray
    cutoffs: np.ndarray
    targets: np.ndarray
    target_groups: dict[int, np.ndarray]
    group_offsets: np.ndarray

    def __len__(self) -> int:
        return len(self.targets)


def build_subsequence_index(sequences: list[np.ndarray]) -> SubsequenceIndex:
    """Index every training prefix/target; sliding truncation happens on access."""
    user_chunks, cutoff_chunks, target_chunks = [], [], []
    for user, sequence in enumerate(sequences):
        if len(sequence) < 2:
            continue
        user_chunks.append(np.full(len(sequence) - 1, user, dtype=np.int64))
        cutoff_chunks.append(np.arange(1, len(sequence), dtype=np.int64))
        target_chunks.append(np.asarray(sequence[1:], dtype=np.int64))
    if not user_chunks:
        empty = np.empty(0, dtype=np.int64)
        return SubsequenceIndex(empty, empty.copy(), empty.copy(), {}, empty.copy())
    users = np.concatenate(user_chunks)
    cutoffs = np.concatenate(cutoff_chunks)
    targets = np.concatenate(target_chunks)
    order = np.argsort(targets, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(targets[order])) + 1)
    target_groups = {int(targets[group[0]]): group for group in groups}
    offsets = np.empty(len(targets), dtype=np.int64)
    for group in groups:
        offsets[group] = np.arange(len(group), dtype=np.int64)
    return SubsequenceIndex(users, cutoffs, targets, target_groups, offsets)


def sample_same_target_partners(
    index: SubsequenceIndex, rows: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly draw a different transition from each target group if possible."""
    partners = np.asarray(rows, dtype=np.int64).copy()
    valid = np.zeros(len(rows), dtype=bool)
    for batch_row, row in enumerate(rows):
        group = index.target_groups[int(index.targets[row])]
        if len(group) > 1:
            draw = int(rng.integers(len(group) - 1))
            # Skip the anchor without rejection loops or popularity-dependent cost.
            draw += int(draw >= index.group_offsets[row])
            partners[batch_row] = group[draw]
            valid[batch_row] = True
    return partners, valid


def pad_prefixes(
    sequences: list[np.ndarray],
    user_indices: np.ndarray,
    cutoffs: np.ndarray | None = None,
    max_length: int = 50,
    padding_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right pad the last window of each prefix, never its target or future."""
    prefixes = []
    for row, user in enumerate(user_indices):
        sequence = sequences[int(user)]
        cutoff = len(sequence) if cutoffs is None else int(cutoffs[row])
        if cutoff < 0 or cutoff > len(sequence):
            raise ValueError("Prefix cutoff is outside the sequence")
        prefixes.append(sequence[max(0, cutoff - max_length) : cutoff])
    lengths = np.asarray([len(prefix) for prefix in prefixes], dtype=np.int64)
    width = max(int(lengths.max(initial=0)), 1)
    padded = np.full((len(prefixes), width), padding_idx, dtype=np.int64)
    for row, prefix in enumerate(prefixes):
        padded[row, : len(prefix)] = prefix
    return torch.from_numpy(padded), torch.from_numpy(lengths)


def masked_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Symmetric Eq. (6): keep the pair, mask all same-label false negatives.

    Dot-product similarities follow the paper. Averaging over the 2B anchors
    differs only by a constant absorbed into the configured loss weights.
    """
    if temperature <= 0:
        raise ValueError("Contrastive temperature must be positive")
    if first.shape != second.shape or first.shape[0] != len(labels):
        raise ValueError("Contrastive views and labels must have matching batches")
    batch = len(first)
    if batch == 0:
        return (first.sum() + second.sum()) * 0.0
    representations = torch.cat((first, second), dim=0)
    logits = representations @ representations.T / temperature
    repeated_labels = torch.cat((labels, labels)).to(logits.device)
    mask = repeated_labels[:, None] == repeated_labels[None, :]
    anchor = torch.arange(2 * batch, device=logits.device)
    positive = (anchor + batch) % (2 * batch)
    mask[anchor, positive] = False
    logits = logits.masked_fill(mask, -torch.inf)
    return F.cross_entropy(logits, positive)


def fit_intent_centers(vectors: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """Fit CPU KMeans, returning the actual centers rather than arbitrary IDs."""
    if len(vectors) == 0:
        raise ValueError("Clustering requires training subsequences")
    count = min(n_clusters, len(vectors))
    fit = KMeans(n_clusters=count, random_state=seed, n_init=3, max_iter=20)
    fit.fit(np.asarray(vectors, dtype=np.float32))
    return np.asarray(fit.cluster_centers_, dtype=np.float32)


def fine_intent_loss(
    encoded: torch.Tensor, centers: torch.Tensor, temperature: float
) -> torch.Tensor:
    """FICL: each prefix queries the very centroids used in its positive pair."""
    if len(centers) == 0:
        raise ValueError("FICL requires fitted intent centers")
    centers = centers.detach()
    # Search unnormalized L2 centers, as in the official FAISS implementation;
    # normalize the selected prototype for its contrastive representation.
    distances = (
        encoded.detach().square().sum(dim=1, keepdim=True)
        - 2 * encoded.detach() @ centers.T
        + centers.square().sum(dim=1)[None, :]
    )
    assignments = distances.argmin(dim=1)
    prototypes = F.normalize(centers, dim=1)[assignments]
    return masked_contrastive_loss(encoded, prototypes, assignments, temperature)


def _encode_rows(
    model: ICSRec, sequences: list[np.ndarray], index: SubsequenceIndex, rows: np.ndarray
) -> torch.Tensor:
    padded, lengths = pad_prefixes(
        sequences, index.user_indices[rows], index.cutoffs[rows],
        max_length=model.max_length, padding_idx=model.padding_idx,
    )
    device = model.item_embedding.weight.device
    return model.encode(padded.to(device), lengths.to(device))


def train_icsrec(
    model: ICSRec,
    sequences: list[np.ndarray],
    observed_sets: list[set[int]],
    config: ICSRecTrainingConfig,
) -> list[dict[str, float]]:
    """Train only on supplied training sequences; full CE needs no negatives.

    ``observed_sets`` remains in the API for existing callers; full-catalog
    cross entropy does not use a sampled-negative pool. Caps, when supplied,
    sample from all training prefixes without filtering by held-out outcomes.
    """
    del observed_sets
    if config.batch_size < 1 or config.cluster_batch_size < 1 or config.epochs < 1:
        raise ValueError("Epoch and batch counts must be positive")
    if config.temperature <= 0 or config.max_length != model.max_length:
        raise ValueError("Use a positive temperature and matching model/config window")
    if config.n_intents != model.n_intents:
        raise ValueError("Model and configuration intent counts must match")
    for cap in (config.max_examples_per_epoch, config.cluster_sample_size):
        if cap is not None and cap < 1:
            raise ValueError("Sample caps must be positive or None")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    index = build_subsequence_index(sequences)
    if len(index) == 0:
        raise ValueError("ICSRec requires a training user with at least two positives")
    if index.targets.min() < 0 or index.targets.max() >= model.n_items:
        raise ValueError("Training targets are outside the model item catalog")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history = []
    device = model.item_embedding.weight.device
    for epoch in range(config.epochs):
        # Each training prefix gets its own representation and cluster query;
        # a label from the complete user sequence is never reused for it.
        model.eval()
        cluster_rows = np.arange(len(index), dtype=np.int64)
        if config.cluster_sample_size is not None and len(index) > config.cluster_sample_size:
            cluster_rows = rng.choice(cluster_rows, config.cluster_sample_size, replace=False)
        vector_batches = []
        with torch.inference_mode():
            for start in range(0, len(cluster_rows), config.cluster_batch_size):
                rows = cluster_rows[start : start + config.cluster_batch_size]
                vector_batches.append(_encode_rows(model, sequences, index, rows).cpu().numpy())
        vectors = np.concatenate(vector_batches)
        centers = fit_intent_centers(vectors, config.n_intents, config.seed + epoch)
        with torch.no_grad():
            model.intent_centers.zero_()
            model.intent_centers[: len(centers)].copy_(torch.as_tensor(centers, device=device))
            model.active_intents.fill_(len(centers))
        del vector_batches, vectors

        order = rng.permutation(len(index))
        if config.max_examples_per_epoch is not None:
            order = order[: config.max_examples_per_epoch]
        totals = {"rec_loss": 0.0, "cicl_loss": 0.0, "ficl_loss": 0.0, "total_loss": 0.0}
        pair_count = 0
        model.train()
        for start in range(0, len(order), config.batch_size):
            rows = order[start : start + config.batch_size]
            partners, paired = sample_same_target_partners(index, rows, rng)
            first = _encode_rows(model, sequences, index, rows)
            second = _encode_rows(model, sequences, index, partners)
            targets = torch.as_tensor(index.targets[rows], dtype=torch.long, device=device)
            rec_loss = F.cross_entropy(model.item_scores(first), targets)
            paired_tensor = torch.as_tensor(paired, device=device)
            cicl_loss = masked_contrastive_loss(
                first[paired_tensor], second[paired_tensor], targets[paired_tensor],
                config.temperature,
            )
            current_centers = model.intent_centers[: int(model.active_intents)]
            ficl_loss = fine_intent_loss(first, current_centers, config.temperature)
            ficl_loss = ficl_loss + fine_intent_loss(second, current_centers, config.temperature)
            loss = (
                rec_loss + config.intent_loss_weight * cicl_loss
                + config.fine_intent_loss_weight * ficl_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite ICSRec joint loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            for name, value in zip(totals, (rec_loss, cicl_loss, ficl_loss, loss)):
                totals[name] += float(value.detach()) * len(rows)
            pair_count += int(paired.sum())
        record = {
            "epoch": float(epoch + 1),
            **{name: value / len(order) for name, value in totals.items()},
            "training_subsequences": float(len(index)),
            "epoch_examples": float(len(order)),
            "cluster_examples": float(len(cluster_rows)),
            "paired_examples": float(pair_count),
        }
        history.append(record)
        print(
            f"ICSRec epoch {epoch + 1:02d}/{config.epochs}: "
            f"CE={record['rec_loss']:.5f}, CICL={record['cicl_loss']:.5f}, "
            f"FICL={record['ficl_loss']:.5f}, examples={len(order)}", flush=True,
        )
    model.eval()
    return history


def candidate_scores(
    model: ICSRec,
    sequences: list[np.ndarray],
    user_indices: np.ndarray,
    candidate_items: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    output = np.empty(candidate_items.shape, dtype=np.float32)
    model.eval()
    device = model.item_embedding.weight.device
    with torch.inference_mode():
        for start in range(0, len(user_indices), batch_size):
            stop = min(start + batch_size, len(user_indices))
            padded, lengths = pad_prefixes(
                sequences, user_indices[start:stop], max_length=model.max_length,
                padding_idx=model.padding_idx,
            )
            encoded = model.encode(padded.to(device), lengths.to(device))
            candidates = torch.as_tensor(candidate_items[start:stop], dtype=torch.long, device=device)
            vectors = model.item_embedding(candidates)
            output[start:stop] = (encoded[:, None, :] * vectors).sum(dim=2).cpu().numpy()
    return output


def all_item_scores(
    model: ICSRec,
    sequences: list[np.ndarray],
    user_indices: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    output = np.empty((len(user_indices), model.n_items), dtype=np.float32)
    model.eval()
    device = model.item_embedding.weight.device
    with torch.inference_mode():
        for start in range(0, len(user_indices), batch_size):
            stop = min(start + batch_size, len(user_indices))
            padded, lengths = pad_prefixes(
                sequences, user_indices[start:stop], max_length=model.max_length,
                padding_idx=model.padding_idx,
            )
            encoded = model.encode(padded.to(device), lengths.to(device))
            output[start:stop] = model.item_scores(encoded).cpu().numpy()
    return output
