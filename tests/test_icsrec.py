"""Regressions for the ICSRec mechanisms required by the project brief."""

from __future__ import annotations

import io

import numpy as np
import torch

from src.models.icsrec import (
    ICSRec,
    ICSRecTrainingConfig,
    all_item_scores,
    build_subsequence_index,
    candidate_scores,
    fine_intent_loss,
    fit_intent_centers,
    masked_contrastive_loss,
    pad_prefixes,
    sample_same_target_partners,
    train_icsrec,
)


def test_sliding_subsequences_keep_early_targets_and_exclude_future() -> None:
    sequences = [np.arange(7), np.array([7, 8]), np.array([], dtype=int)]
    index = build_subsequence_index(sequences)
    assert len(index) == 7
    assert index.targets.tolist() == [1, 2, 3, 4, 5, 6, 8]
    padded, lengths = pad_prefixes(
        sequences, index.user_indices, index.cutoffs, max_length=3, padding_idx=9
    )
    assert padded[0].tolist() == [0, 9, 9]
    assert padded[5].tolist() == [3, 4, 5]
    for row, (user, cutoff, target) in enumerate(
        zip(index.user_indices, index.cutoffs, index.targets)
    ):
        prefix = padded[row, : lengths[row]].numpy()
        np.testing.assert_array_equal(prefix, sequences[user][max(0, cutoff - 3) : cutoff])
        assert target not in prefix  # these test histories have unique items


def test_cicl_partners_are_distinct_transitions_with_same_target() -> None:
    sequences = [np.array([0, 3, 5]), np.array([1, 3, 6]), np.array([2, 3, 5])]
    index = build_subsequence_index(sequences)
    rows = np.arange(len(index))
    partners, valid = sample_same_target_partners(index, rows, np.random.default_rng(42))
    np.testing.assert_array_equal(index.targets, index.targets[partners])
    assert np.all(rows[valid] != partners[valid])
    assert np.all(index.targets[~valid] == 6)
    repeated, repeated_valid = sample_same_target_partners(index, rows, np.random.default_rng(42))
    np.testing.assert_array_equal(partners, repeated)
    np.testing.assert_array_equal(valid, repeated_valid)


def test_false_negative_mask_keeps_positives_but_excludes_same_label_views() -> None:
    first = torch.zeros(2, 3, requires_grad=True)
    second = torch.zeros(2, 3, requires_grad=True)
    # With distinct labels, every anchor sees its positive and two negatives.
    distinct = masked_contrastive_loss(first, second, torch.tensor([1, 2]), 0.2)
    assert np.isclose(float(distinct), np.log(3))
    # With one shared target, every other view is a false negative; the
    # designated positive remains, so the loss is finite and exactly zero.
    shared = masked_contrastive_loss(first, second, torch.tensor([1, 1]), 0.2)
    assert float(shared) == 0.0
    shared.backward()
    assert torch.isfinite(first.grad).all()
    assert torch.isfinite(second.grad).all()


def test_ficl_centers_are_real_means_and_cluster_permutation_is_invariant() -> None:
    vectors = np.array([[0, 0], [0, 2], [10, 0], [10, 2]], dtype=np.float32)
    centers = fit_intent_centers(vectors, 2, 42)
    centers = centers[np.argsort(centers[:, 0])]
    np.testing.assert_allclose(centers, [[0, 1], [10, 1]], atol=1e-6)
    encoded = torch.tensor([[0.1, 1.0], [10.0, 1.0]], requires_grad=True)
    original = fine_intent_loss(encoded, torch.tensor(centers), 0.2)
    permuted = fine_intent_loss(encoded, torch.tensor(centers[::-1].copy()), 0.2)
    torch.testing.assert_close(original, permuted)
    original.backward()
    assert torch.isfinite(encoded.grad).all()


def test_empty_history_is_zero_and_padding_is_not_item_zero() -> None:
    torch.manual_seed(42)
    model = ICSRec(9, embedding_dim=4, hidden_dim=4, n_intents=2)
    sequences = [np.array([], dtype=np.int64), np.array([0, 1])]
    scores = all_item_scores(model, sequences, np.array([0, 1]))
    np.testing.assert_array_equal(scores[0], np.zeros(9))
    padded, lengths = pad_prefixes(sequences, np.array([0, 1]), padding_idx=model.padding_idx)
    assert padded[0].tolist() == [9, 9]
    assert lengths.tolist() == [0, 2]
    assert np.isfinite(scores).all()


def test_three_objective_training_and_checkpoint_round_trip() -> None:
    torch.manual_seed(7)
    model = ICSRec(10, embedding_dim=4, hidden_dim=4, n_intents=12)
    sequences = [np.array([0, 3, 5, 7]), np.array([1, 3, 6, 7]), np.array([2, 3, 5, 8])]
    config = ICSRecTrainingConfig(
        embedding_dim=4, hidden_dim=4, n_intents=12,
        epochs=2, batch_size=9, seed=7,
    )
    before = model.item_embedding.weight.detach().clone()
    history = train_icsrec(model, sequences, [set(s) for s in sequences], config)
    assert len(history) == 2
    assert history[0]["training_subsequences"] == 9
    assert history[0]["epoch_examples"] == 9
    assert history[0]["paired_examples"] > 0
    assert all(np.isfinite(list(row.values())).all() for row in history)
    assert history[0]["rec_loss"] > 0
    assert history[0]["cicl_loss"] > 0
    assert history[0]["ficl_loss"] > 0
    assert not torch.equal(before, model.item_embedding.weight)
    assert "intent_centers" not in dict(model.named_parameters())
    assert int(model.active_intents) == 9
    candidates = np.array([[0, 1, 2], [3, 4, 5]])
    scores = candidate_scores(model, sequences, np.array([0, 1]), candidates)
    full = all_item_scores(model, sequences, np.array([0, 1]))
    np.testing.assert_allclose(scores, np.take_along_axis(full, candidates, axis=1), atol=1e-6)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = ICSRec(10, embedding_dim=4, hidden_dim=4, n_intents=12)
    restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
    np.testing.assert_allclose(
        all_item_scores(restored, sequences, np.array([0, 1])), full, atol=1e-6
    )
