"""Regression checks for the LightFM feature model and BPR sampler."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from src.models.lightfm_adapted import (
    InteractionDataset,
    LightFMAdapted,
    LightFMTrainingConfig,
    candidate_scores,
    train_lightfm,
)


def test_lightfm_scores_include_weighted_feature_embeddings_and_biases() -> None:
    model = LightFMAdapted(2, 3, 2, 2, embedding_dim=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.user_embedding.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        model.item_embedding.weight.copy_(torch.tensor([[2.0, 1.0], [1.0, 0.0], [0.0, 2.0]]))
        model.user_feature_embedding.weight.copy_(torch.eye(2))
        model.item_feature_embedding.weight.copy_(torch.tensor([[0.5, 1.0], [1.0, 0.5]]))
        model.user_bias.weight.copy_(torch.tensor([[0.1], [0.2]]))
        model.item_bias.weight.copy_(torch.tensor([[0.2], [0.3], [0.4]]))
        model.user_feature_bias.weight.copy_(torch.tensor([[0.5], [1.0]]))
        model.item_feature_bias.weight.copy_(torch.tensor([[0.75], [-0.25]]))
    user_features = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    item_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    # Independently calculated from the paper's weighted feature formula.
    expected = torch.tensor([[10.55, 5.65, 11.5], [22.65, 11.25, 28.6]])
    torch.testing.assert_close(
        model.all_item_scores(torch.tensor([0, 1]), user_features, item_features), expected
    )
    users = torch.tensor([0, 0, 0, 1, 1, 1])
    items = torch.tensor([0, 1, 2, 0, 1, 2])
    torch.testing.assert_close(
        model.score(users, items, user_features, item_features), expected.flatten()
    )
    sampled = candidate_scores(
        model, np.array([1, 0]), np.array([[2, 0], [1, 2]]),
        user_features.numpy(), item_features.numpy(), batch_size=1,
    )
    np.testing.assert_allclose(sampled, [[28.6, 22.65], [5.65, 11.5]], rtol=1e-5)


def test_lightfm_metadata_biases_receive_gradients() -> None:
    model = LightFMAdapted(1, 2, 2, 2, embedding_dim=2)
    user_features = torch.tensor([[1.0, 2.0]])
    item_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # Direct score gradients verify both feature-bias paths participate.
    model.score(torch.tensor([0]), torch.tensor([0]), user_features, item_features).sum().backward()
    torch.testing.assert_close(model.user_feature_bias.weight.grad, torch.tensor([[1.0], [2.0]]))
    torch.testing.assert_close(model.item_feature_bias.weight.grad, torch.tensor([[1.0], [0.0]]))
    model.zero_grad()
    positive = model.score(torch.tensor([0]), torch.tensor([0]), user_features, item_features)
    negative = model.score(torch.tensor([0]), torch.tensor([1]), user_features, item_features)
    (-torch.nn.functional.logsigmoid(positive - negative).mean()).backward()
    # BPR pushes up positive metadata and down negative metadata; user-only
    # biases cancel for the same user, which is expected rather than a bug.
    assert model.item_feature_bias.weight.grad[0, 0] < 0
    assert model.item_feature_bias.weight.grad[1, 0] > 0
    torch.testing.assert_close(model.user_feature_bias.weight.grad, torch.zeros(2, 1))


def test_negative_samples_refresh_reproducibly_and_exclude_all_training_positives() -> None:
    users = np.tile(np.arange(3, dtype=np.int64), 40)
    positives = users.copy()
    observed = [{0, 3, 6}, {1, 4}, {2, 5, 7}]
    first = InteractionDataset(users, positives, 16, observed, seed=42)
    second = InteractionDataset(users, positives, 16, observed, seed=42)
    epoch_zero = first.negatives.copy()
    np.testing.assert_array_equal(epoch_zero, second.negatives)
    first.set_epoch(1)
    second.set_epoch(1)
    assert np.any(first.negatives != epoch_zero)
    np.testing.assert_array_equal(first.negatives, second.negatives)
    for index in range(len(first)):
        user, positive, negative = first[index]
        assert positive in observed[user]
        assert negative not in observed[user]
    first.set_epoch(0)
    np.testing.assert_array_equal(first.negatives, epoch_zero)


def test_negative_sampling_rejects_users_without_an_available_negative() -> None:
    # A catalog fully covered by positives must raise instead of looping forever.
    with np.testing.assert_raises(ValueError):
        InteractionDataset(np.array([0]), np.array([0]), 3, [{0, 1, 2}], seed=42)


def test_lightfm_bpr_training_is_repeatable_from_identical_initial_parameters() -> None:
    torch.manual_seed(7)
    first = LightFMAdapted(2, 6, 2, 2, embedding_dim=4)
    second = deepcopy(first)
    user_features = np.eye(2, dtype=np.float32)
    item_features = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0]], dtype=np.float32)
    users = np.array([0, 0, 1, 1])
    items = np.array([0, 1, 2, 3])
    config = LightFMTrainingConfig(embedding_dim=4, epochs=4, batch_size=2, seed=19)
    first_losses = train_lightfm(first, users, items, user_features, item_features, config)
    torch.rand(17)  # Unrelated global RNG use must not change the shuffle stream.
    second_losses = train_lightfm(second, users, items, user_features, item_features, config)
    assert np.isfinite(first_losses).all()
    np.testing.assert_allclose(first_losses, second_losses, rtol=0, atol=0)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    assert torch.count_nonzero(first.item_feature_bias.weight).item() > 0


def test_lightfm_bpr_can_learn_a_simple_preference() -> None:
    torch.manual_seed(5)
    model = LightFMAdapted(1, 4, 1, 2, embedding_dim=4)
    features_u = np.ones((1, 1), dtype=np.float32)
    features_i = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    train_lightfm(
        model, np.array([0, 0]), np.array([0, 1]), features_u, features_i,
        LightFMTrainingConfig(embedding_dim=4, epochs=25, batch_size=2, learning_rate=0.05, seed=7),
    )
    scores = candidate_scores(model, np.array([0]), np.array([[0, 1, 2, 3]]), features_u, features_i)[0]
    assert scores[:2].min() > scores[2:].max()
