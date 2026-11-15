import numpy as np
import pytest

from src.metrics import rank_scores, ranking_metrics


def test_single_positive_precision_and_recall_have_different_denominators():
    scores = np.tile(np.arange(12, 0, -1), (2, 1))
    labels = np.zeros_like(scores)
    labels[0, 0] = labels[1, 10] = 1
    result = ranking_metrics(scores, labels, k=10)
    assert result["HR@10"] == result["Recall@10"] == 0.5
    assert result["Precision@10"] == 0.05
    assert result["MRR@10"] == 0.5
    assert result["MRR"] == pytest.approx((1 + 1 / 11) / 2)


def test_ties_follow_candidate_order_instead_of_favoring_positive():
    labels = np.eye(100, dtype=np.int8)
    scores = np.zeros((100, 100))
    np.testing.assert_array_equal(rank_scores(scores, labels), np.arange(1, 101))
    result = ranking_metrics(scores, labels)
    assert result["HR@10"] == 0.1
    assert result["Precision@10"] == 0.01


@pytest.mark.parametrize("scores,labels", [
    (np.array([[np.nan, 1.0]]), np.array([[1, 0]])),
    (np.ones((1, 2)), np.array([[0.5, 0.5]])),
    (np.empty((0, 2)), np.empty((0, 2))),
])
def test_invalid_evaluation_inputs_are_rejected(scores, labels):
    with pytest.raises(ValueError):
        ranking_metrics(scores, labels, k=1)
