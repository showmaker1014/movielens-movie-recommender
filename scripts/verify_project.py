"""Verify saved metrics and product examples against saved model scores."""
from __future__ import annotations
import argparse
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import ALGORITHMS, load_bundle, score_all, user_index_for
from src.data import make_candidates
from src.metrics import ranking_metrics
from src.models.lightfm_adapted import candidate_scores as light_scores
from src.models.icsrec import candidate_scores as intent_scores
from src.recommendation import example_user_ids, recommendation_examples


def verify():
    torch.set_num_threads(4)
    bundle = load_bundle()
    meta = bundle["metadata"]
    ratings = bundle["ratings"]
    bounds = ratings.groupby(["user_id", "split"])["timestamp"].agg(["min", "max"]).unstack("split")
    assert (bounds[("max", "train")] <= bounds[("min", "validation")]).all()
    assert (bounds[("max", "validation")] <= bounds[("min", "test")]).all()
    candidates = make_candidates(ratings, meta["user_ids"], meta["movie_ids"], meta["user_to_idx"], meta["movie_to_idx"], "test")
    users = np.array([meta["user_to_idx"][int(uid)] for uid in candidates["user_ids"]])
    items = np.searchsorted(meta["movie_ids"], candidates["candidate_items"])
    scores = {
        "ItemKNN": bundle["item_knn"].score_candidates(users, items),
        "LightFM-BPR": light_scores(bundle["lightfm"], users, items, bundle["user_features"], bundle["item_features"]),
        "ICSRec (GRU)": intent_scores(bundle["icsrec"], bundle["sequences"], users, items),
    }
    saved = pd.read_csv(ROOT / "results/final_test_metrics.csv").set_index("algorithm")
    all_metrics = pd.read_csv(ROOT / "results/metrics.csv")
    assert set(saved.index) == set(ALGORITHMS)
    evidence = {"python": platform.python_version(), "models": {}}
    for algorithm, values in scores.items():
        computed = ranking_metrics(values, candidates["labels"])
        keys = ["HR@10", "Recall@10", "Precision@10", "NDCG@10", "MRR", "MRR@10"]
        difference = max(abs(computed[key] - saved.loc[algorithm, key]) for key in keys)
        assert difference < 1e-7, (algorithm, difference)
        validation = all_metrics[(all_metrics.algorithm == algorithm) & (all_metrics["split"] == "validation")]
        best = validation.sort_values(["NDCG@10", "HR@10"], ascending=False).iloc[0]
        assert json.loads(best.config) == json.loads(saved.loc[algorithm, "config"])
        evidence["models"][algorithm] = {"test_users": computed["test_users"], "max_metric_difference": difference}
    chosen = example_user_ids(bundle["users"]["user_id"].to_numpy())
    full = {algorithm: np.stack([score_all(algorithm, user_index_for(int(uid), bundle), bundle) for uid in chosen]) for algorithm in ALGORITHMS}
    observed = ratings.groupby("user_id")["movie_id"].agg(set).to_dict()
    expected = recommendation_examples(bundle["users"], bundle["movies"], chosen, full, observed)
    actual = pd.read_csv(ROOT / "results/sample_user_recommendations.csv")
    columns = ["user_id", "algorithm", "rank", "movie_id", "title", "genres", "release_year"]
    pd.testing.assert_frame_equal(actual[columns], expected[columns], check_dtype=False)
    np.testing.assert_allclose(actual.model_score, expected.model_score, atol=1e-6, rtol=1e-6)
    assert actual.merge(ratings[["user_id", "movie_id"]], on=["user_id", "movie_id"]).empty
    counts = actual.groupby(["user_id", "algorithm"]).size()
    assert counts.ge(5).all()
    evidence["examples"] = {"rows": len(actual), "groups": len(counts), "minimum_movies": int(counts.min()), "observed_overlap": 0}
    evidence["status"] = "pass"
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/verification.json")
    args = parser.parse_args()
    evidence = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))
