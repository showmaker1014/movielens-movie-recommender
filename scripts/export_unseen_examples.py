"""Re-export product examples from saved models, without repeating training."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import ALGORITHMS, load_bundle, score_all, user_index_for
from src.recommendation import example_user_ids, recommendation_examples


def main():
    bundle = load_bundle()
    chosen = example_user_ids(bundle["users"]["user_id"].to_numpy())
    scores = {algorithm: np.stack([score_all(algorithm, user_index_for(int(uid), bundle), bundle) for uid in chosen]) for algorithm in ALGORITHMS}
    observed = bundle["ratings"].groupby("user_id")["movie_id"].agg(set).to_dict()
    result = recommendation_examples(bundle["users"], bundle["movies"], chosen, scores, observed)
    for row in result.itertuples():
        assert row.movie_id not in observed[row.user_id]
    result.to_csv(ROOT / "results/sample_user_recommendations.csv", index=False)


if __name__ == "__main__":
    main()
