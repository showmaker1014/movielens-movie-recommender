"""Export compact measured results for the five-minute presentation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary = json.loads(
        (ROOT / "data" / "processed" / "dataset_summary.json").read_text(
            encoding="utf-8"
        )
    )
    training = json.loads(
        (ROOT / "results" / "training_info.json").read_text(encoding="utf-8")
    )
    final = pd.read_csv(ROOT / "results" / "final_test_metrics.csv")
    validation = pd.read_csv(ROOT / "results" / "hyperparameter_tuning.csv")
    ratings = pd.read_csv(ROOT / "data" / "processed" / "ratings_split.csv.gz")
    samples = pd.read_csv(ROOT / "results" / "sample_user_recommendations.csv")

    rating_counts = (
        ratings["rating"].value_counts().sort_index().astype(int).to_dict()
    )
    sample_user = int(samples["user_id"].min())
    sample_rows = (
        samples[samples["user_id"] == sample_user]
        .sort_values(["algorithm", "rank"])
        .groupby("algorithm", as_index=False)
        .head(3)
    )
    payload = {
        "dataset": {
            "shapes": summary["shapes"],
            "rating_mean": summary["rating_mean"],
            "split_rows": summary["split_rows"],
            "null_columns": summary["null_columns"],
            "rating_counts": rating_counts,
        },
        "training": training,
        "final_metrics": final[
            ["algorithm", "HR@10", "NDCG@10", "MRR", "test_users"]
        ].to_dict(orient="records"),
        "validation_metrics": validation[
            ["algorithm", "config", "HR@10", "NDCG@10", "MRR"]
        ].to_dict(orient="records"),
        "sample_user": sample_user,
        "sample_recommendations": sample_rows[
            ["algorithm", "rank", "title", "genres", "release_year"]
        ].to_dict(orient="records"),
    }
    output = ROOT / "results" / "presentation_summary.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

