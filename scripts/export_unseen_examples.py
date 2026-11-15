"""Re-export product examples from saved models, without repeating training."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import ALGORITHMS, load_bundle, recommendations


def main():
    bundle = load_bundle()
    user_ids = bundle["users"]["user_id"].to_numpy()
    chosen = np.sort(np.random.default_rng(42).choice(user_ids, min(20, len(user_ids)), replace=False))
    frames = []
    for uid in chosen:
        seen = set(bundle["ratings"].loc[bundle["ratings"]["user_id"] == uid, "movie_id"])
        for algorithm in ALGORITHMS:
            rec, _ = recommendations(int(uid), algorithm, 10, "All genres", 0, bundle)
            assert len(rec) >= 5 and not seen.intersection(rec.movie_id)
            rec.insert(0, "user_id", int(uid))
            rec.insert(1, "algorithm", algorithm)
            frames.append(rec)
    pd.concat(frames, ignore_index=True).to_csv(ROOT / "results/sample_user_recommendations.csv", index=False)


if __name__ == "__main__":
    main()
