"""Train, tune, evaluate, and export the three recommenders."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    POSITIVE_THRESHOLD,
    SEED,
    build_index_maps,
    build_side_features,
    build_user_item_matrix,
    make_candidates,
    sequences_from_frame,
)
from src.metrics import ranking_metrics
from src.models.icsrec import (
    ICSRecInspired,
    ICSRecTrainingConfig,
    all_item_scores as ics_all_item_scores,
    candidate_scores as ics_candidate_scores,
    train_icsrec,
)
from src.models.item_knn import ItemKNNRecommender
from src.models.lightfm_adapted import (
    LightFMAdapted,
    LightFMTrainingConfig,
    candidate_scores as lightfm_candidate_scores,
    train_lightfm,
)
from src.plotting import make_result_figures, save_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "figures",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer epochs for a smoke run; do not use these numbers in the report.",
    )
    return parser.parse_args()


def evaluate_scores(
    algorithm: str,
    split: str,
    scores: np.ndarray,
    labels: np.ndarray,
    config: dict,
    elapsed_seconds: float,
    selected: bool = False,
) -> dict:
    row = ranking_metrics(scores, labels)
    return {
        "algorithm": algorithm,
        "split": split,
        "config": json.dumps(config, sort_keys=True),
        **row,
        "training_seconds": round(elapsed_seconds, 2),
        "selected": selected,
    }


def choose_best(rows: list[dict], algorithm: str) -> dict:
    candidates = [row for row in rows if row["algorithm"] == algorithm and row["split"] == "validation"]
    return max(candidates, key=lambda row: (row["NDCG@10"], row["HR@10"]))


def mark_selected(rows: list[dict], best: dict) -> None:
    for row in rows:
        if row["algorithm"] != best["algorithm"]:
            continue
        row["selected"] = bool(
            row["config"] == best["config"] and row["split"] in {"validation", "test"}
        )


def candidate_indices(
    candidate_user_ids: np.ndarray,
    user_to_idx: dict[int, int],
) -> np.ndarray:
    return np.asarray([user_to_idx[int(uid)] for uid in candidate_user_ids], dtype=np.int64)


def sample_recommendations(
    users: pd.DataFrame,
    movies: pd.DataFrame,
    user_ids: np.ndarray,
    scores: dict[str, np.ndarray],
    candidate_items: np.ndarray,
    top_k: int = 10,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    chosen = np.sort(rng.choice(user_ids, size=min(20, len(user_ids)), replace=False))
    movie_lookup = movies.set_index("movie_id")
    user_lookup = users.set_index("user_id")
    rows: list[dict] = []
    for uid in chosen:
        row_index = np.where(user_ids == uid)[0][0]
        for algorithm, score_matrix in scores.items():
            top = np.argsort(-score_matrix[row_index])[:top_k]
            for rank, candidate_index in enumerate(top, start=1):
                movie_id = int(candidate_items[row_index, candidate_index])
                movie = movie_lookup.loc[movie_id]
                rows.append(
                    {
                        "user_id": int(uid),
                        "gender": user_lookup.loc[int(uid), "gender"],
                        "age": int(user_lookup.loc[int(uid), "age"]),
                        "algorithm": algorithm,
                        "rank": rank,
                        "movie_id": movie_id,
                        "title": movie["title"],
                        "genres": movie["genres"],
                        "release_year": int(movie["release_year"]),
                        "score": float(score_matrix[row_index, candidate_index]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    processed = args.processed_dir.resolve()
    artifacts = args.artifacts_dir.resolve()
    results_dir = args.results_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    users = pd.read_csv(processed / "users.csv")
    movies = pd.read_csv(processed / "movies.csv")
    split = pd.read_csv(processed / "ratings_split.csv.gz")
    summary = json.loads((processed / "dataset_summary.json").read_text(encoding="utf-8"))
    user_to_idx, movie_to_idx, user_ids, movie_ids = build_index_maps(users, movies)
    users = users.sort_values("user_id").reset_index(drop=True)
    movies = movies.sort_values("movie_id").reset_index(drop=True)

    train_positive = split[(split["split"] == "train") & (split["rating"] >= POSITIVE_THRESHOLD)]
    train_matrix = build_user_item_matrix(train_positive, user_to_idx, movie_to_idx)
    user_features, item_features, user_feature_names, item_feature_names = build_side_features(
        users,
        movies,
    )
    candidates = {
        name: make_candidates(
            split,
            user_ids,
            movie_ids,
            user_to_idx,
            movie_to_idx,
            name,
        )
        for name in ["validation", "test"]
    }
    candidate_user_indices = {
        name: candidate_indices(data["user_ids"], user_to_idx)
        for name, data in candidates.items()
    }
    candidate_item_indices = {
        name: np.searchsorted(movie_ids, data["candidate_items"]).astype(np.int32)
        for name, data in candidates.items()
    }

    metadata = {
        "seed": SEED,
        "positive_threshold": POSITIVE_THRESHOLD,
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx,
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "user_feature_names": user_feature_names,
        "item_feature_names": item_feature_names,
        "dataset_summary": summary,
        "software": {
            "python": "3.12",
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
    }
    with (artifacts / "metadata.pkl").open("wb") as stream:
        pickle.dump(metadata, stream)

    metric_rows: list[dict] = []
    hyperparameter_rows: list[dict] = []
    training_info: dict = {"models": {}, "candidate_generation": {}}
    for name, data in candidates.items():
        training_info["candidate_generation"][name] = {
            "users": int(len(data["user_ids"])),
            "candidates_per_user": int(data["candidate_items"].shape[1]),
            "positive_items": int(data["labels"].sum()),
        }

    # 1) Lecture algorithm: item-based CF. Tune neighborhood size on validation.
    print("\n=== Item-based collaborative filtering ===")
    item_knn_configs = [30, 60, 120] if not args.quick else [30]
    item_knn_models: dict[int, ItemKNNRecommender] = {}
    for neighbor_count in item_knn_configs:
        started = time.perf_counter()
        model = ItemKNNRecommender(
            neighbor_count=neighbor_count,
            popularity_weight=0.02,
            shrink=5.0,
        ).fit(train_matrix)
        elapsed = time.perf_counter() - started
        validation_scores = model.score_candidates(
            candidate_user_indices["validation"],
            candidate_item_indices["validation"],
        )
        row = evaluate_scores(
            "ItemKNN",
            "validation",
            validation_scores,
            candidates["validation"]["labels"],
            {"neighbors": neighbor_count, "popularity_weight": 0.02},
            elapsed,
        )
        metric_rows.append(row)
        hyperparameter_rows.append(row)
        item_knn_models[neighbor_count] = model
        print(f"neighbors={neighbor_count}: val NDCG@10={row['NDCG@10']:.4f}")
    best_knn = choose_best(metric_rows, "ItemKNN")
    best_knn_k = int(json.loads(best_knn["config"])["neighbors"])
    mark_selected(metric_rows, best_knn)
    item_knn_model = item_knn_models[best_knn_k]
    started = time.perf_counter()
    test_scores = item_knn_model.score_candidates(
        candidate_user_indices["test"],
        candidate_item_indices["test"],
    )
    metric_rows.append(
        evaluate_scores(
            "ItemKNN",
            "test",
            test_scores,
            candidates["test"]["labels"],
            json.loads(best_knn["config"]),
            time.perf_counter() - started,
            selected=True,
        )
    )
    training_info["models"]["ItemKNN"] = json.loads(best_knn["config"])
    if item_knn_model.result is None:
        raise RuntimeError("ItemKNN result is unexpectedly empty")
    np.savez_compressed(
        artifacts / "item_knn.npz",
        similarity_data=item_knn_model.result.similarity.data,
        similarity_indices=item_knn_model.result.similarity.indices,
        similarity_indptr=item_knn_model.result.similarity.indptr,
        train_data=item_knn_model.result.train_matrix.data,
        train_indices=item_knn_model.result.train_matrix.indices,
        train_indptr=item_knn_model.result.train_matrix.indptr,
        popularity=item_knn_model.result.popularity,
        matrix_shape=np.asarray(item_knn_model.result.train_matrix.shape),
    )

    # 2) Reference algorithm: LightFM-style metadata embeddings.
    print("\n=== LightFM-style metadata embeddings ===")
    observed_train = [set() for _ in range(len(user_ids))]
    for user, item in zip(
        train_positive["user_id"].map(user_to_idx).astype(int),
        train_positive["movie_id"].map(movie_to_idx).astype(int),
    ):
        observed_train[user].add(item)
    train_users = train_positive["user_id"].map(user_to_idx).to_numpy(np.int64)
    train_items = train_positive["movie_id"].map(movie_to_idx).to_numpy(np.int64)
    lightfm_configs = [32, 64] if not args.quick else [32]
    lightfm_models: dict[int, LightFMAdapted] = {}
    lightfm_losses: dict[int, list[float]] = {}
    epochs = 3 if args.quick else 10
    for dimension in lightfm_configs:
        started = time.perf_counter()
        model = LightFMAdapted(
            n_users=len(user_ids),
            n_items=len(movie_ids),
            n_user_features=user_features.shape[1],
            n_item_features=item_features.shape[1],
            embedding_dim=dimension,
        )
        config = LightFMTrainingConfig(
            embedding_dim=dimension,
            epochs=epochs,
            batch_size=4096,
            learning_rate=0.01,
            seed=SEED,
        )
        losses = train_lightfm(
            model,
            train_users,
            train_items,
            user_features,
            item_features,
            config,
        )
        elapsed = time.perf_counter() - started
        validation_scores = lightfm_candidate_scores(
            model,
            candidate_user_indices["validation"],
            candidate_item_indices["validation"],
            user_features,
            item_features,
        )
        config_dict = {
            "embedding_dim": dimension,
            "epochs": epochs,
            "batch_size": 4096,
            "learning_rate": 0.01,
            "loss": "BPR",
        }
        row = evaluate_scores(
            "LightFM-adapted",
            "validation",
            validation_scores,
            candidates["validation"]["labels"],
            config_dict,
            elapsed,
        )
        metric_rows.append(row)
        hyperparameter_rows.append(row)
        lightfm_models[dimension] = model
        lightfm_losses[dimension] = losses
        print(f"dim={dimension}: val NDCG@10={row['NDCG@10']:.4f}")
    best_lfm = choose_best(metric_rows, "LightFM-adapted")
    best_lfm_dim = int(json.loads(best_lfm["config"])["embedding_dim"])
    mark_selected(metric_rows, best_lfm)
    best_lfm_model = lightfm_models[best_lfm_dim]
    started = time.perf_counter()
    test_scores = lightfm_candidate_scores(
        best_lfm_model,
        candidate_user_indices["test"],
        candidate_item_indices["test"],
        user_features,
        item_features,
    )
    metric_rows.append(
        evaluate_scores(
            "LightFM-adapted",
            "test",
            test_scores,
            candidates["test"]["labels"],
            json.loads(best_lfm["config"]),
            time.perf_counter() - started,
            selected=True,
        )
    )
    training_info["models"]["LightFM-adapted"] = json.loads(best_lfm["config"])
    torch.save(
        {
            "state_dict": best_lfm_model.state_dict(),
            "config": json.loads(best_lfm["config"]),
            "n_users": len(user_ids),
            "n_items": len(movie_ids),
            "n_user_features": user_features.shape[1],
            "n_item_features": item_features.shape[1],
        },
        artifacts / "lightfm_adapted.pt",
    )

    # 3) Recent algorithm: ICSRec-inspired latent-intent sequence model.
    print("\n=== ICSRec-inspired sequential recommender ===")
    sequences = sequences_from_frame(
        train_positive,
        user_ids,
        movie_to_idx,
        max_length=50,
    )
    ics_configs = [(32, 8), (64, 16)] if not args.quick else [(32, 8)]
    ics_models: dict[tuple[int, int], ICSRecInspired] = {}
    ics_histories: dict[tuple[int, int], list[dict[str, float]]] = {}
    ics_epochs = 2 if args.quick else 6
    for embedding_dim, n_intents in ics_configs:
        started = time.perf_counter()
        config = ICSRecTrainingConfig(
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim,
            n_intents=n_intents,
            epochs=ics_epochs,
            batch_size=256,
            learning_rate=0.003,
            seed=SEED,
        )
        model = ICSRecInspired(
            n_items=len(movie_ids),
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim,
            n_intents=n_intents,
        )
        history = train_icsrec(model, sequences, observed_train, config)
        elapsed = time.perf_counter() - started
        validation_scores = ics_candidate_scores(
            model,
            sequences,
            candidate_user_indices["validation"],
            candidate_item_indices["validation"],
        )
        config_dict = {
            "embedding_dim": embedding_dim,
            "n_intents": n_intents,
            "epochs": ics_epochs,
            "batch_size": 256,
            "learning_rate": 0.003,
            "loss": "BPR + intent InfoNCE",
        }
        row = evaluate_scores(
            "ICSRec-inspired",
            "validation",
            validation_scores,
            candidates["validation"]["labels"],
            config_dict,
            elapsed,
        )
        metric_rows.append(row)
        hyperparameter_rows.append(row)
        ics_models[(embedding_dim, n_intents)] = model
        ics_histories[(embedding_dim, n_intents)] = history
        print(
            f"dim={embedding_dim}, intents={n_intents}: "
            f"val NDCG@10={row['NDCG@10']:.4f}"
        )
    best_ics = choose_best(metric_rows, "ICSRec-inspired")
    best_ics_config = json.loads(best_ics["config"])
    best_ics_key = (
        int(best_ics_config["embedding_dim"]),
        int(best_ics_config["n_intents"]),
    )
    mark_selected(metric_rows, best_ics)
    best_ics_model = ics_models[best_ics_key]
    started = time.perf_counter()
    test_scores = ics_candidate_scores(
        best_ics_model,
        sequences,
        candidate_user_indices["test"],
        candidate_item_indices["test"],
    )
    metric_rows.append(
        evaluate_scores(
            "ICSRec-inspired",
            "test",
            test_scores,
            candidates["test"]["labels"],
            best_ics_config,
            time.perf_counter() - started,
            selected=True,
        )
    )
    training_info["models"]["ICSRec-inspired"] = best_ics_config
    torch.save(
        {
            "state_dict": best_ics_model.state_dict(),
            "config": best_ics_config,
            "n_items": len(movie_ids),
        },
        artifacts / "icsrec_inspired.pt",
    )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(results_dir / "metrics.csv", index=False)
    pd.DataFrame(hyperparameter_rows).to_csv(
        results_dir / "hyperparameter_tuning.csv",
        index=False,
    )
    selected_test = metrics_frame[
        (metrics_frame["split"] == "test") & (metrics_frame["selected"] == True)
    ]
    selected_test.to_csv(results_dir / "final_test_metrics.csv", index=False)
    (results_dir / "training_info.json").write_text(
        json.dumps(training_info, indent=2),
        encoding="utf-8",
    )

    # Export examples and score matrices for the comparison view.
    sample_user_indices = candidate_user_indices["test"]
    score_by_algorithm: dict[str, np.ndarray] = {}
    item_knn_scores = item_knn_model.score_candidates(
        sample_user_indices,
        candidate_item_indices["test"],
    )
    lightfm_scores = lightfm_candidate_scores(
        best_lfm_model,
        sample_user_indices,
        candidate_item_indices["test"],
        user_features,
        item_features,
    )
    ics_scores = ics_candidate_scores(
        best_ics_model,
        sequences,
        sample_user_indices,
        candidate_item_indices["test"],
    )
    score_by_algorithm = {
        "ItemKNN": item_knn_scores,
        "LightFM-adapted": lightfm_scores,
        "ICSRec-inspired": ics_scores,
    }
    sample_recommendations(
        users,
        movies,
        candidates["test"]["user_ids"],
        score_by_algorithm,
        candidates["test"]["candidate_items"],
        top_k=10,
    ).to_csv(results_dir / "sample_user_recommendations.csv", index=False)

    best_ics_history = ics_histories[best_ics_key]
    make_result_figures(metrics_frame, figures_dir)
    save_training_curves(
        lightfm_losses,
        best_ics_history,
        figures_dir,
    )
    print("\nFinal held-out test metrics:")
    print(
        selected_test[
            ["algorithm", "HR@10", "NDCG@10", "MRR", "test_users"]
        ].to_string(index=False)
    )
    print(f"\nArtifacts: {artifacts}")
    print(f"Results: {results_dir}")


if __name__ == "__main__":
    main()

