"""Data loading, profiling, temporal splitting, and evaluation candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SEED = 42
POSITIVE_THRESHOLD = 4
NEGATIVES_PER_USER = 99


def _read_dat(path: Path, names: list[str]) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=names,
        encoding="latin-1",
    )


def load_movielens(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the MovieLens 1M users, movies, and ratings tables."""
    users = _read_dat(raw_dir / "users.dat", ["user_id", "gender", "age", "occupation", "zip"])
    movies = _read_dat(raw_dir / "movies.dat", ["movie_id", "title", "genres"])
    ratings = _read_dat(
        raw_dir / "ratings.dat",
        ["user_id", "movie_id", "rating", "timestamp"],
    )
    users["zip_token"] = users["zip"].astype("category").cat.codes
    return users, movies, ratings


def extract_release_year(title: str) -> int | None:
    """Return the four-digit release year encoded at the end of a title."""
    if isinstance(title, str) and title.endswith(")") and len(title) >= 6:
        token = title[-5:-1]
        if token.isdigit():
            return int(token)
    return None


def prepare_movies(movies: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean movie metadata and expand the multi-label genre column."""
    out = movies.copy()
    out["release_year"] = out["title"].map(extract_release_year)
    out["release_year"] = out["release_year"].fillna(out["release_year"].median()).astype(int)
    out["decade"] = (out["release_year"] // 10 * 10).astype(str) + "s"
    genre_series = out["genres"].str.get_dummies(sep="|").astype("int8")
    genre_cols = sorted(genre_series.columns.tolist())
    out = pd.concat([out, genre_series[genre_cols]], axis=1)
    return out, genre_cols


def temporal_user_split(ratings: pd.DataFrame) -> pd.DataFrame:
    """Build a per-user chronological 80/10/10 split without leakage.

    Sorting happens independently inside each user's history. The earliest
    90% is visible for model fitting and validation tuning; the final 10%
    is reserved for the once-only test evaluation.
    """
    out = ratings.sort_values(["user_id", "timestamp", "movie_id"]).copy()
    out["rank_in_user"] = out.groupby("user_id").cumcount()
    out["user_count"] = out.groupby("user_id")["movie_id"].transform("size")
    relative = out["rank_in_user"] / out["user_count"]
    out["split"] = np.select(
        [relative < 0.8, relative < 0.9],
        ["train", "validation"],
        default="test",
    )
    out["relevant"] = (out["rating"] >= POSITIVE_THRESHOLD).astype("int8")
    return out.sort_values(["user_id", "timestamp", "movie_id"]).reset_index(drop=True)


def profile_tables(
    users: pd.DataFrame,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Create the dataset-inspection evidence requested by the brief."""
    tables = {
        "users": users.drop(columns=["zip_token"], errors="ignore"),
        "movies": movies.drop(
            columns=[
                "release_year",
                "decade",
                *[c for c in movies.columns if c not in {"movie_id", "title", "genres"}],
            ],
            errors="ignore",
        ),
        "ratings": ratings,
    }
    summary = {
        "shapes": {name: list(frame.shape) for name, frame in tables.items()},
        "null_counts": {name: frame.isna().sum().to_dict() for name, frame in tables.items()},
        "null_columns": {
            name: frame.columns[frame.isna().any()].tolist() for name, frame in tables.items()
        },
        "columns": {name: frame.columns.tolist() for name, frame in tables.items()},
        "dtypes": {
            name: {column: str(dtype) for column, dtype in frame.dtypes.items()}
            for name, frame in tables.items()
        },
        "unique_values": {
            name: {column: int(frame[column].nunique(dropna=False)) for column in frame.columns}
            for name, frame in tables.items()
        },
        "rating_min": float(ratings["rating"].min()),
        "rating_max": float(ratings["rating"].max()),
        "rating_mean": float(ratings["rating"].mean()),
        "timestamp_min": int(ratings["timestamp"].min()),
        "timestamp_max": int(ratings["timestamp"].max()),
        "duplicate_user_movie_pairs": int(ratings.duplicated(["user_id", "movie_id"]).sum()),
    }
    comparison_rows = []
    for name, frame in tables.items():
        comparison_rows.append(
            {
                "table": name,
                "rows": len(frame),
                "columns": frame.shape[1],
                "null_cells": int(frame.isna().sum().sum()),
                "duplicate_rows": int(frame.duplicated().sum()),
                "memory_mb": round(frame.memory_usage(deep=True).sum() / 1024**2, 3),
            }
        )
    return summary, pd.DataFrame(comparison_rows)


def save_processed(
    users: pd.DataFrame,
    movies: pd.DataFrame,
    split: pd.DataFrame,
    summary: dict,
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    users.to_csv(output_dir / "users.csv", index=False)
    movies.to_csv(output_dir / "movies.csv", index=False)
    split.to_csv(output_dir / "ratings_split.csv.gz", index=False, compression="gzip")
    comparison.to_csv(output_dir / "dataset_comparison.csv", index=False)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_index_maps(
    users: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int], np.ndarray, np.ndarray]:
    user_ids = np.sort(users["user_id"].unique())
    movie_ids = np.sort(movies["movie_id"].unique())
    user_to_idx = {int(value): idx for idx, value in enumerate(user_ids)}
    movie_to_idx = {int(value): idx for idx, value in enumerate(movie_ids)}
    return user_to_idx, movie_to_idx, user_ids, movie_ids


def build_user_item_matrix(
    frame: pd.DataFrame,
    user_to_idx: dict[int, int],
    movie_to_idx: dict[int, int],
) -> np.ndarray:
    """Return a dense binary matrix; use only after confirming memory is appropriate."""
    matrix = np.zeros((len(user_to_idx), len(movie_to_idx)), dtype=np.float32)
    rows = frame["user_id"].map(user_to_idx).to_numpy()
    cols = frame["movie_id"].map(movie_to_idx).to_numpy()
    matrix[rows, cols] = 1.0
    return matrix


def build_side_features(
    users: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Create standardized user and multi-hot item metadata features."""
    user_frame = users.copy()
    age_mean = user_frame["age"].mean()
    age_std = user_frame["age"].std(ddof=0)
    user_frame["age_standardized"] = (user_frame["age"] - age_mean) / age_std
    gender = pd.get_dummies(user_frame["gender"], prefix="gender", dtype=np.float32)
    occupation = pd.get_dummies(
        user_frame["occupation"].astype("category"),
        prefix="occupation",
        dtype=np.float32,
    )
    user_numeric = user_frame[["age_standardized"]].astype(np.float32)
    user_features = pd.concat([user_numeric, gender, occupation], axis=1).to_numpy(np.float32)

    movie_frame = movies.copy()
    year_mean = movie_frame["release_year"].mean()
    year_std = max(float(movie_frame["release_year"].std(ddof=0)), 1.0)
    movie_frame["year_standardized"] = (
        movie_frame["release_year"] - year_mean
    ) / year_std
    excluded = {"movie_id", "title", "genres", "decade", "release_year"}
    genre_cols = [c for c in movie_frame.columns if c not in excluded and movie_frame[c].dtype != "O"]
    item_features = np.column_stack(
        [
            movie_frame[["year_standardized"]].to_numpy(np.float32),
            movie_frame[genre_cols].to_numpy(np.float32),
        ]
    )
    user_feature_names = [
        *["age_standardized"],
        *gender.columns.tolist(),
        *occupation.columns.tolist(),
    ]
    item_feature_names = ["year_standardized", *genre_cols]
    return user_features, item_features, user_feature_names, item_feature_names


def make_candidates(
    split: pd.DataFrame,
    user_ids: np.ndarray,
    movie_ids: np.ndarray,
    user_to_idx: dict[int, int],
    movie_to_idx: dict[int, int],
    split_name: str,
    negatives_per_user: int = NEGATIVES_PER_USER,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """Create one temporal positive and shared sampled negatives per user.

    Negatives are sampled from movies the user never rated in any period.
    The same users, positives, negative items, and order are reused by all
    algorithms, making model comparisons paired and fair.
    """
    rng = np.random.default_rng(seed + (0 if split_name == "validation" else 1))
    split_frame = split[split["split"] == split_name]
    positives = (
        split_frame[split_frame["relevant"] == 1]
        .sort_values(["user_id", "timestamp", "movie_id"])
        .groupby("user_id", as_index=False)
        .tail(1)
        .sort_values("user_id")
    )
    all_seen = split.groupby("user_id")["movie_id"].agg(set).to_dict()
    eligible_users: list[int] = []
    candidate_items: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    positive_items: list[int] = []

    for _, row in positives.iterrows():
        uid = int(row["user_id"])
        positive = int(row["movie_id"])
        seen = all_seen.get(uid, set())
        unseen = np.array([mid for mid in movie_ids if int(mid) not in seen], dtype=np.int32)
        if len(unseen) < negatives_per_user:
            continue
        negatives = rng.choice(unseen, size=negatives_per_user, replace=False)
        items = np.concatenate([[positive], negatives]).astype(np.int32)
        target = np.zeros(negatives_per_user + 1, dtype=np.int8)
        target[0] = 1
        order = rng.permutation(len(items))
        eligible_users.append(uid)
        candidate_items.append(items[order])
        labels.append(target[order])
        positive_items.append(positive)

    del user_ids, user_to_idx, movie_to_idx  # kept in signature for API clarity
    return {
        "user_ids": np.asarray(eligible_users, dtype=np.int32),
        "candidate_items": np.stack(candidate_items),
        "labels": np.stack(labels),
        "positive_items": np.asarray(positive_items, dtype=np.int32),
    }


def positives_for_split(split: pd.DataFrame, name: str) -> pd.DataFrame:
    return split[(split["split"] == name) & (split["relevant"] == 1)].copy()


def sequences_from_frame(
    frame: pd.DataFrame,
    user_ids: np.ndarray,
    movie_to_idx: dict[int, int],
    max_length: int,
) -> list[np.ndarray]:
    """Return chronological positive-item sequences in internal item-index space."""
    source = frame.sort_values(["user_id", "timestamp", "movie_id"])
    grouped = {
        int(uid): group
        for uid, group in source.groupby("user_id", sort=False)
    }
    sequences: list[np.ndarray] = []
    for uid in user_ids:
        group = grouped.get(int(uid))
        if group is None or len(group) == 0:
            sequences.append(np.array([], dtype=np.int64))
            continue
        values = group["movie_id"].map(movie_to_idx).to_numpy(dtype=np.int64)
        sequences.append(values[-max_length:])
    return sequences


def split_counts(split: pd.DataFrame) -> pd.DataFrame:
    return (
        split.groupby(["split", "relevant"], observed=True)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["split", "relevant"])
    )


def ensure_columns(frame: Iterable[str], required: Iterable[str]) -> None:
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

