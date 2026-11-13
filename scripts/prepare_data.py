"""Download, inspect, and preprocess MovieLens 1M."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    load_movielens,
    prepare_movies,
    profile_tables,
    save_processed,
    temporal_user_split,
)
from src.plotting import make_eda_figures

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def downloadable_archive(raw_dir: Path) -> Path:
    archive = raw_dir / "ml-1m.zip"
    if archive.exists():
        return archive
    raw_dir.mkdir(parents=True, exist_ok=True)
    import urllib.request

    print(f"Downloading {DATASET_URL}")
    urllib.request.urlretrieve(DATASET_URL, archive)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "ml-1m",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "figures",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the official ZIP if the raw directory is empty.",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    required = [raw_dir / "ratings.dat", raw_dir / "movies.dat", raw_dir / "users.dat"]
    archive_hash = None
    if not all(path.exists() for path in required):
        if not args.download:
            raise FileNotFoundError(
                f"Raw MovieLens files are missing under {raw_dir}. "
                "Run again with --download or point --raw-dir at an extracted ml-1m folder."
            )
        archive = downloadable_archive(PROJECT_ROOT / "data" / "raw")
        archive_hash = sha256(archive)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(PROJECT_ROOT / "data" / "raw")

    users, movies, ratings = load_movielens(raw_dir)
    movies_prepared, genre_columns = prepare_movies(movies)
    summary, comparison = profile_tables(users, movies_prepared, ratings)
    split = temporal_user_split(ratings)
    summary.update(
        {
            "positive_threshold": 4,
            "split_rows": {
                name: int((split["split"] == name).sum())
                for name in ["train", "validation", "test"]
            },
            "genre_columns": genre_columns,
            "archive_sha256": archive_hash,
            "citation": (
                "F. Maxwell Harper and Joseph A. Konstan. 2015. "
                "The MovieLens Datasets: History and Context. "
                "ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19."
            ),
            "dataset_doi": "https://doi.org/10.1145/2827872",
            "dataset_url": "https://grouplens.org/datasets/movielens/1m/",
        }
    )
    save_processed(users, movies_prepared, split, summary, comparison, processed_dir)
    make_eda_figures(
        split,
        users,
        movies_prepared,
        genre_columns,
        args.figures_dir.resolve(),
    )
    print(json.dumps(summary["shapes"], indent=2))
    print("Null columns:", summary["null_columns"])
    print("Split rows:", summary["split_rows"])
    print(f"Processed files written to {processed_dir}")


if __name__ == "__main__":
    main()

