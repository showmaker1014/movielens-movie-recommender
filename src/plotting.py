"""Static EDA and result figures for the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "ink": "#172033",
    "blue": "#2F6BFF",
    "teal": "#16A6A1",
    "orange": "#F59E0B",
    "rose": "#E15B64",
    "grid": "#DDE3EC",
    "muted": "#667085",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def make_eda_figures(
    split: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    genre_columns: list[str],
    output_dir: Path,
) -> None:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    rating_counts = split["rating"].value_counts().sort_index()
    axes[0].bar(
        rating_counts.index.astype(str),
        rating_counts.values,
        color=COLORS["blue"],
        width=0.68,
    )
    axes[0].set_title("Rating distribution")
    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel("Interactions")
    rating_share = split["rating"].value_counts(normalize=True).sort_index() * 100
    axes[1].bar(
        rating_share.index.astype(str),
        rating_share.values,
        color=COLORS["teal"],
        width=0.68,
    )
    axes[1].set_title("Rating share")
    axes[1].set_xlabel("Rating")
    axes[1].set_ylabel("Percent of interactions")
    fig.suptitle("MovieLens-1M feedback is skewed toward positive ratings", x=0.05, ha="left")
    fig.tight_layout()
    fig.savefig(output_dir / "rating_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    user_counts = split.groupby("user_id").size()
    item_counts = split.groupby("movie_id").size()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].hist(user_counts, bins=45, color=COLORS["blue"])
    axes[0].set_xlabel("Ratings per user")
    axes[0].set_ylabel("Users")
    axes[0].set_title("User activity")
    axes[1].hist(item_counts, bins=55, color=COLORS["orange"])
    axes[1].set_xlabel("Ratings per movie")
    axes[1].set_ylabel("Movies")
    axes[1].set_title("Item popularity long tail")
    fig.suptitle("Activity is uneven across users and items", x=0.05, ha="left")
    fig.tight_layout()
    fig.savefig(output_dir / "activity_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    genre_counts = movies[genre_columns].sum().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(genre_counts.index, genre_counts.values, color=COLORS["teal"])
    ax.set_title("Movies carrying each genre")
    ax.set_xlabel("Number of movies")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "genre_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    split_by_day = split.assign(
        date=pd.to_datetime(split["timestamp"], unit="s").dt.to_period("M").dt.to_timestamp()
    )
    monthly = split_by_day.groupby(["date", "split"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4.3))
    for name, color in [
        ("train", COLORS["blue"]),
        ("validation", COLORS["orange"]),
        ("test", COLORS["rose"]),
    ]:
        if name in monthly:
            ax.plot(monthly.index, monthly[name], label=name, color=color, linewidth=2)
    ax.set_title("Per-user temporal split follows time")
    ax.set_ylabel("Interactions")
    ax.set_xlabel("")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "temporal_split.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    age_counts = users["age"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(age_counts.index.astype(str), age_counts.values, color=COLORS["rose"])
    ax.set_title("User age groups")
    ax.set_xlabel("Age")
    ax.set_ylabel("Users")
    fig.tight_layout()
    fig.savefig(output_dir / "age_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_result_figures(
    metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    test = metrics[(metrics["split"] == "test") & (metrics["selected"] == True)].copy()
    if test.empty:
        return
    test = test.sort_values("NDCG@10", ascending=True)
    colors = [COLORS["blue"], COLORS["teal"], COLORS["orange"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(test))
    ax.barh(y - 0.18, test["HR@10"], height=0.34, color=COLORS["blue"], label="HR@10")
    ax.barh(y + 0.18, test["NDCG@10"], height=0.34, color=COLORS["orange"], label="NDCG@10")
    ax.set_yticks(y, test["algorithm"], fontweight="600")
    ax.set_xlim(0, max(test["HR@10"].max(), test["NDCG@10"].max()) * 1.2)
    ax.set_title("Held-out test ranking quality")
    ax.set_xlabel("Metric value")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(
    lightfm_losses: dict[int, list[float]],
    icsrec_history: list[dict[str, float]],
    output_dir: Path,
) -> None:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for dim, losses in lightfm_losses.items():
        axes[0].plot(range(1, len(losses) + 1), losses, marker="o", label=f"{dim}d")
    axes[0].set_title("LightFM-adapted BPR loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False)
    if icsrec_history:
        axes[1].plot(
            [row["epoch"] for row in icsrec_history],
            [row["bpr_loss"] for row in icsrec_history],
            marker="o",
            label="BPR",
        )
        axes[1].plot(
            [row["epoch"] for row in icsrec_history],
            [row["intent_loss"] for row in icsrec_history],
            marker="x",
            label="Intent InfoNCE",
        )
    axes[1].set_title("ICSRec-inspired losses")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

