"""Streamlit interface for the MovieLens hybrid recommender."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from scipy import sparse

from src.data import POSITIVE_THRESHOLD, sequences_from_frame
from src.models.icsrec import ICSRecInspired, all_item_scores as ics_all_item_scores
from src.models.item_knn import ItemKNNRecommender, ItemKNNResult
from src.models.lightfm_adapted import LightFMAdapted


ROOT = Path(__file__).resolve().parent
ALGORITHMS = ["ItemKNN", "LightFM-adapted", "ICSRec-inspired"]
AGE_GROUPS = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}


@st.cache_resource(show_spinner=False)
def load_bundle() -> dict:
    metadata_path = ROOT / "artifacts" / "metadata.pkl"
    required = [
        metadata_path,
        ROOT / "artifacts" / "item_knn.npz",
        ROOT / "artifacts" / "lightfm_adapted.pt",
        ROOT / "artifacts" / "icsrec_inspired.pt",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing trained artifacts: "
            + ", ".join(missing)
            + ". Run `python scripts/prepare_data.py --download` and "
            "`python scripts/train_models.py` first."
        )
    with metadata_path.open("rb") as stream:
        metadata = pickle.load(stream)
    users = pd.read_csv(ROOT / "data" / "processed" / "users.csv")
    movies = pd.read_csv(ROOT / "data" / "processed" / "movies.csv")
    ratings = pd.read_csv(ROOT / "data" / "processed" / "ratings_split.csv.gz")
    users = users.sort_values("user_id").reset_index(drop=True)
    movies = movies.sort_values("movie_id").reset_index(drop=True)

    item_data = np.load(ROOT / "artifacts" / "item_knn.npz")
    shape = tuple(int(v) for v in item_data["matrix_shape"])
    train_matrix = sparse.csr_matrix(
        (
            item_data["train_data"],
            item_data["train_indices"],
            item_data["train_indptr"],
        ),
        shape=shape,
    )
    similarity = sparse.csr_matrix(
        (
            item_data["similarity_data"],
            item_data["similarity_indices"],
            item_data["similarity_indptr"],
        ),
        shape=(shape[1], shape[1]),
    )
    item_knn = ItemKNNRecommender()
    item_knn.result = ItemKNNResult(
        neighbor_count=0,
        similarity=similarity,
        train_matrix=train_matrix,
        popularity=item_data["popularity"],
    )

    lightfm_bundle = torch.load(
        ROOT / "artifacts" / "lightfm_adapted.pt",
        map_location="cpu",
        weights_only=False,
    )
    lightfm = LightFMAdapted(
        lightfm_bundle["n_users"],
        lightfm_bundle["n_items"],
        lightfm_bundle["n_user_features"],
        lightfm_bundle["n_item_features"],
        lightfm_bundle["config"]["embedding_dim"],
    )
    lightfm.load_state_dict(lightfm_bundle["state_dict"])
    lightfm.eval()
    from src.data import build_side_features

    user_features, item_features, _, _ = build_side_features(users, movies)

    ics_bundle = torch.load(
        ROOT / "artifacts" / "icsrec_inspired.pt",
        map_location="cpu",
        weights_only=False,
    )
    ics_config = ics_bundle["config"]
    icsrec = ICSRecInspired(
        n_items=ics_bundle["n_items"],
        embedding_dim=ics_config["embedding_dim"],
        hidden_dim=ics_config["embedding_dim"],
        n_intents=ics_config["n_intents"],
    )
    icsrec.load_state_dict(ics_bundle["state_dict"])
    icsrec.eval()

    train_positive = ratings[
        (ratings["split"] == "train") & (ratings["rating"] >= POSITIVE_THRESHOLD)
    ]
    sequences = sequences_from_frame(
        train_positive,
        metadata["user_ids"],
        metadata["movie_to_idx"],
        max_length=50,
    )
    return {
        "metadata": metadata,
        "users": users,
        "movies": movies,
        "ratings": ratings,
        "item_knn": item_knn,
        "lightfm": lightfm,
        "user_features": user_features,
        "item_features": item_features,
        "icsrec": icsrec,
        "sequences": sequences,
    }


def user_index_for(user_id: int, bundle: dict) -> int:
    return int(bundle["metadata"]["user_to_idx"][int(user_id)])


def score_all(algorithm: str, user_index: int, bundle: dict) -> np.ndarray:
    if algorithm == "ItemKNN":
        return bundle["item_knn"].score_users(np.asarray([user_index]))[0]
    if algorithm == "LightFM-adapted":
        user_t = torch.tensor([user_index], dtype=torch.long)
        user_features = torch.as_tensor(bundle["user_features"], dtype=torch.float32)
        item_features = torch.as_tensor(bundle["item_features"], dtype=torch.float32)
        with torch.inference_mode():
            return (
                bundle["lightfm"]
                .all_item_scores(user_t, user_features, item_features)[0]
                .cpu()
                .numpy()
            )
    return ics_all_item_scores(
        bundle["icsrec"],
        bundle["sequences"],
        np.asarray([user_index]),
    )[0]


def recommendations(
    user_id: int,
    algorithm: str,
    top_n: int,
    genre: str,
    minimum_year: int,
    bundle: dict,
) -> tuple[pd.DataFrame, np.ndarray]:
    user_index = user_index_for(user_id, bundle)
    movies = bundle["movies"]
    ratings = bundle["ratings"]
    scores = score_all(algorithm, user_index, bundle)
    observed = set(
        ratings.loc[ratings["user_id"] == user_id, "movie_id"].astype(int).tolist()
    )
    user_train = ratings[
        (ratings["user_id"] == user_id) & (ratings["split"] == "train")
    ]
    train_seen = user_train["movie_id"].astype(int).tolist()
    mask = ~movies["movie_id"].isin(observed).to_numpy()
    mask &= movies["release_year"].to_numpy() >= minimum_year
    if genre != "All genres":
        mask &= movies[genre].to_numpy() == 1
    candidate_idx = np.flatnonzero(mask)
    if len(candidate_idx) == 0:
        return pd.DataFrame(), scores
    candidate_scores = scores[candidate_idx]
    order = candidate_idx[np.argsort(-candidate_scores)[:top_n]]
    result = movies.loc[order, ["movie_id", "title", "genres", "release_year"]].copy()
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result.insert(len(result.columns), "model_score", scores[order])
    result = result.reset_index(drop=True)
    result.attrs["train_seen"] = train_seen
    return result, scores


def history_for(user_id: int, bundle: dict) -> pd.DataFrame:
    ratings = bundle["ratings"]
    movies = bundle["movies"].set_index("movie_id")
    history = ratings[
        (ratings["user_id"] == user_id) & (ratings["split"] == "train")
    ].sort_values("timestamp", ascending=False)
    history = history[["movie_id", "rating", "timestamp"]].copy()
    history["title"] = history["movie_id"].map(movies["title"])
    history["genres"] = history["movie_id"].map(movies["genres"])
    history["rated_at"] = pd.to_datetime(history["timestamp"], unit="s").dt.date
    return history[["title", "genres", "rating", "rated_at"]].head(12)


def metric_cards(metrics: pd.DataFrame) -> None:
    test = metrics[(metrics["split"] == "test") & (metrics["selected"] == True)]
    best = test.loc[test["NDCG@10"].idxmax()]
    columns = st.columns(3)
    columns[0].metric("Best algorithm", best["algorithm"])
    columns[1].metric("Test NDCG@10", f"{best['NDCG@10']:.3f}")
    columns[2].metric("Test users", f"{int(best['test_users']):,}")


def main() -> None:
    st.set_page_config(
        page_title="CineMatch | MovieLens Recommender",
        page_icon="🎬",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2rem;}
        .hero {
            background: linear-gradient(120deg, #0f1f3d 0%, #203b75 58%, #227b7a 100%);
            color: white; padding: 1.8rem 2rem; border-radius: 22px;
            box-shadow: 0 18px 45px rgba(18, 38, 80, .18); margin-bottom: 1.2rem;
        }
        .hero h1 {margin: 0 0 .35rem 0; font-size: 2.35rem; color: white;}
        .hero p {margin: 0; color: #dce7ff; max-width: 820px;}
        .eyebrow {font-size: .74rem; letter-spacing: .16em; text-transform: uppercase;
          color: #8fe5df; font-weight: 700; margin-bottom: .55rem;}
        div[data-testid="stMetric"] {
            background: #fff; border: 1px solid #e4e9f2; padding: .75rem 1rem;
            border-radius: 14px;
        }
        .small-note {color: #667085; font-size: .88rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">MovieLens 1M · Three-model comparison</div>
          <h1>CineMatch</h1>
          <p>Explore unseen movie recommendations from item-based collaborative filtering,
          hybrid metadata embeddings, and latent-intent sequential modeling.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        bundle = load_bundle()
    except FileNotFoundError as error:
        st.error(str(error))
        st.code(
            "python scripts/prepare_data.py --download\n"
            "python scripts/train_models.py\n"
            "streamlit run app.py",
            language="bash",
        )
        st.stop()

    metrics = pd.read_csv(ROOT / "results" / "metrics.csv")
    metric_cards(metrics)
    users = bundle["users"]
    movies = bundle["movies"]

    with st.sidebar:
        st.header("Recommendation setup")
        user_id = st.selectbox(
            "User",
            options=users["user_id"].astype(int).tolist(),
            help="Choose one of the anonymized MovieLens users.",
        )
        algorithm = st.selectbox("Algorithm", ALGORITHMS)
        top_n = st.slider("Number of movies", 5, 20, 10)
        genre = st.selectbox(
            "Genre filter",
            ["All genres"]
            + [c for c in movies.columns if movies[c].dtype == np.int64 and c not in {"movie_id", "release_year"}],
        )
        minimum_year = st.slider(
            "Released on or after",
            min_value=int(movies["release_year"].min()),
            max_value=int(movies["release_year"].max()),
            value=1980,
            step=5,
        )
        st.caption(f"Random seed: {bundle['metadata']['seed']} · Positive threshold: 4★")

    tab_recommend, tab_compare, tab_metrics, tab_data = st.tabs(
        ["Recommendations", "Compare models", "Evaluation", "Dataset evidence"]
    )
    with tab_recommend:
        left, right = st.columns([2.15, 1])
        rec, _ = recommendations(
            user_id,
            algorithm,
            top_n,
            genre,
            minimum_year,
            bundle,
        )
        selected_user = users.loc[users["user_id"] == user_id].iloc[0]
        with left:
            st.subheader(f"Top {top_n} for user {user_id}")
            st.caption(
                f"{selected_user['gender']} · {AGE_GROUPS.get(int(selected_user['age']), 'Unknown age group')} · "
                f"{algorithm} · movies already observed are excluded"
            )
            if rec.empty:
                st.warning("No unseen movies satisfy the filters. Try another genre or year.")
            else:
                st.dataframe(
                    rec[["rank", "title", "genres", "release_year", "model_score"]],
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "rank": st.column_config.NumberColumn("#", width="small"),
                        "model_score": st.column_config.NumberColumn(
                            "Model score", format="%.4f"
                        ),
                    },
                )
                st.download_button(
                    "Download recommendations",
                    rec.to_csv(index=False).encode("utf-8"),
                    file_name=f"recommendations_user_{user_id}.csv",
                    mime="text/csv",
                )
        with right:
            count = int(((bundle["ratings"]["user_id"] == user_id) & (bundle["ratings"]["split"] == "train")).sum())
            positive_count = int(
                (
                    (bundle["ratings"]["user_id"] == user_id)
                    & (bundle["ratings"]["split"] == "train")
                    & (bundle["ratings"]["rating"] >= 4)
                ).sum()
            )
            st.metric("Training ratings", count)
            st.metric("Liked training movies", positive_count)
            genre_counts = (
                bundle["movies"]
                .set_index("movie_id")
                .loc[
                    bundle["ratings"]
                    .query("user_id == @user_id and split == 'train' and rating >= 4")[
                        "movie_id"
                    ]
                ]["genres"]
                .str.split("|", regex=False)
                .explode()
                .value_counts()
                .head(6)
            )
            st.write("**Favourite genres**")
            st.bar_chart(genre_counts, horizontal=True, color="#16A6A1")
        st.write("**Recent training history**")
        st.dataframe(history_for(user_id, bundle), hide_index=True, width="stretch")

    with tab_compare:
        st.subheader("Same-user, three-algorithm comparison")
        st.caption(
            "All algorithms exclude every observed movie. The table shows each model's "
            "own top five under the selected filters."
        )
        compare_cols = st.columns(3)
        for column, name in zip(compare_cols, ALGORITHMS):
            rec, _ = recommendations(
                user_id,
                name,
                max(top_n, 5),
                genre,
                minimum_year,
                bundle,
            )
            with column:
                st.markdown(f"**{name}**")
                st.dataframe(
                    rec.head(5)[["title", "genres", "release_year"]],
                    hide_index=True,
                    width="stretch",
                )

    with tab_metrics:
        test = metrics[(metrics["split"] == "test") & (metrics["selected"] == True)]
        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Held-out test metrics")
            st.dataframe(
                test[["algorithm", "HR@10", "NDCG@10", "MRR", "test_users"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "HR@10": st.column_config.NumberColumn(format="%.4f"),
                    "NDCG@10": st.column_config.NumberColumn(format="%.4f"),
                    "MRR": st.column_config.NumberColumn(format="%.4f"),
                },
            )
            st.info(
                "There is one held-out relevant item per user, so Precision@10 and "
                "Recall@10 coincide with HR@10. NDCG and MRR capture ranking position."
            )
        with right:
            figure = ROOT / "figures" / "model_comparison.png"
            if figure.exists():
                st.image(str(figure), width="stretch")
        st.subheader("Validation selection")
        validation = metrics[metrics["split"] == "validation"]
        st.dataframe(
            validation[["algorithm", "config", "HR@10", "NDCG@10", "MRR"]],
            hide_index=True,
            width="stretch",
        )

    with tab_data:
        summary = bundle["metadata"]["dataset_summary"]
        columns = st.columns(3)
        for column, (name, shape) in zip(columns, summary["shapes"].items()):
            column.metric(name.title(), f"{shape[0]:,} × {shape[1]}")
        st.subheader("Requested inspection outputs")
        st.write("**Shapes:**", summary["shapes"])
        st.write("**Total null values:**", summary["null_counts"])
        st.write("**Null columns:**", summary["null_columns"])
        st.write("**Unique values per column:**", summary["unique_values"])
        with st.expander("Data types and comparison dataset"):
            st.json(summary["dtypes"])
            comparison = pd.read_csv(
                ROOT / "data" / "processed" / "dataset_comparison.csv"
            )
            st.dataframe(comparison, hide_index=True, width="stretch")
        st.subheader("Exploratory figures")
        figure_names = [
            ("Rating distribution", "rating_distribution.png"),
            ("Activity and popularity", "activity_distributions.png"),
            ("Temporal split", "temporal_split.png"),
            ("Genre distribution", "genre_distribution.png"),
        ]
        for title, filename in figure_names:
            path = ROOT / "figures" / filename
            if path.exists():
                st.markdown(f"**{title}**")
                st.image(str(path), width="stretch")


if __name__ == "__main__":
    main()

