"""Generate a technical report in Markdown and PDF from measured results."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BLUE = colors.HexColor("#2F6BFF")
NAVY = colors.HexColor("#172033")
TEAL = colors.HexColor("#16A6A1")
MUTED = colors.HexColor("#667085")
LIGHT = colors.HexColor("#EEF3FB")
BORDER = colors.HexColor("#DDE3EC")


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:{floatfmt}}")
    headers = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([headers, divider, *rows])


def pdf_table(
    data: list[list[str]],
    widths: list[float],
    font_size: float = 8.3,
    header: bool = True,
) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("TEXTCOLOR", (0, 0), (-1, -1), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
        style.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]))
    table.setStyle(TableStyle(style))
    return table


def scaled_image(path: Path, max_width: float, max_height: float | None = None) -> Image:
    from PIL import Image as PILImage

    with PILImage.open(path) as image:
        width, height = image.size
    scale = max_width / width
    if max_height and height * scale > max_height:
        scale = max_height / height
    return Image(str(path), width=width * scale, height=height * scale)


def page_decor(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.6)
    canvas.line(doc.leftMargin, 0.55 * inch, letter[0] - doc.rightMargin, 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 0.34 * inch, "MovieLens Hybrid Movie Recommender")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.34 * inch, str(doc.page))
    canvas.restoreState()


def build_markdown(
    summary: dict,
    metrics: pd.DataFrame,
    validation: pd.DataFrame,
    sample: pd.DataFrame,
    training_info: dict,
) -> str:
    shapes = summary["shapes"]
    final = metrics[(metrics["split"] == "test") & (metrics["selected"] == True)]
    winner = final.loc[final["NDCG@10"].idxmax()]
    lines = [
        "# MovieLens Hybrid Movie Recommender",
        "",
        "**Technical report - Group Project, Project 1**",
        "",
        "Date: 13 November 2026",
        "",
        "## Abstract",
        "",
        (
            "This project builds and evaluates a movie recommendation system on the "
            "MovieLens 1M dataset. The system uses three distinct models: item-based "
            "collaborative filtering, a LightFM-style hybrid metadata factorizer, and "
            "an ICSRec-inspired sequential recommender with latent intents. All models "
            "share a per-user chronological split and identical sampled candidates. "
            f"The strongest held-out test model by NDCG@10 is **{winner['algorithm']}** "
            f"at {winner['NDCG@10']:.4f}, with HR@10 of {winner['HR@10']:.4f}."
        ),
        "",
        "The project also includes a Streamlit interface that accepts an anonymized user "
        "and displays unseen movie titles and genres.",
        "",
        "## 1. Task and success criteria",
        "",
        "Given a user, the recommender must rank at least five movies that the user has "
        "not rated. The deliverable must cover dataset evidence, preprocessing, exactly "
        "three algorithms, comparative evaluation, and an interactive application.",
        "",
        "The operational success criteria are:",
        "",
        "- Reproducible data split and random seed.",
        "- No held-out ratings in training features or learned popularity counts.",
        "- A shared evaluation protocol for all algorithms.",
        "- An interface that accepts a user and shows titles and genres.",
        "",
        "## 2. Dataset evidence",
        "",
        "The official archive provides three relational tables.",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "table": name,
                        "rows": shapes[name][0],
                        "columns": shapes[name][1],
                        "null cells": sum(summary["null_counts"][name].values()),
                        "null columns": len(summary["null_columns"][name]),
                    }
                    for name in ["users", "movies", "ratings"]
                ]
            )
        ),
        "",
        f"Ratings range from {summary['rating_min']:.0f} to {summary['rating_max']:.0f} "
        f"with a mean of {summary['rating_mean']:.3f}. The tables contain "
        f"{summary['duplicate_user_movie_pairs']:,} duplicate user-movie pairs. "
        "The source files have no missing cells, so no imputation was required. "
        "Metadata preprocessing still includes release-year extraction, genre "
        "multi-hot encoding, age standardization, and gender/occupation one-hot encoding.",
        "",
        "The requested inspection evidence is reproduced by `scripts/prepare_data.py` "
        "in `data/processed/dataset_summary.json` and `dataset_comparison.csv`. It "
        "includes shapes, null totals, null columns, unique-value counts, dtypes, and a "
        "table comparison.",
        "",
        "![Rating distribution](../figures/rating_distribution.png)",
        "",
        "![Activity distributions](../figures/activity_distributions.png)",
        "",
        "## 3. Preprocessing and split",
        "",
        (
            f"The processed split contains {summary['split_rows']['train']:,} training, "
            f"{summary['split_rows']['validation']:,} validation, and "
            f"{summary['split_rows']['test']:,} test ratings. Interactions are ordered by "
            "timestamp inside each user, then divided into 80/10/10 partitions. This "
            "prevents a future interaction from becoming a training feature. User and "
            "movie feature statistics, item popularity, sequence histories, and item-item "
            "similarities are fitted from the training partition only."
        ),
        "",
        (
            "A rating of 4 or 5 is converted to a positive relevance label. For evaluation, "
            "each user contributes the latest positive rating from the relevant split plus "
            "99 movies that the user never rated in any period. A fixed seed shuffles the "
            "100 candidates. Hiding the future label from model fitting prevents leakage, "
            "while using the complete observed set for candidate exclusion matches the "
            "product requirement that recommendations be unseen."
        ),
        "",
        "![Temporal split](../figures/temporal_split.png)",
        "",
        "## 4. Algorithms",
        "",
        "### 4.1 Item-based collaborative filtering (lecture algorithm)",
        "",
        "A binary user-item matrix is built from training ratings of 4 or higher. Each "
        "movie column is L2-normalized and multiplied by the transpose to obtain cosine "
        "item-item similarity. A user's score for an unseen movie is the sum of "
        "similarities from movies the user liked, plus a small normalized log-popularity "
        "prior. Neighbor counts of 30, 60, and 120 are compared on validation NDCG@10.",
        "",
        "### 4.2 LightFM-style metadata embeddings (reference algorithm)",
        "",
        "The LightFM paper represents users and items as additive sums of identity and "
        "metadata embeddings. This implementation follows that decomposition:",
        "",
        "`score(u,i) = b_u + b_i + <e_u + F_u w_u, e_i + F_i w_i>`",
        "",
        "User metadata includes standardized age plus gender and occupation indicators. "
        "Item metadata includes genre indicators and standardized release year. The model "
        "is trained with Bayesian Personalized Ranking (BPR), which increases the margin "
        "between an observed positive and a uniformly sampled unobserved negative. "
        "Embedding dimensions 32 and 64 are compared on validation NDCG@10.",
        "",
        "### 4.3 ICSRec-inspired latent-intent sequence model (recent algorithm)",
        "",
        "This is a compact coursework adaptation of ICSRec (WSDM 2024). A GRU encodes up "
        "to the last 50 positive training items. KMeans refreshes 8 or 16 intent clusters "
        "from the current sequence representations. An InfoNCE objective aligns each "
        "sequence with its assigned latent-intent prototype, while a BPR next-item loss "
        "trains item ranking. The simplification avoids multi-view augmentation and full "
        "paper-scale training; it retains intent clustering and contrastive alignment.",
        "",
        "![Genre distribution](../figures/genre_distribution.png)",
        "",
        "## 5. Evaluation protocol",
        "",
        "Every algorithm receives the same users, positive items, 99 negatives, and random "
        "order. Metrics are:",
        "",
        "- **HR@10:** whether the positive appears in the top 10.",
        "- **NDCG@10:** position-sensitive ranking quality.",
        "- **MRR:** reciprocal rank of the positive.",
        "- **Precision@10 / Recall@10:** both equal HR@10 in this one-positive protocol.",
        "",
        "Hyperparameters are selected using validation data only. The selected "
        "configuration is evaluated once on the test partition. No test metric is used "
        "for model selection.",
        "",
        "## 6. Results",
        "",
        markdown_table(
            final[["algorithm", "HR@10", "NDCG@10", "MRR", "test_users"]],
            floatfmt=".4f",
        ),
        "",
        "![Model comparison](../figures/model_comparison.png)",
        "",
        "Validation selection table:",
        "",
        markdown_table(
            validation[["algorithm", "config", "HR@10", "NDCG@10", "MRR"]],
            floatfmt=".4f",
        ),
        "",
        "## 7. Interpretation and error analysis",
        "",
        (
            f"The best test NDCG@10 is {winner['NDCG@10']:.4f}. The result should be read "
            "as performance on a controlled 100-candidate task rather than full-catalog "
            "quality. A score below one is expected because the positive is one item among "
            "99 plausible alternatives and many users have broad tastes."
        ),
        "",
        "ItemKNN is transparent and learns local co-consumption patterns, but its dense "
        "movie-to-movie neighborhoods can over-reward mainstream films with many co-raters. "
        "The LightFM-style model uses genre and user metadata, which helps generalize when "
        "direct interaction data is sparse. The intent model conditions on sequence order; "
        "it can capture changing preferences, but its small CPU budget and short history "
        "limit temporal depth.",
        "",
        "Concrete error modes are:",
        "",
        "- **Popularity bias:** common genres and frequently rated movies have more "
        "neighbors and stronger prior support.",
        "- **Sparse users:** users with few positives provide weak collaborative and "
        "sequence evidence.",
        "- **Cold items:** movies with few training ratings have unstable item-item "
        "neighbors but can still receive metadata scores.",
        "- **Taste drift:** one static user representation cannot fully separate genres "
        "watched in different periods.",
        "- **Ratings below 4:** the current implicit formulation discards future low "
        "ratings during fitting. A future version should model graded negative feedback.",
        "",
        "## 8. Recommendation examples",
        "",
        "The following is a sample from `results/sample_user_recommendations.csv`. The full "
        "file contains 20 users and top-10 recommendations from all three algorithms.",
        "",
        markdown_table(
            sample.head(18)[
                ["user_id", "algorithm", "rank", "title", "genres", "release_year"]
            ]
        ),
        "",
        "## 9. Application",
        "",
        "The Streamlit application accepts a user, algorithm, genre, release-year filter, "
        "and result count. It excludes every movie observed for that user, displays the "
        "top titles and genres, compares the three algorithms side by side, and exposes "
        "validation/test metrics and dataset evidence. Users can download their result "
        "table as CSV.",
        "",
        "## 10. Limitations and future work",
        "",
        "The main limitation is sampled evaluation. A production system should rank the "
        "full unseen catalog and measure recall, NDCG, catalog coverage, and novelty. "
        "The project does not model repeated viewing, time decay, user text reviews, or "
        "item content beyond genre/year. The LightFM and ICSRec methods are transparent "
        "adaptations rather than exact paper reproductions. Future work should add graded "
        "feedback, popularity calibration, cold-start stress tests, and online A/B tests.",
        "",
        "## 11. Reproducibility",
        "",
        f"Random seed: `42`. Positive threshold: `4`. Selected configurations: "
        f"`{json.dumps(training_info['models'], sort_keys=True)}`.",
        "",
        "Reference environment: Python 3.12, pandas 3.1.1, NumPy 2.2.6, PyTorch, Streamlit, "
        "matplotlib, SciPy, and PyArrow. Run instructions, dependencies, seeds, data "
        "citation, and output locations are in `README.md`.",
        "",
        "Data files are intentionally excluded from Git. Rebuild with:",
        "",
        "```bash",
        "python scripts/prepare_data.py --download",
        "python scripts/train_models.py",
        "python scripts/build_report.py",
        "streamlit run app.py",
        "```",
        "",
        "## References",
        "",
        "1. F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: "
        "History and Context. ACM TiiS 5(4), Article 19. https://doi.org/10.1145/2827872",
        "2. Maciej Kula. 2015. Metadata Embeddings for User and Item Cold-start "
        "Recommendations (LightFM). arXiv:1507.08439.",
        "3. Xiuyuan Qin et al. 2024. Intent Contrastive Learning with Cross "
        "Subsequences for Sequential Recommendation (ICSRec). WSDM 2024. "
        "https://doi.org/10.1145/3616855.3635773",
        "",
    ]
    return "\n".join(lines)


def build_pdf(
    summary: dict,
    metrics: pd.DataFrame,
    validation: pd.DataFrame,
    sample: pd.DataFrame,
    training_info: dict,
    output_path: Path,
) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=32,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=MUTED,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1x",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=NAVY,
            spaceBefore=16,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2x",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=16,
            textColor=BLUE,
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Bodyx",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=NAVY,
            spaceAfter=7,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Bulletx",
            parent=styles["Bodyx"],
            leftIndent=14,
            firstLineIndent=-9,
            bulletIndent=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Captionx",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            textColor=MUTED,
            spaceAfter=9,
            alignment=TA_CENTER,
        )
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.68 * inch,
        leftMargin=0.68 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.72 * inch,
        title="MovieLens Hybrid Movie Recommender - Technical Report",
        author="Group Project",
    )
    story = []
    story.append(Spacer(1, 1.25 * inch))
    story.append(Paragraph("MovieLens Hybrid<br/>Movie Recommender", styles["ReportTitle"]))
    story.append(
        Paragraph(
            "Technical report for Project 1: a three-algorithm comparison and interactive system",
            styles["ReportSubtitle"],
        )
    )
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        pdf_table(
            [
                ["Dataset", "MovieLens 1M"],
                ["Evaluation", "Chronological 80/10/10, shared 100-candidate ranking"],
                ["Algorithms", "ItemKNN, LightFM-style BPR, ICSRec-inspired sequence model"],
                ["Date", "13 November 2026"],
            ],
            widths=[1.3 * inch, 5.6 * inch],
            font_size=9,
            header=False,
        )
    )
    story.append(PageBreak())

    final = metrics[(metrics["split"] == "test") & (metrics["selected"] == True)].copy()
    winner = final.loc[final["NDCG@10"].idxmax()]

    def h1(text: str) -> None:
        story.append(Paragraph(escape(text), styles["H1x"]))

    def h2(text: str) -> None:
        story.append(Paragraph(escape(text), styles["H2x"]))

    def body(text: str) -> None:
        story.append(Paragraph(text, styles["Bodyx"]))

    def bullet(text: str) -> None:
        story.append(Paragraph(text, styles["Bulletx"], bulletText="\u2022"))

    h1("Abstract")
    body(
        "This project builds and evaluates a movie recommendation system on MovieLens 1M. "
        "The system uses item-based collaborative filtering, a LightFM-style hybrid "
        "metadata factorizer, and an ICSRec-inspired latent-intent sequence recommender. "
        "All models share a per-user chronological split and identical sampled candidates. "
        f"The strongest held-out model by NDCG@10 is <b>{escape(winner['algorithm'])}</b> "
        f"at <b>{winner['NDCG@10']:.4f}</b>, with HR@10 of <b>{winner['HR@10']:.4f}</b>."
    )
    body(
        "A Streamlit interface accepts an anonymized user, excludes observed movies, and "
        "shows recommended titles and genres. The repository includes data inspection, "
        "trained model export, tests, figures, and full reproduction instructions."
    )

    h1("1. Task and success criteria")
    body(
        "Given a user, recommend at least five movies not present in that user's observed "
        "history. The implementation also must cover dataset evidence, preprocessing, "
        "exactly three algorithms, comparative evaluation, and an interactive interface."
    )
    for text in [
        "Reproducible temporal split and fixed random seed.",
        "No held-out ratings in training features or learned popularity counts.",
        "Identical candidates and metrics across all three algorithms.",
        "User-visible titles, genres, and an exploratory application.",
    ]:
        bullet(text)

    h1("2. Dataset evidence")
    body(
        "The official MovieLens 1M archive contains users, movies, and ratings tables. The "
        "tables have no missing cells. Release year is parsed from titles, genres are "
        "multi-hot encoded, age is standardized, and gender and occupation are one-hot "
        "encoded. All later data-derived statistics use training rows only."
    )
    table_rows = [["Table", "Rows", "Columns", "Null cells", "Null columns"]]
    for name in ["users", "movies", "ratings"]:
        table_rows.append(
            [
                name,
                f"{summary['shapes'][name][0]:,}",
                str(summary["shapes"][name][1]),
                f"{sum(summary['null_counts'][name].values()):,}",
                str(len(summary["null_columns"][name])),
            ]
        )
    story.append(pdf_table(table_rows, [1.0 * inch, 1.0 * inch, 1.0 * inch, 1.2 * inch, 1.3 * inch]))
    story.append(Spacer(1, 6))
    body(
        f"Ratings span {summary['rating_min']:.0f} to {summary['rating_max']:.0f} with "
        f"mean {summary['rating_mean']:.3f}. The source contains "
        f"{summary['duplicate_user_movie_pairs']:,} duplicate user-movie pairs. The script "
        "exports shapes, total nulls, null columns, unique counts, dtypes, and a table "
        "comparison to the processed data directory."
    )
    story.append(scaled_image(ROOT / "figures/rating_distribution.png", 6.85 * inch, 3.1 * inch))
    story.append(Paragraph("Figure 1. Ratings are concentrated in the upper half of the scale.", styles["Captionx"]))
    story.append(scaled_image(ROOT / "figures/activity_distributions.png", 6.85 * inch, 3.1 * inch))
    story.append(Paragraph("Figure 2. User activity and item popularity both have long right tails.", styles["Captionx"]))

    h1("3. Preprocessing and split")
    body(
        f"The split contains {summary['split_rows']['train']:,} training, "
        f"{summary['split_rows']['validation']:,} validation, and "
        f"{summary['split_rows']['test']:,} test ratings. Interactions are sorted by "
        "timestamp inside each user and divided 80/10/10. Item popularity, item-item "
        "similarity, sequence histories, and feature normalization are fitted from "
        "training rows only."
    )
    body(
        "Ratings 4 and 5 become positive relevance labels. For ranking evaluation, each "
        "user contributes the latest relevant validation or test item and 99 movies never "
        "rated by that user. The shared seed makes candidate order identical for every "
        "model. The model receives no future rating when fitting, and the final test "
        "metrics are not used for hyperparameter choice."
    )
    story.append(scaled_image(ROOT / "figures/temporal_split.png", 6.85 * inch, 2.8 * inch))
    story.append(Paragraph("Figure 3. The per-user split follows interaction time.", styles["Captionx"]))

    h1("4. Algorithms")
    h2("4.1 Item-based collaborative filtering")
    body(
        "A binary training matrix uses ratings of 4 or higher. L2-normalized movie columns "
        "produce cosine item-item similarity. A user's unseen-item score sums similarities "
        "from movies the user liked and adds a small normalized log-popularity prior. "
        "Neighbor counts 30, 60, and 120 are compared on validation NDCG@10."
    )
    h2("4.2 LightFM-style metadata embeddings")
    body(
        "The LightFM paper uses additive identity and metadata embeddings. This project "
        "implements score(u,i) = b_u + b_i + &lt;e_u + F_u w_u, e_i + F_i w_i&gt;. User "
        "features are age, gender, and occupation. Item features are genre and release "
        "year. Bayesian Personalized Ranking trains each observed positive above a sampled "
        "unobserved negative. Dimensions 32 and 64 are compared."
    )
    h2("4.3 ICSRec-inspired sequential model")
    body(
        "A GRU encodes up to 50 positive training items. KMeans refreshes 8 or 16 latent "
        "intent clusters from sequence representations. An InfoNCE loss aligns sequences "
        "with intent prototypes, and a BPR next-item loss trains ranking. This is a compact "
        "adaptation of ICSRec, not an exact reproduction of every augmentation and training "
        "detail in the paper."
    )
    story.append(scaled_image(ROOT / "figures/genre_distribution.png", 5.2 * inch, 3.5 * inch))
    story.append(Paragraph("Figure 4. Drama and comedy dominate the catalog, increasing popularity bias risk.", styles["Captionx"]))

    h1("5. Evaluation protocol")
    body(
        "All models score the same 100-item candidate set per user. Metrics are HR@10, "
        "NDCG@10, and MRR. With one relevant item per user, Precision@10 and Recall@10 "
        "equal HR@10. Validation NDCG@10 selects one configuration per algorithm, then the "
        "selected configuration receives a single final test evaluation."
    )

    h1("6. Results")
    result_rows = [["Algorithm", "HR@10", "NDCG@10", "MRR", "Users"]]
    for _, row in final.iterrows():
        result_rows.append(
            [
                str(row["algorithm"]),
                f"{row['HR@10']:.4f}",
                f"{row['NDCG@10']:.4f}",
                f"{row['MRR']:.4f}",
                f"{row['test_users']:.0f}",
            ]
        )
    story.append(pdf_table(result_rows, [2.0 * inch, 1.0 * inch, 1.1 * inch, 1.0 * inch, 1.0 * inch]))
    story.append(Spacer(1, 8))
    body(
        f"The best held-out NDCG@10 is {winner['NDCG@10']:.4f}. This is a controlled "
        "100-candidate comparison and should not be interpreted as full-catalog accuracy. "
        "The test table is written to results/final_test_metrics.csv."
    )
    story.append(scaled_image(ROOT / "figures/model_comparison.png", 6.3 * inch, 3.1 * inch))
    story.append(Paragraph("Figure 5. Held-out HR@10 and NDCG@10 for the selected models.", styles["Captionx"]))

    h2("Validation selection")
    validation_rows = [["Algorithm", "HR@10", "NDCG@10", "MRR", "Configuration"]]
    for _, row in validation.iterrows():
        validation_rows.append(
            [
                str(row["algorithm"]),
                f"{row['HR@10']:.4f}",
                f"{row['NDCG@10']:.4f}",
                f"{row['MRR']:.4f}",
                str(row["config"]).replace('"', "").replace("{", "").replace("}", "")[:48],
            ]
        )
    story.append(pdf_table(validation_rows, [1.45 * inch, 0.8 * inch, 1.0 * inch, 0.8 * inch, 2.55 * inch], font_size=7.5))

    h1("7. Interpretation and error analysis")
    body(
        "ItemKNN is transparent and captures co-consumption, but mainstream titles gather "
        "more co-ratings and can receive overly broad neighborhoods. Metadata embeddings "
        "generalize better for sparse items and users, while the sequence model can adapt "
        "to order and changing preferences. The sequence model is limited by a short "
        "CPU training budget and a 50-item history."
    )
    for text in [
        "<b>Popularity bias:</b> frequent items have more training evidence and stronger prior support.",
        "<b>Sparse users:</b> a small positive history weakens both collaborative and sequence estimates.",
        "<b>Cold items:</b> item-item neighborhoods are unstable, although metadata still supplies signal.",
        "<b>Taste drift:</b> a static embedding may blur preferences from different time periods.",
        "<b>Implicit labels:</b> ratings below 4 are excluded from fitting, losing negative preference detail.",
    ]:
        bullet(text)

    h1("8. Recommendation examples")
    body(
        "The table below shows representative outputs from the exported top-10 sample. The "
        "full file contains 20 users and all three algorithms."
    )
    sample_rows = [["User", "Model", "#", "Movie", "Genres", "Year"]]
    for row in sample.head(16).itertuples():
        sample_rows.append(
            [
                str(row.user_id),
                row.algorithm,
                str(row.rank),
                str(row.title)[:43],
                str(row.genres)[:26],
                str(row.release_year),
            ]
        )
    story.append(pdf_table(sample_rows, [0.48 * inch, 1.08 * inch, 0.3 * inch, 2.42 * inch, 1.5 * inch, 0.55 * inch], font_size=7.2))

    h1("9. Application")
    body(
        "The Streamlit application accepts a user, algorithm, genre, minimum release year, "
        "and result count. It excludes every observed title, displays movie metadata, "
        "compares all three algorithms for one user, exposes model metrics, and downloads "
        "recommendations as CSV. Start it with `streamlit run app.py`."
    )

    h1("10. Limitations and future work")
    body(
        "Sampled evaluation does not measure full-catalog recall or business value. A "
        "production study should assess catalog coverage, novelty, fairness across user "
        "groups, and temporal decay. Text reviews and richer metadata could improve cold "
        "start. The two recent/reference methods are transparent adaptations, so parameter "
        "results are project-specific. Future work should include graded feedback, "
        "popularity calibration, cold-start experiments, and online A/B testing."
    )

    h1("11. Reproducibility")
    body(
        "Random seed is 42 and the positive threshold is 4. The complete selected "
        f"configuration is {escape(json.dumps(training_info['models'], sort_keys=True))}. "
        "The repository README lists software requirements, run commands, dataset citation, "
        "and all result locations. Raw and processed MovieLens files are excluded from Git."
    )

    h1("References")
    for text in [
        "F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM TiiS 5(4), Article 19. doi:10.1145/2827872.",
        "Maciej Kula. 2015. Metadata Embeddings for User and Item Cold-start Recommendations (LightFM). arXiv:1507.08439.",
        "Xiuyuan Qin et al. 2024. Intent Contrastive Learning with Cross Subsequences for Sequential Recommendation (ICSRec). WSDM 2024. doi:10.1145/3616855.3635773.",
    ]:
        body(text)

    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)


def main() -> None:
    processed = ROOT / "data" / "processed"
    summary = json.loads((processed / "dataset_summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(ROOT / "results" / "metrics.csv")
    validation = metrics[metrics["split"] == "validation"].copy()
    sample = pd.read_csv(ROOT / "results" / "sample_user_recommendations.csv")
    training_info = json.loads(
        (ROOT / "results" / "training_info.json").read_text(encoding="utf-8")
    )
    docs = ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    markdown = build_markdown(summary, metrics, validation, sample, training_info)
    (docs / "technical_report.md").write_text(markdown, encoding="utf-8")
    build_pdf(
        summary,
        metrics,
        validation,
        sample,
        training_info,
        docs / "technical_report.pdf",
    )
    print(f"Wrote {docs / 'technical_report.md'}")
    print(f"Wrote {docs / 'technical_report.pdf'}")


if __name__ == "__main__":
    main()

