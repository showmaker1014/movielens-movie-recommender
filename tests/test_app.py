"""Filter regression checks without starting a Streamlit server."""
import ast
from pathlib import Path
import numpy as np
import pandas as pd


def test_empty_filters_keep_schema_and_all_observed_ratings_are_excluded():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8"))
    names = {"recommendations", "recommendation_notice", "genre_options"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"np": np, "pd": pd, "user_index_for": lambda uid,b: 0,
          "score_all": lambda a,u,b: np.arange(len(b["movies"]), dtype=float)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py", "exec"), ns)
    movies = pd.DataFrame({"movie_id": [1,2,3], "title": ["A","B","C"],
        "genres": ["Drama","Comedy|Drama","Comedy"], "release_year": [1990,2000,2010]})
    ratings = pd.DataFrame({"user_id": [1,1], "movie_id": [1,3], "split": ["train","test"]})
    bundle = {"movies": movies, "ratings": ratings}
    result,_ = ns["recommendations"](1,"ItemKNN",10,"All genres",1900,bundle)
    assert result.movie_id.tolist() == [2]
    empty,_ = ns["recommendations"](1,"ItemKNN",10,"Drama",2020,bundle)
    assert empty.empty and {"title","genres","release_year"}.issubset(empty.columns)
    assert ns["recommendation_notice"](0,10)
    assert "at least five" in ns["recommendation_notice"](1,10)
    assert ns["genre_options"](movies) == ["All genres","Comedy","Drama"]
