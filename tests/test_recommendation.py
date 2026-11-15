import numpy as np
import pandas as pd
from src.recommendation import rank_unseen, example_user_ids, recommendation_examples


def test_shared_export_and_ranking_agree_with_ties_and_observed_movies():
    movies = pd.DataFrame({"movie_id":[3,1,2], "title":["C","A","B"], "genres":["Drama"]*3, "release_year":[2000]*3})
    users = pd.DataFrame({"user_id":[1], "gender":["F"], "age":[25]})
    scores = np.ones(3)
    ranked = rank_unseen(movies, scores, {3}, 10)
    assert ranked.movie_id.tolist() == [1,2]
    exported = recommendation_examples(users, movies, [1], {"model":scores[None,:]}, {1:{3}})
    assert exported.movie_id.tolist() == ranked.movie_id.tolist()
    assert exported.model_score.tolist() == ranked.model_score.tolist()
    empty = rank_unseen(movies, scores, {1,2,3})
    assert empty.empty and list(empty.columns) == list(ranked.columns)


def test_example_selection_is_independent_of_input_order():
    ids = np.arange(1,101)
    np.testing.assert_array_equal(example_user_ids(ids), example_user_ids(ids[::-1]))
