import numpy as np
import pandas as pd
from src.data import build_side_features, prepare_movies, sequences_from_frame

def test_year_once_and_constant_ages():
    users=pd.DataFrame({"user_id":[1,2],"age":[25,25],"gender":["M","F"],"occupation":[1,2]})
    movies,_=prepare_movies(pd.DataFrame({"movie_id":[1,2],"title":["A (1990)","B (2000)"],"genres":["Comedy","Drama"]}))
    uf,items,_,names=build_side_features(users,movies)
    assert names.count("year_standardized")==1
    assert len(names)==len(set(names))==items.shape[1]
    assert np.isfinite(uf).all()

def test_full_history():
    frame=pd.DataFrame({"user_id":[1]*60,"movie_id":range(60),"timestamp":range(60)})
    assert len(sequences_from_frame(frame,np.array([1]),{i:i for i in range(60)},None)[0])==60
