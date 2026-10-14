import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import NOMINAL_COLS, NUMERIC_COLS, RATING_COLS   # use your own import style


def learn_preprocessing(X_tr, ratings="onehot"):
    """Learn everything from TRAINING data only. Returns a dict of learned things."""
    state = {"ratings": ratings}

    # 0 means unrated -> treat as missing, learn the median of the real ratings
    rated = X_tr[RATING_COLS].replace(0, np.nan)
    state["rating_medians"] = rated.median()
    filled = rated.fillna(state["rating_medians"])

    if ratings == "onehot":
        state["rating_encoder"] = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(filled)

    state["nominal_encoder"] = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_tr[NOMINAL_COLS])

    state["numeric_medians"] = X_tr[NUMERIC_COLS].median()
    state["scaler"] = StandardScaler().fit(X_tr[NUMERIC_COLS].fillna(state["numeric_medians"]))
    return state


def apply_preprocessing(X, state):
    """Apply what was learned to ANY data (train, validation or test)."""
    # ratings: flag unrated first, then fill with the learned median
    unrated = (X[RATING_COLS] == 0).astype(int).add_suffix("_unrated")
    ratings = X[RATING_COLS].replace(0, np.nan).fillna(state["rating_medians"])
    if state["ratings"] == "onehot":
        enc = state["rating_encoder"]
        ratings = pd.DataFrame(enc.transform(ratings), columns=enc.get_feature_names_out(RATING_COLS), index=X.index)

    enc = state["nominal_encoder"]
    nominal = pd.DataFrame(enc.transform(X[NOMINAL_COLS]), columns=enc.get_feature_names_out(NOMINAL_COLS), index=X.index)

    numeric = X[NUMERIC_COLS].fillna(state["numeric_medians"])
    numeric = pd.DataFrame(state["scaler"].transform(numeric), columns=NUMERIC_COLS, index=X.index)

    return pd.concat([ratings, unrated, nominal, numeric], axis=1)