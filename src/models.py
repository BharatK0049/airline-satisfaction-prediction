"""Model factories. Add new models here so notebooks never build them inline."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import SEED
# from .preprocessing import build_preprocessor


def make_logreg(**kwargs):
    params = {"max_iter": 1000, "random_state": SEED}
    params.update(kwargs)
    return LogisticRegression(**params)


# def make_pipeline(model, rating_mode="numeric"):
#     """Full pipeline: raw dataframe in, probability out."""
#     return Pipeline([("prep", build_preprocessor(rating_mode)), ("model", model)])
