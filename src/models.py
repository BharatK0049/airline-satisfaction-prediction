"""Model factories. Add new models here so notebooks never build them inline."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from .config import SEED
# from .preprocessing import build_preprocessor


def make_logreg(**kwargs):
    params = {"max_iter": 1000, "random_state": SEED}
    params.update(kwargs)
    return LogisticRegression(**params)

def make_random_forest(**kwargs):
    params = {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": SEED,
        "n_jobs": -1,
    }
    params.update(kwargs)
    return RandomForestClassifier(**params)

def make_lightgbm(**kwargs):
    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,      
    }
    params.update(kwargs)
    return LGBMClassifier(**params)
