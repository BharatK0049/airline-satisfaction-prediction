"""Evaluation, experiment tracking and saving."""
from datetime import datetime

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import EXPERIMENT_LOG, MODEL_DIR, SEED


def report(pipeline, X_val, y_val):
    """Print the validation report; return ROC-AUC (the competition metric)."""
    proba = pipeline.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_val, proba)
    print(f"Validation ROC-AUC : {auc:.4f}")
    print(f"Validation accuracy: {accuracy_score(y_val, pred):.4f}\n")
    print("Confusion matrix:\n", confusion_matrix(y_val, pred), "\n")
    print(classification_report(y_val, pred))
    return auc

# def cv_auc(pipeline, X, y, n_splits=5):
#     """Stratified k-fold ROC-AUC. Returns (mean, std). Pipeline is refit per fold."""
#     cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
#     scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
#     return scores.mean(), scores.std()


# def log_result(name, val_auc, cv_mean=None, cv_std=None, lb=None, notes=""):
#     """Append one row to experiments/results.csv (creates it if missing)."""
#     row = pd.DataFrame([{
#         "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
#         "experiment": name,
#         "val_auc": round(val_auc, 5),
#         "cv_auc_mean": None if cv_mean is None else round(cv_mean, 5),
#         "cv_auc_std": None if cv_std is None else round(cv_std, 5),
#         "leaderboard": lb,
#         "notes": notes,
#     }])
#     EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
#     row.to_csv(EXPERIMENT_LOG, mode="a", header=not EXPERIMENT_LOG.exists(), index=False)


# def run_experiment(name, pipeline, X_tr, y_tr, X_val, y_val, notes="", save=True):
#     """Fit -> report -> save one artifact -> log. Returns validation AUC."""
#     pipeline.fit(X_tr, y_tr)
#     auc = report(pipeline, X_val, y_val)
#     if save:
#         MODEL_DIR.mkdir(parents=True, exist_ok=True)
#         joblib.dump(pipeline, MODEL_DIR / f"{name}.joblib")
#     log_result(name, auc, notes=notes)
#     return auc