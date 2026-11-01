"""Evaluation, experiment tracking and saving."""
from datetime import datetime

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import EXPERIMENT_LOG, MODEL_DIR, SEED
from .preprocessing import learn_preprocessing, apply_preprocessing

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

def cross_validate(X, y, make_model, n_splits=5):
    sk_fold = StratifiedKFold(n_splits=n_splits, random_state = SEED, shuffle=True)
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(sk_fold.split(X, y), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        state = learn_preprocessing(X_tr, ratings="numeric", scaling="no")
        X_tr_enc = apply_preprocessing(X_tr, state)
        X_val_enc = apply_preprocessing(X_val, state)

        model = make_model()
        model.fit(X_tr_enc, y_tr)

        auc = roc_auc_score(y_val, model.predict_proba(X_val_enc)[:, 1])
        scores.append(auc)
        print(f"Fold {fold}: {auc:.5f}")

    print(f"Mean: {np.mean(scores):.5f}  Std: {np.std(scores):.5f}")
    return scores


def get_oof_predictions(make_model, X, y, X_test, ratings="numeric", n_splits=5):
    """Out-of-fold predictions for the train rows, fold-averaged predictions for the test rows."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # learn preprocessing from this fold's training rows only
        state = learn_preprocessing(X_tr, ratings=ratings)
        model = make_model()
        model.fit(apply_preprocessing(X_tr, state), y_tr)

        oof[val_idx] = model.predict_proba(apply_preprocessing(X_val, state))[:, 1]
        test_pred += model.predict_proba(apply_preprocessing(X_test, state))[:, 1] / n_splits

        print(f"fold {fold}: AUC = {roc_auc_score(y_val, oof[val_idx]):.5f}")

    print(f"pooled OOF AUC = {roc_auc_score(y, oof):.5f}")
    return oof, test_pred