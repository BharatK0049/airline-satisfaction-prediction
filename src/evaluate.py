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