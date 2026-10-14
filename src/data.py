"""Loading data and creating the shared train/validation split."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, ID_COL, SEED, TARGET, TEST_SIZE


def load_train():
    """Return training features X and the binary target y (0/1)."""
    df = pd.read_csv(DATA_DIR / "train.csv")
    return df.drop(columns=[ID_COL, TARGET]), df[TARGET].astype(int)


def load_test():
    """Return test ids and test features."""
    df = pd.read_csv(DATA_DIR / "test.csv")
    return df[ID_COL], df.drop(columns=[ID_COL])


def get_split(X, y):
    """The shared stratified split. Every experiment must use this one."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
