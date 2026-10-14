import pandas as pd

from .config import DATA_DIR, SUBMISSION_DIR


def save_submission(test_ids, proba, filename):
    """Build the Kaggle submission, check it, save it to submissions/<filename>."""
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

    submission = sample_submission.copy()
    submission["id"] = test_ids.to_numpy()
    submission["satisfaction"] = proba

    # Validate the output
    assert submission.shape == sample_submission.shape
    assert submission.columns.tolist() == sample_submission.columns.tolist()
    assert submission["id"].equals(sample_submission["id"])
    assert submission["satisfaction"].notna().all()

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMISSION_DIR / filename
    submission.to_csv(path, index=False)

    print("Saved to:", path)
    print(submission["satisfaction"].describe())
    return submission