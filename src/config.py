"""Paths, constants and column groups. The single source of truth for the project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
EXPERIMENT_LOG = PROJECT_ROOT / "experiments" / "results.csv"

ID_COL = "id"
TARGET = "satisfaction"
SEED = 42
TEST_SIZE = 0.2

# 1-5 service ratings; 0 means "not rated"
RATING_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]

NOMINAL_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]

NUMERIC_COLS = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]
