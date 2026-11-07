# Airline Passenger Satisfaction (Kaggle Playground S6E10)

The task was to predict whether a customer would be satisfied or not given their age, travel class, purpose of travel, distance travelled, departure/arrival delay and 13 service features rated on a scale from 1 to 5. The final evaluation score is on AUC-ROC (Area under ROC curve) between predicted probability and observed target. 
I first tried to understand the data by looking through how each features affects passenger satisfaction and how correlated they are.
I first trained a logistic regression model, with some preprocessing, and then moved to blending models to get an aggregate score.
My final score on the private leaderboard (80% testing data) was 0.95900 earning me a rank of 2944 out of 3734 teams.

## Problem and data
- Binary classification: predict whether a passenger was satisfied. Metric: ROC-AUC.
- About 700k training rows and 300k test rows, 21 features: 13 service ratings (1-5, with 0 meaning "not rated"), 4 categorical (gender, customer type, type of travel, class) and 4 numeric (age, flight distance, two delays).
- About 44% of passengers are satisfied.

## Approach
**EDA**
- Rating `0` means "not rated", not the lowest score (confirmed against the [original dataset](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k/data)). `Baggage handling` has no zeros.
- About 91% of departures have zero delay, with a long right tail.
- Delays lower satisfaction, but 55% of passengers with no departure delay were still unsatisfied. Arrival delay shows the same pattern.
- Most ratings rise steadily with satisfaction. Wi-Fi, online booking, gate location and online boarding are non-monotonic.
- Loyal customers are more satisfied than disloyal ones and more forgiving of low ratings.
- Business travellers are more satisfied than personal travellers in every class. Most passengers travel for business (mainly ages 20-59); under-20s and 60+ mostly travel for personal reasons. Average age is about 39.

**Logistic regression**
- Baseline (0.9398): one-hot categoricals, scaled numeric features, and each `0` rating flagged with an `_unrated` column and filled with the training median.
- One-hot ratings (0.9476): each rating level gets its own weight, so the model can fit the non-monotonic patterns that a single linear weight per rating cannot.

**Error analysis**
- Errors are spread evenly: 7-13% across every Type of Travel × Class group, with only small gaps by delay (8.2% vs 9.3%), loyalty (8.9% vs 10.1%) and unrated services (about 11% vs 9.0%).
- The highest-error groups are tiny (816 rows in the smallest), so fixing them would remove about 100 of the 12.8k mistakes. No segment-specific feature was worth building.
- About 65% of mistakes were confident (probability outside 0.3-0.7). That suggests logistic regression misses feature interactions, such as a poor seat only mattering when legroom is also poor. Label noise could also contribute.

**Tree models**
- Trees need no scaling and learn interactions and non-linear effects directly. A single depth-12 tree (0.9446) nearly matched logistic regression. Random Forest (0.9544), XGBoost (0.9572) and LightGBM (0.9576) beat it.
- Tuning LightGBM plateaued around 0.958.

**Blending and CatBoost**
- Averaging LightGBM and XGBoost gave +0.00007, which is inside the noise. Adding Random Forest or logistic regression made the average worse. Leaderboard: 0.95800, the same as LightGBM alone.
- Catboost gave 0.95796, which was worse than LightGBM.

## Results

| Experiment | Validation AUC | Leaderboard |
|---|---|---|
| Logistic regression, numeric ratings | 0.9398 | 0.94031 |
| Logistic regression, one-hot ratings | 0.9476 | 0.94799 |
| Decision tree (max_depth 12) | 0.94557 | 0.94446 |
| Random Forest | 0.9544 | 0.95379 |
| XGBoost | 0.9572 | 0.95744 |
| LightGBM | 0.9576 | 0.95800 |
| LightGBM + XGBoost average | 0.95767 | 0.95800 |
| CatBoost | 0.95796 | 0.95710 |

Final private score: 0.95900, rank 2944.

## Project structure
- `notebooks/` - one notebook per step, numbered in the order they were done (01 intro and 02 EDA through 09 XGBoost)
- `src/` - shared code: `config.py` (paths and column lists), `data.py` (loading and splitting), `preprocessing.py`, `models.py`, `evaluate.py`, `submission.py`
- `LOG.md` - day-by-day experiment log and reasoning

## How to run
1. `pip install -r requirements.txt`
2. Download the data from the Kaggle competition page and put `train.csv`, `test.csv` and `sample_submission.csv` in a `data/` folder (it is not included in the repo).
3. Run the notebooks in order. `00_submission` needs the models saved by `03` and `04`.

## What I'd do differently
- Explore Cross-fold validation and OOFs
- Explore more on distance travelled feature as most top submissions identified that as a top feature affecting satisfiability.
- Improve preprocessing and feature engineering.