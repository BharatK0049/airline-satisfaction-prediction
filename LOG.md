# Experiment log

## 2026-10-08

### Setup
- Downloaded Playground Series S6E10 airline satisfaction dataset.
- Set up Python virtual environment.
- Added initial project structure.

### Observations
- `Arrival Delay in Minutes` is the only column with 292 missing values
- Dataset has a mixture of categorical and numerical features
- Mean of `satisfaction` was around 44%

### Initial Hypotheses
- For satisfaction, clearly the arrival and departure delay plays a huge role because arriving at the destination on time is the biggest factor for travel satisfaction
- Mini things like Inflight wifi service, entertainment, food and drink, gate location, type of travel, although small do contribute towards the satisfaction.
- A loyal customer is likely to adjust more to a few shortcomings like the lack of minor perks above, whereas a disloyal customer might be nitpicky.
- Some niche things depending on the type of travel, leg space might have a correlation. A business travel might prefer more leg room space.

## 2026-10-09

### First checks for EDA
- Need to first observe which columns are categorical
- Found out that for `departure delay in minutes`, around 90.78% of departures have zero recorded delay; the distribution has a long right tail. Whether using a binary indicator (delay or not) adds predictive value or not; to test its relationship with satisfaction.
- Business travel is more prevalent among passengers aged 20–59, while personal travel is more prevalent among passengers under 20 and those aged 60 or older.
- Satisfaction decreases as delay increases but even among passengers with zero departure delay, 55.20% are not satisfied. Departure punctuality alone does not explain passenger satisfaction.
- Arrival delays are associated with lower satisfaction than departure delays for some delay groups.
- Food and drink, seat comfort, inflight entertainment, on-board service, leg room, baggage handling, check-in service, cleanliness *generally increasing*
- Inflight Wi-Fi, online booking, gate location, online boarding are *non-monotonic*
- Departure/Arrival Time convenient is insconsistent
- Baggage handling has no 0 rating, meaning it is mandatory or made to look mandatory
- Several service ratings have strong associations with satisfaction, with notable nonlinearities and exceptions.

## 2026-10-10

### Final EDA for the day before next phase
- Loyal customers show more satisfaction than disloyal customers
- Disloyal customers are much less likely to be satisfied when Wi-Fi ratings are low whereas loyal customers show comparatively higher satisfaction even at low Wi-Fi ratings.
- I want to be able to examine all 13 service features amongst loyal and disloyal customers. We introduce "gap", the average satisfaction of high ratings (4-5) minus the avg satisfaction of low ratings (1-3). Then compare the gap between loyal and disloyal customers.
- 0 rating is excluded but later, we can engineer a feature accounting users who did not rate at all.
- Across most service features and rating levels, loyal customers generally have higher satisfaction. The relationship between service ratings and satisfaction differs across customer types and features, with some exceptions at the highest rating.
- Travel purpose has a large association with satisfaction. Business travellers have higher satisfaction rates in every class.
- Within business travel, class matters descriptively. Business-class passengers have a much higher satisfaction rate than Eco and Eco Plus passengers. Within personal travel, satisfaction remains low across all three classes.

## 2026-12-10

### Next step - Creating the baseline
- First, a simple baseline with some basic preprocessing and feature engineering for the model to make a prediction
- Arrival Delay in Minutes was the only feature with missing values (292 rows). The median calculated from the training split was 0.
- One hot encoded nominal categories - gender, customer type, class and type of travel.
- Despite class having a hierarchy between the classes, Business-class passengers travelling for business had 73.25% satisfaction, compared with 11.19% among personal travellers in Business class. This suggests class should not be assumed to have a simple, equally spaced relationship with satisfaction.
- Created a binary _unrated indicator for each rating column in both training and validation
- Considered replacing unrated zeros with NaN and imputing them using the median of rated observations in the training split.
- same training-derived median would be used for both training and validation data to prevent validation leakage.
- Keeping the _unrated indicators would preserve the distinction between a genuine rating and an imputed value.
- validation set is supposed to simulate unseen data. We want to learn our preprocessing rules from training data and apply those same rules to validation data, so it is imputed with learnt data.
- All numeric columns on different scales, decided to use StandardScaler for baseline.
- Investigated potential outliers using box plots. For departure delay, Q1 = 0, Q3 = 0, and IQR = 0, making the conventional IQR outlier threshold uninformative for this zero-heavy distribution.

### Concerns
- Why class is not encoded ordinally with ranking
- whether imputing a rating while retaining an unrated flag could confuse logistic regression about which value to trust.
- delay features still had highly skewed distributions after scaling. The 25th percentile, median, and 75th percentile were all approximately -0.20, while maximum scaled values exceeded 80.

### Results
Validation Results

ROC-AUC: 0.9398

Accuracy: 89.13%

Confusion Matrix:

True Negatives: 70,456

False Positives: 7,403

False Negatives: 7,809

True Positives: 54,259

Class 0 (Dissatisfied): Precision 0.90, Recall 0.90, F1-score 0.90.

Class 1 (Satisfied): Precision 0.88, Recall 0.87, F1-score 0.88.

First submission: 0.75987 (Did not scale)
Second submission: 0.94031 (Scaled)

## 2026-13-10

- One hot encoding all feature ratings and making 91 columns improves validation auc by a small amount.
- Validation AUC: 0.9476
- Validation Accuracy: 0.9085

Confusion Matrix:
[[72001  5858]
 [ 6946 55122]]

Classification Report:
              precision    recall  f1-score   support

           0       0.91      0.92      0.92     77859
           1       0.90      0.89      0.90     62068

    accuracy                           0.91    139927
   macro avg       0.91      0.91      0.91    139927
weighted avg       0.91      0.91      0.91    139927

Leaderboard score: 0.94799

## 2026-14-10

- Updated codebase with new codefiles with helper functions.
- Removed common.py
- Made workflow modular

## 2026-21-10
- Created decision Tree classifier, max performance caps at max_depth = 12.
- Score was 0.94446. Did not beat the benchmark.

| Model | Training ROC-AUC | Testing ROC-AUC |
|---|---:|---:|
| Logistic Regression (One-Hot Ratings) | 0.9489 | **0.94799** |
| Decision Tree (max_depth=12, One-Hot Ratings) | 0.9539 | 0.94446 |

Observation: Logistic regression achieves a higher validation ROC-AUC (0.9476 vs. 0.9446), despite having a slightly lower training ROC-AUC. This suggests better generalization on the validation split, while the decision tree shows a larger training-validation gap.

Also for why not use KNN or SVM is because there are too many features for KNN to check for distance and SVM uses kernels and hyperparameters which are expensive for large datasets.

Used a RandomForestClassifier model with parameters - 200 n_estimators, max_depth 12, "min_samples_leaf": 2, "max_features": "sqrt", "random_state": SEED, "n_jobs": -1

RandomForest - Training ROC-AUC: 0.9583; Testing ROC-AUC: 0.95379

RandomForestClassifier when fed with Service Features without one hot encoding showed improvement in performance.
Scaling not needed in Random Forest as it barely matters.

Finally Results so far:

| Model | Configuration | Training ROC-AUC | Validation ROC-AUC | Kaggle Score |
|---|---|---:|---:|---:|
| Logistic Regression | One-hot encoded ratings | 0.9489 | 0.9476 | 0.94799 |
| Decision Tree | One-hot encoded ratings, `max_depth=12` | 0.9539 | 0.9446 | 0.94446 |
| Random Forest | One-hot ratings, 200 trees, `max_depth=12`, `min_samples_leaf=2` | 0.9583 | 0.9533 | 0.95379 |
| **Random Forest 2** | **Numeric ratings, 200 trees, `max_depth=12`, `min_samples_leaf=2`** | **0.9597** | **0.9544** | **0.95486** |

Observation - Random Forest with numerically encoded service ratings currently performs best, achieving a validation ROC-AUC of 0.9544 and a Kaggle score of 0.95486. Compared with one-hot-encoded ratings, numeric encoding improved validation ROC-AUC by 0.0011 and Kaggle score by 0.00107, with the same model configuration. This suggests numeric encoding is a promising representation for Random Forest on this dataset.

## 2026-22-10

- Created two model submissions for LightGBM Classifier, one with numeric and one hot encoding of service ratings.
- Improvements seen from RandomForest Classifier, with numeric being the best so far.
- Took the same two configurations and removed scaling and numeric no scaling was the best so far.

| Model    | Configuration                                                | Training ROC-AUC | Validation ROC-AUC | Kaggle Score |
| -------- | ------------------------------------------------------------ | ---------------- | ------------------ | ------------ |
| LightGBM | Service ratings numerically encoded; numeric features scaled | 0.9629           | 0.9574             | 0.95763      |
| LightGBM | Service ratings one-hot encoded; numeric features scaled     | 0.9629           | 0.9573             | 0.95748      |
| LightGBM | Service ratings numerically encoded; no scaling              | 0.9628           | 0.9573             | 0.95786      |
| LightGBM | Service ratings one-hot encoded; no scaling                  | 0.9628           | 0.9573             | 0.95741      |
| XGBoost  | Service ratings numerically encoded; no scaling              | 0.9630           | 0.9572             | 0.95744      |

Observation: The numerical-encoding, no-scaling configuration achieved the highest Kaggle score of 0.95786. The results suggest that scaling has little effect on LightGBM's validation performance, while the encoding and scaling variations produce small differences in Kaggle scores.
Also, stick to numeric no scaling configuration here on, that produces the best result.
XGBoost, LightGBM are saturating in performance, which suggests that changing model isn't the option. Catboost would be optional.