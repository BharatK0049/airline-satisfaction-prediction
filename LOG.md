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