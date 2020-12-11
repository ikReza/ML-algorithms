# Ensemble Techniques (Boosting & Bagging)

## Random Forest

Random forest is one of the most popular `sklearn` algorithm because it performs better on any problem(classification/regression) than other algorithms by default.

- [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

The differences:

- criterion: `gini/entropy`(classification), `mse/mae`(regressor)

**gini vs entropy**

- These are the functions to measure the quality of a split
- `gini` for Gini Impurity
  - formula:
  ```
  1 - (the probability of yes)² - (the probability of no)²
  ```
- `entropy` for Information Gain
  - formula:
  ```
  (the probability of yes)*log(the probability of yes) - (the probability of no)*log((the probability of no))
  ```
