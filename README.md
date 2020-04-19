# cafeen

This repository presents an approach used for solving [Kaggle Categorical Feature Encoding Challenge II](https://www.kaggle.com/c/cat-in-the-dat-ii).

### Cross-validation scheme

To validate the results, I splitted train dataset (600000 rows) into two sets 
having 300000 rows each. I repeated this operation 4 times using 
different `random_seed` and calculated CV score as a mean score over 4 iterations. 

```python
from cafeen import config, steps

scores = []

for seed in [0, 1, 2, 3]:
    train_x, test_x, train_y, test_y, test_id = steps.make_data(
        path_to_train=config.path_to_train,
        seed=seed)

    scores += [steps.train_predict(train_x, train_y, test_x, test_y=test_y)]
```
 
### Encoding pipeline

The full encoding pipeline [can be seen here](https://github.com/viktorsapozhok/cafeen/blob/642bf49b79e44d88e38f8ec7b0a37f6431ded18b/cafeen/steps.py#L79). 
 
### Score improvements

#### Baseline

As a baseline model, I used logistic regression with default parameters and `liblinear` solver. 
All features in dataset are one-hot encoded.

CV: 0.78130, private score: 0.78527

#### Tuning hyperparameters

After hyperparameters optimization, I found the following configuration yields a highest CV score.

```python
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(
    C=0.049,
    class_weight={0: 1, 1: 1.42},
    solver='liblinear',
    fit_intercept=True,
    penalty='l2')
```    

CV: 0.78519, private score: 0.78704

#### Drop bin_3

I dropped `bin_3` feature, as it seems to be not really important, and keeping
it in the dataset doesn't improve the score.

CV: 0.78520, private score: 0.78704

#### Ordinal encoding

I used ordinal encoding for `ord_0`, `ord_1`, `ord_4`, `ord_5`, approximating 
categories target mean with a linear function. For `ord_4` and `ord_5` I removed outliers, 
categories with small amount of observations, before applying the linear regression. 

CV: 0.78582, private score: 0.78727

#### Grouping

For `nom_6` feature I removed all categories which have less than 90 observations (replaced it with `NaN`).
Then using target encoding with cross-validation, converted it to numeric and
grouped in three groups with `qcut`.

```python
import pandas as pd

x['nom_6'] = pd.qcut(x['nom_6'], 3, labels=False, duplicates='drop')
```

CV: 0.78691, private score: 0.78796

#### Filtering

For `nom_9` feature I removed all categories which have less than 60 observations (replaced it with `NaN`)
and combined together categories which have equal target average. 

CV: 0.78691, private score: 0.78797

#### Missing values

For one-hot encoded features (all features except `ord_0`, `ord_1`, `ord_4`, `ord_5`), 
I replaced missing values with `-1`. For ordinal encoded features, I replaced it with 
the target probability, `0.18721`.  

### Results

That's it, though I haven't chosen the best submission for final score and the official
results are a bit worse.

Private score 0.78795 (110 place)
Public score 0.78669 (22 place)
