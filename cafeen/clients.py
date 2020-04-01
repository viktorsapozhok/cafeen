import logging

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from . import config, steps, utils

logger = logging.getLogger('cafeen')


def validate(n_valid_rows=0):
    scores = []

    for seed in [0, 1, 2, 2020]:
        scores += [predict(n_valid_rows=n_valid_rows, seed=seed)]

    logger.debug(f'score: {np.mean(scores):.6f}')


def predict(n_valid_rows=0, seed=2020):
    df, valid_y = utils.read_data(n_valid_rows=n_valid_rows, seed=seed)

    train_x, train_y, test_x, test_id = encode(df)

    estimator = LogisticRegression(
        random_state=2020,
        C=0.049,
        class_weight={0: 1, 1: 1.42},
        solver='liblinear',
        max_iter=2020,
        fit_intercept=True,
        penalty='l2',
        verbose=0)

    if valid_y is None:
        submitter = steps.Submitter(estimator, config.path_to_data)
    else:
        submitter = steps.Submitter(estimator)

    if isinstance(train_y, pd.Series):
        train_y = train_y.values

    y_pred = submitter.fit(train_x, train_y).predict_proba(
        test_x, test_id=test_id)

    if valid_y is not None:
        _valid_y = valid_y.merge(y_pred[['id', 'target']], how='left', on='id')
        score = roc_auc_score(
            _valid_y['y_true'].values,
            _valid_y['target'].values)
        logger.info(f'score: {score:.6f}')
    else:
        score = 0

    return score


def encode(df):
    df = df.drop(columns=['bin_3'])

    df.loc[df['day'] == 5, 'day'] = 3
    df.loc[df['day'] == 6, 'day'] = 2
    df.loc[df['day'] == 7, 'day'] = 1
    df.loc[df['nom_1'] == 'Square', 'nom_1'] = 'Triangle'
    df.loc[df['nom_4'] == 'Oboe', 'nom_4'] = 'Theremin'
    df.loc[df['ord_0'].isna(), 'ord_0'] = 2
    df.loc[df['month'] == 10, 'month'] = 12
    df.loc[df['month'] == 7, 'month'] = 9
    df.loc[df['month'] == 6, 'month'] = 12
    df.loc[df['month'].isna(), 'month'] = 8

    df['ord_1'] = df['ord_1'].map({
        'Novice': 1,
        'Contributor': 2,
        'Expert': 3,
        'Master': 4,
        'Grandmaster': 5
    })

    na_value = df[df['target'] > -1]['target'].mean()

    extra_features = ['nom_8', 'ord_3', 'ord_2', 'nom_1', 'nom_2']
    linear_features = ['ord_0', 'ord_1', 'ord_4', 'ord_5']
    cardinal_features = ['nom_6', 'nom_9']
    ohe_features = [
        'nom_1', 'nom_2', 'nom_5',
        'nom_6', 'nom_7', 'nom_8', 'nom_9',
        'ord_2', 'ord_3', 'day', 'month']

    for feature in extra_features:
        feature_ = feature + '_'
        df[feature_] = 0

        enc = df[df['target'] > -1].groupby(feature)['target'].mean()

        df.loc[df[feature].isin(enc[enc > na_value].index), feature_] = 1
        df.loc[df[feature].isin(enc[enc < na_value].index), feature_] = -1
        df.loc[df[feature].isna(), feature_] = 0

    feature_params = {
        'ord_0': {'alpha': 0},
        'ord_1': {'alpha': 0},
        'ord_4': {'alpha': 0.2},
        'ord_5': {'alpha': 0.1},
        'nom_6': {
            'count_min': 90,
            'cv': StratifiedKFold(n_splits=3, shuffle=True, random_state=2020),
            'n_groups': 3
        },
        'nom_9': {
            'count_min': 52,
            'std_max': 0.1,
            'eps': (0.0000001, 7)
        }
    }

    encoder = steps.Encoder(
        linear_features=linear_features,
        cardinal_features=cardinal_features,
        feature_params=feature_params,
        na_value=na_value)

    df = encoder.fit_transform(df)

    train, test = utils.split_data(df)

    train_y = train['target']
    test_id = test['id']

    ordinal_features = [f for f in encoder.features if f not in ohe_features]

    encoder = OneHotEncoder(sparse=True)
    encoder.fit(df[ohe_features])

    train_x = encoder.transform(train[ohe_features])
    train_x = sparse.hstack((train_x, train[ordinal_features].values)).tocsr()
    test_x = encoder.transform(test[ohe_features])
    test_x = sparse.hstack((test_x, test[ordinal_features].values)).tocsr()

    return train_x, train_y, test_x, test_id
