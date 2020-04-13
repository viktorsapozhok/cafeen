__all__ = [
    'cross_val',
    'train_predict',
    'encode',
    'make_data',
    'get_feature_names'
]

import logging

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

from . import config, encoders

logger = logging.getLogger('cafeen')

nominal_features = [
    'bin_0', 'bin_1', 'bin_2', 'bin_4',
    'nom_0', 'nom_1', 'nom_2', 'nom_3',
    'nom_4', 'nom_5', 'nom_7', 'nom_8',
    'ord_2', 'ord_3', 'day', 'month'
]


def cross_val():
    for seed in [0, 1, 2, 3]:
        train_x, test_x, train_y, test_y, test_id = \
            make_data(config.path_to_train, seed=seed, drop_features=['bin_3'])

        score = train_predict(train_x, train_y, test_x, test_y=test_y)
        logger.info(f'score: {score:.6f}')


def train_predict(train_x, train_y, test_x, test_y=None):
    is_val = test_y is not None
    train_x, test_x = encode(train_x, train_y, test_x, is_val=is_val)

    estimator = LogisticRegression(
        random_state=2020,
        C=0.049,
        class_weight={0: 1, 1: 1.42},
        solver='liblinear',
        max_iter=2020,
        fit_intercept=True,
        penalty='l2',
        verbose=0)

    predicted = estimator.fit(train_x, train_y).predict_proba(test_x)
    score = roc_auc_score(test_y.values, predicted[:, 1])

    return score


def encode(train_x, train_y, test_x, is_val=False):
    test_y = pd.Series(data=[np.nan] * len(test_x), index=test_x.index)
    fill_value = train_y.mean()

    nom_6_encoder = encoders.TargetEncoderCV(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=2020),
        columns=['nom_6'], fill_value=fill_value, min_count=90, n_groups=3)

    nom_9_encoder = encoders.TargetEncoder(
        fill_value=fill_value, normalize=False, min_count=60, tol='0.0000001')

    encoder = make_column_transformer(
        (encoders.LinearEncoder(
            alpha=0, fill_value=fill_value), ['ord_0', 'ord_1']),
        (encoders.LinearEncoder(
            alpha=0.2, fill_value=fill_value), ['ord_4']),
        (encoders.LinearEncoder(
            alpha=0.1, fill_value=fill_value), ['ord_5']),
        (encoders.TargetEncoder(
            fill_value=fill_value), nominal_features),
        (nom_9_encoder, ['nom_9']),
        remainder='passthrough')

    x = pd.concat([train_x, test_x])
    y = pd.concat([train_y, test_y])

    x = nom_6_encoder.fit(train_x, train_y).transform(x, y, is_val=is_val)
    x = encoder.fit(train_x, train_y).transform(x)

    ohe_encoder = make_column_transformer(
        (OneHotEncoder(sparse=True), list(range(4, 21))),
        remainder='passthrough')

    train_x = ohe_encoder.fit(x).transform(x[~y.isna()])
    test_x = ohe_encoder.fit(x).transform(x[y.isna()])

    return train_x, test_x


def make_data(path_to_train, path_to_test=None, seed=0, drop_features=None):
    """Read data and prepare dataset.

    If ``path_to_test`` is not initialized, the data is reading
    from train.csv and then splitting into train and validation sets
    contained 300000 rows each.

    Args:
        path_to_train:
            Path to train.csv.
        path_to_test:
            Path to test.csv. If not initialized, then split
            train set into train and validation sets.
        seed:
            Random seed used in ``train_test_split``.
        drop_features:
            List of column names to be removed from data.
    """

    train = pd.read_csv(path_to_train)

    if path_to_test is not None:
        test_x = pd.read_csv(config.path_to_test)
        train_x = train[get_feature_names(train)]
        train_y = train['target']
        test_y = None
    else:
        train_x, test_x, train_y, test_y = train_test_split(
            train,
            train['target'],
            test_size=0.5,
            shuffle=True,
            random_state=seed,
            stratify=train['target'])

    test_id = test_x['id']

    # remove abundant features
    if drop_features is not None:
        train_x = train_x.drop(columns=drop_features)
        test_x = test_x.drop(columns=drop_features)

    features = get_feature_names(train_x)

    # convert ord_1 to numeric
    train_x = ord_1_to_ordinal(train_x[features])
    test_x = ord_1_to_ordinal(test_x[features])

    return train_x, test_x, train_y, test_y, test_id


def get_feature_names(x):
    """Return list of feature names.
    """

    return [col for col in x.columns if col not in ['id', 'target']]


def ord_1_to_ordinal(x):
    """Convert ``ord_1`` to numeric.
    """

    df = x.copy()
    df['ord_1'] = x['ord_1'].map({
        'Novice': 1,
        'Contributor': 2,
        'Expert': 3,
        'Master': 4,
        'Grandmaster': 5})
    return df
