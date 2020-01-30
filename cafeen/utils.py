from collections import defaultdict
import logging
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import eli5
from eli5.sklearn import PermutationImportance
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from cafeen import config

logger = logging.getLogger('cafeen')


def get_features(features):
    return [feat for feat in features if feat not in ['id', 'target']]


def read_data(nrows=None, test=True):
    logger.info('reading train')
    train = pd.read_csv(config.path_to_train, nrows=nrows)

    if test:
        logger.info('reading test')
        test = pd.read_csv(config.path_to_test, nrows=nrows)
        test['target'] = -1

        return pd.concat([train, test])
    return train


def split_data(df):
    return df[df['target'] > -1], df.loc[df['target'] == -1]


def encode_features(df, features=None, keep_na=False):
    if features is None:
        features = get_features(df.columns)

    obj_cols = [col for col in features if df[col].dtype == np.object]
    num_cols = [col for col in features if df[col].dtype == np.float64]

    if not keep_na:
        logger.info('fill nans in numeric columns')
        df[num_cols] = df[num_cols].fillna(value=-1)

    for col in tqdm(obj_cols, ascii=True, desc='encoding', ncols=70):
        df[col] = df[col].fillna(value='-1')

        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])

        if keep_na:
            na_encoded = encoder.transform(['-1'])[0]
            df[col] = df[col].replace(na_encoded, np.nan)

    return df


def encode_ordinal_features(df, features):
    train = df[df['target'] > -1]

    for feature in features:
        encoded = train.groupby(feature)['target'].agg(
            ['mean', 'count']).sort_values(by='mean')
        encoded['count'] = range(1, len(encoded) + 1)
        df[feature] = df[feature].map(encoded['count'])

    return df


def fill_na(df, features):
    imputer = IterativeImputer(verbose=2, random_state=0, tol=1e-6)
    df[features] = imputer.fit_transform(df[features])
    df[features] = np.round(df[features].values)
    return df


def group_features(df, features, n_groups):
    for feature in features:
#        df[feature + '_' + str(n_groups)] = \
#            pd.qcut(df[feature], n_groups, labels=False)
        df[feature] = pd.qcut(df[feature], n_groups, labels=False)

    return df


def show_weights(expl):
    for row in expl.itertuples():
        print(f'{row.feature:8s}: '
              f'{100*row.weight:5.2f} +- '
              f'{100*2*row.std:.3f}')


def eval_weights(train, features):
    train_x, test_x, train_y, test_y = train_test_split(
        train[features],
        train['target'],
        shuffle=True,
        train_size=0.7,
        random_state=42)

    estimator = lgb.LGBMClassifier(n_estimators=50)
    estimator.fit(train_x, train_y)

    # initialize permutation importance class
    perm_estimator = PermutationImportance(
        estimator=estimator,
        scoring=make_scorer(roc_auc_score, needs_proba=True),
        n_iter=3)
    perm_estimator.fit(test_x, test_y)

    # calculate feature weights and return it as DataFrame
    expl = eli5.format_as_dataframe(
        eli5.explain_weights(
            perm_estimator,
            top=None,
            feature_names=features
        )
    )

    show_weights(expl)
