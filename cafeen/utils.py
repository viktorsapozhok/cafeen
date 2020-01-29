from collections import defaultdict

import eli5
from eli5.sklearn import PermutationImportance
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def group_feature(train, test, feature, n_groups=10):
    nobs = len(train) / n_groups
    new_feature = feature + '_' + str(n_groups)

    grouped = \
        train.groupby(feature)['target'].agg(['mean', 'count']).sort_values(
            by='mean', ascending=False)

    grouped[new_feature] = np.floor(grouped['count'].cumsum() / nobs)
    groups = grouped[new_feature].unique()
    grouped.loc[grouped[new_feature] == groups[-1], new_feature] = groups[-2]
    grouped.reset_index(level=feature, inplace=True)

    train_columns = [
        col for col in train.columns if col not in [new_feature]]
    test_columns = [
        col for col in test.columns if col not in [new_feature]]

    train = train[train_columns].merge(
        grouped[[feature, new_feature]], how='left', on=feature)
    test = test[test_columns].merge(
        grouped[[feature, new_feature]], how='left', on=feature)

    train[new_feature] = train[new_feature].astype('int')
    test[new_feature] = test[new_feature].astype('int')

    return train, test


def group_feature_train(train, feature, n_groups=10):
    nobs = len(train) / n_groups
    new_feature = feature + '_' + str(n_groups)

    grouped = \
        train.groupby(feature)['target'].agg(['mean', 'count']).sort_values(
            by='mean', ascending=False)

    grouped[new_feature] = np.floor(grouped['count'].cumsum() / nobs)
    groups = grouped[new_feature].unique()
    grouped.loc[grouped[new_feature] == groups[-1], new_feature] = groups[-2]
    grouped.reset_index(level=feature, inplace=True)

    train_columns = [
        col for col in train.columns if col not in [new_feature]]

    train = train[train_columns].merge(
        grouped[[feature, new_feature]], how='left', on=feature)

    train[new_feature] = train[new_feature].astype('int')

    return train


def add_woe_feature(train, test, feature, verbose=True):
    n_events = train['target'].sum()
    n_non_events = len(train) - n_events

    bins = train.groupby(feature)['target'].agg(['sum', 'count'])
    bins['n_non_events'] = bins['count'] - bins['sum']
    bins['p_event'] = bins['sum'] / n_events
    bins['p_non_event'] = bins['n_non_events'] / n_non_events
    bins['woe'] = np.log(bins['p_event'] / bins['p_non_event'])

    train[feature + '_woe'] = train[feature].map(bins['woe'].to_dict())
    test[feature + '_woe'] = test[feature].map(bins['woe'].to_dict())

    if verbose:
        iv = ((bins['p_event'] - bins['p_non_event']) * bins['woe']).sum()
        print(f'{feature}: IV {iv:.2f}')

    return train, test


def add_woe_max(train, features):
    train['min_woe'] = train[features].min(axis=1)
    train['max_woe'] = train[features].max(axis=1)
    train['woe'] = train['max_woe']
    mask = train['min_woe'].abs() > train['max_woe'].abs()
    train.loc[mask, 'woe'] = train.loc[mask, 'min_woe']
    return train


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


def impute_nans(estimator, train, test, features):
    _train = train.copy()
    _test = test.copy()
    _train['is_train'] = 1
    _test['is_train'] = 0

    df = pd.concat([
        _train[features + ['is_train']],
        _test[features + ['is_train']]]
    ).reset_index(drop=True)

    del _train, _test

    nans = df.isna()

    obj_cols = [col for col in features if df[col].dtype == np.object]
    df[obj_cols] = df[obj_cols].fillna(value='-1')

    num_cols = [col for col in features if df[col].dtype == np.float64]
    df[num_cols] = df[num_cols].fillna(value=-1)

    encoders = defaultdict()

    for col in obj_cols:
        encoders[col] = LabelEncoder()
        encoders[col].fit(df[col])
        df[col] = encoders[col].transform(df[col])

    for feature in tqdm(features, ascii=True, ncols=70):
        _features = [f for f in features if f not in [feature]]

        imputed = estimator.fit(
            df.loc[~nans[feature], _features],
            df.loc[~nans[feature], feature]
        ).predict(df[_features])

        df.loc[nans[feature], feature] = imputed[nans.index[nans[feature]]]

    for col in obj_cols:
        df[col] = encoders[col].inverse_transform(df[col])

    train[features] = df.loc[df['is_train'] == 1, features]
    test[features] = df.loc[df['is_train'] == 0, features]

    return train, test
