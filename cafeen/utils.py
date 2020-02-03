import logging
from random import choices, randrange
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from category_encoders import OneHotEncoder, TargetEncoder
import eli5
from eli5.sklearn import PermutationImportance
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split, StratifiedKFold
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
    mask = df['target'] > -1
#    mask = (df['target'] > -1) & (df.isnull().sum(axis=1) <= 2)

#    logger.info(f'removed {(df.isnull().sum(axis=1) > 2).sum()} rows from train')

#    mask = (df['target'] > -1) & (df.isnull().sum(axis=1) < 4)
#    mask = (df['target'] > -1) & \
#           (~df['ord_3'].isnull()) & \
#           (~df['ord_2'].isnull()) & \
#           (~df['ord_5'].isnull())
    return df.loc[mask], df.loc[df['target'] == -1]


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


def replace_na(df, features):
    obj_cols = [col for col in features if df[col].dtype == np.object]
    num_cols = [col for col in features if df[col].dtype == np.float64]

    df[obj_cols] = df[obj_cols].fillna(value='-1')
    df[num_cols] = df[num_cols].fillna(value=-1)

    return df


def apply_ordinal_encoder(df, features):
    for feature in features:
        test_vals = df.loc[df['target'] == -1, feature].unique()
        train_vals = df.loc[df['target'] > -1, feature].unique()
        unknown_vals = [v for v in test_vals if v not in train_vals]
        df.loc[df[feature].isin(unknown_vals), feature] = np.nan

    train_mask = df['target'] > -1
    df = replace_na(df, features)

    for feature in features:
        encoder = LabelEncoder()
        df[feature] = encoder.fit(df[feature]).transform(df[feature])

    return df


def encode_ordinal_features(df, features, handle_missing='value'):
    train = df[df['target'] > -1]

    for feature in features:
        if handle_missing == 'value':
            if df[feature].dtype == np.object:
                df[feature] = df[feature].fillna(value='-1')
                train[feature] = train[feature].fillna(value='-1')
            else:
                df[feature] = df[feature].fillna(value=-1)
                train[feature] = train[feature].fillna(value=-1)

        mapping = train.groupby(feature)['target'].agg(['mean', 'count'])
        mapping = mapping.sort_values(by='mean', ascending=True)
        mapping['value'] = [i / (len(mapping) - 1) for i in range(len(mapping))]
        df[feature] = df[feature].map(mapping['value'].to_dict())

    return df


def one_hot_encoding(df, features):
    encoder = OneHotEncoder(cols=features, return_df=True, use_cat_names=True)
    encoded = encoder.fit_transform(df[features])

    df.drop(features, axis=1, inplace=True)
    df = pd.concat([df, encoded], axis=1, sort=False)

    return df


def target_encoding(df, features,
                    smoothing=0.2, handle_missing='value'):
    train, test = split_data(df)
    del df

    train.sort_index(inplace=True)
#    train_id = train['id']
#    target = train['target']

    encoded = []

    if not isinstance(smoothing, list):
        smoothing = [smoothing]

    for _iter in range(len(smoothing)):
        logger.debug(f'iteration {_iter + 1}')

        _encoded = pd.DataFrame()
        cv = StratifiedKFold(
            n_splits=5,
            random_state=2020,
            shuffle=True)

        for fold, (train_index, valid_index) in enumerate(
                cv.split(train[features], train['target'])):
            logger.info(f'target encoding on fold {fold + 1}')

            encoder = TargetEncoder(
                cols=features,
                smoothing=smoothing[_iter],
                handle_missing=handle_missing,
                handle_unknown='value')

            encoder.fit(train.iloc[train_index][features],
                        train.iloc[train_index]['target'])

            _encoded = _encoded.append(
                encoder.transform(train.iloc[valid_index][features]),
                ignore_index=False)

        encoded += [_encoded.sort_index()]

    encoder = TargetEncoder(
        cols=features,
        smoothing=np.mean(smoothing),
        handle_missing=handle_missing,
        handle_unknown='value')

    encoder.fit(train[features], train['target'])
    test[features] = encoder.transform(test[features])

    train[features] = pd.concat(encoded).groupby(level=0).mean()
#    train['id'] = train_id.values
#    train['target'] = target.values
#    train = encoded.sort_index()

    df = pd.concat([train[test.columns], test])

    return df


def encode_na(df, features):
    encoded = dict()

    train = df[df['target'] > -1]

    for feature in features:
        mask = train[feature].isna()
        grouped = train[~mask].groupby(feature)['target'].agg(['mean', 'count'])
        encoded[feature] = \
            (grouped['mean'] * grouped['count']).sum() / \
            grouped['count'].sum()

    return encoded


def fill_na(df, features, initial_strategy='mean'):
#    estimator = ExtraTreesRegressor(n_estimators=10, random_state=0)

    imputer = IterativeImputer(
#        estimator=estimator,
        initial_strategy=initial_strategy,
        verbose=2,
        random_state=0,
        tol=1e-4)

    df[features] = imputer.fit_transform(df[features])
    return df


def simulate_na(df, features):
    for feature in features:
        counts = df[df['target'] > -1].groupby(feature)['target'].count()
        mask = df[feature].isna()
        df.loc[mask, feature] = \
            choices(counts.index.values, weights=counts.values, k=mask.sum())

        logger.info(f'{feature}: imputed {mask.sum()} values')

    return df


def group_features(df, features, n_groups):
    for feature in features:
#        df[feature + '_' + str(n_groups)] = \
#            pd.qcut(df[feature], n_groups, labels=False)
        df[feature] = pd.qcut(df[feature], n_groups, labels=False)

    return df


def add_counts(df, features):
    for feature in features:
        counts = df.groupby(feature)['target'].count()
        df[feature + '_count'] = df[feature].map(counts)

    return df


def mark_as_na(df, features, threshold=0):
    for feature in features:
        counts = df.groupby(feature)['target'].count()
        categories = list(counts[counts < threshold].index)
        n_nan = df[feature].isna().sum()

        df.loc[df[feature].isin(categories), feature] = np.nan

        logger.info(
            f'{feature}: {df[feature].isna().sum() - n_nan} marked as NaN')

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
