import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import CategoricalNB, BernoulliNB
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler, OneHotEncoder

from cafeen import config, steps, utils

logger = logging.getLogger('cafeen')


def submit_1(**kwargs):
    nrows = kwargs.get('nrows', None)

    ordinal_features = ['ord_4', 'ord_5']
    splits = [3, 3]
    groups = [51, 27]
    filters = {
        'nom_5': [0, 0, 0, 0.5],
        'nom_6': [126, 0, 0, 0.5],
        'nom_9': [12, 0, 0.044, 0.398]
    }

    cardinal_encoding = dict()

    for i, feature in enumerate(['nom_6', 'nom_9']):
        cardinal_encoding[feature] = dict()
        cardinal_encoding[feature]['cv'] = StratifiedKFold(
            n_splits=splits[i],
            shuffle=True,
            random_state=2020)
        cardinal_encoding[feature]['n_groups'] = groups[i]
        cardinal_encoding[feature]['filter'] = filters[feature]

    correct_features = {'ord_4': True, 'ord_5': False, 'day': True, 'nom_7': True}

    df, valid_y = utils.read_data(
        nrows=nrows,
        valid_rows=kwargs.get('valid_rows', 0))

    encoder = steps.Encoder(
        ordinal_features=ordinal_features,
        cardinal_encoding=cardinal_encoding,
        filters=filters,
        handle_missing=True,
        log_alpha=0,
        one_hot_encoding=True,
        correct_features=correct_features,
        verbose=True)

    train_x, train_y, test_x, test_id = encoder.fit_transform(df)

    estimator = LogisticRegression(
        random_state=2020,
        C=0.054,
        class_weight={0: 1, 1: 2.01},
        solver='liblinear',
        tol=1e-4,
        max_iter=2020,
        fit_intercept=True,
        penalty='l2',
        verbose=1)

    if valid_y is None:
        submitter = steps.Submitter(estimator, config.path_to_data)
    else:
        submitter = steps.Submitter(estimator)

    if isinstance(train_y, pd.Series):
        train_y = train_y.values

    y_pred = submitter.fit(train_x, train_y).predict_proba(test_x, test_id=test_id)

    if valid_y is not None:
        _valid_y = valid_y.merge(y_pred[['id', 'target']], how='left', on='id')
        score = roc_auc_score(_valid_y['y_true'].values, _valid_y['target'].values)
        logger.debug(f'score: {score}')


def submit_2(**kwargs):
    nrows = kwargs.get('nrows', None)

    df, valid_y = utils.read_data(
        nrows=nrows,
        valid_rows=kwargs.get('valid_rows', 0))

    if nrows is None:
        df = utils.mark_as_na(df, ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'], threshold=50)

    na_value = df[df['target'] > -1]['target'].mean()
    df_copy = df.copy()

    logger.info(f'na_value: {na_value}')

    df.loc[df['ord_4'] == 'J', 'ord_4'] = 'K'
    df.loc[df['ord_4'] == 'L', 'ord_4'] = 'M'
    df.loc[df['ord_4'] == 'S', 'ord_4'] = 'R'
    df['ord_5'] = df_copy['ord_5'].str[0]
    df.loc[df_copy['ord_5'].isna(), 'ord_5'] = np.nan
    df.loc[df['ord_5'] == 'Z', 'ord_5'] = 'Y'
    df.loc[df['ord_5'] == 'K', 'ord_5'] = 'L'
    df.loc[df['ord_5'] == 'E', 'ord_5'] = 'D'

    df.loc[(df['month'] == 2) & (df['day'] == 2), 'day'] = 6
    df.loc[(df['month'] == 4) & (df['day'] == 6), 'day'] = 7
    df.loc[(df['month'] == 5) & (df['day'] == 7), 'day'] = 1
    df.loc[(df['month'] == 10) & (df['day'] == 2), 'day'] = 1
    df.loc[(df['month'] == 10) & (df['day'] == 6), 'day'] = 1
    df.loc[(df['month'] == 10) & (df['day'] == 7), 'day'] = 1

    features = utils.get_features(df.columns)
    ohe_features = [
        'ord_3', 'ord_4', 'ord_5',
        'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
        'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4',
        'ord_0', 'ord_1', 'ord_2', 'day', 'month']
    te_features = [
        'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

    df = utils.target_encoding(df, ohe_features)
    df = utils.target_encoding_cv(df, te_features, cv=KFold(n_splits=5))
    df = utils.group_features(df, te_features, n_groups=50, min_group_size=2000)

    logger.info('')

    for feature in features:
        df.loc[df[feature].isna(), feature] = -1

    del df_copy

    logger.info('amount of unique values')
    for feature in features:
        logger.info(f'{feature}: {df[feature].nunique()}')
    logger.info('')

    assert df.isnull().sum().sum() == 0

    logger.debug(f'{len(features)} features in dataset')
    logger.debug(f'train: {df.shape}, test: {df.shape}')

    encoder = OneHotEncoder(sparse=True)
    encoder.fit(df[ohe_features + te_features])
    train, test = utils.split_data(df)
    del df

    train_x = encoder.transform(train[ohe_features + te_features])
    test_x = encoder.transform(test[ohe_features + te_features])

    estimator = LogisticRegression(
        random_state=2020,
        C=0.1,
        class_weight='balanced',
        solver='liblinear',
        max_iter=2020,
        fit_intercept=True,
        penalty='l2',
        verbose=1)

    estimator.fit(train_x, train['target'].values)

    res = pd.DataFrame()
    res['id'] = test['id'].values
    res['y_pred'] = estimator.predict_proba(test_x)[:, 1]

    valid_y = valid_y.merge(res, how='left', on='id')
    score = roc_auc_score(valid_y['y_true'].values, valid_y['y_pred'].values)
    logger.info('')
    logger.debug(f'score: {score}')
    logger.info('')


def submit_4(**kwargs):
    logger.info('reading train')
    train = pd.read_csv(config.path_to_train, nrows=kwargs.get('nrows', None))

    bs = steps.BayesSearch(
        n_trials=500,
        n_folds=kwargs.get('folds'),
        verbose=kwargs.get('verbose'))

    bs.fit(train)


def _submit(train, test, valid_y=None, **kwargs):
#    estimator = steps.OneColClassifier(
#        estimator=lgb.LGBMClassifier(
#            objective='binary',
#            metric='auc',
#            is_unbalance=True,
#            boost_from_average=False,
#            n_estimators=kwargs.get('n_estimators', 100),
#            learning_rate=kwargs.get('eta', 0.1),
#            num_leaves=57,
#            min_child_samples=35,
#            colsample_bytree=0.3,
#            reg_alpha=0.8,
#            reg_lambda=1),
#        n_splits=1)

#    estimator = LogisticRegression(solver='liblinear', C=0.095, verbose=1)

    features = utils.get_features(train.columns)

#    estimator.fit(train[features], train['target'])
#    y_pred = estimator.predict_proba(test[features + ['id']])

    estimator = LogisticRegression(
        random_state=2020,
        C=0.1,
        class_weight='balanced',
        solver='liblinear',
        max_iter=2020,
        fit_intercept=True,
        penalty='l2',
        verbose=1)

#    estimator = LogisticRegression(
#        random_state=1,
#        solver='lbfgs',
#        max_iter=2020,
#        fit_intercept=True,
#        penalty='none',
#        verbose=1)

#    estimator = CategoricalNB(alpha=0)
#    estimator = BernoulliNB(alpha=1)

    clf = steps.Classifier(estimator)
#    clf.cross_val(train[features].values, train['target'].values, n_splits=6, corr=False)
#    submitter = steps.Submitter(clf)

    if valid_y is None:
        submitter = steps.Submitter(clf, config.path_to_data)
    else:
        submitter = steps.Submitter(clf)

    y_pred = submitter.fit(
        train[features], train['target']).predict_proba(test)

    if valid_y is not None:
#        valid_y = valid_y.merge(y_pred[['id', 'mean']], how='left', on='id')
        valid_y = valid_y.merge(y_pred[['id', 'target']], how='left', on='id')
        score = roc_auc_score(valid_y['y_true'].values, valid_y['target'].values)
        logger.info('')
        logger.debug(f'score: {score}')
        logger.info('')
