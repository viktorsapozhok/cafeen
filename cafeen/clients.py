import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import CategoricalNB, BernoulliNB
from sklearn.preprocessing import StandardScaler

from cafeen import config, steps, utils

logger = logging.getLogger('cafeen')


def submit_1(**kwargs):
    df, valid_y = utils.read_data(
        nrows=kwargs.get('nrows', None),
        valid_rows=kwargs.get('valid_rows', 0))
    na_value = df[df['target'] > -1]['target'].mean()
    df_copy = df.copy()

    logger.info(f'na_value: {na_value}')

    df = utils.mark_as_na(df, ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'], threshold=17)

    df.loc[df['ord_4'] == 'J', 'ord_4'] = 'K'
    df.loc[df['ord_4'] == 'L', 'ord_4'] = 'M'
    df.loc[df['ord_4'] == 'S', 'ord_4'] = 'R'
#    df.loc[df['month'] == 10, 'month'] = 9
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
    oe_features = [
        'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']
    ohe_features = [
        'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
        'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'day', 'month']
    te_features = [f for f in features if (f not in oe_features) and (f not in ohe_features)]

#    df = utils.label_encoding(df, oe_features)
#    scaler = StandardScaler(copy=True)
#    scaler.fit(df[oe_features])
#    df[oe_features] = scaler.transform(df[oe_features])

    df = utils.target_encoding(df, oe_features + ohe_features, na_value=na_value)
#    df = utils.target_encoding_cv(df, te_features, cv=KFold(n_splits=5, random_state=1), n_rounds=1, na_value=na_value)
#    df = utils.target_encoding_cv(df, te_features, cv=KFold(n_splits=5), n_rounds=1, na_value=na_value)

    df = utils.target_encoding_cv(df, te_features, cv=KFold(n_splits=5))
    df = utils.group_features(df, te_features, n_groups=20, min_group_size=5000)
    df = utils.target_encoding(df, te_features)

    logger.info('')

    for feature in features:
        if na_value is not None:
            df.loc[df_copy[feature].isna(), feature] = na_value
        logger.info(f'{feature}: {df[feature].min():.4f} - {df[feature].max():.4f}')
        df[feature] = np.log(0.1 + df[feature])
    logger.info('')

    logger.info('amount of unique values')
    for feature in features:
        logger.info(f'{feature}: {df[feature].nunique()}')
    logger.info('')

#    df = utils.one_hot_encoding(df, features)

    assert df.isnull().sum().sum() == 0

    train, test = utils.split_data(df)
    del df

    features = utils.get_features(train.columns)
    logger.debug(f'{len(features)} features in dataset')
    logger.debug(f'train: {train.shape}, test: {test.shape}')

    _submit(
        train[features + ['target']],
        test[features + ['target', 'id']],
        valid_y=valid_y,
        n_estimators=kwargs.get('n_estimators', 100),
        eta=kwargs.get('eta', 0.1))


def submit_2(n_estimators=100, eta=0.1, nrows=None):
    train = pd.read_csv(config.path_to_train)
    test = pd.read_csv(config.path_to_test)
    train.sort_index(inplace=True)
    train_y = train['target']
    test_id = test['id']
    train.drop(['target', 'id'], axis=1, inplace=True)
    test.drop('id', axis=1, inplace=True)

    cat_feat_to_encode = train.columns.tolist()
    smoothing = 0.20

    import category_encoders as ce
    oof = pd.DataFrame([])

    from sklearn.model_selection import StratifiedKFold

    for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=2020, shuffle=True).split(train, train_y):
        logger.info('target encoding')
        ce_target_encoder = ce.TargetEncoder(cols=cat_feat_to_encode, smoothing=smoothing)
        ce_target_encoder.fit(train.iloc[tr_idx, :], train_y.iloc[tr_idx])
        oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)

    ce_target_encoder = ce.TargetEncoder(cols=cat_feat_to_encode, smoothing=smoothing)
    ce_target_encoder.fit(train, train_y)
    train = oof.sort_index()
    test = ce_target_encoder.transform(test)

    from sklearn import linear_model

    estimator = linear_model.LogisticRegression(random_state=1, solver='lbfgs', max_iter=2020, fit_intercept=True,
                                          penalty='none', verbose=1)

    clf = steps.Classifier(estimator)
    clf.cross_val(train.values, train_y.values, n_splits=6)

    test['id'] = test_id
    features = utils.get_features(train)
    submitter = steps.Submitter(clf, config.path_to_data)
    submitter.fit(train, train_y).predict_proba(test, features)


def submit_3(**kwargs):
    df, valid_y = utils.read_data(
        nrows=kwargs.get('nrows', None),
        valid_rows=kwargs.get('valid_rows', 0))
    na_value = df[df['target'] > -1]['target'].mean()
    df_copy = df.copy()

    logger.info(f'na_value: {na_value}')

    df = utils.mark_as_na(df, ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'], threshold=17)

    df.loc[df['ord_4'] == 'J', 'ord_4'] = 'K'
    df.loc[df['ord_4'] == 'L', 'ord_4'] = 'M'
    df.loc[df['ord_4'] == 'S', 'ord_4'] = 'R'
    df.loc[df['month'] == 10, 'month'] = 9
    df['ord_5'] = df['ord_5'].str[0]
    df.loc[df_copy['ord_5'].isna(), 'ord_5'] = np.nan
    df.loc[df['ord_5'] == 'Z', 'ord_5'] = 'Y'
    df.loc[df['ord_5'] == 'K', 'ord_5'] = 'L'
    df.loc[df['ord_5'] == 'E', 'ord_5'] = 'D'

    features = utils.get_features(df.columns)
    oe_features = [
        'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']
    ohe_features = [
        'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
        'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4',
        'day', 'month']
    te_features = [f for f in features if (f not in oe_features) and (f not in ohe_features)]

    df = utils.target_encoding(df, oe_features + ohe_features, na_value=na_value)
    df = utils.target_encoding_cv(df, te_features, cv=KFold(n_splits=8, random_state=1), n_rounds=2, na_value=na_value)

    logger.info('')

    for feature in features:
        df.loc[df_copy[feature].isna(), feature] = na_value
        logger.info(f'{feature}: {df[feature].min():.4f} - {df[feature].max():.4f}')
#        df[feature] = np.log(0.1 + df[feature])
    logger.info('')

    logger.info('amount of unique values')
    for feature in features:
        logger.info(f'{feature}: {df[feature].nunique()}')
    logger.info('')

    df = utils.group_features(df, ['nom_7', 'nom_8'], n_groups=20, min_group_size=3000)
    df = utils.group_features(df, ['nom_5', 'nom_6', 'nom_9'], n_groups=10, min_group_size=5000)
    df = utils.label_encoding(df, features)

    logger.info('amount of unique after grouping')
    for feature in te_features:
        logger.info(f'{feature}: {df[feature].nunique()}')
    logger.info('')

    assert df.isnull().sum().sum() == 0

    train, test = utils.split_data(df)
    del df

    features = utils.get_features(train.columns)
    logger.debug(f'{len(features)} features in dataset')
    logger.debug(f'train: {train.shape}, test: {test.shape}')

#    steps.BayesSearch(50).fit(train[features], train['target'])
#    features = utils.get_features(train.columns)

#    selector = steps.FeatureSelector(estimator, threshold=0)
#    selector.fit(train[features], train['target'])
#    features = selector.get_features()

    _submit(
        train[features + ['target']],
        test[features + ['target', 'id']],
        valid_y=valid_y)


def _submit(train, test, valid_y=None, **kwargs):
    estimator = steps.OneColClassifier(
        estimator=lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            is_unbalance=True,
            boost_from_average=False,
            n_estimators=kwargs.get('n_estimators', 100),
            learning_rate=kwargs.get('eta', 0.1),
            num_leaves=57,
            min_child_samples=35,
            colsample_bytree=0.3,
            reg_alpha=0.8,
            reg_lambda=1),
        n_splits=1)

#    estimator = LogisticRegression(solver='liblinear', C=0.095, verbose=1)

    features = utils.get_features(train.columns)

    estimator.fit(train[features], train['target'])
    y_pred = estimator.predict_proba(test[features + ['id']])

#    estimator = LogisticRegression(
#        solver='liblinear',
#        C=0.1,
#       class_weight='balanced',
#        max_iter=1000,
#        tol=1e-9,
#        penalty='l2',
#        verbose=1)

#    estimator = LogisticRegression(
#        random_state=1,
#        solver='lbfgs',
#        max_iter=2020,
#        fit_intercept=True,
#        penalty='none',
#        verbose=1)

#    estimator = CategoricalNB(alpha=0)
#    estimator = BernoulliNB(alpha=1)

#    clf = steps.Classifier(estimator)
#    clf.cross_val(train[features].values, train['target'].values, n_splits=6, corr=False)
#    submitter = steps.Submitter(clf)

#    if valid_y is None:
#        submitter = steps.Submitter(clf, config.path_to_data)
#    else:
#        submitter = steps.Submitter(clf)

#    y_pred = submitter.fit(
#        train[features], train['target']).predict_proba(test)

    if valid_y is not None:
        valid_y = valid_y.merge(y_pred[['id', 'mean']], how='left', on='id')
        score = roc_auc_score(valid_y['y_true'].values, valid_y['mean'].values)
        logger.info('')
        logger.debug(f'score: {score}')
        logger.info('')
