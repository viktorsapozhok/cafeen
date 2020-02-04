import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB

from cafeen import config, steps, utils

logger = logging.getLogger('cafeen')


def submit_1(**kwargs):
    nrows = kwargs.get('nrows', None)
    valid_rows = kwargs.get('valid_rows', 0)

    df, valid_y = utils.read_data(nrows=nrows, valid_rows=valid_rows)
    na_value = df[df['target'] > -1]['target'].mean()
    df_copy = df.copy()

    logger.info(f'na_value: {na_value}')

#    df.drop(['bin_3'], axis=1, inplace=True)

    df = utils.mark_as_na(df, ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'], threshold=17)
#    df.loc[df['month'] == 10, 'month'] = np.nan

    df.loc[df['ord_4'] == 'J', 'ord_4'] = 'K'
    df.loc[df['ord_4'] == 'L', 'ord_4'] = 'M'
    df.loc[df['ord_4'] == 'S', 'ord_4'] = 'R'
    df.loc[df['month'] == 10, 'month'] = 9
    df['ord_5'] = df['ord_5'].str[0]
    df.loc[df_copy['ord_5'].isna(), 'ord_5'] = np.nan
    df.loc[df['ord_5'] == 'Z', 'ord_5'] = 'Y'
    df.loc[df['ord_5'] == 'K', 'ord_5'] = 'L'
    df.loc[df['ord_5'] == 'E', 'ord_5'] = 'D'

    #    df['n_nan'] = df.isnull().sum(axis=1)
#    df['ord_5_1'] = df['ord_5'].str[0]
#    df['ord_5_2'] = df['ord_5'].str[1]
#    df['day_month'] = df['day'].map(str) + '-' + df['month'].map(str)
#    df['ord_3_4'] = df['ord_3'].map(str) + '-' + df['ord_4'].map(str)
#    df['ord3_nom7'] = df['ord_3'].map(str) + '-' + df['nom_7'].map(str)
#    df['ord3_nom8'] = df['ord_3'].map(str) + '-' + df['nom_8'].map(str)

    features = utils.get_features(df.columns)
#    oe_features = [f for f in features if 'ord' in f]
    oe_features = [
        'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']
    ohe_features = [
        'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
        'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4',
        'day', 'month']
#    oe_features = []
#    ohe_features = []
    te_features = [f for f in features if (f not in oe_features) and (f not in ohe_features)]

#    df = utils.simulate_na(df, features)
#    df = utils.label_encoding(df, oe_features + ohe_features)
    df = utils.target_encoding(df, oe_features + ohe_features, na_value=na_value)
    df = utils.target_encoding_cv(df, te_features, n_rounds=3, na_value=na_value)

    for feature in features:
        df.loc[df_copy[feature].isna(), feature] = na_value
#        min_pos_value = df.loc[df[feature] > 0, feature].min()
#        df.loc[df[feature] == 0, feature] = min_pos_value
        logger.info(f'{feature}: {df[feature].min():.4f} - {df[feature].max():.4f}')
        df[feature] = np.log(0.01 + df[feature])

    for feature in features:
        logger.info(f'{feature}: {df[feature].nunique()}')

#    na_encoded = utils.encode_na(df, te_features)
#    df = utils.encode_ordinal_features(df, oe_features, handle_missing='value')
#    df = utils.one_hot_encoding(df, oe_features + ohe_features)
#    df = utils.target_encoding_alt(df, oe_features + ohe_features)
#    df = utils.target_encoding(df, te_features, smoothing=[0.2], handle_missing='value')

#    for feature in te_features:
#        df.loc[df[feature].isna(), feature] = na_encoded[feature]

#    df = utils.fill_na(df, features, initial_strategy='mean')
    assert df.isnull().sum().sum() == 0

    train, test = utils.split_data(df)
    del df

    features = utils.get_features(train.columns)
    logger.debug(f'{len(features)} features in dataset')
    logger.debug(f'train: {train.shape}, test: {test.shape}')

#    steps.BayesSearch(50).fit(train[features], train['target'])
#    features = utils.get_features(train.columns)


#    estimator = lgb.LGBMClassifier(
#            objective='binary',
#            metric='auc',
#            is_unbalance=True,
#            boost_from_average=False,
#            n_estimators=100)

#    selector = steps.FeatureSelector(estimator, threshold=0)
#    selector.fit(train[features], train['target'])
#    features = selector.get_features()

    _submit(
        train[features + ['target']],
        test[features + ['target', 'id']],
        valid_y=valid_y)


def submit_2(n_estimators=100, eta=0.1, nrows=None):
    train = pd.read_csv(config.path_to_train)
    test = pd.read_csv(config.path_to_test)
    train.sort_index(inplace=True)
    train_y = train['target']
    test_id = test['id']
    train.drop(['target', 'id'], axis=1, inplace=True)
    test.drop('id', axis=1, inplace=True)

    from sklearn.metrics import roc_auc_score
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


#    glm.fit(train, train_y)
#    pd.DataFrame({'id': test_id, 'target': glm.predict_proba(test)[:, 1]}).to_csv('submission.csv', index=False)


def _submit(train, test, valid_y=None, **kwargs):
#    estimator = steps.Classifier(
#        lgb.LGBMClassifier(
#            objective='binary',
#            metric='auc',
#            is_unbalance=True,
#            boost_from_average=False,
#            n_estimators=n_estimators,
#            learning_rate=eta,
#            num_leaves=57,
#            min_child_samples=35,
#            colsample_bytree=0.3,
#            reg_alpha=0.8,
#            reg_lambda=1),
#        n_splits=4)

#    estimator = LogisticRegression(solver='liblinear', C=0.095, verbose=1)

    features = utils.get_features(train.columns)

#    estimator = LogisticRegression(
#        solver='liblinear',
#        C=1,
#        class_weight='balanced',
#        max_iter=1000,
#        tol=1e-5,
#        penalty='l2',
#        verbose=1)

    estimator = LogisticRegression(
        random_state=1,
        solver='lbfgs',
        max_iter=2020,
        fit_intercept=True,
        penalty='none',
        verbose=1)

#    estimator = CategoricalNB(alpha=1)

    clf = steps.Classifier(estimator)
#    clf.cross_val(train[features].values, train['target'].values, n_splits=6, corr=False)
#    submitter = steps.Submitter(clf)

    submitter = steps.Submitter(clf, config.path_to_data)
#    submitter.fit(train[features], train['target']).predict(test, features)

    y_pred = submitter.fit(
        train[features], train['target']).predict_proba(test)

    if valid_y is not None:
        valid_y = valid_y.merge(y_pred, how='left', on='id')
        score = roc_auc_score(valid_y['y_true'].values, valid_y['target'].values)
        logger.info('')
        logger.debug(f'score: {score}')
        logger.info('')
