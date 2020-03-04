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
    verbose = kwargs.get('verbose', False)

    ordinal_features = ['ord_4', 'ord_5', 'ord_0', 'ord_1',
                        'bin_0', 'bin_1', 'bin_2', 'bin_4',
                        'nom_0', 'nom_4', 'nom_3']

    cardinal_encoding = dict()
    cardinal_encoding['nom_6'] = dict()
    cardinal_encoding['nom_6']['cv'] = StratifiedKFold(n_splits=3, shuffle=True, random_state=2020)
    cardinal_encoding['nom_6']['n_groups'] = 3
    cardinal_encoding['nom_9'] = dict()

    correct_features = {'ord_4': False, 'ord_5': False, 'day': False, 'nom_7': False}

    df, valid_y = utils.read_data(
        nrows=nrows,
        valid_rows=kwargs.get('valid_rows', 0))

    filters = {
        'nom_9': [0.0000001, 7, 29]
    }

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

    for w in np.arange(1.2, 1.6, 0.05):
        estimator = LogisticRegression(
            random_state=2020,
            C=0.049,
            class_weight={0: 1, 1: w},
            solver='liblinear',
            max_iter=2020,
            fit_intercept=True,
            penalty='l2',
            verbose=1 * verbose)

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
            logger.debug(f'{w:.2f}, score: {score:.6f}')


def submit_4(**kwargs):
    logger.info('reading train')
    train = pd.read_csv(config.path_to_train, nrows=kwargs.get('nrows', None))
    train = train.drop(columns=['bin_3'])

    bs = steps.BayesSearch(
        n_trials=kwargs.get('trials'),
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
