import logging

import lightgbm as lgb
import pandas as pd
from sklearn.impute import MissingIndicator

from cafeen import config, steps, utils

logger = logging.getLogger('cafeen')


def submit_0(n_estimators=100):
    train, test = utils.split_data(
        utils.encode_features(utils.read_data()))

    _submit(train, test, n_estimators)


def submit_1(n_estimators=100):
    df = utils.read_data()
    features = utils.get_features(df.columns)

    ind_features = MissingIndicator().fit_transform(df[features])
    ind_columns = ['ind_' + str(i) for i in range(ind_features.shape[1])]
    df[ind_columns] = pd.DataFrame(ind_features).astype('int')

    df = utils.encode_ordinal_features(df, features)
    df = utils.fill_na(df, features)

    train, test = utils.split_data(utils.encode_features(df))

#    steps.BayesSearch(50).fit(train[features], train['target'])

    _submit(train, test, n_estimators)


def submit_2(n_estimators=100):
    df = utils.read_data()
    features = utils.get_features(df.columns)

    ind_features = MissingIndicator().fit_transform(df[features])
    ind_columns = ['ind_' + str(i) for i in range(ind_features.shape[1])]
    df[ind_columns] = pd.DataFrame(ind_features).astype('int')

    df = utils.encode_ordinal_features(df, features)

    grouped_features = ['nom_' + str(i) for i in range(5, 10)] + ['ord_5']
    df = utils.group_features(df, grouped_features, 20)
    df = utils.fill_na(df, features)

    train, test = utils.split_data(utils.encode_features(df))

    _submit(train, test, n_estimators)


def _submit(train, test, n_estimators=100):
    estimator = steps.Classifier(
        lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.01,
            num_leaves=63,
            min_child_samples=1,
            colsample_bytree=0.5,
            reg_alpha=0.2,
            reg_lambda=0.9),
        n_splits=4)

    features = utils.get_features(train.columns)

    submitter = steps.Submitter(estimator, config.path_to_data)
    submitter.fit(
        train[features],
        train['target'],
        categorical_feature='auto',
        feature_name='auto'
    ).predict(test, features)
