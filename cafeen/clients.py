import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from cafeen import config, steps, utils

logger = logging.getLogger('cafeen')


def encode_data(train, test, features=None):
    if features is None:
        features = test.columns

    obj_cols = [
        col for col in features
        if (col not in ['id']) and (test[col].dtype == np.object)]
    num_cols = [
        col for col in features
        if (col not in ['id']) and (test[col].dtype == np.float64)]

    logger.info('fill nans in numeric columns')
    train[num_cols] = train[num_cols].fillna(value=-1)
    test[num_cols] = test[num_cols].fillna(value=-1)

    for col in tqdm(obj_cols, ascii=True, desc='encoding', ncols=70):
        # fill nans replace nans by most frequent value
        # works much faster than SimpleImputer
        train[col] = train[col].fillna(value='NAN')
        test[col] = test[col].fillna(value='NAN')

        # encode values in column
        encoder = LabelEncoder()
        encoder.fit(pd.concat((train[col], test[col])))
        train[col] = encoder.transform(train[col])
        test[col] = encoder.transform(test[col])

    return train, test


def submit_0():
    logger.info('reading train')
    train = pd.read_csv(config.path_to_train)

    logger.info('reading test')
    test = pd.read_csv(config.path_to_test)

    train, test = encode_data(train, test)

    features = [col for col in test.columns if col not in ['target', 'id']]

    estimator = steps.Classifier(
        lgb.LGBMClassifier(n_estimators=100),
        n_splits=4)

    submitter = steps.Submitter(estimator, config.path_to_data)
    submitter.fit(train[features], train['target']).predict(test, features)


def submit_1():
    logger.info('reading train')
    train = pd.read_csv(config.path_to_train)

    logger.info('reading test')
    test = pd.read_csv(config.path_to_test)

    train, test = encode_data(train, test)
    train, test = utils.group_feature(train, test, 'nom_5', n_groups=20)
    train, test = utils.group_feature(train, test, 'nom_9', n_groups=20)

    features = [col for col in train.columns if col not in ['id', 'target']]

    estimator = steps.Classifier(
        lgb.LGBMClassifier(n_estimators=200),
        n_splits=4)

    submitter = steps.Submitter(estimator, config.path_to_data)
    submitter.fit(train[features], train['target']).predict(test, features)


def submit_2():
    logger.info('reading train')
    train = pd.read_csv(config.path_to_train)

    logger.info('reading test')
    test = pd.read_csv(config.path_to_test)

    train['ord3_x_ord2'] = train['ord_3'].map(str) + '_x_' + train['ord_2']
    test['ord3_x_ord2'] = test['ord_3'].map(str) + '_x_' + test['ord_2']

    train, test = encode_data(train, test)
    train, test = utils.group_feature(train, test, 'nom_5', n_groups=20)
    train, test = utils.group_feature(train, test, 'nom_9', n_groups=20)

    features = [col for col in train.columns if col not in ['id', 'target']]

    estimator = steps.Classifier(
        lgb.LGBMClassifier(n_estimators=100),
        n_splits=4)

    submitter = steps.Submitter(estimator, config.path_to_data)
    submitter.fit(train[features], train['target']).predict(test, features)


def submit_3():
    logger.info('reading train')
    train = pd.read_csv(config.path_to_train)

    logger.info('reading test')
    test = pd.read_csv(config.path_to_test)

    train['ord3_x_ord2'] = train['ord_3'].map(str) + '_x_' + train['ord_2']
    test['ord3_x_ord2'] = test['ord_3'].map(str) + '_x_' + test['ord_2']
    train.loc[train['month'] == 10, 'month'] = 11
    test.loc[train['month'] == 10, 'month'] = 11

    train, test = encode_data(train, test)
    train, test = utils.group_feature(train, test, 'nom_5', n_groups=20)
    train, test = utils.group_feature(train, test, 'nom_9', n_groups=20)
    train, test = utils.group_feature(train, test, 'ord_5', n_groups=20)

    features = [col for col in train.columns if col not in ['id', 'target']]

    estimator = steps.Classifier(
        lgb.LGBMClassifier(n_estimators=100),
        n_splits=3)

    submitter = steps.Submitter(estimator, config.path_to_data)
    submitter.fit(train[features], train['target']).predict(test, features)


def submit_4(n_estimators=100):
    logger.info('reading train')
    train = pd.read_csv(config.path_to_train)

    logger.info('reading test')
    test = pd.read_csv(config.path_to_test)

    features = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
                'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4',
                'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4',
                'day', 'month']

    for feature in features:
        train, test = utils.add_woe_feature(train, test, feature)

    features_woe = [feature + '_woe' for feature in features]
    train = utils.add_woe_max(train, features_woe)
    test = utils.add_woe_max(test, features_woe)

    train, test = encode_data(train, test)
    train, test = utils.group_feature(train, test, 'nom_5', n_groups=20)
    train, test = utils.group_feature(train, test, 'nom_9', n_groups=20)
    train, test = utils.group_feature(train, test, 'ord_5', n_groups=20)

    for feature in ['nom_5_20', 'nom_9_20', 'ord_5_20']:
        train, test = utils.add_woe_feature(train, test, feature)

    features = [col for col in train.columns if 'woe' in col] + \
               ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_5']
    cat_features = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_5']

#    steps.BayesSearch(30, 1).fit(train[features], train['target'])

    estimator = steps.Classifier(
        lgb.LGBMClassifier(
            n_estimators=n_estimators,
            num_leaves=251,
            learning_rate=0.05,
            min_child_samples=1,
            colsample_bytree=0.5,
            reg_alpha=0.3,
            reg_lambda=0.6),
        n_splits=3)

    submitter = steps.Submitter(estimator, config.path_to_data)
    submitter.fit(
        train[features],
        train['target']
    ).predict(test, features)
