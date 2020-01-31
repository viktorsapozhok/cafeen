import logging

import lightgbm as lgb
import pandas as pd
from sklearn.impute import MissingIndicator
from sklearn.model_selection import StratifiedKFold

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

    train, test = utils.split_data(df)

    _submit(train, test, n_estimators)


def submit_3(n_estimators=100):
    df = utils.read_data()
    features = utils.get_features(df.columns)

    df = utils.mark_as_na(df, ['nom_5', 'nom_6', 'nom_9'], threshold=85)

    ind_features = MissingIndicator().fit_transform(df[features])
    ind_columns = ['ind_' + str(i) for i in range(ind_features.shape[1])]
    df[ind_columns] = pd.DataFrame(ind_features).astype('int')

    df = utils.encode_ordinal_features(df, features)
    df = utils.fill_na(df, features, initial_strategy='most_frequent')
    df = utils.add_counts(
        df, ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_5'])

    train, test = utils.split_data(df)

    _submit(train, test, n_estimators)


def submit_4(n_estimators=100):
    df = utils.read_data()
    features = utils.get_features(df.columns)

#    df = utils.mark_as_na(df, ['nom_5', 'nom_6', 'nom_9'], threshold=100)

    df = utils.target_encoding(
        df, features, smoothing=0.2, handle_missing='return_nan')

    df = utils.fill_na(df, features, initial_strategy='mean')

    train, test = utils.split_data(df)

#    steps.BayesSearch(50).fit(train[features], train['target'])

#    estimator = lgb.LGBMClassifier(
#            objective='binary',
#            metric='auc',
#            is_unbalance=True,
#            boost_from_average=False,
#            n_estimators=200)

#    selector = steps.FeatureSelector(estimator, threshold=0)
#    selector.fit(train[features], train['target'])
#    features = selector.get_features()

    _submit(
        train[features + ['target']],
        test[features + ['target', 'id']],
        n_estimators)


def _submit(train, test, n_estimators=100):
    estimator = steps.Classifier(
        lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            is_unbalance=True,
            boost_from_average=False,
            n_estimators=n_estimators,
            learning_rate=0.005,
            num_leaves=57,
            min_child_samples=35,
            colsample_bytree=0.3,
            reg_alpha=0.8,
            reg_lambda=1),
        n_splits=6)

    features = utils.get_features(train.columns)

    submitter = steps.Submitter(estimator, config.path_to_data)
    submitter.fit(
        train[features],
        train['target'],
        categorical_feature='auto',
        feature_name='auto'
    ).predict(test, features)
