import logging

import lightgbm as lgb
import pandas as pd
from sklearn.impute import MissingIndicator
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

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


def submit_4(n_estimators=100, eta=0.1, nrows=None):
    df = utils.read_data(nrows=nrows)

#    df.drop(['bin_3'], axis=1, inplace=True)

#    df = utils.mark_as_na(df, ['nom_5', 'nom_6', 'nom_9'], threshold=15)

#    df['n_nan'] = df.isnull().sum(axis=1)
#    df['ord_5_1'] = df['ord_5'].str[0]
#    df['ord_5_2'] = df['ord_5'].str[1]
#    df['day_month'] = df['day'].map(str) + '-' + df['month'].map(str)
#    df['ord_3_4'] = df['ord_3'].map(str) + '-' + df['ord_4'].map(str)
#    df['ord3_nom7'] = df['ord_3'].map(str) + '-' + df['nom_7'].map(str)
#    df['ord3_nom8'] = df['ord_3'].map(str) + '-' + df['nom_8'].map(str)

    features = utils.get_features(df.columns)
    oe_features = [f for f in features if 'ord' in f]
    ohe_features = [
        'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
        'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4',
        'day', 'month']
#    oe_features = []
#    ohe_features = []
    te_features = [f for f in features if (f not in oe_features) and (f not in ohe_features)]

    df = utils.simulate_na(df, oe_features + te_features)

#    na_encoded = utils.encode_na(df, te_features)
    df = utils.encode_ordinal_features(df, oe_features, handle_missing='return_na')
    df = utils.one_hot_encoding(df, ohe_features)

    df = utils.target_encoding(df, te_features, smoothing=[0.2], handle_missing='return_na')

#    for feature in te_features:
#        df.loc[df[feature].isna(), feature] = na_encoded[feature]

    features = utils.get_features(df.columns)
    logger.debug(f'{len(features)} features in dataset')

#    df = utils.fill_na(df, features, initial_strategy='mean')
    assert df.isnull().sum().sum() == 0

    train, test = utils.split_data(df)
    del df

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
        n_estimators, eta)


def _submit(train, test, n_estimators=100, eta=0.1):
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

    estimator = steps.LogReg(solver='liblinear', C=1, class_weight='balanced', max_iter=1000, tol=1e-4, penalty='l2')
    estimator.cross_val(train[features].values, train['target'].values, n_splits=6)

#    submitter = steps.Submitter(estimator, config.path_to_data)
#    submitter.fit(train[features], train['target']).predict(test, features)

#    submitter.fit(train[features], train['target']).predict_proba(test, features)
