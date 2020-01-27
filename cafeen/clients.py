import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from cafeen import config, steps

logger = logging.getLogger('cafeen')


def encode_data(train, test):
    obj_cols = [
        col for col in test.columns
        if (col not in ['id']) and (test[col].dtype == np.object)]
    num_cols = [
        col for col in test.columns
        if (col not in ['id']) and (test[col].dtype == np.float64)]

    logger.info('fill nans in numeric columns')
    train[num_cols] = train[num_cols].fillna(value=-9999)
    test[num_cols] = test[num_cols].fillna(value=-9999)

    for col in tqdm(obj_cols, ascii=True, desc='encoding'):
        # fill nans replace nans by most frequent value
        # works much faster than SimpleImputer
        train[col] = train[col].fillna(value=train[col].mode()[0])
        test[col] = test[col].fillna(value=test[col].mode()[0])

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
