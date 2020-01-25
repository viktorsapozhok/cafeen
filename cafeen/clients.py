from collections import defaultdict
import logging

import pandas as pd
from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from cafeen import config

logger = logging.getLogger('cafeen')


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._encoders = defaultdict()

    def fit(self, X, y=None):
        for col in tqdm(X.columns, ascii=True, desc='encoding'):
            self._encoders[col] = LabelEncoder().fit(X)
        return self

    def transform(self, X):
        for col in X.columns:
            X[col] = self._encoders[col].transform(X[col])
        return X


def encode_files():
    logger.info('reading train')
    train = pd.read_csv(config.path_to_train, nrows=1000)

    logger.info('reading test')
    test = pd.read_csv(config.path_to_test, nrows=1000)

    le = LabelEncoder()

    for col in tqdm(test.columns, ascii=True):
        le.fit(pd.concat((train[col], test[col])))

        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    train.to_csv(config.path_to_train_enc)
    test.to_csv(config.path_to_test_enc)
