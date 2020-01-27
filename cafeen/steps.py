from datetime import datetime
import logging
from os import path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

logger = logging.getLogger('cafeen')


class Classifier(BaseEstimator):
    def __init__(self, estimator, n_splits):
        self.estimators = [estimator] * n_splits
        self.n_splits = n_splits

    def fit(self, x, y=None, early_stopping_rounds=50, verbose=20):
        cv = KFold(n_splits=self.n_splits, random_state=42)

        for fold, (train_index, valid_index) in enumerate(cv.split(x, y)):
            train_x, train_y = x[train_index], y[train_index]
            valid_x, valid_y = x[valid_index], y[valid_index]

            logger.info(f'started training on fold {fold}')
            logger.info(f'train size: {train_x.shape}')
            logger.info(f'valid size: {valid_x.shape}')

            self.estimators[fold].fit(
                train_x,
                train_y,
                eval_set=[(valid_x, valid_y)],
                eval_metric=self.roc_auc_score,
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose)

        return self

    def predict(self, x):
        predicted = np.zeros(len(x))

        for estimator in self.estimators:
            p = estimator.predict_proba(x)
            predicted += p[:, 1] / self.n_splits

        return predicted

    @staticmethod
    def roc_auc_score(y_true, y_score):
        return 'roc_auc', -roc_auc_score(y_true, y_score), False


class Submitter(BaseEstimator):
    def __init__(self, estimator, path_to_data):
        self.estimator = estimator
        self.path_to_data = path_to_data
        self.results = pd.DataFrame()

    def fit(self, x, y=None):
        self.estimator.fit(x.values, y.values)
        return self

    def predict(self, x, features):
        self.results['id'] = x['id'].astype('int')
        self.results['target'] = self.estimator.predict(x[features].values)

        now = datetime.now().strftime('%Y_%m_%d_%H_%M')
        path_to_file = path.join(
            self.path_to_data,
            'results_' + now + '.csv')

        self.results.to_csv(
            path_to_file,
            index=False,
            float_format='%.5f')
