from datetime import datetime
import logging
from os import path
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

optuna.logging.set_verbosity(optuna.logging.ERROR)
logger = logging.getLogger('cafeen')


class Classifier(BaseEstimator):
    def __init__(self, estimator, n_splits):
        self.estimators = [estimator] * n_splits
        self.n_splits = n_splits

    def fit(self, x, y=None,
            categorical_feature='auto',
            feature_name='auto',
            early_stopping_rounds=50,
            verbose=20):
        score = 0

        cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=0)

        for fold, (train_index, valid_index) in enumerate(cv.split(x, y)):
            train_x, train_y = x[train_index], y[train_index]
            valid_x, valid_y = x[valid_index], y[valid_index]

            logger.info('')
            logger.debug(f'started training on fold {fold}')
            logger.info('')

            self.estimators[fold].fit(
                train_x,
                train_y,
                categorical_feature=categorical_feature,
                feature_name=feature_name,
                eval_set=[(valid_x, valid_y)],
                eval_metric=self.roc_auc_score,
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose)

            score += -self.estimators[fold].best_score_['valid_0']['roc_auc']

        logger.info('')
        logger.debug(f'score: {score / self.n_splits:.3f}')
        logger.info('')

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

    def fit(self, x, y=None,
            categorical_feature='auto',
            feature_name='auto'):
        self.estimator.fit(
            x.values,
            y.values,
            categorical_feature=categorical_feature,
            feature_name=feature_name)
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


class BayesSearch(BaseEstimator, TransformerMixin):
    def __init__(self, n_trials=10):
        self.n_trials = n_trials

    def fit(self, x, y=None):
        def _evaluate(trial):
            estimator = lgb.LGBMClassifier(
#                n_jobs=self.n_jobs,
                num_leaves=trial.suggest_int(
                    'num_leaves', 10, 1000),
#                learning_rate=trial.suggest_loguniform(
#                    'learning_rate', 0.005, 0.3),
#                n_estimators=trial.suggest_int(
#                    'n_estimators', 50, 500),
                min_child_samples=trial.suggest_int(
                    'min_child_samples', 1, 50),
                colsample_bytree=trial.suggest_discrete_uniform(
                    'colsample_bytree', 0.5, 0.9, 0.1),
                reg_alpha=trial.suggest_discrete_uniform(
                    'reg_alpha', 0, 1, 0.1),
                reg_lambda=trial.suggest_discrete_uniform(
                    'reg_lambda', 0, 1, 0.1))

            y_score = estimator.fit(train_x, train_y).predict_proba(test_x)
            score = -roc_auc_score(test_y, y_score[:, 1])
            self._print_trial(trial, score)

            return score

        def _pack_best_trial(trial):
            pass

        train_x, test_x, train_y, test_y = \
            train_test_split(x, y, shuffle=True, train_size=0.8)
        study = optuna.create_study()
        study.optimize(_evaluate, n_trials=self.n_trials)

        # compare best_trial vs model from file and save the best one
        _pack_best_trial(study.best_trial)

        return self

    def transform(self, x, y=None):
        return x

    def _print_trial(self, trial, score):
        info = f'trial {trial.number:2.0f}: '
        info += self._params2str(trial.params)
        info += f'score: {score:6.4f}'

        is_best = False

        if trial.number > 0:
            info += f', best: {min(score, trial.study.best_value):6.4f}'

            if score < trial.study.best_value:
                is_best = True
        elif trial.number < 0:
            if score < trial.study_best_value:
                is_best = True

        if is_best:
            logger.debug(info)
        else:
            logger.info(info)

    @staticmethod
    def _create_trial(estimator, number, best_value):
        trial = optuna.trial.FixedTrial(estimator.get_params())
        setattr(trial, 'number', number)
        setattr(trial, 'study_best_value', best_value)
        return trial

    @staticmethod
    def _params2str(params):
        info = ''
        for key in params:
            if params[key] is None:
                info += f'{key}: {params[key]}, '
            else:
                if isinstance(params[key], int):
                    info += f'{key}: {params[key]:4.0f}, '
                else:
                    info += f'{key}: {params[key]:4.3f}, '
        return info

    @staticmethod
    def _get_params(params):
        _params = dict(params)
        return _params
