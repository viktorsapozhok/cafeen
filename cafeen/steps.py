from datetime import datetime
import logging
from os import path
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import eli5
from eli5.sklearn import PermutationImportance
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer
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
            early_stopping_rounds=100,
            verbose=20):
        score = 0

        if self.n_splits > 1:
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
                    eval_metric='auc',
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=verbose)

                score += self.estimators[fold].best_score_['valid_0']['auc']

            logger.info('')
            logger.debug(f'score: {score / self.n_splits:.4f}')
            logger.info('')
        else:
            self.estimators[0].fit(
                x, y,
                categorical_feature=categorical_feature,
                feature_name=feature_name,
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


class LogReg(BaseEstimator):
    def __init__(self, solver, C, class_weight, max_iter, tol, penalty):
        self.solver = solver
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.penalty = penalty
        self._estimator = LogisticRegression(
            penalty=self.penalty,
            solver=self.solver,
            C=self.C,
            class_weight=self.class_weight,
            tol=self.tol,
            max_iter=self.max_iter)

    def fit(self, x, y=None, **fit_params):
        self._estimator.fit(x, y)
        return self

    def cross_val(self, x, y, n_splits):
        score = 0

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=0)

        for fold, (train_index, valid_index) in enumerate(cv.split(x, y)):
            train_x, train_y = x[train_index], y[train_index]
            valid_x, valid_y = x[valid_index], y[valid_index]

            logger.info('')
            logger.debug(f'started training on fold {fold}')
            logger.info('')

            estimator = LogisticRegression(
                solver=self.solver,
                max_iter=self.max_iter,
                C=self.C,
                tol=self.tol,
                class_weight=self.class_weight,
                random_state=0,
                verbose=1)

            estimator.fit(train_x, train_y)
            y_pred = estimator.predict_proba(valid_x)
            _score = roc_auc_score(valid_y, y_pred[:, 1])
            logger.info(f'score: {_score:.4f}')

            score += _score

        logger.info('')
        logger.debug(f'score: {score / n_splits:.4f}')
        logger.info('')

    def predict_proba(self, x):
        return self._estimator.predict_proba(x)


class Submitter(BaseEstimator):
    def __init__(self, estimator, path_to_data):
        self.estimator = estimator
        self.path_to_data = path_to_data
        self.results = pd.DataFrame()

    def fit(self, x, y=None, **fit_params):
        self.estimator.fit(x.values, y.values, **fit_params)
        return self

    def predict(self, x, features):
        self.results['id'] = x['id'].astype('int')
        self.results['target'] = self.estimator.predict(x[features].values)

        now = datetime.now().strftime('%Y_%m_%d_%H_%M')
        path_to_file = path.join(
            self.path_to_data,
            'results_' + now + '.csv')

        self.results.to_csv(path_to_file, index=False)

    def predict_proba(self, x, features):
        self.results['id'] = x['id'].astype('int')
        p = self.estimator.predict_proba(x[features].values)
        self.results['target'] = p[:, 1]

        now = datetime.now().strftime('%Y_%m_%d_%H_%M')
        path_to_file = path.join(
            self.path_to_data,
            'results_' + now + '.csv')

        self.results.to_csv(path_to_file, index=False)


class BayesSearch(BaseEstimator, TransformerMixin):
    def __init__(self, n_trials=10):
        self.n_trials = n_trials

    def fit(self, x, y=None):
        def _evaluate(trial):
#            estimator = lgb.LGBMClassifier(
#                objective='binary',
#                metric='auc',
#                is_unbalance=True,
#                boost_from_average=False,
#                num_leaves=trial.suggest_int(
#                    'num_leaves', 10, 1000),
#                min_child_samples=trial.suggest_int(
#                    'min_child_samples', 1, 50),
#                colsample_bytree=trial.suggest_discrete_uniform(
#                    'colsample_bytree', 0.1, 0.9, 0.1),
#                reg_alpha=trial.suggest_discrete_uniform(
#                    'reg_alpha', 0, 1, 0.1),
#                reg_lambda=trial.suggest_discrete_uniform(
#                    'reg_lambda', 0, 1, 0.1))

            estimator = LogisticRegression(
                solver='liblinear',
                random_state=0,
                C=trial.suggest_discrete_uniform('C', 0.05, 1, 0.05),
                class_weight=trial.suggest_categorical('class_weight', ['balanced', None]))

            y_score = estimator.fit(train_x, train_y).predict_proba(test_x)
            score = -roc_auc_score(test_y, y_score[:, 1])
            self._print_trial(trial, score)

            return score

        def _pack_best_trial(trial):
            pass

        train_x, test_x, train_y, test_y = \
            train_test_split(x, y, shuffle=True, train_size=0.7, random_state=0)
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
        info += f'score: {score:6.5f}'

        is_best = False

        if trial.number > 0:
            info += f', best: {min(score, trial.study.best_value):6.5f}'

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
                if isinstance(params[key], str):
                    info += f'{key}: {params[key]}, '
                else:
                    info += f'{key}: {params[key]:4.3f}, '
        return info

    @staticmethod
    def _get_params(params):
        _params = dict(params)
        return _params


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, threshold=0.0002):
        self.estimator = estimator
        self.threshold = threshold
        self.features = None

    def get_features(self):
        return self.features

    def fit(self, x, y=None):
        train_x, test_x, train_y, test_y = \
            train_test_split(x, y, shuffle=True, train_size=0.7)

        self.estimator.fit(train_x, train_y)

        perm_estimator = PermutationImportance(
            estimator=self.estimator,
#            scoring=make_scorer(roc_auc_score, needs_proba=True),
            n_iter=5).fit(test_x, test_y)

        # calculate feature weights and return it as DataFrame
        expl = eli5.format_as_dataframe(
            eli5.explain_weights(
                perm_estimator,
                top=None,
                feature_names=x.columns.to_list()))

        # select features with weights above the threshold
        selected = self.select_features(expl)
        self.features = selected['feature'].to_list()

        return self

    def transform(self, x, y=None):
        return x[self.features]

    def select_features(self, expl, verbose=True):
        expl = expl.sort_values(by='weight', ascending=False)

        if verbose:
            logger.info('')

            for row in expl.itertuples():
                msg = f'{row.weight:7.4f} +- {2*row.std:5.3f} {row.feature}'

                if row.weight >= self.threshold:
                    logger.debug(msg)
                else:
                    logger.info(msg)

            logger.info('')

        mask = expl['weight'] >= self.threshold

        return expl.loc[mask, ['feature', 'weight', 'std']]
