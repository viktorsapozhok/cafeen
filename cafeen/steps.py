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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

from cafeen import utils

optuna.logging.set_verbosity(optuna.logging.ERROR)
logger = logging.getLogger('cafeen')


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            cardinal_encoding=None,
            handle_missing=True,
            min_category_size=17,
            log_alpha=0.1,
            one_hot_encoding=True,
            correct_features=None
    ):
        self.cardinal_encoding = cardinal_encoding
        self.handle_missing = handle_missing
        self.min_category_size = min_category_size
        self.correct_features = correct_features
        self.log_alpha = log_alpha
        self.one_hot_encoding = one_hot_encoding

        self.nominal_features = None
        self.ordinal_features = None
        self.cardinal_features = None

    def fit(self, x, y=None):
        features = self.get_features(x.columns)

        self.nominal_features = [
            'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',
            'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4',
            'day', 'month'
        ]
        self.ordinal_features = [
            'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5'
        ]
        self.cardinal_features = [
            feature for feature in features
            if feature not in self.nominal_features + self.ordinal_features]

        return self

    def transform(self, x):
        _x = x.copy()

        features = self.get_features(_x.columns)

        if self.min_category_size > 0:
            _x = utils.mark_as_na(
                _x, self.cardinal_features, threshold=self.min_category_size)

        _x['ord_5'] = x['ord_5'].str[0]
        _x.loc[x['ord_5'].isna(), 'ord_5'] = np.nan

        if self.correct_features['ord_4']:
            _x = self.correct_ord_4(_x)
        if self.correct_features['ord_5']:
            _x = self.correct_ord_5(_x)
        if self.correct_features['day']:
            _x = self.correct_day(_x)

        _x = utils.target_encoding(
            _x, self.nominal_features + self.ordinal_features)

        for feature in self.cardinal_features:
            cv = self.cardinal_encoding[feature]['cv']
            n_groups = self.cardinal_encoding[feature]['n_groups']
            min_group_size = self.cardinal_encoding[feature]['min_group_size']

            _x = utils.target_encoding_cv(_x, [feature], cv=cv)

            if n_groups > 0:
                _x = utils.group_features(
                    _x, [feature],
                    n_groups=n_groups,
                    min_group_size=min_group_size)
                _x = utils.target_encoding(_x, [feature])

        na_value = self.get_na_value(x)

        for feature in features:
            if self.handle_missing:
                _x.loc[x[feature].isna(), feature] = na_value
            _x.loc[_x[feature].isna(), feature] = na_value
        logger.info('')

        for feature in features:
            logger.info(
                f'{feature}: {_x[feature].min():.4f} - {_x[feature].max():.4f}')

            if self.log_alpha > 0:
                _x[feature] = np.log(self.log_alpha + _x[feature])

        logger.info('')
        logger.info('amount of unique values')
        for feature in features:
            logger.info(f'{feature}: {_x[feature].nunique()}')

        logger.info('')
        logger.info(f'data size: {_x.shape}')
        logger.info('')

        assert _x.isnull().sum().sum() == 0

        _train, _test = utils.split_data(_x)
        _train_y = _train['target'].values
        _train_x = _train[features].values
        _test_x = _test[features].values
        _test_id = _test['id'].values

        if self.one_hot_encoding:
            encoder = OneHotEncoder(sparse=True)
            encoder.fit(_x[features])
            del _x

            _train_x = encoder.transform(_train[features])
            _test_x = encoder.transform(_test[features])

        return _train_x, _train_y, _test_x, _test_id

    @staticmethod
    def get_na_value(x):
        return x[x['target'] > -1]['target'].mean()

    @staticmethod
    def correct_day(x):
        x.loc[(x['month'] == 2) & (x['day'] == 2), 'day'] = 6
        x.loc[(x['month'] == 4) & (x['day'] == 6), 'day'] = 7
        x.loc[(x['month'] == 5) & (x['day'] == 7), 'day'] = 1
        x.loc[(x['month'] == 10) & (x['day'] == 2), 'day'] = 1
        x.loc[(x['month'] == 10) & (x['day'] == 6), 'day'] = 1
        x.loc[(x['month'] == 10) & (x['day'] == 7), 'day'] = 1
        return x

    @staticmethod
    def correct_ord_4(x):
        x.loc[x['ord_4'] == 'J', 'ord_4'] = 'K'
        x.loc[x['ord_4'] == 'L', 'ord_4'] = 'M'
        x.loc[x['ord_4'] == 'S', 'ord_4'] = 'R'
        return x

    @staticmethod
    def correct_ord_5(x):
        x.loc[x['ord_5'] == 'Z', 'ord_5'] = 'Y'
        x.loc[x['ord_5'] == 'K', 'ord_5'] = 'L'
        x.loc[x['ord_5'] == 'E', 'ord_5'] = 'D'
        return x

    @staticmethod
    def get_features(features):
        return [feature for feature in features
                if feature not in ['id', 'target']]


class OneColClassifier(BaseEstimator):
    def __init__(self, estimator, n_splits=1):
        self.estimators = [estimator] * n_splits
        self.n_splits = n_splits

    def fit(self, x, y=None, early_stopping_rounds=100, verbose=20):
        x, y = self.union_features(x, y)

        if self.n_splits > 1:
            score = 0
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=1)

            for fold, (train_index, valid_index) in enumerate(cv.split(x, y)):
                train_x, train_y = x[train_index], y[train_index]
                valid_x, valid_y = x[valid_index], y[valid_index]

                logger.info('')
                logger.debug(f'started training on fold {fold}')
                logger.info('')

                self.estimators[fold].fit(
                    train_x,
                    train_y,
                    eval_set=[(valid_x, valid_y)],
                    eval_metric='auc',
                    early_stopping_rounds=early_stopping_rounds,
                    categorical_feature=[0],
                    verbose=verbose)

                score += self.estimators[fold].best_score_['valid_0']['auc']

                logger.info('')
                logger.debug(f'score: {score / self.n_splits:.4f}')
                logger.info('')
        else:
            self.estimators[0].fit(
                x, y,
                categorical_feature=['feature'],
                feature_name=['feature', 'value'],
                verbose=verbose)

        return self

    def predict_proba(self, x, valid):
        features = list(x.columns)
        x, _ = self.union_features(x, index=x['id'], return_df=True)

        predicted = np.zeros(len(x))

        for estimator in self.estimators:
            p = estimator.predict_proba(x[['feature', 'value']])
            predicted += p[:, 1] / self.n_splits

        y_pred = self.decompose(x, predicted)

        train = valid.merge(y_pred, how='left', on='id')

        _estimator = LogisticRegression(
            random_state=1,
            solver='lbfgs',
            max_iter=2020,
            fit_intercept=True,
            penalty='none',
            verbose=1)

        y_pred['y_pred'] = _estimator.fit(
            train[['mean', 'percentile_10', 'percentile_90']], train['y_true']).predict_proba(
            train[['mean', 'percentile_10', 'percentile_90']])[:, 1]

        return y_pred

    @staticmethod
    def roc_auc_score(y_true, y_score):
        return 'roc_auc', -roc_auc_score(y_true, y_score), False

    @staticmethod
    def union_features(x, y=None, index=None, return_df=False):
        x_new = pd.DataFrame()

        for i, col in enumerate(x.columns):
            _x = pd.DataFrame()

            if index is not None:
                _x['id'] = x['id']

            _x['value'] = x[col]

            if y is not None:
                _x['target'] = y

            _x['feature'] = i
            x_new = x_new.append(_x)

        x_new['feature'] = x_new['feature'].astype('int')

        if return_df:
            x = x_new[['id', 'feature', 'value']]
        else:
            x = x_new[['feature', 'value']].values

        if y is not None:
            if return_df:
                y = x_new['target']
            else:
                y = x_new['target'].values

        return x, y

    def decompose(self, x, predicted):
        x['target'] = predicted
        y_pred = x.groupby('id')['target'].agg(['mean', self.percentile(10), self.percentile(90)])
        return y_pred.reset_index(level='id')

    @staticmethod
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_


class LgbClassifier(BaseEstimator):
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

    def predict_proba(self, x):
        predicted = np.zeros(len(x))

        for estimator in self.estimators:
            p = estimator.predict_proba(x)
            predicted += p[:, 1] / self.n_splits

        return predicted

    @staticmethod
    def roc_auc_score(y_true, y_score):
        return 'roc_auc', -roc_auc_score(y_true, y_score), False


class Classifier(BaseEstimator):
    def __init__(self, estimator):
        self._estimator = estimator

    def fit(self, x, y=None, **fit_params):
        self._estimator.fit(x, y, **fit_params)
        return self

    def cross_val(self, x, y, n_splits, corr=False):
        scores = []

        if n_splits > 1:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

            for fold, (train_index, valid_index) in enumerate(cv.split(x, y)):
                train_x, train_y = x[train_index], y[train_index]
                valid_x, valid_y = x[valid_index], y[valid_index]

                if corr:
                    corr_index = np.all(valid_x <= np.max(train_x, axis=0), axis=1)
                    valid_x = valid_x[corr_index]
                    valid_y = valid_y[corr_index]

                logger.info('')
                logger.debug(f'started training on fold {fold}')
                logger.info('')

                estimator = self._estimator
                estimator.fit(train_x, train_y)

                y_pred = estimator.predict_proba(valid_x)
                score = roc_auc_score(valid_y, y_pred[:, 1])
                scores += [score]

                logger.info(f'score: {score:.4f}')
        else:
            y_pred = self._estimator.fit(x, y).predict_proba(x)
            scores += [roc_auc_score(y, y_pred[:, 1])]

        logger.info('')
        logger.debug(f'score: {np.mean(scores):.5f}')
        logger.info('')

    def predict_proba(self, x):
        return self._estimator.predict_proba(x)


class Submitter(BaseEstimator):
    def __init__(self, estimator, path_to_data=None):
        self.estimator = estimator
        self.path_to_data = path_to_data
        self.results = pd.DataFrame()

    def fit(self, x, y=None, **fit_params):
        self.estimator.fit(x, y, **fit_params)
        return self

    def predict_proba(self, x, test_id=None):
        self.results['id'] = test_id
        p = self.estimator.predict_proba(x)

        if len(p.shape) == 2:
            self.results['target'] = p[:, 1]
        else:
            self.results['target'] = p

        if self.path_to_data is not None:
            now = datetime.now().strftime('%Y_%m_%d_%H_%M')
            file_name = 'results_' + now + '.csv'
            path_to_file = path.join(self.path_to_data, file_name)
            self.results.to_csv(path_to_file, index=False)

        return self.results


class BayesSearch(BaseEstimator, TransformerMixin):
    def __init__(self, n_trials=10):
        self.n_trials = n_trials

    def fit(self, x, y=None, valid=None):
        def _evaluate(trial):
            n_groups = [17, 18, 19, 20, 21, 22, 23, 24, 25]
            group_size = [1000, 2000, 5000]
            n_splits = [3, 4, 5, 6, 7, 8]

            cardinal_encoding = dict()

            for feature in ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']:
                fid = feature[-1]
                cardinal_encoding[feature] = dict()
                cardinal_encoding[feature]['cv'] = KFold(
                    n_splits=trial.suggest_categorical('cv_' + str(fid), n_splits))
                cardinal_encoding[feature]['n_groups'] = \
                    trial.suggest_categorical('groups_' + str(fid), n_groups)
                cardinal_encoding[feature]['min_group_size'] = \
                    trial.suggest_categorical('size_' + str(fid), group_size)

            min_cat_size = [2, 5, 10, 15, 20, 25, 30, 50]
            correct_features = {'ord_4': True, 'ord_5': True, 'day': True}

            encoder = Encoder(
                cardinal_encoding=cardinal_encoding,
                handle_missing=True,
                min_category_size=trial.suggest_categorical('min_cat_size', min_cat_size),
                log_alpha=0,
                one_hot_encoding=True,
                correct_features=correct_features)

            estimator = LogisticRegression(
                random_state=1,
                C=0.1,
                class_weight='balanced',
                solver='liblinear',
                max_iter=2020,
                fit_intercept=True,
                penalty='l2',
                verbose=1)

#            estimator = OneColClassifier(
#                estimator=lgb.LGBMClassifier(
#                    objective='binary',
#                    metric='auc',
#                    is_unbalance=True,
#                    boost_from_average=False,
#                    n_estimators=trial.suggest_int('n_estimators', 5, 50),
#                    learning_rate=trial.suggest_uniform('learning_rate', 0.005, 0.1),
#                    num_leaves=trial.suggest_int('num_leaves', 3, 500),
#                    min_child_samples=trial.suggest_int('min_child_samples', 1, 10000),
#                    colsample_bytree=1),
#                n_splits=1)

            _train_x, _train_y, _test_x, _test_id = encoder.fit_transform(x)

            predicted = pd.DataFrame()
            predicted['id'] = _test_id
            predicted['y_pred'] = estimator.fit(_train_x, _train_y).predict_proba(_test_x)[:, 1]
            del _train_x, _test_x, _train_y, _test_id

            _valid = valid.merge(predicted, how='left', on='id')
            score = -roc_auc_score(_valid['y_true'].values, _valid['y_pred'].values)
            del _valid, predicted

            self._print_trial(trial, score)

            return score

        def _pack_best_trial(trial):
            pass

#        train_x = x.copy
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

        logger.info('')

        if is_best:
            logger.debug(info)
        else:
            logger.info(info)

        logger.info('')

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
                info += f'{key}: None, '
            else:
                if isinstance(params[key], int):
                    info += f'{key}: {params[key]:.0f}, '
                elif isinstance(params[key], str):
                    info += f'{key}: {params[key]}, '
                elif isinstance(params[key], bool):
                    info += f'{key}: {str(params[key])}, '
                else:
                    info += f'{key}: {params[key]:.3f}, '
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


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, na_value=None):
        self.na_value = na_value
        self.target_mean = {}

    def fit(self, x, y=None):
        features = list(x.columns)
        _x = x.copy()
        _x['target'] = y
        mask = _x['target'] > -1

        for feature in features:
            _x.loc[_x[feature].isna(), feature] = '-1'
            target_mean = _x[mask].groupby(feature)['target'].mean()

            if self.na_value is not None:
                target_mean['-1'] = self.na_value

            self.target_mean[feature] = target_mean

        return self

    def transform(self, x):
        for feature in list(x.columns):
            x.loc[x[feature].isna(), feature] = '-1'

            values = x[feature].unique()
            encoded_values = list(self.target_mean[feature].index)
            unknown_values = [val for val in values if val not in encoded_values]
            x.loc[x[feature].isin(unknown_values), feature] = '-1'

            x[feature] = x[feature].map(self.target_mean[feature].to_dict())
        return x
