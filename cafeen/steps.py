from datetime import datetime
import gc
import logging
from os import path
import random
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import eli5
from eli5.sklearn import PermutationImportance
import numpy as np
import optuna
import pandas as pd
from scipy import sparse
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
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
            ordinal_features=None,
            cardinal_encoding=None,
            handle_missing=True,
            log_alpha=0.1,
            one_hot_encoding=True,
            correct_features=None,
            verbose=True
    ):
        self.cardinal_encoding = cardinal_encoding
        self.handle_missing = handle_missing
        self.correct_features = correct_features
        self.log_alpha = log_alpha
        self.one_hot_encoding = one_hot_encoding
        self.verbose = verbose

        if ordinal_features is None:
            self.ordinal_features = []
        else:
            self.ordinal_features = ordinal_features

        self.nominal_features = None
        self.cardinal_features = None

    def fit(self, x, y=None):
        features = self.get_features(x.columns)

        self.cardinal_features = list(self.cardinal_encoding.keys())
        self.nominal_features = [f for f in features
                                 if f not in self.cardinal_features + self.ordinal_features]

        return self

    def transform(self, x):
        _x = x.copy()
#        _x = self.augment_train(_x)
        na_value = self.get_na_value(x)
        features = self.get_features(_x.columns)

        _x['ord_5'] = x['ord_5'].str[0]
        _x.loc[x['ord_5'].isna(), 'ord_5'] = np.nan

        if self.correct_features['ord_4']:
            _x = self.correct_ord_4(_x)
        if self.correct_features['ord_5']:
            _x = self.correct_ord_5(_x)
        if self.correct_features['day']:
            _x = self.correct_day(_x)
        if self.correct_features['nom_7']:
            _x = self.correct_nom_7(_x)

        _x = self.target_encoding(_x, self.nominal_features, na_value=na_value)

        ordinal_features = [f for f in features
                            if f in self.ordinal_features and f not in self.cardinal_features]
        _x = self.encode_ordinal(_x, ordinal_features, na_value)

#        _x = self.target_encoding(_x, ['ord_5'], na_value=na_value)
#        _x = self.group_features(_x, ['ord_5'], n_groups=26, min_group_size=None)

        for feature in self.cardinal_features:
            cv = self.cardinal_encoding[feature]['cv']
            n_groups = self.cardinal_encoding[feature]['n_groups']

            _x = self.filter_feature(_x, feature, self.cardinal_encoding[feature]['filter'])
            _x = self.target_encoding_cv(_x, [feature], cv, na_value=na_value)

            if n_groups > 0:
                _x = self.group_features(_x, [feature], n_groups=n_groups)

        _x = self.target_encoding(_x, self.cardinal_features, na_value=na_value)

        if self.verbose:
            logger.info(f'na_value: {na_value:.5f}')
            logger.info('')

            for feature in features:
                try:
                    logger.info(
                        f'{feature}: {_x[feature].min():.4f} - {_x[feature].max():.4f}')
                except (TypeError, ValueError):
                    continue

        for feature in features:
            if self.handle_missing:
#                _x.loc[x[feature].isna(), feature] = -1
                if feature not in self.ordinal_features:
                    _x.loc[x[feature].isna(), feature] = na_value
#            _x.loc[_x[feature].isna(), feature] = na_value
            _x.loc[_x[feature].isna(), feature] = -1

            if self.log_alpha > 0:
                _x[feature] = np.log(self.log_alpha + _x[feature])

        if self.verbose:
            logger.info('')
            logger.info('amount of unique values')
            for feature in features:
                logger.info(f'{feature}: {_x[feature].nunique()}')

        assert _x[features].isnull().sum().sum() == 0

        _train, _test = utils.split_data(_x)
#        _train_y = _train['target'].values
#        _train_x = _train[features].values
#        _test_x = _test[features].values
#        _test_id = _test['id'].values

        _train_y = _train['target']
        _train_x = _train[features]
        _test_x = _test[features]
        _test_id = _test['id']

        if self.one_hot_encoding:
            ohe_features = [f for f in features if f not in self.ordinal_features] + \
                           [f for f in self.ordinal_features if _x[f].nunique() > 100]
            ordinal_features = [f for f in features if f not in ohe_features]

            encoder = OneHotEncoder(sparse=True)
            encoder.fit(_x[ohe_features])
            del _x
            gc.collect()

            _train_x = encoder.transform(_train[ohe_features])
            _train_x = sparse.hstack((_train_x, _train[ordinal_features].values)).tocsr()
            _test_x = encoder.transform(_test[ohe_features])
            _test_x = sparse.hstack((_test_x, _test[ordinal_features].values)).tocsr()

        if self.verbose:
            logger.info('')
            logger.info(f'train: {_train_x.shape}, test: {_test_x.shape}')
            logger.info('')

        return _train_x, _train_y, _test_x, _test_id

    def augment_train(self, x):
        features = self.get_features(x.columns)

        train, test = utils.split_data(x)
        fair_count = self.get_fair_na_count(len(train))

        if self.verbose:
            n_obs = (train[features].isna().sum() - fair_count).abs().sum()
            logger.info(f'conflicts before: {n_obs}')

        for feature in features:
            na_count = train[feature].isna().sum()

            index = random.sample(
                list(train[train[feature].isna()].index),
                int(np.abs(na_count - fair_count)))

            if na_count > fair_count:
                train = train.drop(index).reset_index(drop=True)
            elif na_count < fair_count:
                train = train.append(train.iloc[index]).reset_index(drop=True)

        if self.verbose:
            na_counts = train[features].isna().sum()
            n_obs = (na_counts - fair_count).abs().sum()
            logger.info(f'conflicts after: {n_obs}')

        x = pd.concat([train, test])

        return x

    @staticmethod
    def get_na_value(x):
        return x[x['target'] > -1]['target'].mean()

    @staticmethod
    def get_fair_na_count(n_obs):
        return 0.03 * n_obs

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
    def correct_nom_7(x):
        x.loc[x['nom_7'] == 'b39008216', 'nom_7'] = '230229e51'
        return x

    @staticmethod
    def get_features(features):
        return [feature for feature in features
                if feature not in ['id', 'target']]

    @staticmethod
    def encode_ordinal(x, features, na_value):
        train = x[x['target'] > -1].reset_index(drop=True)

        for feature in features:
            x.loc[x[feature].isna(), feature] = -1
            train.loc[train[feature].isna(), feature] = -1

            if feature in ['ord_0', 'ord_1', 'ord_2']:
                p_min = train.groupby(feature)['target'].mean().min()
                p_max = train.groupby(feature)['target'].mean().max()
                encoding = train.groupby(feature)['target'].mean()
                encoding.iloc[0] = na_value

                encoding = (encoding - p_min) / (p_max - p_min)

                x[feature] = x[feature].map(encoding.to_dict())
            else:
                encoding = train.groupby(feature)['target'].agg(['mean', 'count'])
                encoding['count'] = list(range(len(encoding)))

                nan_pos = (na_value - encoding['mean'].iloc[1]) / \
                          (encoding['mean'].iloc[-1] - encoding['mean'].iloc[1])

                p_min = encoding['count'].iloc[1]
                p_max = encoding['count'].iloc[-1]
                encoding.loc[encoding.index == -1, 'count'] = p_min + nan_pos * (p_max - p_min)
                encoding['count'] = (encoding['count'] - p_min) / (p_max - p_min)

                x[feature] = x[feature].map(encoding['count'].to_dict())

        return x

    @staticmethod
    def target_encoding(x, features, na_value=None):
        mask = x['target'] > -1

        for feature in features:
            x.loc[x[feature].isna(), feature] = '-1'
            target_mean = x[mask].groupby(feature)['target'].mean()

            if na_value is not None:
                target_mean['-1'] = na_value

            x[feature] = x[feature].map(target_mean.to_dict())

        return x

    def target_encoding_cv(self, x, features, cv, n_rounds=1, na_value=None):
        train, test = utils.split_data(x)
        del x

        train.sort_index(inplace=True)
        encoded = []

        for _iter in range(n_rounds):
            if self.verbose:
                logger.debug(f'iteration {_iter + 1}')

            _encoded = pd.DataFrame()

            for fold, (train_index, valid_index) in enumerate(
                    cv.split(train[features], train['target'])):
                if self.verbose:
                    logger.info(f'target encoding on fold {fold + 1}')

                encoder = TargetEncoder(na_value=na_value)

                encoder.fit(train.iloc[train_index][features],
                            train.iloc[train_index]['target'])

                _encoded = _encoded.append(
                    encoder.transform(train.iloc[valid_index][features]),
                    ignore_index=False)

            encoded += [_encoded.sort_index()]

        encoder = TargetEncoder(na_value=na_value)
        encoder.fit(train[features], train['target'])

        _test = test.copy()
        _test[features] = encoder.transform(test[features].copy())

        _train = train.copy()
        _train[features] = pd.concat(encoded).groupby(level=0).mean()

        x = pd.concat([_train[test.columns], _test])

        return x

    @staticmethod
    def group_features(x, features, n_groups, min_group_size=None):
        for feature in features:
            if min_group_size is not None:
                groups = x.groupby(feature)['target'].count()
                group_index = list(groups[groups >= min_group_size].index)
                grouped = ~x[feature].isin(group_index)

                x.loc[grouped, feature] = pd.qcut(
                    x.loc[grouped, feature],
                    n_groups,
                    labels=False,
                    duplicates='drop')
            else:
                x[feature] = pd.qcut(
                    x[feature], n_groups, labels=False, duplicates='drop')

        return x

    @staticmethod
    def filter_feature( x, feature, filter_):
        mask = x['target'] > -1

        stat = x[mask].groupby(feature)['target'].agg(['count', 'mean'])

        na_filter = list(stat[stat['count'] < filter_[0]].index)
        mask = x[feature].isin(na_filter)
        x.loc[mask, feature] = 'np.nan'
        n_na = mask.sum()

        filtered = list(stat[(stat['count'] >= filter_[0]) &
                             (stat['count'] < filter_[1]) &
                             (stat['mean'] < filter_[2])].index)
        mask = x[feature].isin(filtered)
        x.loc[mask, feature] = 'low'
        n_low = mask.sum()

        filtered = list(stat[(stat['count'] >= filter_[0]) &
                             (stat['count'] < filter_[1]) &
                             (stat['mean'] > filter_[2])].index)
        mask = x[feature].isin(filtered)
        x.loc[mask, feature] = 'high'
        n_high = mask.sum()

        logger.info(f'{feature}: {n_na} na, {n_low} low, {n_high} high')

        return x


class BayesSearch(BaseEstimator, TransformerMixin):
    def __init__(self, n_trials=10, n_folds=3, verbose=True):
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.verbose = verbose

    def fit(self, x, y=None):
        def _evaluate(trial):
            ordinal_features = ['ord_4', 'ord_5']

#            for i in range(5, 10):
#                as_ordinal = trial.suggest_categorical('nom_' + str(i), [False, False])
#                if as_ordinal:
#                    ordinal_features += ['nom_' + str(i)]

            n_splits = [3, 4, 5]

            groups = {
                'nom_5': [11, 12],
                'nom_6': [50, 51],
                'nom_9': [27, 28],
            }

            filters = {
                'nom_5': [
                    trial.suggest_int('nom_5_na_count', 5, 10),
                    trial.suggest_int('nom_5_min_count', 50, 60),
                    trial.suggest_uniform('nom_5_min_avg', 0.125, 0.135),
                    trial.suggest_uniform('nom_5_max_avg', 0.275, 0.285)
                ],
                'nom_6': [
                    trial.suggest_int('nom_6_na_count', 8, 13),
                    trial.suggest_int('nom_6_min_count', 27, 37),
                    trial.suggest_uniform('nom_6_min_avg', 0.08, 0.09),
                    trial.suggest_uniform('nom_6_max_avg', 0.28, 0.29)
                ],
                'nom_9': [
                    trial.suggest_int('nom_9_na_count', 13, 18),
                    trial.suggest_int('nom_9_min_count', 21, 31),
                    trial.suggest_uniform('nom_9_min_avg', 0.127, 0.137),
                    trial.suggest_uniform('nom_9_max_avg', 0.23, 0.33)
                ]
            }

            cardinal_encoding = dict()

            for feature in ['nom_5', 'nom_6', 'nom_9']:
                fid = feature[-1]
                cardinal_encoding[feature] = dict()
                cardinal_encoding[feature]['cv'] = StratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=2020)
                cardinal_encoding[feature]['n_groups'] = \
                    trial.suggest_int('groups_' + str(fid), groups[feature][0], groups[feature][1])
                cardinal_encoding[feature]['filter'] = filters[feature]

            correct_features = {
                'ord_4': True,
                'ord_5': False,
                'day': True,
                'nom_7': True
            }

            encoder = Encoder(
                ordinal_features=ordinal_features,
                cardinal_encoding=cardinal_encoding,
                handle_missing=True,
                log_alpha=0,
                one_hot_encoding=True,
                correct_features=correct_features,
                verbose=self.verbose)

            estimator = LogisticRegression(
                random_state=2020,
                C=trial.suggest_uniform('C', 0.052, 0.054),
                class_weight={0: 1, 1: trial.suggest_int('class_1', 1, 10)},
                solver='liblinear',
                max_iter=2020,
                fit_intercept=True,
                penalty='l2',
                verbose=0)

#            estimator = NaiveBayes(na_value=-1, correct_features=['ord_3', 'ord_4', 'month', 'ord_0'])

            scores = []

            for fold in range(self.n_folds):
                if self.verbose:
                    logger.info('')
                    logger.debug(f'fold {fold} started')
                    logger.info('')

                train, valid, _, _ = train_test_split(
                    x, x['target'],
                    test_size=0.5,
                    shuffle=True,
                    random_state=fold,
                    stratify=x['target'])

                _valid = valid.copy()
                _valid['target'] = -1
                train = train.append(_valid).reset_index(drop=True)
                del _valid

                _train_x, _train_y, _test_x, _test_id = encoder.fit_transform(train)

                predicted = pd.DataFrame()
#                predicted['id'] = _test_id
                predicted['id'] = _test_id.values
                predicted['y_pred'] = estimator.fit(_train_x, _train_y).predict_proba(_test_x)[:, 1]
                del _train_x, _test_x, _train_y, _test_id

                _valid = valid.merge(predicted, how='left', on='id')
                score = -roc_auc_score(_valid['target'].values, _valid['y_pred'].values)
                scores += [score]
                del _valid, predicted
                gc.collect()

                if self.verbose:
                    logger.debug(f'score: {score:.5f}')

            logger.info('')
            logger.debug(f"scores: {', '.join([str(np.round(score, 5)) for score in scores])}, avg: {np.mean(scores):.6f}")

            self._print_trial(trial, np.mean(scores))

            return np.mean(scores)

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


class NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, na_value=-1, correct_features=None):
        self.na_value = na_value

        if correct_features is None:
            self.correct_features = []
        else:
            self.correct_features = correct_features

        self.class_count_ = np.zeros(2)
        self.class_prior_ = np.zeros(2)
        self.posterior_ = dict()

    def fit(self, x, y=None):
        self.class_count_[0] = (y == 0).sum()
        self.class_count_[1] = (y == 1).sum()
        self.class_prior_ = self.class_count_ / np.sum(self.class_count_)

        for fid, feature in enumerate(x.columns):
            counts_0 = x[y == 0].groupby(feature).size()
            counts_1 = x[y == 1].groupby(feature).size()

            for category, category_count in counts_0.iteritems():
                if category == self.na_value:
                    self.posterior_[fid, category, 0] = self.na_prob()
                else:
                    self.posterior_[fid, category, 0] = category_count / self.class_count_[0]

            for category, category_count in counts_1.iteritems():
                if category == self.na_value:
                    self.posterior_[fid, category, 1] = self.na_prob()
                else:
                    self.posterior_[fid, category, 1] = category_count / self.class_count_[1]

        if 'ord_3' in self.correct_features:
            fid = x.columns.get_loc('ord_3')
            counts = x.groupby('ord_3').size()

            for class_ in range(2):
                self.posterior_[fid, 'g', class_] = self.mid_prob(
                    counts['g'], counts['f'], counts['h'], self.posterior_[fid, 'f', class_], self.posterior_[fid, 'h', class_])
                self.posterior_[fid, 'j', class_] = self.mid_prob(
                    counts['j'], counts['i'], counts['k'], self.posterior_[fid, 'i', class_], self.posterior_[fid, 'k', class_])

        if 'ord_4' in self.correct_features:
            fid = x.columns.get_loc('ord_4')
            counts = x.groupby('ord_4').size()

            for class_ in range(2):
                self.posterior_[fid, 'G', class_] = self.mid_prob(
                    counts['G'], counts['F'], counts['H'], self.posterior_[fid, 'F', class_], self.posterior_[fid, 'H', class_])
                self.posterior_[fid, 'J', class_] = self.mid_prob(
                    counts['J'], counts['I'], counts['K'], self.posterior_[fid, 'I', class_], self.posterior_[fid, 'K', class_])
                self.posterior_[fid, 'L', class_] = self.mid_prob(
                    counts['L'], counts['K'], counts['M'], self.posterior_[fid, 'K', class_], self.posterior_[fid, 'M', class_])
                self.posterior_[fid, 'S', class_] = self.mid_prob(
                    counts['S'], counts['R'], counts['T'], self.posterior_[fid, 'R', class_], self.posterior_[fid, 'T', class_])

        if 'month' in self.correct_features:
            fid = x.columns.get_loc('month')
            counts = x.groupby('month').size()

            for class_ in range(2):
                self.posterior_[fid, 10, class_] = self.mid_prob(
                    counts[10], counts[9], counts[11], self.posterior_[fid, 9, class_], self.posterior_[fid, 11, class_])

        if 'ord_0' in self.correct_features:
            fid = x.columns.get_loc('ord_0')
            counts = x.groupby('ord_0').size()

            for class_ in range(2):
                self.posterior_[fid, 2, class_] = self.mid_prob(
                    counts[2], counts[1], counts[3], self.posterior_[fid, 1, class_], self.posterior_[fid, 3, class_])

        return self

    def predict_proba(self, x):
        y = np.zeros((len(x), 2))

        for i, features in enumerate(x.values):
            for k in range(2):
                p = 1

                for j, val in enumerate(features):
                    try:
                        p *= self.posterior_[j, val, k]
                    except KeyError:
                        p *= self.posterior_[j, self.na_value, k]

                y[i, k] = self.class_prior_[k] * p

            y_sum = y[i, 0] + y[i, 1]
            y[i, 0] /= y_sum
            y[i, 1] /= y_sum

        return y

    @staticmethod
    def na_prob():
        return 0.03

    @staticmethod
    def mid_prob(p_mid, p_1, p_2, p_1_y, p_2_y):
        return 0.5 * p_mid * ((p_1_y / p_1) + (p_2_y / p_2))


class OneColClassifier(BaseEstimator):
    def __init__(self, estimator, n_splits=1):
        self.estimators = [estimator] * n_splits
        self.n_splits = n_splits

    def fit(self, x, y=None, early_stopping_rounds=100, verbose=20):
        x, y = self.union_features(x, y)

        if self.n_splits > 1:
            score = 0
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=2020)

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
            random_state=2020,
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
            verbose=200):
        score = 0

        if self.n_splits > 1:
            cv = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=2020)

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
        predicted = np.zeros(x.shape[0])

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
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)

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

