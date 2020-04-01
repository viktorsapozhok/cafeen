from datetime import datetime
from os import path
from typing import Any, Dict, List
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm

from . import utils

warnings.simplefilter(action='ignore', category=FutureWarning)


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            linear_features: List[str],
            cardinal_features: List[str],
            feature_params: Dict[str, Dict[str, Any]],
            na_value: float
    ) -> None:
        self.linear_features = linear_features
        self.cardinal_features = cardinal_features
        self.params = feature_params
        self.na_value = na_value

        self.features = None
        self.nominal_features = None

    def fit(self, x, y=None):
        self.features = self.get_features(x)

        self.nominal_features = [
            feature for feature in self.features
            if feature not in self.linear_features + self.cardinal_features
        ]

        return self

    def transform(self, x):
        x = self.target_encoding(x, self.nominal_features)

        for feature in self.linear_features:
            x = self.linear_encoding(
                x, feature, alpha=self.params[feature]['alpha'])

        for feature in self.cardinal_features:
            if 'count_min' in self.params[feature].keys():
                count_min = self.params[feature]['count_min']
                x = self.count_filter(x, feature, count_min=count_min)

            if 'cv' in self.params[feature].keys():
                cv = self.params[feature]['cv']
                x = self.target_encoding_cv(x, feature, cv=cv)

            if 'n_groups' in self.params[feature].keys():
                n_groups = self.params[feature]['n_groups']
                x = self.group_feature(x, feature, n_groups=n_groups)

            if 'std_max' in self.params[feature].keys():
                std_max = self.params['nom_9']['std_max']
                x = self.std_filter(x, feature, std_max=std_max)

            if 'eps' in self.params[feature].keys():
                eps =self.params['nom_9']['eps']
                x = self.binning(x, feature, eps=eps)

        for feature in self.features:
            x.loc[x[feature].isna(), feature] = -1

        return x

    @staticmethod
    def get_features(x):
        return [c for c in x.columns if c not in ['id', 'target']]

    def linear_encoding(self, x, feature, alpha):
        train = x[x['target'] > -1].reset_index(drop=True)

        train.loc[train[feature].isna(), feature] = -1

        encoding = train.groupby(feature)['target'].agg(['mean', 'count'])
        encoding['x'] = range(len(encoding))

        min_count = encoding['count'].quantile(alpha)
        mask = (encoding.index != -1) & (encoding['count'] >= min_count)

        y = encoding.loc[mask, 'mean']
        X = sm.add_constant(encoding.loc[mask, 'x'])

        model = sm.OLS(y, X).fit()

        encoding['mean'] = model.predict(sm.add_constant(encoding['x']))
        encoding.loc[encoding.index == -1, 'mean'] = self.na_value

        return self.encode_feature(x, feature, encoding['mean'])

    def target_encoding(self, x, features):
        train = x[x['target'] > -1].reset_index(drop=True)

        for feature in features:
            train.loc[train[feature].isna(), feature] = -1
            encoding = train.groupby(feature)['target'].mean()
            encoding.loc[encoding.index == -1] = self.na_value
            x = self.encode_feature(x, feature, encoding)

        return x

    def target_encoding_cv(self, x, feature, cv):
        train, test = utils.split_data(x)
        del x

        train = train.sort_index()
        encoded = pd.DataFrame()

        for fold, (train_index, valid_index) in enumerate(
                cv.split(train[feature], train['target'])):
            encoder = TargetEncoder(na_value=self.na_value)

            encoder.fit(train.iloc[train_index][[feature]],
                        train.iloc[train_index]['target'])

            encoded = encoded.append(
                encoder.transform(train.iloc[valid_index][[feature]]),
                ignore_index=False)

        encoder = TargetEncoder(na_value=self.na_value)
        encoder.fit(train[[feature]], train['target'])

        _test = test.copy()
        _test[feature] = encoder.transform(test[[feature]].copy())

        _train = train.copy()
        _train[feature] = encoded.groupby(level=0).mean()

        x = pd.concat([_train[test.columns], _test])
        return x

    @staticmethod
    def group_feature(x, feature, n_groups):
        mask = x[feature] >= 0

        x.loc[mask, feature] = pd.qcut(
            x.loc[mask, feature], n_groups, labels=False, duplicates='drop')

        return x

    @staticmethod
    def count_filter(x, feature, count_min=0):
        if len(x) > 600000:
            mask = x['target'] > -1
            counts = x[mask].groupby(feature).size()
        else:
            counts = x.groupby(feature).size()

        index = counts[counts < count_min].index
        x.loc[x[feature].isin(index), feature] = np.nan

        return x

    @staticmethod
    def std_filter(x, feature, std_max=1, n_folds=3):
        train = x.loc[x['target'] > -1, :]

        for i in range(n_folds):
            cv = StratifiedKFold(
                n_splits=2, shuffle=True, random_state=2020 * i)

            for fold, (train_index, valid_index) in enumerate(
                    cv.split(train[feature], train['target'])):
                enc_1 = train.iloc[
                    train_index].groupby(feature)['target'].mean()
                enc_1 = enc_1.rename('mean_' + str(2 * i + 1))

                enc_2 = train.iloc[
                    valid_index].groupby(feature)['target'].mean()
                enc_2 = enc_2.rename('mean_' + str(2 * i + 2))

                if i == 0:
                    enc = enc_1.to_frame().merge(
                        enc_2.to_frame(),
                        how='outer', left_index=True, right_index=True)
                else:
                    enc = enc.merge(
                        enc_1.to_frame(),
                        how='outer', left_index=True, right_index=True)
                    enc = enc.merge(
                        enc_2.to_frame(),
                        how='outer', left_index=True, right_index=True)
                break

        columns = ['mean_' + str(1 + i) for i in range(n_folds)]
        enc['std'] = enc[columns].std(axis=1)

        index = enc.loc[enc['std'] > std_max].index
        x.loc[x[feature].isin(index), feature] = np.nan

        return x

    @staticmethod
    def binning(x, feature, eps=(0.001, 3)):
        mask = x['target'] > -1

        encoding = x[mask].groupby(feature)['target'].mean()

        if eps is not None:
            encoding = ((encoding / eps[0]).round() * eps[0]).round(eps[1])

        x[feature] = x[feature].map(encoding.to_dict())
        return x

    @staticmethod
    def encode_feature(x, feature, encoding):
        x.loc[x[feature].isna(), feature] = -1
        encoding = (encoding - encoding.min()) / \
                   (encoding.max() - encoding.min())
        x[feature] = x[feature].map(encoding.to_dict())
        x[feature] = (x[feature] - x[feature].mean()) / x[feature].std()
        return x


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


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, na_value=None):
        self.na_value = na_value
        self.encoding = dict()

    def fit(self, x, y=None):
        _x = x.copy()
        _x['target'] = y

        mask = _x['target'] > -1

        for feature in x.columns:
            _x.loc[_x[feature].isna(), feature] = '-1'
            encoding = _x[mask].groupby(feature)['target'].mean()

            if self.na_value is not None:
                encoding['-1'] = self.na_value

            self.encoding[feature] = encoding

        return self

    def transform(self, x):
        for feature in x.columns:
            x.loc[x[feature].isna(), feature] = '-1'
            values = x[feature].unique()
            encoded = self.encoding[feature].index
            unknown = [v for v in values if v not in encoded]
            x.loc[x[feature].isin(unknown), feature] = '-1'
            x[feature] = x[feature].map(self.encoding[feature].to_dict())
        return x
