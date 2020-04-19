__all__ = [
    'LinearEncoder',
    'TargetEncoder',
    'TargetEncoderCV',
]

import abc

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import statsmodels.api as sm


class BaseEncoder(TransformerMixin, BaseEstimator):
    """Base class for all encoders.

    Args:
        fill_value:
            Value to use to fill the missing values.
        normalize:
            If true, then normalize features by removing the mean
            and scaling to variance.
        min_count:
            Use positive value to replace with NaN all the categories
            with amount of observations less than ``min_count``.

    Attributes:
        encoding:
            Dictionary of various encodings applied to features of
            the dataset. Feature names are used as dictionary keys.
    """

    def __init__(self, fill_value=None, normalize=True, min_count=0):
        self.fill_value = fill_value
        self.normalize = normalize
        self.min_count = min_count
        self.encoding = dict()

    @abc.abstractmethod
    def fit(self, x, y=None):
        raise NotImplementedError()

    def transform(self, x):
        """Apply encoding to features.

        (1) Replace with NaN categories not found in train dataset,
        (2) filter outliers by replacing with NaN categories with
        small amount of observations, (3) apply encodings to features,
        (4) fill missing values using ``fill_value``,
        (5) normalize features.
        """

        df = x.copy()
        # replace unknown categories with nan
        df = self.handle_unknown(df)
        # replace categories with small amount of observations with nan
        if self.min_count > 0:
            df = self.handle_outliers(df)

        for col in df.columns:
            # apply encoding
            df[col] = df[col].map(self.encoding[col].to_dict())
            # replace nans
            if self.fill_value is not None:
                df.loc[df[col].isna(), col] = self.fill_value
            # normalize feature
            if self.normalize:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        return df

    def get_feature_names(self):
        """Return names of the encoded features.
        """

        return list(self.encoding.keys())

    def handle_unknown(self, x):
        """Handle unknown categories.

        Replace with NaN all categories not found in ``train``.
        """

        for col in x.columns:
            mask = ~x[col].isin(self.encoding[col].index)
            x.loc[mask, col] = np.nan
        return x

    def handle_outliers(self, x):
        """Filter outliers.

        Replace with NaN all categories with amount of
        observations less than ``self.min_count``.
        """

        for col in x.columns:
            counts = x.groupby(col).size()
            index = counts[counts < self.min_count].index
            x.loc[x[col].isin(index), col] = np.nan
        return x


class LinearEncoder(BaseEncoder):
    """Modified target encoding.

    It applies target encoding to categories and then fits
    linear regression using categories ordinal position as an independent
    variable and calculated target mean as a dependent variable.
    Each category is then encoded with a linear model prediction.

    Args:
        alpha:
            Value between 0 <= alpha <= 1, used to filter outliers
            before fitting linear regression. Categories with
            amount of observations less than alpha-level quantile
            are not used when fitting the linear model.
        fill_value:
            Value to use to fill the missing values.
        normalize:
            If true, then normalize features by removing the mean
            and scaling to variance.
    """

    def __init__(self, alpha=0, fill_value=None, normalize=True):
        super().__init__(fill_value, normalize)
        self.alpha = alpha

    def fit(self, x, y=None):
        for col in x.columns:
            mask = ~x[col].isna()
            feature = x.loc[mask, col]
            target = y[mask]

            df = pd.concat([feature, target], axis=1)
            encoding = df.groupby(feature.name)[target.name].agg(['mean', 'count'])
            encoding['index'] = range(len(encoding))

            mask = encoding['count'] >= encoding['count'].quantile(self.alpha)
            mean = encoding.loc[mask, 'mean']
            index = sm.add_constant(encoding.loc[mask, 'index'])
            model = sm.OLS(mean, index).fit()

            self.encoding[col] = model.predict(sm.add_constant(encoding['index']))
        return self


class TargetEncoder(BaseEncoder):
    """Target encoding.

    Replace each category with its corresponding probability
    of the target.

    Args:
        fill_value:
            Value to use to fill the missing values.
        normalize:
            If true, then normalize features by removing the mean
            and scaling to variance.
        min_count:
            Use positive value to replace with NaN all the categories
            with amount of observations less than ``min_count``.
        tol:
            Tolerance used to round the encoded values.
    """

    def __init__(self, fill_value=None, normalize=True, min_count=0, tol=None):
        super().__init__(fill_value, normalize, min_count)
        self.tol = tol

    def fit(self, x, y=None):
        df = pd.concat([x, y], axis=1)

        for col in x.columns:
            mask = ~df[col].isna()
            self.encoding[col] = df[mask].groupby(col)['target'].mean()

            if self.tol is not None:
                tol = float(self.tol)
                n_digits = self.tol[::-1].find('.')
                self.encoding[col] = \
                    ((self.encoding[col] / tol).round() * tol).round(n_digits)
        return self


class TargetEncoderCV(TransformerMixin, BaseEstimator):
    def __init__(self, cv, columns=None, fill_value=None, min_count=0,
                 n_groups=None):
        self.cv = cv
        self.columns = columns
        self.fill_value = fill_value
        self.min_count = min_count
        self.n_groups = n_groups

        self.fold_encoders = dict()
        self.test_encoders = dict()

    def fit(self, x, y=None):
        for col in self.columns:
            self.fold_encoders[col] = []

            for train_index, valid_index in self.cv.split(x[col], y):
                encoder = TargetEncoder(normalize=False)
                encoder.fit(x.iloc[train_index][[col]], y.iloc[train_index])
                self.fold_encoders[col] += [encoder]

            self.test_encoders[col] = TargetEncoder(normalize=False)
            self.test_encoders[col].fit(x[[col]], y)
        return self

    def transform(self, x, y=None, is_val=False):
        if self.min_count > 0:
            if is_val:
                x = self.handle_outliers(x)
            else:
                x = self.handle_outliers(x, y)

        train_x, train_y, test_x = self.split_data(x, y)

        for col in self.columns:
            train_col = pd.DataFrame()

            for fold, (train_index, valid_index) in enumerate(
                    self.cv.split(train_x[col], train_y)):
                train_col = train_col.append(
                    self.fold_encoders[col][fold].transform(
                        train_x.iloc[valid_index][[col]]))

            test_col = self.test_encoders[col].transform(test_x[[col]])

            train_x = self.replace_column(train_x, col, train_col)
            test_x = self.replace_column(test_x, col, test_col)
            x = pd.concat([train_x, test_x])

            if self.n_groups is not None:
                mask = ~x[col].isna()
                x.loc[mask, col] = pd.qcut(x.loc[mask, col], self.n_groups,
                                           labels=False, duplicates='drop')

            if self.fill_value is not None:
                x.loc[x[col].isna(), col] = self.fill_value

        return x

    def handle_outliers(self, x, y=None):
        df = x.copy()
        for col in self.columns:
            if y is None:
                counts = x.groupby(col).size()
            else:
                counts = x[~y.isna()].groupby(col).size()
            index = counts[counts < self.min_count].index
            df.loc[df[col].isin(index), col] = np.nan
        return df

    @staticmethod
    def split_data(x, y):
        return x.loc[~y.isna()], y.loc[~y.isna()], x.loc[y.isna()]

    @staticmethod
    def replace_column(x, col, encoded):
        encoded.columns = [col + '_']
        x = x.merge(encoded, how='left', left_index=True, right_index=True)
        x[col] = x[col + '_']
        return x.drop(columns=[col + '_'])
