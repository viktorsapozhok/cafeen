from collections import defaultdict

from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class Encoder(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value):
        self.fill_value
        self._encoders = defaultdict()

    def fit(self, x, y=None):
        for col in tqdm(x.columns, ascii=True, desc='encoding'):
            self._encoders[col] = LabelEncoder().fit(x[col])
        return self

    def transform(self, x):
        for col in range(x.columns):
            x[col] = self._encoders[col].transform(x[col])
        return x
