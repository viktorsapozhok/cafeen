import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from cafeen import config

logger = logging.getLogger('cafeen')


def encode_files():
    logger.info('reading files')

    train = pd.read_csv(config.path_to_train, nrows=1000)
    test = pd.read_csv(config.path_to_test, nrows=1000)

    le = LabelEncoder()

    for col in tqdm(test.columns, ascii=True):
        le.fit(pd.concat((train[col], test[col])))

        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    train.to_csv(config.path_to_train_enc)
    test.to_csv(config.path_to_test_enc)
