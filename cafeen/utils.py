import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def read_data(n_valid_rows=0, seed=2020):
    train = pd.read_csv(config.path_to_train)

    if n_valid_rows > 0:
        test_size = n_valid_rows / len(train)

        train, test, train_y, test_y = train_test_split(
            train,
            train['target'],
            test_size=test_size,
            shuffle=True,
            random_state=seed,
            stratify=train['target'])

        test['target'] = -1
        train['target'] = train_y.values
        train = train.append(test)

        test['target'] = test_y.values
        test.rename(columns={'target': 'y_true'}, inplace=True)

        return train, test
    else:
        test = pd.read_csv(config.path_to_test)
        test['target'] = -1

        return pd.concat([train, test]), None


def split_data(df):
    return df.loc[df['target'] > -1], df.loc[df['target'] == -1]
