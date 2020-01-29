import numpy as np


def group_feature(train, test, feature, n_groups=10):
    nobs = len(train) / n_groups
    new_feature = feature + '_' + str(n_groups)

    grouped = \
        train.groupby(feature)['target'].agg(['mean', 'count']).sort_values(
            by='mean', ascending=False)

    grouped[new_feature] = np.floor(grouped['count'].cumsum() / nobs)
    groups = grouped[new_feature].unique()
    grouped.loc[grouped[new_feature] == groups[-1], new_feature] = groups[-2]
    grouped.reset_index(level=feature, inplace=True)

    train_columns = [
        col for col in train.columns if col not in [new_feature]]
    test_columns = [
        col for col in test.columns if col not in [new_feature]]

    train = train[train_columns].merge(
        grouped[[feature, new_feature]], how='left', on=feature)
    test = test[test_columns].merge(
        grouped[[feature, new_feature]], how='left', on=feature)

    train[new_feature] = train[new_feature].astype('int')
    test[new_feature] = test[new_feature].astype('int')

    return train, test


def group_feature_train(train, feature, n_groups=10):
    nobs = len(train) / n_groups
    new_feature = feature + '_' + str(n_groups)

    grouped = \
        train.groupby(feature)['target'].agg(['mean', 'count']).sort_values(
            by='mean', ascending=False)

    grouped[new_feature] = np.floor(grouped['count'].cumsum() / nobs)
    groups = grouped[new_feature].unique()
    grouped.loc[grouped[new_feature] == groups[-1], new_feature] = groups[-2]
    grouped.reset_index(level=feature, inplace=True)

    train_columns = [
        col for col in train.columns if col not in [new_feature]]

    train = train[train_columns].merge(
        grouped[[feature, new_feature]], how='left', on=feature)

    train[new_feature] = train[new_feature].astype('int')

    return train


def add_woe_feature(train, test, feature, verbose=True):
    n_events = train['target'].sum()
    n_non_events = len(train) - n_events

    bins = train.groupby(feature)['target'].agg(['sum', 'count'])
    bins['n_non_events'] = bins['count'] - bins['sum']
    bins['p_event'] = bins['sum'] / n_events
    bins['p_non_event'] = bins['n_non_events'] / n_non_events
    bins['woe'] = np.log(bins['p_event'] / bins['p_non_event'])

    train[feature + '_woe'] = train[feature].map(bins['woe'].to_dict())
    test[feature + '_woe'] = test[feature].map(bins['woe'].to_dict())

    if verbose:
        iv = ((bins['p_event'] - bins['p_non_event']) * bins['woe']).sum()
        print(f'{feature}: IV {iv:.2f}')

    return train, test


def add_woe_max(train, features):
    train['min_woe'] = train[features].min(axis=1)
    train['max_woe'] = train[features].max(axis=1)
    train['woe'] = train['max_woe']
    mask = train['min_woe'].abs() > train['max_woe'].abs()
    train.loc[mask, 'woe'] = train.loc[mask, 'min_woe']
    return train
