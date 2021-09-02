from collections import Counter
import numpy as np
import pandas as pd


def make_column_numeric(df, column_index):
    """
    Transforms non-numeric column to numeric.

    :param df: Input dataset
    :param column_index: Index of column which will be transformed.
    :return: Returns nothing.
    """
    col = df.iloc[:, column_index]
    c = Counter(col)
    mapped = dict(zip(c, range(len(c))))
    new_col = np.array([mapped[val] for val in col])
    df.iloc[:, column_index] = new_col


def make_test_data(train_df, target_df, n):
    """
    Randomly generates n-sized test data from train and target dataset.

    :param train_df: Train dataset.
    :param target_df: Target dataset.
    :param n: Size of desired test dataset.
    :return: Returns transformed datasets.
    """
    possible_indexes = np.arange(train_df.shape[0])
    test_index = np.random.choice(possible_indexes, n, replace=False)
    test_df = train_df.loc[test_index]
    test_df.loc[:, ["target"]] = target_df.loc[test_index]
    new_train_df = train_df.drop(labels=test_index, axis=0)
    new_target_df = target_df.drop(labels=test_index, axis=0,)
    return new_train_df, new_target_df, test_df


def cross_validate(model, train_df, target_df, k=10):
    """
    Cross validates data to pick the best model.

    :param model: Model used.
    :param train_df: Training dataset.
    :param target_df: Target dataset.
    :param k: validate data size.
    :return: Returns best model object.
    """
    dataset = pd.concat([train_df, target_df], axis=1)
    sample_amount = dataset.shape[0]
    models = {}
    for i in range(0, sample_amount, k):
        m = model()
        validate_data_indexes = np.arange(i, i+k if i+k <= sample_amount else sample_amount)
        validate_data = dataset.iloc[validate_data_indexes]
        new_dataset = dataset.drop(validate_data_indexes)
        train_df, target_df = new_dataset.iloc[:, :-1], new_dataset.iloc[:, -1]
        m.fit(train_df, target_df)
        predictions = m.predict(validate_data)
        models[i//k] = [m.prediction_score(predictions, validate_data, i//k), m]
    min_rss_key = min(models, key=lambda x: models[x])
    return models[min_rss_key][1]
