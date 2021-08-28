from collections import Counter
import numpy as np


def make_column_numeric(df, column_index):
    col = df.iloc[:, column_index]
    c = Counter(col)
    mapped = dict(zip(c, range(len(c))))
    new_col = np.array([mapped[val] for val in col])
    df.iloc[:, column_index] = new_col


def make_test_data(train_df, target_df, n):
    possible_indexes = np.arange(train_df.shape[0])
    test_index = np.random.choice(possible_indexes, n, replace=False)
    test_df = train_df.loc[test_index]
    test_df.loc[:, ["target"]] = target_df.loc[test_index]
    new_train_df = train_df.drop(labels=test_index, axis=0)
    new_target_df = target_df.drop(labels=test_index, axis=0,)
    return new_train_df, new_target_df, test_df
