from collections import Counter
import numpy as np


def make_column_numeric(df, column_index):
    col = df.iloc[:, column_index]
    c = Counter(col)
    mapped = dict(zip(c, range(len(c))))
    new_col = np.array([mapped[val] for val in col])
    df.iloc[:, column_index] = new_col
