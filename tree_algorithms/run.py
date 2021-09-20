import pandas as pd
from sklearn.datasets import load_iris, load_boston

from functions import make_test_data, cross_validate, generate_nan
from algorithms import ClassificationTree, RegressionTree, RandomForest

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from time import time

from itertools import product

if __name__ == '__main__':
    iris = load_iris()
    train_df = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])
    target_df = pd.DataFrame(data=iris["target"], columns=["target"])
    # train_df, target_df, test_df = make_test_data(train_df, target_df, 140)
    train_df = generate_nan(train_df.copy(), 10)
    dataset = pd.concat([train_df, target_df], axis=1)

    s = time()
    rf = RandomForest(dataset, tree_type="classification", min_sample_split=2, max_depth=100)
    rf.build_forest(100, fast=True, iter_amount=3)
    e = time()
    print(e-s, "seconds")
    print(rf.accuracy_estimation)
    rf.plot_proximity()
    print(dataset.iloc[69, :])

    my_sample = np.array([[5, 2, 3.5, 1]])
    predictions = rf.predict(my_sample)
    print(predictions)
    print(30*"#")
    print("predicted:", predictions[0].value)
