import pandas as pd
from sklearn.datasets import load_iris, load_boston

from functions import make_test_data, cross_validate
from algorithms import ClassificationTree, RegressionTree

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    # iris = load_iris()
    # train_df = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])
    # target_df = pd.DataFrame(data=iris["target"], columns=["target"])
    # train_df, target_df, test_df = make_test_data(train_df, target_df, 10)
    # print(train_df.shape)
    # print(target_df.shape)
    # print(test_df.shape)
    # t1 = ClassificationTree()
    # t1.fit(train_df, target_df)
    # predictions = t1.predict(test_df)
    # print("predicted:", predictions)
    # print("prediction score:", t1.prediction_score(predictions, test_df))

    boston = load_boston()
    train_df = pd.DataFrame(data=boston["data"], columns=boston["feature_names"])
    target_df = pd.DataFrame(data=boston["target"], columns=["target"])
    # train_df, target_df, test_df = make_test_data(train_df, target_df, 20)
    # print(train_df.shape)
    # print(target_df.shape)
    # print(test_df.shape)
    # t2 = RegressionTree()
    # t2.fit(train_df, target_df)
    # predictions = t2.predict(test_df)
    # print(t2.prediction_score(predictions, test_df.iloc[:, -1]))

    best_model = cross_validate(RegressionTree, train_df, target_df)
