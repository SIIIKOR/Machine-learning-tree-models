import pandas as pd
from sklearn.datasets import load_iris, load_boston

from functions import make_test_data, cross_validate
from algorithms import ClassificationTree, RegressionTree, RandomForest

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from time import time

if __name__ == '__main__':
    iris = load_boston()
    train_df = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])
    target_df = pd.DataFrame(data=iris["target"], columns=["target"])

    # train_df, target_df, test_df = make_test_data(train_df, target_df, 140)
    print(train_df.shape)
    print(target_df.shape)
    # print(test_df.shape)
    dataset = pd.concat([train_df, target_df], axis=1)
    s = time()
    rf = RandomForest(dataset, tree_type="regression", min_sample_split=2, max_depth=100)
    # rf.build_trees(100)
    rf.build_forest(100)
    e = time()
    print(e-s, "seconds")
    # rf.predict(test_df)













    # print(train_df)
    # train_df, target_df, test_df = make_test_data(train_df, target_df, 120)
    # print(train_df.shape)
    # print(target_df.shape)
    # print(test_df.shape)
    # t1 = ClassificationTree(min_sample_split=20, max_depth=100)
    # t1.fit(train_df, target_df, mode="vis")
    # # t1.print_tree()
    # # t1.print_tree(train_df, target_df)
    # predictions = t1.predict(test_df)
    # print("predicted:", predictions)
    # print("prediction score:", t1.prediction_score(predictions, test_df))

    # boston = load_boston()
    # train_df = pd.DataFrame(data=boston["data"], columns=boston["feature_names"])
    # target_df = pd.DataFrame(data=boston["target"], columns=["target"])
    # train_df, target_df, test_df = make_test_data(train_df, target_df, 50)
    # print(train_df.shape)
    # print(target_df.shape)
    # print(test_df.shape)
    # t2 = RegressionTree(min_sample_split=20, max_depth=100)
    # t2.fit(train_df, target_df)
    # predictions = np.array(t2.predict(test_df))
    # print(t2.prediction_score(predictions, test_df, mode="rss"))

    # base_scores = []
    # pruned_scores = []
    # for i in range(10):
    #     train_df = pd.DataFrame(data=boston["data"], columns=boston["feature_names"])
    #     target_df = pd.DataFrame(data=boston["target"], columns=["target"])
    #     train_df, target_df, test_df = make_test_data(train_df, target_df, 100)
    #     t2 = RegressionTree(min_sample_split=2)
    #     t2.fit(train_df, target_df, mode="prune")
    #     predictions = np.array(t2.predict(test_df))
    #     base_scores.append(t2.prediction_score(predictions, test_df, mode="rss"))
    #
    #     pruned_tree = t2.prune()
    #     pruned_predictions = pruned_tree.predict(test_df)
    #     pruned_scores.append(pruned_tree.prediction_score(pruned_predictions, test_df, mode="rss"))
    # print("base", base_scores)
    # print("base_avg", sum(base_scores)/len(base_scores))
    # print("prune", pruned_scores)
    # print("prune_avg", sum(pruned_scores)/len(pruned_scores))



    # best_model = cross_validate(RegressionTree, train_df, target_df)
    # print(best_model)
