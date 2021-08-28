import pandas as pd
from sklearn.datasets import load_iris, load_boston

from functions import make_test_data
from algorithms import ClassificationTree, RegressionTree
def a():
    return 1
if __name__ == '__main__':
    # iris = load_iris()
    # train_df = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])
    # target_df = pd.DataFrame(data=iris["target"], columns=["target"])
    # train_df, target_df, test_df = make_test_data(train_df, target_df, 140)
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
    print(train_df.shape)
    print(target_df.shape)
    t2 = RegressionTree()
    a = t2.get_optimal_split(train_df.to_numpy(), train_df.shape[1]-1, t2.rss_score)
    print(a)
