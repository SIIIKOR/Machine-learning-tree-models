import pandas as pd
from sklearn.datasets import load_iris

from functions import make_test_data
from clasification_tree import ClassificationTree

if __name__ == '__main__':
    iris = load_iris()
    train_df = pd.DataFrame(data=iris["data"], columns=iris["feature_names"])
    target_df = pd.DataFrame(data=iris["target"], columns=["target"])
    train_df, target_df, test_df = make_test_data(train_df, target_df, 140)
    print(train_df.shape)
    print(target_df.shape)
    print(test_df.shape)
    t1 = ClassificationTree()
    t1.fit(train_df, target_df)
    predictions = t1.predict(test_df)
    print("predicted:", predictions)
    print("prediction score:", t1.prediction_score(predictions, test_df))
