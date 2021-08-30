from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, split_score=None):
        """
        Decision node.
        if feature value <= threshold:
            go to the left node
        else:
            go to the right node

        :param feature_index: Index of feature which will split data.
        :param threshold: Value by which data will be split.
        :param left: Dataset with column containing values smaller than threshold.
        :param right: Dataset with column containing values greater than threshold.
        :param split_score: Score of the split.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.split_score = split_score

    def __str__(self):
        return f"feature_index: {self.feature_index}\nthreshold: {self.threshold}"


class Leaf(Node):
    def __init__(self, value, size=None):
        """
        Leaf node.
        At the end, every sample lands in one.
        It tells what it have been classified as.

        :param value: Prediction output.
        :param size: Amount of samples.
        """
        super().__init__()
        self.value = value
        self.size = size

    def __str__(self):
        return f"value: {self.value}\nsize: {self.size}"


class ClassificationTree:
    def __init__(self, dataset=None, min_sample_split=2, max_depth=100):
        """
        Classification Tree - ML classification algorithm.

        :param dataset: Pandas dataframe with numeric training and target data(must be the last column).
        :param min_sample_split: Minimal samples required to split dataset.
        :param max_depth: Maximal depth of the tree.
        """
        self.dataset = dataset
        self.root = None
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.eval_func_split = self.gini_index
        self.eval_func_leaf = self.get_most_common_value

    @property
    def features(self):
        """
        Returns column names of the dataset.

        :return: List of names of the columns.
        """
        return self.dataset.columns

    @property
    def numpy_data(self):
        """
        Returns np.array with the data.

        :return: dataset transformed to np.array.
        """
        return self.dataset.to_numpy()

    @staticmethod
    def generate_thresholds(column):
        """
        Generates thresholds which are the mean of every two values.

        :param column: Numeric 1d np.array.
        :return: Yields threshold.
        """
        for i in range(len(column)):
            if i != len(column) - 1:
                yield (column[i] + column[i + 1]) / 2

    @staticmethod
    def get_most_common_value(dataset):
        """
        Returns most frequently occurring value in a column.

        :param dataset: Dataset with merged train and target columns.
        :return: Most frequently occurring value in given dataset target column.
        """
        count = Counter(dataset[:, -1])
        return max(count, key=lambda x: count[x])

    @staticmethod
    def split(dataset, feature_index, threshold):
        """
        Splits the data in two.

        :param dataset: Dataset with merged train and target columns. often smaller size than original.
        :param feature_index: Column by which threshold dataset will be split.
        :param threshold: Numeric value by which dataset will be split.
        :return: dataset_left containing column with values smaller than threshold and
         dataset_right with column containing values bigger than threshold
        """
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def get_optimal_split(self, dataset, feature_range, eval_func_split):
        """
        Returns the best split for the given dataset.

        :param dataset: Dataset with merged train and target columns. often smaller size than original.
        :param feature_range: Range of training features.
        :param eval_func_split: Function used to evaluate the split.
        :return: Dictionary with information about best split: feature_index, threshold,
         left_dataset, right_dataset, split_score, is_final(is it the last split that needs to be done).
        """
        best_score = float("inf")
        best_split = None
        for feature_index in range(feature_range):
            thresholds = self.generate_thresholds(np.unique(sorted(dataset[:, feature_index])))
            for threshold in thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                eval_output = eval_func_split(dataset_left, dataset_right, feature_index)
                score = eval_output["score"]
                if score < best_score:
                    best_score = score
                    best_split = {"feature_index": feature_index,
                                  "threshold": threshold,
                                  "dataset_left": dataset_left,
                                  "dataset_right": dataset_right,
                                  "split_score": best_score,
                                  "is_final": False}
                    if "is_final" in eval_output:
                        best_split["is_final"] = eval_output["is_final"]
                    elif "datasets_rss" in eval_output:
                        best_split["datasets_rss"] = eval_output["datasets_rss"]
                if score == 0:
                    return best_split
        return best_split

    def fit(self, x=None, target=None):
        """
        Builds up the tree.

        :param x: Training dataset.
        :param target: Target dataset.
        :return: Returns nothing.
        """
        def build_tree(dataset, dataset_rss=None, depth=0):
            """
            Recursive function for building tree.

            :param dataset: Dataset with merged train and target columns. often smaller size than original.
            :param dataset_rss: If this is regression tree, this contains rss of the dataset
            :param depth: Current depth of this instance.
            :return: Node object if more splits will be required of Leaf object if no more split will be required.
            """
            sample_amount, feature_amount = dataset.shape
            if sample_amount >= self.min_sample_split and depth <= self.max_depth:
                optimal_split = self.get_optimal_split(dataset, feature_amount - 1, self.eval_func_split)
                dataset_left = optimal_split["dataset_left"]
                dataset_right = optimal_split["dataset_right"]
                feature_index = optimal_split["feature_index"]
                threshold = optimal_split["threshold"]
                split_score = optimal_split["split_score"]
                if isinstance(self, ClassificationTree):
                    is_final = optimal_split["is_final"]
                    if split_score > 0 or (split_score == 0 and not is_final):
                        left_subtree = build_tree(dataset_left, depth=depth + 1)
                        right_subtree = build_tree(dataset_right, depth=depth + 1)
                        return Node(feature_index, threshold, left_subtree, right_subtree, split_score)
                else:
                    dataset_rss = optimal_split["dataset_rss"]
                    if split_score > 0:
                        left_subtree = build_tree(dataset_left, dataset_rss[0], depth=depth + 1)
                        right_subtree = build_tree(dataset_right, dataset_rss[1], depth=depth + 1)
                        return Node(feature_index, threshold, left_subtree, right_subtree, split_score)

            if isinstance(self, ClassificationTree):
                return Leaf(self.get_most_common_value(dataset), len(dataset))
            return Leaf(dataset_rss, len(dataset))

        if not self.dataset:
            self.dataset = x
        self.dataset["target"] = target
        self.root = build_tree(self.dataset.to_numpy())

    @staticmethod
    def gini_index(*datasets):
        """
        Calculates gini index for given datasets and decides whether it was the last split.

        :param datasets: Two datasets with column containing values smaller than soe threshold.
        :return: weighted gini index, information whether this split should be final.
        """
        dataset_left, dataset_right = datasets[0], datasets[1]
        scores = []
        counts = []
        for dataset in [dataset_left, dataset_right]:
            count = Counter(dataset[:, -1])
            counts.append(count)
            score = 0
            for value in count:
                score += (count[value] / len(dataset)) ** 2
            scores.append(1 - score)

        is_final = False
        is_single = True
        for count in counts:
            if len(count) != 1:
                is_single = False
        if is_single:
            if counts[0].keys() == counts[1].keys():
                is_final = True
        weighted_gini = (len(dataset_left) * scores[0] + len(dataset_right) * scores[1]) / (
                    len(dataset_left) + len(dataset_right))
        return {"score": weighted_gini, "is_final": is_final}

    def predict(self, x):
        """
        Function to predict values from given data.

        :param x: Dataset with data to predict.
        :return: Returns predictions.
        """
        def make_prediction(sample, node=None):
            """
            Recursive function for data prediction.

            :param sample: Vector with data to predict.
            :param node: Current node.
            :return: Prediction.
            """
            if node is None:
                node = self.root
            if isinstance(node, Leaf):
                return node.value
            else:
                feature_index = node.feature_index
                threshold = node.threshold
                if sample[feature_index] <= threshold:
                    return make_prediction(sample, node.left)
                else:
                    return make_prediction(sample, node.right)

        test, target = x.iloc[:, :-1].to_numpy(), x.iloc[:, -1].to_numpy()
        predictions = [make_prediction(sample) for sample in test]
        return predictions

    @staticmethod
    def prediction_score(predictions, target):
        """
        Function for calculating Success rate of prediction.

        :param predictions: List with predictions.
        :param target: Real values.
        :return: Success rate of prediction.
        """
        target = target.iloc[:, -1].to_numpy()
        return len([el for el in zip(predictions, target) if el[0] == el[1]])/len(predictions)

    def print_tree(self, indent="-", target_names=None):
        """
        Prints out the tree structure.

        :param indent: Symbol used to make indents.
        :param target_names: List with names of target classes.
        :return: String representation of the tree.
        """
        def _print_tree(node=None, multi=1):
            """
            Recursive function for creating string representation of the tree.

            :param node: Current worked on node.
            :param multi: Amount of indents required to represent this node.
            :return: Returns nothing.
            """
            if node is None:
                node = self.root
            if isinstance(node, Leaf):
                print(f"class: {target_names[int(node.value)] if target_names else node.value}, size: {node.size}")
            else:
                print(self.features[node.feature_index], "<=", node.threshold, "?", node.split_score)
                print(f"{multi * indent}left: ", end="")
                _print_tree(node.left, multi=multi + 1, )
                print(f"{multi * indent}right: ", end="")
                _print_tree(node.right, multi=multi + 1)

        return _print_tree()


class RegressionTree(ClassificationTree):
    """
    Regression Tree - Ml algorithm.
    """
    def __init__(self, dataset=None, min_sample_split=2, max_depth=100):
        super().__init__(dataset, min_sample_split, max_depth)
        self.eval_func_split = self.rss_score

    @staticmethod
    def rss_calc(dataset, feature_index):
        column = dataset[:, feature_index]
        mean = column.mean()
        rss = 0
        for value in column:
            rss += (value - mean)**2
        return rss

    def rss_score(self, dataset_left, dataset_right, feature_index):
        datasets_rss = [self.rss_calc(dataset, feature_index) for dataset in [dataset_left, dataset_right]]
        return {"score": sum(datasets_rss), "datasets_rss": datasets_rss}

    @staticmethod
    def prediction_score(predictions, target):
        vis_df = pd.DataFrame()
        vis_df["predicted"] = predictions
        vis_df["actual"] = target
        sns.scatterplot(data=vis_df)
        plt.show()
        rss_score_final = sum((vis_df["actual"] - vis_df["predicted"]) ** 2) / vis_df.shape[0]
        return rss_score_final
