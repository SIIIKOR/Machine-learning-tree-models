from collections import Counter, deque
from abc import ABC

import numpy as np
import pandas as pd


class Node(ABC):
    pass


class DecisionNode(Node):
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, **kwargs):
        """
        Decision node.

        if feature value <= threshold:
            go to the left node
        else:
            go to the right node

        :param feature_index: Index of feature by which decision will be made.
        :param threshold: Value by which decision will be made.
        :param left: DecisionNode|Leaf.
        :param right: DecisionNode|Leaf.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right


class VisDecisionNode(DecisionNode):
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, **kwargs):
        """
        Decision node that will be used for visualisation.

        :param feature_index: Index of feature by which decision will be made.
        :param threshold: Value by which decision will be made.
        :param left: DecisionNode|Leaf.
        :param right: DecisionNode|Leaf.
        :param split_score: Value indicating quality of the split.
        """
        super().__init__(feature_index, threshold, left, right)
        self.split_score = kwargs["split_score"]

    def __str__(self):
        return f"feature_index: {self.feature_index}\nthreshold: {self.threshold}"


class PruningDecisionNode(DecisionNode):
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, **kwargs):
        """
        Decision Node that will be used during pruning.

        It has additional parameter parent, which is pointing at the parent of this node

        :param feature_index: Index of feature by which decision will be made.
        :param threshold: Value by which decision will be made.
        :param left: DecisionNode|Leaf.
        :param right: DecisionNode|Leaf.
        :param parent: DecisionNode
        """
        super().__init__(feature_index, threshold, left, right)
        self.parent = kwargs["parent"]


class Leaf(Node):
    def __init__(self, value=None, **kwargs):
        """
        Leaf node.

        At the end, every sample lands in one.
        It tells what it have been classified as.

        :param value: Prediction output.
        """
        self.value = value


class VisLeaf(Leaf):
    def __init__(self, value=None, **kwargs):
        """
        Leaf node used for visualisation.

        :param value: Prediction output.
        :param size: Amount of samples.
        """
        super().__init__(value)
        self.size = kwargs["size"]

    def __str__(self):
        return f"value: {self.value}\nsize: {self.size}"


class PruningLeaf(Leaf):
    def __init__(self, value=None, **kwargs):
        """
        Leaf node used for pruning tree.

        It contains score which will be used to replace decision nodes with leaf.

        :param value: Prediction output.
        :param score: Leaf evaluation score.
        """
        super().__init__(value)
        self.score = kwargs["leaf_eval_score"]


class Tree(ABC):
    def __init__(self, dataset=None, min_sample_split=2, max_depth=5):
        """
        Abstract class used to inherit from for classification tree and regression tree.

        :param dataset: Pandas dataframe with numeric training and target data(must be the last column).
        :param min_sample_split: Pre-pruning - Minimal samples required to split dataset.
        :param max_depth: Maximal depth of the tree.
        """
        self.dataset = dataset
        self.root = None
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth

        self.eval_func_split = None
        self.get_leaf_prediction_value = None
        self.eval_func_leaf = None

        self.node_type = DecisionNode
        self.leaf_type = Leaf

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
                                  "split_score": best_score}
                    if "is_final" in eval_output:
                        best_split["is_final"] = eval_output["is_final"]
                if score == 0:
                    return best_split
        return best_split

    def fit(self, x=None, target=None):
        """
        Builds up the tree.

        Iterative algorithm based on bfs.

        :param x: Training dataset.
        :param target: Target dataset.
        :return: Returns nothing.
        """

        def build_tree_iterative(dataset):
            # left_child, current dataset, parent
            queue = deque([(False, dataset, None)])
            depth = 0
            while queue:
                # for every node on current level
                # print(30*"#")
                for _ in range(len(queue)):
                    # if depth == 3:
                    #     queue = False
                    is_left, current_dataset, current_parent = queue.popleft()
                    # print(current_dataset.shape)
                    sample_amount, feature_amount = current_dataset.shape
                    # if we can we proceed to split the data
                    if sample_amount >= self.min_sample_split and depth <= self.max_depth:
                        # splitting
                        optimal_split = self.get_optimal_split(current_dataset, feature_amount - 1,
                                                               self.eval_func_split)
                        # unpacking the data
                        dataset_left = optimal_split["dataset_left"]
                        dataset_right = optimal_split["dataset_right"]
                        # print("left", dataset_left.shape)
                        # print("right", dataset_right.shape)
                        feature_index = optimal_split["feature_index"]
                        threshold = optimal_split["threshold"]
                        split_score = optimal_split["split_score"]
                        # if this is classification tree, we have different condition
                        # for creating leaf rather than decision node
                        if isinstance(self, ClassificationTree):
                            is_final = optimal_split["is_final"]
                            condition = split_score > 0 or (split_score == 0 and not is_final)
                        else:
                            condition = True

                        if condition:  # decision node
                            current_node = self.node_type(feature_index, threshold,
                                                          split_score=split_score,
                                                          parent=current_parent)
                        else:  # leaf, only used in classification tree if node is pure
                            current_node = self.leaf_type(self.get_leaf_prediction_value(current_dataset),
                                                          leaf_eval_score=self.eval_func_leaf(current_dataset),
                                                          size=len(current_dataset))
                        # if this is not a leaf we proceed to split the data
                        if not isinstance(current_node, Leaf):
                            queue.append((True, dataset_left, current_node))
                            queue.append((False, dataset_right, current_node))
                    else:  # if we are not allowed to split the data, we create leaf
                        current_node = self.leaf_type(self.get_leaf_prediction_value(current_dataset),
                                                      leaf_eval_score=self.eval_func_leaf(current_dataset),
                                                      size=len(current_dataset))
                    # if this is not root
                    if not (not is_left and current_parent is None):
                        if is_left:  # we assign newly created node to be left child of parent
                            current_parent.left = current_node
                        else:
                            current_parent.right = current_node
                    else:
                        self.root = current_node
                depth += 1  # after iterating over entire level we add 1 to the depth
            return self.root

        def build_tree_recursive(dataset, depth=0):
            """
            Recursive function for building tree.

            :param dataset: Dataset with merged train and target columns. often smaller size than original.
            :param depth: Current depth of this instance.
            :return: DecisionNode if more splits will be required of Leaf object if no more split will be required.
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
                        left_subtree = build_tree_recursive(dataset_left, depth=depth + 1)
                        right_subtree = build_tree_recursive(dataset_right, depth=depth + 1)
                        return self.node_type(feature_index, threshold, left_subtree, right_subtree,
                                              split_score=split_score)
                else:
                    left_subtree = build_tree_recursive(dataset_left, depth=depth + 1)
                    right_subtree = build_tree_recursive(dataset_right, depth=depth + 1)
                    return self.node_type(feature_index, threshold, left_subtree, right_subtree,
                                          split_score=split_score)

            return self.leaf_type(self.get_leaf_prediction_value(dataset),
                                  leaf_eval_score=self.eval_func_leaf(dataset), size=len(dataset))

        if not self.dataset:
            self.dataset = x
        self.dataset["target"] = target
        self.root = build_tree_iterative(self.dataset.to_numpy())

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

    def print_tree(self, x=None, target=None, indent="-", target_names=None):
        """
        Prints out the tree structure.
        Have to create tree with specific Nodes to see size and split_score.

        :param x: Train data for fit function.
        :param target: Target data for fit function.
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
            if isinstance(node, VisLeaf):
                print(f"class: {target_names[int(node.value)] if target_names else node.value}, size: {node.size}")
            if isinstance(node, Leaf):
                print(f"class: {target_names[int(node.value)] if target_names else node.value}")
            else:
                if isinstance(node, VisDecisionNode):
                    print(self.features[node.feature_index], "<=", node.threshold, "?", node.split_score)
                else:
                    print(self.features[node.feature_index], "<=", node.threshold)
                print(f"{multi * indent}left: ", end="")
                _print_tree(node.left, multi=multi + 1, )
                print(f"{multi * indent}right: ", end="")
                _print_tree(node.right, multi=multi + 1)

        if x is not None and target is not None:
            self.node_type = VisDecisionNode
            self.leaf_type = VisLeaf
            self.fit(x, target)
        return _print_tree()


class ClassificationTree(Tree):
    def __init__(self, dataset=None, min_sample_split=2, max_depth=100):
        """
        Classification Tree - ML classification algorithm.

        :param dataset: Pandas dataframe with numeric training and target data(must be the last column).
        :param min_sample_split: Minimal samples required to split dataset.
        :param max_depth: Maximal depth of the tree.
        """
        super().__init__(dataset, min_sample_split, max_depth)
        self.eval_func_split = self.gini_index
        self.get_leaf_prediction_value = self.get_most_common_value
        self.eval_func_leaf = self.get_amount_of_most_common_value

    @staticmethod
    def get_most_common_value(dataset):
        """
        Function that sets prediction value for a leaf.

        :param dataset: Dataset with merged train and target columns.
        :return: Most frequently occurring value in given dataset target column.

        """
        count = Counter(dataset[:, -1])
        return max(count, key=lambda x: count[x])

    @staticmethod
    def get_amount_of_most_common_value(dataset):
        """
        Function used for leaf evaluation for potential pruning.

        :param dataset: Dataset with merged train and target columns.
        :return: Amount of most commonly occurring value in target column.
        """
        count = Counter(dataset[:, -1])
        return count[max(count, key=lambda x: count[x])]

    @staticmethod
    def gini_index(*datasets):
        """
        Calculates gini index for given datasets and decides whether it was the last split.

        :param datasets: Two datasets with column containing values smaller than some threshold.
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


class RegressionTree(Tree):
    def __init__(self, dataset=None, min_sample_split=2, max_depth=100):
        """
        Regression Tree - Ml algorithm.

        :param dataset: Pandas dataframe with numeric training and target data(must be the last column).
        :param min_sample_split: Minimal samples required to split dataset.
        :param max_depth: Maximal depth of the tree.
        """
        super().__init__(dataset, min_sample_split, max_depth)

        self.eval_func_split = self.rss_score
        self.get_leaf_prediction_value = self.get_avg_value
        self.eval_func_leaf = self.rss_calc

    @staticmethod
    def rss_calc(dataset, feature_index=-1):
        """
        Calculates rss of column in dataset.

        :param dataset: Dataset with training data.
        :param feature_index: Index of column to calculate.
        :return: Float rss score.
        """
        column = dataset[:, feature_index]
        mean = column.mean()
        rss = 0
        for value in column:
            rss += (value - mean)**2
        return rss

    def rss_score(self, dataset_left, dataset_right, feature_index):
        """
        Calculates sum of rss for datasets

        :param dataset_left: Dataset with column smaller than some threshold.
        :param dataset_right: Dataset with column grater than some threshold.
        :param feature_index: Index of feature by which rss will be calculated.
        :return: Returns rss of datasets.
        """
        datasets_rss = [self.rss_calc(dataset, feature_index) for dataset in [dataset_left, dataset_right]]
        return {"score": sum(datasets_rss), "datasets_rss": datasets_rss}

    @staticmethod
    def prediction_score(predictions, target):
        """
        Calculates mse of prediction.

        :param predictions: np.array of model predictions.
        :param target: Dataframe with target data.
        :return: Float mse score
        """
        vis_df = pd.DataFrame()
        vis_df["predicted"] = predictions
        vis_df["actual"] = target.iloc[:, -1].to_numpy()
        mse_score_final = sum((vis_df["actual"] - vis_df["predicted"]) ** 2) / vis_df.shape[0]
        return mse_score_final

    @staticmethod
    def get_avg_value(dataset):
        """
        Function that sets prediction value for a leaf.

        :param dataset: Dataset with training data.
        :return: Mean of target column or if single sample then just the value.
        """
        if dataset.shape[0] > 1:
            return dataset[:, -1].mean()
        else:
            return dataset[:, -1][0]
