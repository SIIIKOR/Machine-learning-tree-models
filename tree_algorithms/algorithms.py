from collections import Counter, deque
from abc import ABC, abstractmethod
import copy

import numpy as np

from functions import cross_validate


class Node(ABC):
    pass


class DecisionNode(Node):
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, **kwargs):
        """
        Decision root_node.

        Values of feature which are smaller than some threshold go to the left. Greater go to the right.

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
        Decision root_node that will be used for visualisation.

        :param feature_index: Index of feature by which decision will be made.
        :param threshold: Value by which decision will be made.
        :param left: DecisionNode|Leaf.
        :param right: DecisionNode|Leaf.
        :param split_score: Value indicating quality of the split.
        """
        super().__init__(feature_index, threshold, left, right)
        self.split_score = kwargs["split_score"]


class PruningDecisionNode(DecisionNode):
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, **kwargs):
        """
        Decision Node that will be used during pruning.

        It has additional parameter parent, which is pointing at the parent of this node.

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
        It tells what given sample have been classified as.

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


class PruningLeaf(Leaf):
    def __init__(self, value=None, **kwargs):
        """
        Leaf node used in pruning tree.

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

        :param dataset: numpy array with numeric training and target data(must be the last column).
        :param min_sample_split: Pre-pruning - Minimal samples required to split dataset.
        :param max_depth: Maximal depth of the tree.
        """
        self.dataset = dataset
        self.root = None
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth

        self.node_type = DecisionNode
        self.leaf_type = Leaf

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

    def get_optimal_split(self, dataset, feature_range):
        """
        Returns the best split for the given dataset.

        Uses evaluation function which is different for classification and regression tree.
        But classification can have multiple evaluation functions.

        :param dataset: Dataset with merged train and target columns. often smaller size than original.
        :param feature_range: Range of training features.
        :return: Dictionary with information about best split: feature_index, threshold,
         left_dataset, right_dataset, split_score, is_final(is it the last split that needs to be done).
        """
        best_score = float("inf")
        best_split = None
        if isinstance(feature_range, int):
            feature_range = range(feature_range)
        for feature_index in feature_range:
            thresholds = self.generate_thresholds(np.unique(sorted(dataset[:, feature_index])))
            thresholds = list(thresholds)
            for threshold in thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                eval_output = self.evaluate_split(dataset_left=dataset_left, dataset_right=dataset_right,
                                                  feature_index=feature_index)
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

    @abstractmethod
    def evaluate_split(self, **kwargs):
        """Function used to rate given split."""
    @abstractmethod
    def set_prediction(self, **kwargs):
        """Function used to set prediction value for a leaf."""
    @abstractmethod
    def evaluate_leaf(self, **kwargs):
        """Function used to rate leaf."""

    def fit(self, x=None, target=None, mode=None, k_parameter=None):
        """
        Builds up the trees.

        :param x: Training dataset.
        :param target: Target dataset.
        :param mode: If user want's to choose type of nodes used. For example for pruning.
        :param k_parameter: Parameter for random forest.
         Specifies amount of features to be randomly picked at each step.
        :return: Returns nothing.
        """

        def build_tree_iterative(dataset):
            """
            Iterative tree building algorithm basing on bfs.

            Currently the the only one working for this project.
            Recursive variant hasn't been updated for random forest and bootstrapping.

            :param dataset: Dataset used to build tree.
            :return: Root node to which rest of the tree is attached.
            """
            # left_child, current dataset, parent
            queue = deque([(False, dataset, None)])
            depth = 0
            while queue:
                # for every root_node on current level
                for _ in range(len(queue)):
                    is_left, current_dataset, current_parent = queue.popleft()
                    sample_amount, feature_amount = current_dataset.shape
                    # if this is random forest
                    if k_parameter:
                        possible_index = np.arange(feature_amount-1)
                        random_features = np.random.choice(possible_index, k_parameter, replace=False)
                        feature_range = random_features
                    else:
                        feature_range = feature_amount - 1
                    # if we can we proceed to split the data
                    if sample_amount >= self.min_sample_split and depth < self.max_depth:
                        # splitting
                        optimal_split = self.get_optimal_split(current_dataset, feature_range)
                        # if optimal split is None then we could not split the data hence we create leaf.
                        condition = False
                        if optimal_split is not None:  # means if we can split the data.
                            split_score = optimal_split["split_score"]
                            # if this is regression tree condition will always be true,
                            # because if we can split the data we will always do so
                            condition = True
                            # if this is classification tree, we have different condition
                            if isinstance(self, ClassificationTree):
                                is_final = optimal_split["is_final"]
                                condition = split_score > 0 or (split_score == 0 and not is_final)

                        if condition:  # create decision node
                            current_node = self.node_type(optimal_split["feature_index"], optimal_split["threshold"],
                                                          split_score=split_score, parent=current_parent)
                            queue.append((True, optimal_split["dataset_left"], current_node))
                            queue.append((False, optimal_split["dataset_right"], current_node))
                        else:  # create leaf
                            current_node = self.leaf_type(self.set_prediction(dataset=current_dataset),
                                                          leaf_eval_score=self.evaluate_leaf(dataset=current_dataset),
                                                          size=len(current_dataset))
                    else:  # if we are not allowed to split the data, we create leaf
                        current_node = self.leaf_type(self.set_prediction(dataset=current_dataset),
                                                      leaf_eval_score=self.evaluate_leaf(dataset=current_dataset),
                                                      size=len(current_dataset))
                    # if this is not root
                    if not (not is_left and current_parent is None):
                        if is_left:  # we assign newly created root_node to be left child of parent
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

            Not updated for random forest.

            :param dataset: Dataset with merged train and target columns. often smaller size than original.
            :param depth: Current depth of this instance.
            :return: DecisionNode if more splits will be required of Leaf object if no more split will be required.
            """
            sample_amount, feature_amount = dataset.shape
            if sample_amount >= self.min_sample_split and depth <= self.max_depth:
                optimal_split = self.get_optimal_split(dataset, feature_amount - 1)
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

            return self.leaf_type(self.set_prediction(dataset),
                                  leaf_eval_score=self.evaluate_split(dataset), size=len(dataset))

        if self.dataset is None:
            self.dataset = np.append(x, target, axis=1)
        if mode == "prune":
            self.node_type = PruningDecisionNode
            self.leaf_type = PruningLeaf
        elif mode == "vis":
            self.node_type = VisDecisionNode
            self.leaf_type = VisLeaf
        self.root = build_tree_iterative(self.dataset)

    def make_prediction(self, sample, node=None):
        """
        Recursive function for traversing the tree and thus data prediction.

        :param sample: Vector with data to predict.
        :param node: Current root_node.
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
                return self.make_prediction(sample, node.left)
            else:
                return self.make_prediction(sample, node.right)

    def predict(self, x):
        """
        Function to predict values from given data.

        :param x: Dataset with data to predict.
        :return: Returns predictions.
        """
        test = x[:, :-1]
        predictions = [self.make_prediction(sample) for sample in test]
        return predictions

    @abstractmethod
    def prediction_score(self):
        """Calculates prediction score for a given tree type."""

    def cost_complexity_pruning(self):
        """
        cost complexity pruning algorithm.

        After changes, doesn't work needs an update.

        Pretty much works terribly. Maybe i did something wrong in the implementation.
        It's surely makes predictions worse so maybe over-fitting is less noticeable.

        :return: Pruned trees with alphas with minimal rss
        """
        self.node_type = PruningDecisionNode
        self.leaf_type = PruningLeaf
        curr_root = self.root
        # leaf_sum_score, leaf_amount, tree
        variants = [[None, None, self]]
        # while root is not leaf
        while not isinstance(curr_root, Leaf):
            # we take lastly added tree and copy it
            curr_tree = copy.deepcopy(variants[-1][2])
            curr_root = curr_tree.root
            # is_left, current_node, corresponding dataset
            stack = [(None, curr_root, self.dataset)]
            while stack:
                is_left, curr, current_dataset = stack.pop()
                # if left child is not a leaf
                if not isinstance(curr.left, Leaf):
                    # we get it's corresponding dataset
                    left_dataset = current_dataset.loc[current_dataset.iloc[:, curr.feature_index] <= curr.threshold]
                    # we push to the stack to traverse further
                    stack.append((True, curr.left, left_dataset))
                if not isinstance(curr.right, Leaf):
                    right_dataset = current_dataset.loc[current_dataset.iloc[:, curr.feature_index] > curr.threshold]
                    stack.append((False, curr.right, right_dataset))
                # if both children are leaves
                if isinstance(curr.left, Leaf) and isinstance(curr.right, Leaf):
                    # we transform dataset to np.array
                    current_dataset = current_dataset.to_numpy()
                    new_leaf = self.leaf_type(self.set_prediction(dataset=current_dataset),
                                              leaf_eval_score=self.evaluate_leaf(dataset=current_dataset))
                    # if this is root
                    if is_left is None:
                        curr_root = new_leaf
                    # if it was left child
                    elif is_left:
                        curr.parent.left = new_leaf
                    else:
                        curr.parent.right = new_leaf
                    # we found candidate for leaf so we stop searching
                    break
            new_tree = self.__class__(dataset=self.dataset)
            new_tree.root = curr_root
            variants.append([None, None, new_tree])
        # the last variant of pruned trees is always a leaf so we don't have to do calculations
        variants[-1][0] = variants[-1][2].root.score
        variants[-1][1] = 1

        # dfs algorithm to calculate sum of rss of all leaves and amount of leaves
        for variant in variants[:-1]:
            tree_variant = variant[2]
            stack = [tree_variant.root]
            sum_score = 0
            leaf_amount = 0
            while stack:
                curr = stack.pop()
                for node in [curr.left, curr.right]:
                    if isinstance(node, PruningLeaf):
                        sum_score += node.score
                        leaf_amount += 1
                    else:
                        stack.append(node)
            variant[0] = sum_score
            variant[1] = leaf_amount
        variants = np.c_[np.full(len(variants), None), np.zeros(len(variants)), np.array(variants)]

        # algorithm to calculate optimal alpha for each tree.
        # trees without alpha are removed, because they are pretty bad i guess.
        alpha = 0
        while variants[len(variants) - 1, 0] is None:
            variants[:, 1] = variants[:, 2] + alpha * variants[:, 3]
            min_cost_index = np.where(variants[:, 1] == min(variants[:, 1]))[0][0]
            if variants[min_cost_index, 0] is None:
                variants[min_cost_index, 0] = alpha
            alpha += 1
        valid_alpha = np.where(variants[:, 0] != None)[0]
        variants = variants[valid_alpha, :]
        return variants[:, [0, 4]]

    def prune(self):
        """
        In theory should return tree pruned in the best way.

        :return: Pruned tree
        """
        initial_variants = self.cost_complexity_pruning()
        models = cross_validate(self.__class__, full_dataset=self.dataset, mode="prune",
                                model_type="regression", min_sample_split=self.min_sample_split)
        alphas = []
        for el in models:
            test_data, model = el
            variants = model.cost_complexity_pruning()
            rss_scores = [m.prediction_score(m.predict(test_data), test_data) for m in variants[:, 1]]
            min_rss_index = np.argmin(rss_scores)
            alphas.append(variants[min_rss_index, 0])
        avg_alpha = sum(alphas)/len(alphas)
        initial_alphas = initial_variants[:, 0]
        final_model = None
        for i in range(len(initial_alphas)-1):
            if initial_alphas[i] < avg_alpha < initial_alphas[i+1]:
                final_model = i
        return initial_variants[final_model, :][1]

    def print_tree(self, root_node=None, x=None, target=None, indent="-", target_names=None):
        """
        Prints out the tree structure.
        Have to create tree with specific Nodes to see size and split_score.

        :param root_node: Root node of a tree to traverse.
        :param x: Train data for fit function.
        :param target: Target data for fit function.
        :param indent: Symbol used to make indents.
        :param target_names: List with names of target classes.
        :return: String representation of the tree.
        """
        def _print_tree(node=None, multi=1):
            """
            Recursive function for creating string representation of the tree.

            :param node: Current worked on root_node.
            :param multi: Amount of indents required to represent this root_node.
            :return: Returns nothing.
            """
            if node is None:
                node = self.root
            if isinstance(node, VisLeaf):
                print(f"class: {target_names[int(node.value)] if target_names else node.value}, size: {node.size}")
            elif isinstance(node, PruningLeaf):
                print(f"class: {target_names[int(node.value)] if target_names else node.value}, score: {node.score}")
            elif isinstance(node, Leaf):
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
        return _print_tree(root_node)


class ClassificationTree(Tree):

    @staticmethod
    def set_prediction(**kwargs):
        """
        Function that sets prediction value for a leaf.

        :return: Most frequently occurring value in given dataset target column.

        """
        dataset = kwargs["dataset"]
        count = Counter(dataset[:, -1])
        return max(count, key=lambda x: count[x])

    @staticmethod
    def evaluate_leaf(**kwargs):
        """
        Function used for leaf evaluation for potential pruning.

        :return: Amount of most commonly occurring value in target column.
        """
        dataset = kwargs["dataset"]
        count = Counter(dataset[:, -1])
        return count[max(count, key=lambda x: count[x])]

    @staticmethod
    def evaluate_split(**kwargs):
        """
        Calculates gini index for given datasets and decides whether it was the last split.

        :return: weighted gini index, information whether this split should be final.
        """
        scores = []
        counts = []
        dataset_left = kwargs["dataset_left"]
        dataset_right = kwargs["dataset_right"]
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
    def prediction_score(predictions, target, **kwargs):
        """
        Function for calculating Success rate of prediction.

        :param predictions: List with predictions.
        :param target: Real values.
        :return: Success rate of prediction.
        """
        target = target[:, -1]
        return len([el for el in zip(predictions, target) if el[0] == el[1]])/len(predictions)


class RegressionTree(Tree):

    @staticmethod
    def set_prediction(**kwargs):
        """
        Function that sets prediction value for a leaf.

        :return: Mean of target column or if single sample then just the value.
        """
        dataset = kwargs["dataset"]
        if dataset.shape[0] > 1:
            return dataset[:, -1].mean()
        else:
            return dataset[:, -1][0]

    @staticmethod
    def evaluate_leaf(**kwargs):
        """
        Calculates rss of column in dataset.

        :return: Float rss score.
        """
        dataset = kwargs["dataset"]
        feature_index = -1
        if "feature_index" in kwargs:
            feature_index = kwargs["feature_index"]
        column = dataset[:, feature_index]
        mean = column.mean()
        rss = 0
        for value in column:
            rss += (value - mean)**2
        return rss

    def evaluate_split(self, **kwargs):
        """
        Calculates sum of rss for datasets

        :return: Returns rss of datasets.
        """
        dataset_left = kwargs["dataset_left"]
        dataset_right = kwargs["dataset_right"]
        feature_index = kwargs["feature_index"]
        datasets_rss = [self.evaluate_leaf(dataset=dataset, feature_index=feature_index)
                        for dataset in [dataset_left, dataset_right]]
        return {"score": sum(datasets_rss), "datasets_rss": datasets_rss}

    @staticmethod
    def prediction_score(predictions, target, **kwargs):
        """
        Calculates mse of prediction.

        :param predictions: np.array of model predictions.
        :param target: Dataframe with target data.
        :param kwargs: Mode - parameter used to chose type of output: rss or mse.
        :return: Float mse score
        """
        target = target.iloc[:, -1].to_numpy()
        if kwargs["mode"] == "mse":
            mse_score_final = sum((target - predictions) ** 2) / len(predictions)
            return mse_score_final
        elif kwargs["mode"] == "rss":
            return sum((target - predictions) ** 2)


class RandomForest:
    def __init__(self, dataset=None, tree_type="classification", min_sample_split=2, max_depth=5):
        """
        Random forest algorithm.

        Based on bootstrapping and randomly picking features at each step at tree building.

        :param dataset: Dataset used to build forest.
        :param tree_type: Classification or regression.
        :param min_sample_split: Minimal samples required to split dataset.
        :param max_depth: Maximal depth of the tree.
        """
        self.dataset = dataset
        if tree_type == "classification":
            tree_type = ClassificationTree
        elif tree_type == "regression":
            tree_type = RegressionTree
        self.tree_type = tree_type
        self.trees = None
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth

    @staticmethod
    def bootstrap_dataset(n, m, sample_amount):
        """
        Function for creating bootstrapped datasets.

        :param n: Amount of new datasets.
        :param m: Amount of samples in each dataset.
        :param sample_amount: Amount of samples in base dataset.
        :return: List of newly created bootstrapped datasets.
        """
        if m is None:
            m = sample_amount
        all_indexes = set(np.arange(sample_amount))
        bt_datasets = []
        for _ in range(n):
            bootstrapped_indexes = np.random.random_integers(0, sample_amount-1, m)
            out_of_bag_indexes = np.fromiter(all_indexes - Counter(bootstrapped_indexes).keys(), int)
            bt_datasets.append((bootstrapped_indexes, out_of_bag_indexes))
        return bt_datasets

    @staticmethod
    def bootstrap_dataset_generator(n, m, sample_amount):
        """
        Function for creating bootstrapped datasets.

        It's a generator so memory efficiency is much better.
        I'm not sure whether generator won't change randomly
        chosen feature indexes during multiple iterations while
        searching for the best k_parameter

        This problem might be solved by setting random seed at the beginning
        of forest building algorithm.

        :param n: Amount of new datasets.
        :param m: Amount of samples in each dataset.
        :param sample_amount: Amount of samples in base dataset.
        :return: List of newly created bootstrapped datasets.
        """
        if m is None:
            m = sample_amount
        all_indexes = set(np.arange(sample_amount))
        for _ in range(n):
            bootstrapped_indexes = np.random.random_integers(0, sample_amount - 1, m)
            out_of_bag_indexes = np.fromiter(all_indexes - Counter(bootstrapped_indexes).keys(), int)
            yield bootstrapped_indexes, out_of_bag_indexes

    def build_trees(self, n, bt_dataset=None, k_parameter=None, min_sample_split=None,
                    max_depth=None, m=None):
        """
        Builds random forest for given k parameter.

        :param n: Amount of trees.
        :param bt_dataset: list of tuples containing array of
         sample indexes for bootstrap dataset and it's corresponding out of bag samples
        :param k_parameter: Amount of features which will be randomly selected at each step while building given tree.
        :param min_sample_split: Minimal samples required to split dataset.
        :param max_depth: Maximal depth of the tree.
        :param m: Optional parameter, you can limit sample amount per bootstrapped dataset
        :return: float accuracy_estimate and list of trees
        """
        if min_sample_split is not None:
            self.min_sample_split = min_sample_split
        if max_depth is not None:
            self.max_depth = max_depth
        if bt_dataset is None:
            sample_amount = len(self.dataset)
            bt_dataset = self.bootstrap_dataset_generator(n, m, sample_amount)
        if k_parameter is None:
            feature_amount = self.dataset.shape[1]
            k_parameter = int(np.sqrt(feature_amount))

        trees = []
        oob_predictions_scores = []
        for el in bt_dataset:
            bootstrapped_index, out_of_bag_indexes = el
            new_tree = self.tree_type(self.dataset[bootstrapped_index],
                                      min_sample_split=self.min_sample_split,
                                      max_depth=self.max_depth)
            new_tree.fit(k_parameter=k_parameter)
            oob_dataset = self.dataset[out_of_bag_indexes]
            oob_prediction = new_tree.predict(oob_dataset)
            score = new_tree.prediction_score(oob_prediction, oob_dataset, mode="rss")
            oob_predictions_scores.append(score)
            trees.append(new_tree)
        rf_accuracy_estimate = sum(oob_predictions_scores)/len(oob_predictions_scores)
        return rf_accuracy_estimate, trees

    def build_forest(self, n, diff=None, min_sample_split=None, max_depth=None, m=None):
        """
        Builds random forest with best k parameter.

        :param n: Amount of trees
        :param diff: Specifies amount of k parameters to check.
        :param min_sample_split: Minimal samples required to split dataset.
        :param max_depth: Maximal depth of the tree.
        :param m: Optional parameter, you can limit sample amount per bootstrapped dataset
        :return: Returns nothing.
        """
        sample_amount, feature_amount = self.dataset.shape
        # setting seed to prevent mistakes at picking best k. I'm not sure if it is a problem or if this is a solution.
        # np.random.default_rng()
        bootstrapped_datasets = self.bootstrap_dataset(n, m, sample_amount)
        # uncomment to use generator for creating bootstrapped dataset but it may be wrong for picking best k
        # bootstrapped_datasets = None
        init_k_para = int(np.sqrt(feature_amount-1))
        if diff is None:  # creating boundaries of k to test.
            diff = init_k_para
        low_boundary = init_k_para-diff if init_k_para-diff >= 1 else 1
        high_boundary = init_k_para+diff+1 if init_k_para+diff+1 < feature_amount else feature_amount-1

        best_rf, best_rf_accuracy_estimate = None, None
        for k in range(low_boundary, high_boundary):  # creating forests with different k parameters.
            accuracy_estimate, trees = self.build_trees(n, bootstrapped_datasets, k,
                                                        min_sample_split, max_depth, m)

            if best_rf_accuracy_estimate is None:
                best_rf_accuracy_estimate = accuracy_estimate
                best_rf = trees
            if isinstance(self.tree_type, ClassificationTree):
                cond = accuracy_estimate > best_rf_accuracy_estimate
            else:
                cond = accuracy_estimate < best_rf_accuracy_estimate
            if cond:
                best_rf_accuracy_estimate = accuracy_estimate
                best_rf = trees
        self.trees = best_rf
        print(best_rf_accuracy_estimate)

    def predict(self, x):
        """
        Function used to predict data in random forest.

        The final prediction is the most occurring one for classification or the average one for regression.
        :param x: Dataset with data to predict from.
        :return: Predictions.
        """
        test, target = x.iloc[:, :-1].to_numpy(), x.iloc[:, -1].to_numpy()
        predictions = []
        for i in range(len(test)):
            sample, sample_target = test[i, :], target[i]
            variants = []
            for tree in self.trees:
                prediction_variant = tree.make_prediction(sample)
                variants.append(prediction_variant)
            if self.tree_type is ClassificationTree:
                count = Counter(variants)
                most_common_value = max(count, key=lambda k: count[k])
                predictions.append((most_common_value, sample_target))
            else:
                predictions.append((sum(variants)/len(variants), sample_target))
        return predictions
