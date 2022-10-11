import numpy as np
from utils import feature_split, calculate_gini

class TreeNode:
    def __init__(self, feature_ix = None, threshold = None,
                 leaf_value = None, left_branch = None, right_branch = None):
        self.feature_ix = feature_ix
        self.threshold = threshold
        self.leaf_value = leaf_value
        self.left_branch = left_branch
        self.right_branch = right_branch

class BinaryDecisionTree(object):
    def __init__(self, min_samples_split=2, min_gini_impurity=999,
                 max_depth=float("inf"), loss=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.mini_gini_impurity = min_gini_impurity
        self.max_depth = max_depth
        self.gini_impurity_calculation = None
        self._leaf_value_calculation = None
        self.loss = loss

    def fit(self, X, y, loss=None):
        self.root = self._build_tree(X, y)
        self.loss=None

    def _build_tree(self, X, y, current_depth=0):
        init_gini_impurity = 999
        best_criteria = None
        best_sets = None

        Xy = np.concatenate((X, y), axis=1)
        n_samples, n_features = X.shape

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                for threshold in unique_values:
                    Xy1, Xy2 = feature_split(Xy, feature_i, threshold)
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]
                        impurity = self.impurity_calculation(y, y1, y2)

                        if impurity < init_gini_impurity:
                            init_gini_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],
                                "lefty": Xy1[:, n_features:],
                                "rightX": Xy2[:, :n_features],
                                "righty": Xy2[:, n_features:]
                                }

        if init_gini_impurity < self.mini_gini_impurity:
            left_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            right_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return TreeNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                                "threshold"], left_branch=left_branch, right_branch=right_branch)

        leaf_value = self._leaf_value_calculation(y)

        return TreeNode(leaf_value=leaf_value)

    def predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root

        if tree.leaf_value is not None:
            return tree.leaf_value

        feature_value = x[tree.feature_i]
        branch = tree.right_branch

        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.left_branch
        elif feature_value == tree.threshold:
            branch = tree.left_branch

        return self.predict_value(x, branch)


    def predict(self, X):
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

class ClassificationTree(BinaryDecisionTree):
    def _calculate_gini_impurity(self, y, y1, y2):
        p = len(y1) / len(y)
        gini = calculate_gini(y)
        gini_impurity = p * calculate_gini(y1) + (1-p) * calculate_gini(y2)
        return gini_impurity

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        self.impurity_calculation = self._calculate_gini_impurity
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)
