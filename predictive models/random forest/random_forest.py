import numpy as np
from CART import ClassificationTree

class RandomForest:
    def __init__(self,n_estimators = 100, min_samples_split = 2,
                 min_gain = 0, max_depth = float("inf"), max_features = None):
        self.n_estimatores = n_estimators
        self.min_samples_split = min_samples_split
        self.min_gain_impurity = min_gain
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

        for _ in range(self.n_estimatores):
            tree = ClassificationTree(min_samples_split = self.min_samples_split,
                                           min_gini_impurity = self.min_gain_impurity,
                                           max_depth = self.max_depth)
            self.trees.append(tree)

    def bootstrap_sampling(self,X,y):
        X_y = np.concatenate([X,y.reshape(-1,1)], axis = 1)
        np.random.shuffle(X_y)
        n_samples = X.shape[0]
        sampling_subset = []

        for _ in range(self.n_estimators):
            idx1 = np.random.choice(n_samples,n_samples,replace=True)
            bootstrap_Xy = X_y[idx1,:]
            bootstrap_X = bootstrap_Xy[:,:-1]
            bootstrap_y = bootstrap_Xy[:,:-1]
            sampling_subset.append([bootstrap_X,bootstrap_y])

        return sampling_subset

    def fit(self,X, y):
        n_features = X.shape[1]
        sub_sets = self.bootstrap_sampling(X,y,self.n_estimators)

        if self.max_features == None:
            self.max_features = int(np.sqrt(n_features))

        for i in range(self.n_estimators):
            sub_X,sub_y = sub_sets[i]
            idx2 = np.random.choice(n_features, self.max_features, replace=True)
            sub_X = sub_X[:, idx2]
            self.trees[i].fit(sub_X,sub_y)
            self.trees[i].feature_indices = idx2
            print("The {}th tree is trained done.".format(i+1))

    def predict(self,X):
        y_pred = []

        for i in range(self.n_estimatores):
            idx = self.trees[i].feature_indices
            sub_X = X[:, idx]
            y_pred = self.trees[i].predict(sub_X)
            y_pred.append(y_pred)

        y_pred = np.array(y_pred).T
        res = []

        for j in y_pred:
            res.append(np.bincount(j.astype('int')).argmax())

        return res


