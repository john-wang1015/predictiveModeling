import numpy as np
import CART
import utils

class XGBoost_Single_Tree(CART.BinaryDecisionTree):
    def node_split(self,y):
        feature = int(np.shape(y)[1]/2)
        y_true, y_pred = y[:,:feature], y[:,feature:]
        return y_true,y_pred

    def gain(self, y, y_pred):
        Gradient = np.power((y* self.loss.gradient(y,y_pred)).sum(), 2)
        Hessian = self.loss.hess(y,y_pred).sum()

        return 0.5*(Gradient/Hessian)

    def gain_xgb(self, y, y1, y2):
        y_true, y_pred = self.node_split(y)
        y1, y1_pred = self.node_split(y1)
        y2, y2_pred = self.node_split(y2)

        true_gain = self.gain(y1, y1_pred)
        false_gain = self.gain(y2, y2_pred)
        gain = self.gain(y_true, y_pred)

        return true_gain + false_gain - gain

    def leaf_weight(self, y):
        y_true, y_pred = self.node_split(y)
        gradient = np.sum((y* self.loss.gradient(y,y_pred)), axis = 0)
        hessian = np.sum(self.loss.gradient(y,y_pred), axis = 0)

        leaf_weight = gradient/hessian

        return leaf_weight

    def fit(self, X, y):
        self.impurity_calculation = self.gain_xgb
        self._leaf_value_calculation = self.leaf_weight
        super(XGBoost_Single_Tree,self).fit(X, y)

class Sigmoid:
    def __call__(self,x):
        return 1/(1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

class LogisticLoss:
    def __init__(self):
        sigmoid = Sigmoid()
        self._func = sigmoid
        self._grad = sigmoid.gradient

    def loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        p = self._func(y_pred)
        return y * np.log(p) + (1 - y) * np.log(1 - p)

    def gradient(self, y, y_pred):
        p = self._func(y_pred)
        return -(y-p)

    def hess(self,y,y_pred):
        p = self._func(y_pred)
        return p*(1-p)

class XGBoost:
    def __init__(self, n_estimators = 200,learning_rate = 1e-4,
                 min_samples_split = 2, mini_gini_impurity = 999,
                 max_depth = 50):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.mini_gini_impurity = mini_gini_impurity
        self.max_depth = max_depth
        self.loss = LogisticLoss()
        self.estimators = []

        for _ in range(n_estimators):
            estimator = XGBoost_Single_Tree(min_samples_split = self.min_samples_split,
                                       mini_gini_impurity = self.mini_gini_impurity,
                                       max_depth = self.max_depth,
                                       loss = self.loss)
            self.estimators.append(estimator)


    def fit(self, X, y):
        y = utils.cat_lable_convert(y)
        y_pred = np.zeros(np.shape(y))

        for i in range(self.n_estimators):
            estimator = self.estimators[i]
            y_true_pred = np.concatenate((y,y_pred), axis = 1)
            estimator.fit(X, y_true_pred)
            iter_pred = estimator.predict(X)
            y_pred -= np.multiply(self.learning_rate,iter_pred)

    def predict(self, X):
        y_pred = None

        for estimator in self.estimators:
            iter_pred = estimator.predict(X)
            if y_pred is None:
                y_pred = np.zeros_like(iter_pred)
            y_pred -= np.multiply(self.learning_rate, iter_pred)

        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis = 1, keepdims = True)
        y_pred = np.armax(y_pred, axis = 1)
        return y_pred
