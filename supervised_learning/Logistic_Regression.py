import numpy as np
import math
from utils import make_diagonal
from deep_learning.activation_functions import Sigmoid


class LogisticRegression():
    """ 逻辑回归 """
    def __init__(self, learning_rate=1e-4, gradient_descent=True):
        self.param = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        self.param = np.random.uniform(low = -1/math.sqrt(n_features), high = 1/math.sqrt(n_features), size = (n_features, ))

    def fit(self, X, y, n_iterations=4000):
        self._initialize_parameters(X)
        for _ in range(n_iterations):
            y_pred = self.sigmoid(np.dot(X, self.param))
            if self.gradient_descent:
                self.param -= self.learning_rate * (y_pred - y).dot(X)
            else:
                # Make a diagonal matrix of the sigmoid gradient column vector
                diag_gradient = make_diagonal(self.sigmoid.gradient(X.dot(self.param)))
                # Batch opt:
                self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T).dot(diag_gradient.dot(X).dot(self.param) + y - y_pred)

    def predict(self, X):
        y_pred = np.round(self.sigmoid(np.dot(X, self.param))).astype(int)
        return y_pred
