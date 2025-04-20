import numpy as np
import math
from deep_learning.activation_functions import Sigmoid
from deep_learning.loss_function import SquareLoss

class Perceptron():
    def __init__(self, n_iterations=2000, learning_rate=0.01, activation_func = Sigmoid, loss = SquareLoss):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.activation_func = activation_func()
        self.loss = loss()
    
    def _initailize_weights(self, X, y):
        n_samples, n_features = X.shape
        _, n_output = y.shape
        limit = 1 / math.sqrt(n_features)
        self.w0 = np.random.uniform(-limit, limit, (n_features, n_output))
        self.b0 = np.zeros((1, n_output))
        
    def fit(self, X, y):
        self._initailize_weights(X, y)
        for i in range(self.n_iterations):
            #前向传播
            linear_output = X.dot(self.w0) + self.b0
            y_pred = self.activation_func(linear_output)
            #反向传播
            error_gradient = self.loss.gradient(y, y_pred)*self.activation_func.gradient(linear_output)
            grad_wrt_w0 = X.T.dot(error_gradient)
            grad_wrt_b0 = np.sum(error_gradient, axis=0, keepdims=True)
            #梯度更新
            self.w0 -= self.learning_rate * grad_wrt_w0
            self.b0 -= self.learning_rate * grad_wrt_b0  
            
    def predict(self, X):
        linear_output = X.dot(self.w0) + self.b0
        y_pred = self.activation_func(linear_output)
        return y_pred