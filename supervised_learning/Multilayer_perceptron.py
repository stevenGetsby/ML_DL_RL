import numpy as np
import math
from deep_learning.activation_functions import ReLU,Softmax,Sigmoid
from deep_learning.loss_function import CrossEntropy

class MultilayerPerceptron():
    def __init__(self, n_hidden, n_iterations, hidden_activation, output_activation,  learning_rate = 0.01):
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = CrossEntropy()
    
    def _initialize_weights(self, X, y):
        n_samples, n_features = X.shape
        _, n_classes = y.shape
        limit = 1 / math.sqrt(n_features)
        self.w0 = np.random.uniform(-limit, limit, (n_features, self.n_hidden))
        self.b0 = np.zeros((1, self.n_hidden))
        limit = 1 / math.sqrt(self.n_hidden)
        self.w1 = np.random.uniform(-limit, limit, (self.n_hidden, n_classes))
        self.b1 = np.zeros((1, n_classes))
    
    def fit(self, X, y):
        self._initialize_weights(X, y)
        for i in range(self.n_iterations):
            #前向传播
            hidden_input = X.dot(self.w0) + self.b0
            hidden_output = self.hidden_activation(hidden_input)
            output_input = hidden_output.dot(self.w1) + self.b1
            y_pred = self.output_activation(output_input)
            #反向传播
            grad_wrt_out_l_input = self.loss.gradient(y, y_pred) * self.output_activation.gradient(output_input)
            grad_w1 = hidden_output.T.dot(grad_wrt_out_l_input)
            grad_b1 = np.sum(grad_wrt_out_l_input, axis=0, keepdims=True)
            
            grad_wrt_hidden_l_input = grad_wrt_out_l_input.dot(self.w1.T) * self.hidden_activation.gradient(hidden_input)
            grad_w0 = X.T.dot(grad_wrt_hidden_l_input)
            grad_b0 = np.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)
            
            #梯度更新
            self.w1 -= self.learning_rate * grad_w1
            self.b1 -= self.learning_rate * grad_b1
            self.w0 -= self.learning_rate * grad_w0
            self.b0 -= self.learning_rate * grad_b0
            
    def predict(self, X):
        hidden_input = X.dot(self.w0) + self.b0
        hidden_output = self.hidden_activation(hidden_input)
        output_input = hidden_output.dot(self.w1) + self.b1
        y_pred = self.output_activation(output_input)
        return y_pred