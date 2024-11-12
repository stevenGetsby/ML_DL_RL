import numpy as np
import math
from utils import train_test_split, normalize

class NaiveBayes():
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.parameters = []
        #初始化每一个类别的特征向量的均值和方差
        for i,c in enumerate(self.classes):
            X_where_c = X[np.where(y == c)]
            self.parameters.append([])
            for col in X_where_c.T:
                parameters = {"mean": col.mean(), "var": col.var()}
                self.parameters[i].append(parameters)
    
    def _calculate_likelihood(self, mean, var, x):
        #计算高斯似然
        eps = 1e-6
        coeff = 1.0/math.sqrt(2.0 * math.pi * var + eps)
        exponent = -math.pow(x - mean, 2)/(2 * var + eps)
        return coeff * math.exp(exponent)
    
    def _calculate_prior(self, c):
        #计算后验概率
        frequency = np.sum(self.y == c)/len(self.y)
        return frequency
    
    def _classify(self, x):
        posteriors = []
        for i,c in enumerate(self.classes):
            posterior = self._calculate_prior(c)
            for feature_val, params in zip(x, self.parameters[i]):
                likelihood = self._calculate_likelihood(params["mean"], params["var"], feature_val)
                posterior *= likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        y_pred = [self._classify(x) for x in X]
        return y_pred