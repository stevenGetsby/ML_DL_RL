import numpy as np
from sklearn import datasets
from utils import normalize, train_test_split, accuracy_score, to_categorical
from supervised_learning.Perceptron import Perceptron
from deep_learning.activation_functions import Sigmoid
from deep_learning.loss_function import CrossEntropy

def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    # One-hot encoding of nominal y-values
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    # Perceptron
    clf = Perceptron(n_iterations=5000, 
                     learning_rate=0.001, 
                     activation_func = Sigmoid, 
                     loss = CrossEntropy)
    
    clf.fit(X_train, y_train)

    y_pred = np.argmax(clf.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)
    
if __name__ == "__main__":
    main()