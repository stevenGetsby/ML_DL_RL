from sklearn import datasets
import numpy as np
from utils import normalize,to_categorical,train_test_split,accuracy_score
from supervised_learning.Multilayer_perceptron import MultilayerPerceptron

def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    # Convert the nominal y values to binary
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)

    # MLP
    clf = MultilayerPerceptron(n_hidden=32,n_iterations=10000,learning_rate=0.01)

    clf.fit(X_train, y_train)
    y_pred = np.argmax(clf.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy:", accuracy)

if __name__ == "__main__":
    main()