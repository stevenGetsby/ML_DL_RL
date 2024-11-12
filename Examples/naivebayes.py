from sklearn import datasets
import numpy as np
from utils import train_test_split, normalize, accuracy_score
from supervised_learning.Naive_bayes import NaiveBayes
from sklearn.naive_bayes import GaussianNB

def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = NaiveBayes()
    clf_s = GaussianNB()
    
    clf.fit(X_train, y_train)
    clf_s.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_pred_s = clf_s.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_s = accuracy_score(y_test, y_pred_s)
    
    print ("my bayes accuracy:", accuracy)
    print ("sklearn bayes accuracy:", accuracy_s)

if __name__ == "__main__":
    main()