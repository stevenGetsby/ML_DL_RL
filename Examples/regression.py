from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from supervised_learning import LinearRegression, LassoRegression
from supervised_learning.Regression import RidgeRegression, ElasticNet
from utils import mean_squared_error


def cp(model_list,X_train,y_train,X_test,y_test):
    error_list = []
    for model in model_list:
        model.fit(X_train,y_train)
        error_list.append(mean_squared_error(model.predict(X_test),y_test))
    return error_list

def main():
    X,y = make_regression(n_samples=100, n_features=5, noise=20)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    model_list = []

    linear_regression = LinearRegression(n_iterations=200,learning_rate=0.005)
    lasso_regression = LassoRegression(l1_ratio=0.1,n_iterations=200,learning_rate=0.005)
    ridge_regression = RidgeRegression(l2_ratio=0.1,n_iterations=200,learning_rate=0.005)
    elastic_regression = ElasticNet(reg_factor=0.1,l1_ratio=0.5,n_iterations=200,learning_rate=0.005)
    model_list.append(linear_regression)
    model_list.append(lasso_regression)
    model_list.append(ridge_regression)
    model_list.append(elastic_regression)
    ls = cp(model_list,X_train,y_train,X_test,y_test)
    print("linear test loss:{:.2f} \n lasso test loss:{:.2f} \n ridge test loss:{:.2f} \n elasticNet test loss:{:.2f}".format(ls[0],ls[1],ls[2],ls[3]))

if __name__ == '__main__':
    main()