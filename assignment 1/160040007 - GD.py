"""
Ridge Regression
using Gradient Descent
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv


def compute_cost(X, y, learning_rate, weigths, lmda):
    """ total Least Square Error """
    """ X.shape[0]) is number of rows """
    slope = 1. / (2. * X.shape[0])
    prediction = X @ weigths
    error = prediction - y
    summation = np.sum((error ** 2))
    # print(summation)
    ridge_coeff = np.sum(weigths.T @ weigths)
    c = (slope) * (summation + lmda * ridge_coeff)
    return c


def gradientDescent(iters, learning_rate, lmda, X, y):
    weights = np.zeros([X.shape[1]+1])
    cost = np.zeros([iters])
    for i in range(iters):
        error = y - X @ weights
        # print(y.shape)
        error_der = X.T @ error
        # print(error.shape)
        weights = weights - learning_rate*error_der/len(y)
        cost[i] = compute_cost(X, y, learning_rate, weights, lmda)

    return weights, cost


# def predict(X, theta):
#     """ Predict values for given X """
#     Xn = np.ndarray.copy(X)

#     Xn -= X_mean
#     Xn /= X_std
#     Xn = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn))

#     return Xn.dot(theta) + y_mean


if __name__ == "__main__":
    data = pd.read_csv('data.csv',  sep=';', decimal=',')
    X = data.iloc[:, 0:-1].values
    Y = data.iloc[:, -1].values
    train = df.sample(frac=0.8)
    test = df.drop(train.index)

    X_train = train.iloc[:, :-1]
    X_test = test.iloc[:, :-1]
    y_train = train['Slowness in traffic (%)']
    y_test = test['Slowness in traffic (%)']

    lmda = 0.01
    z, cost = gradientDescent(30, 0.0001, lmda.01, X_train, y_train)
    # x_coordinate = [i for i in range(len(cost))]
    # plt.plot(x_coordinate, cost)
    # plt.show()
    print(cost[1])
    print(cost[-1])

    print(X.shape)
