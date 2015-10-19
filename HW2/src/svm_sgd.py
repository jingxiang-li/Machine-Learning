"""Summary"""
import numpy as np
import scipy as sp
from numpy import mat, array, zeros, repeat, dot, unique
from numpy.random import choice
from math import sqrt
from data_preprocessor import Data_Preprocessor


def select_workset(X, y, weight, k):
    """Select the workset used for the algorithm,
    We will first select k observations from the entire dataset,
    and then choose those satisfying y <w, x> < 1

    Args:
        X (numpy.array): Design Matrix
        y (numpy.array): Response vector {-1, 1}
        weight (numpy.array): vector whose dimension should match the number of columns of X
        k (int): number of observations to be potentially selected

    Returns:
        (numpy.array, numpy.array): selected working subset of X and y
    """
    n, p = X.shape
    index = choice(n, k)
    X_sub = X[index, :]
    y_sub = y[index]
    sub_index = (dot(X_sub, weight) * y_sub) < 1
    index = index[sub_index]
    return (X[index, :], y[index])


def initialize_weight(p, para_lambda):
    """Initialzie the weight vector used for the algorithm

    Args:
        p (int): number of features, number of cols in X
        para_lambda (float): regularization parameter

    Returns:
        numpy.array: a p dimensional weight vector used for the algorithm
    """
    weight = zeros(p)
    weight.fill(sqrt(1 / (p * para_lambda)))
    neg_index = choice(p, size=(int)(p / 2))
    weight[neg_index] = -weight[neg_index]
    return weight


def update_weight(X, y, weight, para_lambda, k, iter_num):
    """update the weight vector

    Args:
        X (numpy.array): Design Matrix
        y (numpy.array): Response vector {-1, 1}
        weight (numpy.array): vector whose dimension should match the number of columns of X
        para_lambda (float): regularization parameter
        k (int): number of observations to be potentially selected in this iteration
        iter_num (int): current iteration number

    Returns:
        numpy.array: the updated weight vector
    """
    eta = 1 / (para_lambda * iter_num)  # step size
    weight_half = (1 - eta * para_lambda) * weight + eta / k * dot(y, X)
    weight_new = np.minimum(1, 1 / sqrt(para_lambda) /
                            sqrt(sum(weight_half ** 2))) * weight_half
    return weight_new


def calc_loss_function(X, y, weight):
    n, p = X.shape
    tmp_loss = 1 - y * dot(X, weight)
    loss = sum(tmp_loss[tmp_loss > 0]) / n
    return loss

data_MNIST = np.genfromtxt('../res/MNIST-13.csv', delimiter=',')
X = data_MNIST[:, 1:]
data_preprocessor = Data_Preprocessor(X)
X = data_preprocessor.predict(X)
print(X.shape)
y = data_MNIST[:, 0]
y_vals = unique(y)
y[y == y_vals[0]] = -1
y[y == y_vals[1]] = 1
n, p = X.shape
para_lambda = 1
weight = initialize_weight(p, para_lambda)
k = 50

for i in range(1, 10000):
    X_work, y_work = select_workset(X, y, weight, k)
    print(calc_loss_function(X, y, weight))
    weight_new = update_weight(X_work, y_work, weight, para_lambda, k, i)
    if sqrt(sum((weight_new - weight) ** 2)) < 0.001:
        break
    else:
        weight = weight_new

