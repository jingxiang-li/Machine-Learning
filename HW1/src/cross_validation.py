# This file contains functions for cross validation
import numpy as np
from least_square_discriminant import Least_Square_Discriminant
from logistic_regression import Logistic_Regression
from naive_bayes import Naive_Bayes
from multivariate_gaussian import Multivariate_Gaussian
from numpy.random import randint
from math import ceil


def n_fold_cv(X, y, Classifier_, n_fold, n_rep, args=[]):
    n_obs = y.size
    error_mat = np.zeros((n_rep, n_fold))
    index = np.arange(n_obs)
    for rep in range(0, n_rep):
        print("Replicate No. ", rep)
        np.random.shuffle(index)
        index = index[np.argsort(y[index], kind='mergesort')]
        for fold in range(0, n_fold):
            train_index = np.remainder(index, n_fold) != fold
            test_index = np.remainder(index, n_fold) == fold
            X_train = X[train_index, :]
            y_train = y[train_index]
            X_test = X[test_index, :]
            y_test = y[test_index]
            model = Classifier_(X_train, y_train, *args)
            error_mat[rep, fold] = model.validate(X_test, y_test)
    # print(error_mat)
    return error_mat


def fancy_cv(X, y, Classifier_, n_rep, train_percent, args=[]):
    n_obs = y.size
    error_mat = np.zeros((n_rep, train_percent.size))
    for rep in range(0, n_rep):
        print("Replicate No. ", rep)
        # here slipe the whole set as 80% train 20test
        train_index, test_index = train_test_index(y, .8)
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]
        for p in range(0, train_percent.size):
            percent = train_percent[p] / 100
            train_sub_index = train_test_index(y_train, percent)[0]
            X_train_sub = X_train[train_sub_index, :]
            y_train_sub = y_train[train_sub_index]
            model = Classifier_(X_train_sub, y_train_sub, *args)
            error_mat[rep, p] = model.validate(X_test, y_test)
    # print(error_mat)
    return error_mat


def train_test_index(y, train_percent):
    num_obs = y.size
    y_index = np.arange(num_obs)
    y_vals = np.unique(y)
    train_index = np.array([], dtype=int)
    for k in range(0, y_vals.size):
        y_sub_index = y_index[y == y_vals[k]]
        num_train_k = ceil(y_sub_index.size * train_percent)
        train_index_k = np.random.choice(y_sub_index, num_train_k, replace=False)
        train_index = np.concatenate((train_index, train_index_k))
    test_index = np.setdiff1d(y_index, train_index, assume_unique=True)
    return (train_index, test_index)


data_spam = np.genfromtxt('../dataset/spam.csv', delimiter=",", skip_header=0)
X = data_spam[:, 1:]
y = data_spam[:, 0].astype(int)
result = n_fold_cv(X, y, Logistic_Regression, 10, 2, [1])
print(np.mean(result, axis=0))
print(np.std(result, axis=0))

# data_MNIST = np.load('data_MNIST.npy')
# X = data_MNIST[:, 1:]
# y = data_MNIST[:, 0].astype(int)
# result = n_fold_cv(X, y, Least_Square_Discriminant, 10, 1)
# print(result)

# # result = fancy_cv(X, y, Naive_Bayes, 30, np.array([1, 2, 3, 5, 10]))
# print(np.mean(result, axis=0))

# print("Reading data now ...")
# data_MNIST = np.load('data_MNIST.npy')
# print("Complete reading data")
# X = data_MNIST[:, 1:]
# y = data_MNIST[:, 0].astype(int)
# n_fold_cv(X, y, Least_Square_Discriminant, 5, 1)
