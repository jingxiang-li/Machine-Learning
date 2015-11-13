# This file contains functions for cross validation
import numpy as np
from numpy.random import randint
from math import ceil


def n_fold_cv(X, y, Classifier_, n_fold, n_rep, args=[]):
    # print("Running", n_fold, "fold cross validation with", n_rep, "replicates for", Classifier_.__name__)
    n_obs = y.size
    train_error_mat = np.zeros((n_rep, n_fold))
    test_error_mat = np.zeros((n_rep, n_fold))
    index = np.arange(n_obs)
    for rep in range(0, n_rep):
        # print("Runnnig replicate No.", rep, end="\r")
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
            train_error_mat[rep, fold] = model.validate(X_train, y_train)
            test_error_mat[rep, fold] = model.validate(X_test, y_test)
    # print("Cross Validation Complete")
    return (test_error_mat, train_error_mat)


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
