""" This file contains function myBagging2
Author: Jingxiang Li
Date: Wed 11 Nov 2015 09:06:47 PM CST
"""

from randomForest import RandomForest
from cross_validation import n_fold_cv
import sys
import os.path
from numpy import genfromtxt, mean, std, array


def read_file(filename):
    assert os.path.isfile(filename)
    print("Loading", filename)
    data = genfromtxt(filename, delimiter=",", skip_header=0)
    print("Successfully Loaded", filename)
    print("")
    return data


def myBagging(X, y, B, k):

    assert(B > 0 and k > 0)

    n, p = X.shape
    num_trees = B
    num_features = p
    sample_size = int(n * (k - 1) / k)
    depth = 2
    test_error, train_error = n_fold_cv(
        X, y, RandomForest, k, 1, (num_trees, num_features, sample_size, depth))
    for i in range(k):
        print("Train error for fold", i + 1, "with", B,
              "base classifiers: ", round(train_error[0, i], 3))
        print("Test  error for fold", i + 1, "with", B,
              "base classifiers: ", round(test_error[0, i], 3))
    print("------------------------------------------------------")


def myBagging2(filename, B_vec, k):
    k = int(k)
    data = read_file(filename)
    n, p = data.shape
    X = data[:, :(p - 1)]
    y = array(data[:, p - 1], dtype='int')

    for B in B_vec:
        myBagging(X, y, B, k)


def main(argv=sys.argv):
    if len(argv) == 4:
        B_str = argv[2]
        B_str = B_str.replace('[', '')
        B_str = B_str.replace(']', '')
        B_str = B_str.replace(',', ' ')
        B_vec = array(B_str.split(), dtype=int)
        myBagging2(argv[1], B_vec, argv[3])
    else:
        print('Usage: python3 ./myBagging2.py /path/to/dataset.csv number_of_base_learner, num_folds', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
