""" This file contains function myRForest2.py
Author: Jingxiang Li
Date: Wed 11 Nov 2015 09:06:52 PM CST
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


def myRForest(X, y, M, k):

    assert(M > 0 and k > 0)

    n, p = X.shape
    num_trees = 100
    num_features = M
    sample_size = 30
    depth = 2
    test_error, train_error = n_fold_cv(
        X, y, RandomForest, k, 1, (num_trees, num_features, sample_size, depth))
    for i in range(k):
        print("Train error for fold", i + 1, "with", M,
              "random features: ", round(train_error[0, i], 3))
        print("Test  error for fold", i + 1, "with", M,
              "random features: ", round(test_error[0, i], 3))
    print("------------------------------------------------------")


def myRForest2(filename, M_vec, k):
    k = int(k)
    data = read_file(filename)
    n, p = data.shape
    X = data[:, :(p - 1)]
    y = array(data[:, p - 1], dtype='int')

    for M in M_vec:
        myRForest(X, y, M, k)


def main(argv=sys.argv):
    if len(argv) == 4:
        M_str = argv[2]
        M_str = M_str.replace('[', '')
        M_str = M_str.replace(']', '')
        M_str = M_str.replace(',', ' ')
        M_vec = array(M_str.split(), dtype=int)
        myRForest2(argv[1], M_vec, argv[3])
    else:
        print('Usage: python3 ./myRForest2.py /path/to/dataset.csv number_of_features, num_folds', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
