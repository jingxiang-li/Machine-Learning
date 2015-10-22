"""This file contains functions for CSci5525 HW2

    Jingxiang Li
    Wed 21 Oct 2015 01:09:41 AM CDT
"""
from svm_sgd import SVM_SGD
from numpy import genfromtxt, zeros, mean, std, array, savetxt, around
import sys
import time
import pickle
from os.path import isfile


def read_file(filename):
    """Read Data from file

    Args:
        filename (string): path to the file

    Returns:
        numpy.array: dataset
    """
    assert isfile(filename)
    print("Loading", filename)
    data = genfromtxt(filename, delimiter=",", skip_header=0)
    print("Successfully Loaded", filename)
    print("")
    return data


def mysgdsvm(filename, k, numruns):
    """Train SVM with Pegasos algorithm several times and record the training time

    Args:
        filename (string): path to the dataset
        k (int): maximum training sample size for each iteration
        numruns (int): number of times model will be trained

    Returns:
        none:
    """
    numruns = int(numruns)
    k = int(k)
    assert(numruns > 0 and k > 0)
    data = read_file(filename)
    X = data[:, 1:]
    y = data[:, 0]
    para_lambda = 1
    time_array = zeros(numruns)
    log_list = []

    for i in range(numruns):
        print("Training model ", i + 1, "/",
              numruns, ", please wait...", end="\r")
        begin = time.time()
        model = SVM_SGD(X, y, para_lambda, k)
        log_list.append(model.loglist)
        end = time.time()
        time_array[i] = end - begin

    time_avg = mean(time_array)
    time_std = std(time_array, ddof=1)
    with open("./tmp.txt", "w") as f:
        for li in log_list:
            for val in li:
                print(val, end=", ", file=f)
            print(file=f)

    print()
    print()
    print("Avg runtime for ", numruns, " runs with minibatch size of ", k, ":\t", round(time_avg, 1), " sec.")
    print("Std runtime for ", numruns, " runs with minibatch size of ", k, ":\t", round(time_std, 1), " sec.")
    print("Plot data exported to ./tmp.txt")

    return


def main(argv=sys.argv):
    """main function

    Args:
        argv (Name, optional)

    Returns:
        none:
    """
    if len(argv) == 4:
        mysgdsvm(*argv[1:])
    else:
        print(
            'Usage: python3 ./mysgdsvm.py /path/to/dataset.csv k numruns', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
