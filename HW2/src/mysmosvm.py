"""This file contains functions for CSci5525 HW2

    Jingxiang Li
    Wed 21 Oct 2015 01:09:41 AM CDT
"""
from svm_smo import SVM_SMO
from numpy import genfromtxt, zeros, mean, std, array, savetxt
import sys
import time
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


def mysmosvm(filename, numruns):
    """Train SVM with SMO algorithm several times and record the training time

    Args:
        filename (string): path to the dataset
        numruns (int): number of times model will be trained

    Returns:
        none:
    """
    numruns = int(numruns)
    assert(numruns > 0)
    data = read_file(filename)
    X = data[:, 1:]
    y = data[:, 0]
    C = 1
    iter_max = 2000
    time_array = zeros(numruns)
    log_list = []

    for i in range(numruns):
        print("Training model ", i + 1, "/",
              numruns, ", please wait...", end="\r")
        begin = time.time()
        model = SVM_SMO(X, y, C, iter_max)
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
    print("Avg runtime for ", numruns, " runs:\t", round(time_avg, 1), " sec.")
    print("Std runtime for ", numruns, " runs:\t", round(time_std, 1), " sec.")
    print("Plot data exported to ./tmp.txt")

    return


def main(argv=sys.argv):
    """main function

    Args:
        argv (Name, optional):

    Returns:
        none:
    """
    if len(argv) == 3:
        mysmosvm(*argv[1:])
    else:
        print('Usage: python3 ./mysmosvm.py /path/to/dataset.csv numruns',
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
