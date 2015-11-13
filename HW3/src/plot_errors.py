from numpy import arange, array, zeros, genfromtxt, mean, std, around
from cross_validation import n_fold_cv
from randomForest import RandomForest


def myRForest(X, y, M, k=10):
    print(M)
    n, p = X.shape
    num_trees = 100
    num_features = M
    sample_size = int(n * (k - 1) / k)
    depth = 2
    test_error, train_error = n_fold_cv(
        X, y, RandomForest, k, 1, (num_trees, num_features, sample_size, depth))
    return (train_error[0, :], test_error[0, :])


def print_mat():
    data = genfromtxt('../res/ionoshpere3.txt', delimiter=',')
    n, p = data.shape
    X = data[:, :(p - 1)]
    y = array(data[:, p - 1], dtype='int')
    n, p = X.shape

    M_vec = arange(1, p)
    rf_result = zeros((22, M_vec.size))

    for i in range(M_vec.size):
        train_error, test_error = myRForest(X, y, M_vec[i])
        i_train = 0
        i_test = 0
        j = 0
        while j != 20:
            rf_result[j, i] = train_error[i_train]
            rf_result[j + 1, i] = test_error[i_test]
            j += 2
            i_train += 1
            i_test += 1
        rf_result[20, i] = mean(rf_result[:20, i])
        rf_result[21, i] = std(rf_result[:20, i], ddof=1)

    print(rf_result)


print_mat()
