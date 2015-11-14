from numpy import arange, array, zeros, genfromtxt, mean, std, around, savetxt, argmin
from cross_validation import n_fold_cv
from randomForest import RandomForest
import matplotlib.pyplot as plt


def myRForest(X, y, B, k=10):
    print(B)
    n, p = X.shape
    num_trees = B
    num_features = p
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

    B_vec = arange(5, 51, 5)
    rf_result = zeros((24, B_vec.size))

    for i in range(B_vec.size):
        train_error, test_error = myRForest(X, y, B_vec[i])
        i_train = 0
        i_test = 0
        j = 0
        while j != 20:
            rf_result[j, i] = train_error[i_train]
            rf_result[j + 1, i] = test_error[i_test]
            j += 2
            i_train += 1
            i_test += 1
        rf_result[20, i] = mean(train_error)
        rf_result[21, i] = std(train_error, ddof=1)
        rf_result[22, i] = mean(test_error)
        rf_result[23, i] = std(test_error, ddof=1)

    return rf_result


rf_result = print_mat()
savetxt("./bg_result.csv", rf_result, delimiter=',')
print(rf_result)

# B_vec = arange(5, 51, 5)
# rf_result = genfromtxt('./bg_result.csv', delimiter=',')
# n, p = rf_result.shape
# plt.plot(B_vec, rf_result[20, :])

# i = argmin(rf_result[20, :])
# astr = 'Minimum Achieved\n (' + str(B_vec[i]) + ', ' + str(round(rf_result[20, i], 3)) + ')'
# plt.annotate(astr,
#              xy=(B_vec[i], rf_result[20, i]),
#              xytext=(B_vec[i] + 1, rf_result[20, i] + 0.05), arrowprops=dict(facecolor='black', shrink=0.05, width=2))
# plt.ylim(0.05, 0.3)
# plt.ylabel('Average Error Rate')
# plt.xlabel('Number of Trees')
# plt.title('Bagged 2-layer decision trees\n Average Error Rate v.s. number of base classifiers')
# # plt.grid(True)
# plt.tight_layout()
# plt.savefig('../tex/figure/bg.pdf', transparent=True, dpi=600)
