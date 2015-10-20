import numpy as np
from numpy import mat, array, zeros, repeat, dot, unique, genfromtxt, minimum, sum, copy, maximum
from numpy.random import choice, permutation, uniform
from math import sqrt, fabs
from data_preprocessor import Data_Preprocessor


def calc_loss(alpha_array, X, y):
    signed_y = y * alpha_array
    y_hat = dot(X, dot(signed_y, X))
    tmp_loss = 1 - y * y_hat
    loss = sum(tmp_loss[tmp_loss > 0]) / n
    return loss


def get_output(x_new, alpha_array, X, y):
    signed_y = y * alpha_array
    return dot(dot(signed_y, X), x_new)


def calc_objective(alpha_array, X, y):
    K = dot(X, X.T)
    signed_y = y * alpha_array
    return sum(alpha_array) - 0.5 * dot(dot(signed_y, K), signed_y)


def calc_objective_fast(alpha_array, y, K):
    signed_y = y * alpha_array
    return sum(alpha_array) - 0.5 * dot(dot(signed_y, K), signed_y)


def calc_LH(alpha1, alpha2, y1, y2, C):
    L = H = 0
    if y1 != y2:
        L = maximum(0, alpha2 - alpha1)
        H = minimum(C, C + alpha2 - alpha1)
    else:
        L = maximum(0, alpha1 + alpha2 - C)
        H = minimum(C, alpha1 + alpha2)
    return (L, H)


def choose_i1(i2, alpha_index_nonbound, alpha_array, X, y):
    E2 = get_output(X[i2, :], alpha_array, X, y) - y[i2]
    max_Ediff = -1
    i1_ret = -1
    for i1 in alpha_index_nonbound:
        E1 = get_output(X[i1, :], alpha_array, X, y) - y[i1]
        if fabs(E1 - E2) > max_Ediff:
            max_Ediff = fabs(E1 - E2)
            i1_ret = i1
    return i1_ret


def take_step(i1, i2, alpha_array, X, y, C):
    if i1 == i2:
        return False

    eps = 7. / 3 - 4. / 3 - 1
    alpha1, alpha2 = alpha_array[[i1, i2]]
    y1, y2 = y[[i1, i2]]
    X1, X2 = X[[i1, i2]]
    E1 = get_output(X1, alpha_array, X, y) - y1
    E2 = get_output(X2, alpha_array, X, y) - y2
    s = y1 * y2
    L, H = calc_LH(alpha1, alpha2, y1, y2, C)

    if L == H:
        return False

    k11 = dot(X1, X1)
    k12 = dot(X1, X2)
    k22 = dot(X2, X2)
    eta = 2 * k12 - k11 - k22

    if eta < 0:
        a2 = alpha2 - y2 * (E1 - E2) / eta
        if a2 < L:
            a2 = L
        elif a2 > H:
            a2 = H
    else:
        alpha_array[i2] = L
        Lobj = calc_objective(alpha_array, X, y)
        alpha_array[i2] = alpha2

        alpha_array[i2] = H
        Hobj = calc_objective(alpha_array, X, y)
        alpha_array[i2] = alpha2

        if Lobj > Hobj + eps:
            a2 = L
        elif Lobj < Hobj - eps:
            a2 = H
        else:
            a2 = alpha2

    if a2 < 1e-08:
        a2 = 0
    elif a2 > C - 1e-08:
        a2 = C

    if fabs(a2 - alpha2) < eps * (a2 + alpha2 + eps):
        return False

    a1 = alpha1 + s * (alpha2 - a2)

    alpha_array[i1] = a1
    alpha_array[i2] = a2
    return True


def examine_example(i2, alpha_array, X, y, C):
    n, p = X.shape
    y2 = y[i2]
    alpha2 = alpha_array[i2]
    X2 = X[i2]
    E2 = get_output(X2, alpha_array, X, y) - y2
    r2 = E2 * y2
    tol = 1e-03

    if (r2 < -tol and alpha2 < C) or (r2 > tol and alpha2 > 0):
        # find thoses nonbound alphas
        alpha_index_nonbound = [i for i in range(n)
                                if alpha_array[i] != 0 and alpha_array[i] != C]
        num_nonbound = len(alpha_index_nonbound)

        if num_nonbound > 1:
            # heuristic choose i1
            i1 = choose_i1(i2, alpha_index_nonbound, alpha_array, X, y)
            if i1 >= 0 and take_step(i1, i2, alpha_array, X, y, C):
                return True

            # iterate over all nonbound alphas
            alpha_index_nonbound = permutation(alpha_index_nonbound)
            for i1 in alpha_index_nonbound:
                if take_step(i1, i2, alpha_array, X, y, C):
                    return True

        # iterate over all alphas
        alpha_index = permutation(range(n))
        for i1 in alpha_index:
            if take_step(i1, i2, alpha_array, X, y, C):
                return True

    return False


def train_svm(X, y, C):
    n, p = X.shape
    K = dot(X, X.T)
    alpha_array = zeros(n)
    num_changed = 0
    examine_all = True

    iter_max = 100
    while num_changed > 0 or examine_all:
        num_changed = 0

        if examine_all:
            alpha_index = permutation(range(n))
            for i2 in alpha_index:
                if examine_example(i2, alpha_array, X, y, C):
                    num_changed += 1
                    print(calc_objective_fast(alpha_array, y, K))
                    print(calc_loss(alpha_array, X, y))
                    iter_max -= 1
                    if iter_max < 0:
                        break
            if iter_max < 0:
                break
        else:
            alpha_index_nonbound = [i for i in range(n)
                                    if alpha_array[i] != 0 and alpha_array[i] != C]
            alpha_index_nonbound = permutation(alpha_index_nonbound)
            for i2 in alpha_index_nonbound:
                if examine_example(i2, alpha_array, X, y, C):
                    num_changed += 1
                    print(calc_objective(alpha_array, X, y))
                    iter_max -= 1
                    if iter_max < 0:
                        break
            if iter_max < 0:
                break

        if examine_all:
            examine_all = False
        elif num_changed == 0:
            examine_all = True
    return alpha_array


data_MNIST = genfromtxt('../res/MNIST-13.csv', delimiter=',')
X = data_MNIST[:, 1:]
data_preprocessor = Data_Preprocessor(X, False)
X = data_preprocessor.predict(X)
print(X.shape)
y = data_MNIST[:, 0]
y_vals = unique(y)
y[y == y_vals[0]] = -1
y[y == y_vals[1]] = 1
y = array(y, dtype=int)
n, p = X.shape
C = 1
alpha_array = train_svm(X, y, C)
