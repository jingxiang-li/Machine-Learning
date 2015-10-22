"""This file contains class SVM_SMO which implements the SMO algorithm for training (linear) SVM

    Jingxiang Li
    Wed 21 Oct 2015 12:08:09 AM CDT
"""
from numpy import mat, array, zeros, repeat, dot, unique, genfromtxt, minimum, sum, copy, maximum, argmax, fabs, concatenate, arange
from numpy.random import choice, permutation, uniform
from math import sqrt
from data_preprocessor import Data_Preprocessor
from classifier import Classifier


class SVM_SMO (Classifier):
    """class SVM_SMO which implements the SMO algorithm for training (linear) SVM

    Attributes:
        data_preprocessor (Data_Preprocessor): a Data_Preprocessor instance
        loglist (list): store dual objective function value for each iteration
        X (numpy.array): Design Matrix
        y (numpy.array): Response Vector
        y_vals (numpy.array): class labels
    """

    def __init__(self, X, y, C, iter_max):
        """initialzie the classifier and train the model

        Args:
            X (numpy.array): Design Matrix
            y (numpy.array): Response Vector
            C (float): penalty parameter
            iter_max (int): maximum interation for the algorithm
        """
        self.data_preprocessor = Data_Preprocessor(X)
        self.X = self.data_preprocessor.predict(X)
        self.y = copy(y)
        self.y_vals = unique(self.y)
        self.y[self.y == self.y_vals[0]] = -1
        self.y[self.y == self.y_vals[1]] = 1
        self.loglist = []
        self.alpha_array, self.b = self.train_svm(self.X, self.y, C, iter_max)

    def predict(self, X, output=0):
        """make prediction for the new Design Matrix X

        Args:
            X (numpy.array): new Design Matrix
            output (int, optional): Description

        Returns:
            numpy.array: vector of prediction class
        """
        X = self.data_preprocessor.predict(X)
        predicted_score = self.predict_score(X)
        predicted_class = self.predict_class(predicted_score)
        return predicted_class

    def validate(self, X, y, output=0):
        """validate prediction result for the new Desgin Matrix X and new Response Vector y

        Args:
            X (numpy.array): new Design Matrix
            y (numpy.array): new Response Vector
            output (int, optional): Description

        Returns:
            float: prediction error on the new Inputs
        """
        X = self.data_preprocessor.predict(X)
        predicted_score = self.predict_score(X)
        predicted_class = self.predict_class(predicted_score)
        prediction_error = self.calc_predict_error(predicted_class, y)
        return prediction_error

    def predict_score(self, X):
        """calculate prediction score for new Design Matrix X

        Args:
            X (numpy.array): new Design Matrix

        Returns:
            numpy.array: vector of prediction score
        """
        return self.get_output(X, self.alpha_array, self.X, self.y, self.b)

    def predict_class(self, predicted_score):
        """predict the class label for each observation in the new Design Matrix X

        Args:
            predicted_score (numpy.array): vector of prediction score

        Returns:
            numpy.array: vector of predicted class label
        """
        return [self.y_vals[0] if s < 0 else self.y_vals[1] for s in predicted_score]

    def calc_predict_error(self, predicted_class, y):
        """Calculate the prediction error

        Args:
            predicted_class (numpy.array): vector of predicted class label
            y (numpy.array): vector of true class label

        Returns:
            float: overall error rate
        """
        predicted_indicator = array(
            [predicted_class[i] == y[i] for i in range(0, y.size)])
        return 1 - sum(predicted_indicator) / y.size

    def get_output(self, x_new, alpha_array, X, y, b):
        """Calculate f(x_new)

        Args:
            x_new (numpy.array): new design matrix
            alpha_array (numpy.array): current alpha array
            X (numpy.array): Design Matrix
            y (numpy.array): Response Vector
            b (list): threshold

        Returns:
            numpy.array: f(x_new)
        """
        signed_y = y * alpha_array
        return dot(dot(signed_y, X), x_new.T) - b

    def calc_objective_fast(self, alpha_array, y, K):
        """Calculate the dual objective function

        Args:
            alpha_array (numpy.array): current alpha array
            y (numpy.array): Response Vector
            K (numpy.array): kernel Matrix dot(X, X.T)

        Returns:
            float: dual objective function value
        """
        signed_y = y * alpha_array
        return sum(alpha_array) - 0.5 * dot(dot(signed_y, K), signed_y)

    def calc_LH(self, alpha1, alpha2, y1, y2, C):
        """Calculate the lower and upper bound for new alpha2

        Args:
            alpha1 (float):
            alpha2 (float):
            y1 (numpy.array):
            y2 (numpy.array):
            C (float):

        Returns:
            (L, H):
        """
        L = H = 0
        if y1 != y2:
            L = maximum(0, alpha2 - alpha1)
            H = minimum(C, C + alpha2 - alpha1)
        else:
            L = maximum(0, alpha1 + alpha2 - C)
            H = minimum(C, alpha1 + alpha2)
        return (L, H)

    def choose_i1(self, i2, alpha_index_nonbound, alpha_array, X, y, b):
        """Heuristically choose alpha1 to optimize given alpha2

        Args:
            i2 (int): index of alpha2
            alpha_index_nonbound (numpy.array): array of nonbound alpha's index
            alpha_array (numpy.array): current alpha array
            X (numpy.array):
            y (numpy.array):
            b (list):

        Returns:
            int: index of alpha1
        """
        E2 = self.get_output(X[i2, :], alpha_array, X, y, b) - y[i2]
        E1_array = self.get_output(
            X[alpha_index_nonbound, :], alpha_array, X, y, b) - y[alpha_index_nonbound]
        index = argmax(fabs(E1_array - E2[0]))
        return alpha_index_nonbound[index]

    def update_b(self, b_old, alpha1, alpha2, a1, a2, x1, x2, y1, y2, E1, E2, C):
        """update the threshold parameter b

        Args:
            b_old (list):
            alpha1 (float):
            alpha2 (float):
            a1 (float):
            a2 (float):
            x1 (numpy.array):
            x2 (numpy.array):
            y1 (int):
            y2 (int):
            E1 (float):
            E2 (float):

        Returns:
            float: updated parameter b
        """
        b1 = E1 + y1 * (a1 - alpha1) * dot(x1, x1) + y2 * \
            (a2 - alpha2) * dot(x1, x2) + b_old
        b2 = E2 + y1 * (a1 - alpha1) * dot(x1, x2) + y2 * \
            (a2 - alpha2) * dot(x2, x2) + b_old
        if (a1 == C or a1 == 0) and (a2 == C or a2 == 0):
            return (b1 + b2) / 2
        elif a1 == C or a1 == 0:
            return b2
        else:
            return b1

    def take_step(self, i1, i2, alpha_array, X, y, C, b, K):
        """Optimize given alpha1 and alpha2

        Args:
            i1 (int): index of alpha1
            i2 (int): index of alpha2
            alpha_array (numpy.array): current alpha array
            X (numpy.array):
            y (numpy.array):
            C (float):
            b (list):
            K (numpy.array):

        Returns:
            boolean: True if we optimize the alpha pairs, otherwise False
        """
        if i1 == i2:
            return False

        eps = 1e-05
        alpha1, alpha2 = alpha_array[[i1, i2]]
        y1, y2 = y[[i1, i2]]
        X1, X2 = X[[i1, i2]]
        E1 = self.get_output(X1, alpha_array, X, y, b) - y1
        E2 = self.get_output(X2, alpha_array, X, y, b) - y2
        s = y1 * y2
        L, H = self.calc_LH(alpha1, alpha2, y1, y2, C)

        if fabs(L - H) < eps:
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
            Lobj = self.calc_objective_fast(alpha_array, X, y, K)
            alpha_array[i2] = alpha2

            alpha_array[i2] = H
            Hobj = self.calc_objective_fast(alpha_array, X, y, K)
            alpha_array[i2] = alpha2

            if Lobj > Hobj + eps:
                a2 = L
            elif Lobj < Hobj - eps:
                a2 = H
            else:
                a2 = alpha2

        if a2 < eps:
            a2 = 0
        elif a2 > C - eps:
            a2 = C

        if fabs(a2 - alpha2) < eps * (a2 + alpha2 + eps):
            return False

        a1 = alpha1 + s * (alpha2 - a2)

        # update b
        b_old = b[0]
        b_new = self.update_b(b_old, alpha1, alpha2, a1,
                              a2, X1, X2, y1, y2, E1, E2, C)
        b[0] = b_new[0]

        alpha_array[i1] = a1
        alpha_array[i2] = a2
        return True

    def examine_example(self, i2, alpha_array, X, y, C, b, K):
        """Given alpha2, find alpha1 and optimize them

        Args:
            i2 (index): index of alpha2
            alpha_array (numpy.array): current array of alpha
            X (numpy.array):
            y (numpy.array):
            C (float):
            b (list):
            K (numpy.array):

        Returns:
            boolean: True if we optimize alpha2, otherwise False
        """
        n, p = X.shape
        y2 = y[i2]
        alpha2 = alpha_array[i2]
        X2 = X[i2]
        E2 = self.get_output(X2, alpha_array, X, y, b) - y2
        r2 = E2 * y2
        tol = 1e-03

        if (r2 < -tol and alpha2 < C) or (r2 > tol and alpha2 > 0):
            # find thoses nonbound alphas
            alpha_index_nonbound = [i for i in range(n)
                                    if alpha_array[i] != 0 and alpha_array[i] != C]
            num_nonbound = len(alpha_index_nonbound)

            if num_nonbound > 1:
                # heuristicly choose i1
                i1 = self.choose_i1(
                    i2, alpha_index_nonbound, alpha_array, X, y, b)
                if i1 >= 0 and self.take_step(i1, i2, alpha_array, X, y, C, b, K):
                    return True

                # iterate over all nonbound alphas
                start_index = choice(len(alpha_index_nonbound), 1)
                alpha_index_nonbound_modified = concatenate(
                    (alpha_index_nonbound[start_index:],
                     alpha_index_nonbound[0:start_index]))
                for i1 in alpha_index_nonbound_modified:
                    if self.take_step(i1, i2, alpha_array, X, y, C, b, K):
                        return True

            # iterate over all alphas
            start_index = choice(n, 1)
            alpha_index_modified = concatenate(
                (arange(start_index, n),
                 arange(start_index, 2)))
            for i1 in range(n):
                if self.take_step(i1, i2, alpha_array, X, y, C, b, K):
                    return True

        return False

    def train_svm(self, X, y, C, iter_max):
        """Train SVM using SMO algorithm

        Args:
            X (numpy.array): Design Matrix
            y (numpy.array): Response Vector {-1, 1}
            C (float): penalty parameter
            iter_max (int): maximum number of iterations

        Returns:
            (numpy.array, list): (alpha_array, b), trained parameters
        """
        n, p = X.shape
        K = dot(X, X.T)
        alpha_array = zeros(n)
        b = [0]
        num_changed = 0
        examine_all = True

        while num_changed > 0 or examine_all:
            num_changed = 0

            if examine_all:
                alpha_index = permutation(range(n))
                for i2 in alpha_index:
                    if self.examine_example(i2, alpha_array, X, y, C, b, K):
                        num_changed += 1
                        self.loglist.append(
                            self.calc_objective_fast(alpha_array, y, K))
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
                    if self.examine_example(i2, alpha_array, X, y, C, b, K):
                        num_changed += 1
                        self.loglist.append(
                            self.calc_objective_fast(alpha_array, y, K))
                        iter_max -= 1
                        if iter_max < 0:
                            break
                if iter_max < 0:
                    break

            if num_changed < n / 10:
                break

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

        return (alpha_array, b)


# data_MNIST = genfromtxt('../res/MNIST-13.csv', delimiter=',')
# X = data_MNIST[:, 1:]
# y = data_MNIST[:, 0]
# model = SVM_SMO(X, y, 1, 2000)
# print(model.loglist)
# print(model.validate(X, y))
# yhat = model.get_output(model.X, model.alpha_array, model.X, model.y, model.b)
# print(sum(model.y * yhat > 1))
