"""This file contains class SVM_SGD which implements the Pegasos algorithm for training linear SVM.

    Jingxiang Li
    Tue 20 Oct 2015 11:18:16 PM CDT
"""
from numpy import mat, array, zeros, repeat, dot, unique, genfromtxt, minimum, maximum, copy
from numpy.random import choice
from math import sqrt
from data_preprocessor import Data_Preprocessor
from classifier import Classifier


class SVM_SGD (Classifier):
    """class SVM_SGD which implements the Pegasos algorithm for training linear SVM

    Attributes:
        data_preprocessor (Data_Preprocessor): a Data_Preprocessor instance
        loglist (list): store primal objective function value for each iteration
        weight (numpy.array): trained weight
        y_vals (numpy.array): class labels
    """

    def __init__(self, X, y, para_lambda, k):
        """initialzie the classifier and train the model

        Args:
            X (numpy.array): Design Matrix
            y (numpy.array): Response Vector
            para_lambda (float): regularization parameter
            k (int): maximum training sample size for each iteration
        """
        self.data_preprocessor = Data_Preprocessor(X)
        X = self.data_preprocessor.predict(X)
        y = copy(y)
        self.y_vals = unique(y)
        y[y == self.y_vals[0]] = -1
        y[y == self.y_vals[1]] = 1
        self.loglist = []
        self.weight = self.calc_weight(X, y, para_lambda, k)

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

    def calc_weight(self, X, y, para_lambda, k):
        """estimate the weight vector given X, y, para_lambda and k

        Args:
            X (numpy.array): Design Matrix
            y (numpy.array): Response Vector
            para_lambda (float): regularization parameter
            k (int): maximum training sample size for each iteration

        Returns:
            numpy.array: the trained weight vector
        """
        n, p = X.shape
        weight = self.initialize_weight(p, para_lambda)
        for i in range(1, 10000):
            X_work, y_work = self.select_workset(X, y, weight, k)
            self.loglist.append(self.calc_loss_function(
                X, y, weight, para_lambda))
            weight_new = self.update_weight(
                X_work, y_work, weight, para_lambda, k, i)
            if sum((weight_new - weight) ** 2) < 0.01:
                break
            else:
                weight = weight_new
        return weight

    def predict_score(self, X):
        """calculate prediction score for new Design Matrix X

        Args:
            X (numpy.array): new Design Matrix

        Returns:
            numpy.array: vector of prediction score
        """
        return dot(X, self.weight)

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

    def select_workset(self, X, y, weight, k):
        """Select training set for each iteration

        Args:
            X (numpy.array): Design Matrix
            y (numpy.array): Response Vector
            weight (numpy.array): weight vector
            k (int): maximum training sample size for each iteration

        Returns:
            (numpy.array, numpy.array): (X_train, y_train)
        """
        n, p = X.shape
        index = choice(n, k)
        X_sub = X[index, :]
        y_sub = y[index]
        sub_index = (dot(X_sub, weight) * y_sub) < 1
        index = index[sub_index]
        return (X[index, :], y[index])

    def initialize_weight(self, p, para_lambda):
        """initialize the weight vector

        Args:
            p (int): number of features in the Design Matrix
            para_lambda (float): regularization parameter

        Returns:
            numpy.array: a satisfactory weight vector
        """
        weight = zeros(p)
        weight.fill(sqrt(1 / (p * para_lambda)))
        neg_index = choice(p, size=(int)(p / 2))
        weight[neg_index] = -weight[neg_index]
        return weight

    def update_weight(self, X, y, weight, para_lambda, k, iter_num):
        """update the weight vector

        Args:
            X (numpy.array): Design Matrix
            y (numpy.array): Response Vector
            weight (numpy.array): weight vector
            para_lambda (float): regularization parameter
            k (int): maximum training sample size for each iteration
            iter_num (int): current iteration number

        Returns:
            numpy.array: an updated weight vector
        """
        eta = 1 / (para_lambda * iter_num)  # step size
        weight_half = (1 - eta * para_lambda) * weight + eta / k * dot(y, X)
        if sum(weight_half ** 2) < 1e-07:
            weight_half = maximum(weight_half, 1e-04)
        weight_new = minimum(1, 1 / sqrt(para_lambda) /
                             sqrt(sum(weight_half ** 2))) * weight_half
        return weight_new

    def calc_loss_function(self, X, y, weight, para_lambda):
        """calcualte the primal objective function for linear SVM

        Args:
            X (numpy.array): Design Matrix
            y (numpy.array): Response Vector
            weight (numpy.array): weight parameter
            para_lambda (float): regularization parameter

        Returns:
            float: current value of the primal objective function
        """
        n, p = X.shape
        tmp_loss = 1 - y * dot(X, weight)
        loss = sum(tmp_loss[tmp_loss > 0]) / n + \
            para_lambda / 2 * dot(weight, weight)
        return loss


# data_MNIST = genfromtxt('../res/MNIST-13.csv', delimiter=',')
# X = data_MNIST[:, 1:]
# y = data_MNIST[:, 0]
# model = SVM_SGD(X, y, 1, 5)
# print(model.loglist)
# print(model.validate(X, y))
