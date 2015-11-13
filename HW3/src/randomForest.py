""" This file contains class RandomForest
Author: Jingxiang Li
Date: Tue 10 Nov 2015 11:52:29 PM CST
"""

from numpy import array, asarray, bincount, count_nonzero, genfromtxt, unique, argsort, zeros_like, sort, sum, percentile, all, arange, argmax, linspace, amin, zeros
from numpy.random import choice
from math import log, floor, sqrt
from tree_node import Tree_Node


class RandomForest:
    def __init__(self, X, y, num_trees, num_features, sample_size, depth):
        self.forest = []
        self.y_labels = unique(y)
        self.num_trees = num_trees

        n, p = X.shape
        for i in range(num_trees):
            sample_array = choice(n, sample_size)
            tree = Tree_Node(X, y, num_features, sample_array, depth)
            self.forest.append(tree)

    def predict(self, X):
        predicted_score = self.predict_score(X)
        predicted_class = self.predict_class(predicted_score)
        return predicted_class

    def validate(self, X, y):
        predicted_score = self.predict_score(X)
        predicted_class = self.predict_class(predicted_score)
        prediction_error = self.calc_predict_error(predicted_class, y)
        return prediction_error

    def calc_predict_error(self, predicted_class, y):
        return 1 - sum(y == predicted_class) / y.size

    def predict_score(self, X):
        n, p = X.shape
        prediction_result = zeros((self.num_trees, n))
        for i in range(self.num_trees):
            prediction_result[i, :] = self.forest[i].predict_class(X)

        scores = zeros((n, self.y_labels.size))
        for i in range(n):
            for j in range(self.y_labels.size):
                scores[i, j] = sum(prediction_result[:, i] == self.y_labels[j])
        scores = scores / self.num_trees
        return scores

    def predict_class(self, predicted_score):
        result = [self.y_labels[argmax(s)] for s in predicted_score]
        return result

# data = genfromtxt('../res/ionoshpere3.txt', delimiter=',')
# n, p = data.shape
# X = data[:, 0:(p - 1)]
# y = asarray(data[:, p - 1], dtype='int')
# n, p = X.shape
# rf = RandomForest(X, y, 100, p, n, 2)
# print(rf.validate(X, y))
