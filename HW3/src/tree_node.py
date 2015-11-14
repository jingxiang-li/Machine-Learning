"""This file contains a class Tree_Node and a function entropy

Author: Jingxiang Li
Date: Tue 10 Nov 2015 07:39:14 PM CST
"""

from numpy import array, asarray, bincount, count_nonzero, genfromtxt, unique, argsort, zeros_like, sort, sum, percentile, all, arange, argmax, linspace, amin, zeros
from numpy.random import permutation, choice
from math import log, floor, ceil


class Tree_Node:
    """A node in the binary split tree

    Attributes:
        cur_entropy (float): entropy of current node
        depth (int): depth of current node
        eps (float): machine epsilon
        left_node (Tree_Node): Tree_Node if less than split_val
        predict_label (int): predicted label if using current node
        right_node (Tree_Node): Tree_Node if greater than split_val
        split_id (id): index of feature used for splitting
        split_val (float): value used for splitting
        splittable (bool): if the current node is splittable
        y_labels (numpy.array): unique labels of y
    """

    def __init__(self, X, y, num_features, sample_array, depth, y_labels=None, predict_label=None):
        """Constructor the Tree_Node

        Args:
            X (numpy.array): Design Matrix
            y (numpy.array): Response Vector
            num_features (int): number of features used for training the binary split tree
            sample_array (numpy.array): indicators of data used to train the binary split tree
            depth (int): target depth of the tree
            y_labels (TYPE, optional): Description
            predict_label (TYPE, optional): Description
        """
        # initialize class members
        self.y_labels = y_labels
        self.predict_label = predict_label
        self.depth = depth

        self.splittable = False
        self.split_id = -1
        self.split_val = -float('inf')
        self.left_node = None
        self.right_node = None
        self.eps = 7. / 3 - 4. / 3 - 1
        self.cur_entropy = None

        assert(sample_array.size != 0 or self.predict_label is not None)

        if sample_array.size == 0:
            return

        # if sample size is not 0, calc predict_label

        # first get y_labels
        if self.y_labels is None:
            self.y_labels = unique(y)

        # then find the most frequent label in y_sub
        X_sub = X[sample_array, :]
        y_sub = y[sample_array]

        label, counts = unique(y_sub, return_counts=True)

        if counts.size == 1:
            self.predict_label = y_sub[0]
            return

        self.predict_label = self.y_labels[argmax(counts)]

        if self.depth != 0:
            self.splittable = True
        else:
            return

        # split the node from here

        n, p = X_sub.shape
        # make sure the number of features is valid
        assert(num_features > 0 and num_features <= p)
        # randomly choose features
        if p == num_features:
            feature_array = arange(p)
        else:
            feature_array = choice(p, num_features, replace=False)

        # calculate current entropy
        self.cur_entropy = self.entropy(y_sub)

        left_index, right_index = self.split(X_sub, y_sub, feature_array)
        sample_array_left = sample_array[left_index]
        sample_array_right = sample_array[right_index]

        self.left_node = Tree_Node(
            X, y, num_features, sample_array_left, depth - 1, self.y_labels, self.predict_label)
        self.right_node = Tree_Node(
            X, y, num_features, sample_array_right, depth - 1, self.y_labels, self.predict_label)

    def predict_class(self, X_new):
        """predict class label for X_new

        Args:
            X_new (numpy.array): Design Matirx to be predicted

        Returns:
            numpy.array: array of class labels
        """
        result = [self.predict_class_single(x) for x in X_new]
        return result

    def predict_class_single(self, x_new):
        """Summary

        Args:
            x_new (TYPE): Description

        Returns:
            TYPE: Description
        """
        if not self.splittable:
            return self.predict_label
        else:
            if x_new[self.split_id] < self.split_val:
                return self.left_node.predict_class_single(x_new)
            else:
                return self.right_node.predict_class_single(x_new)

    def split(self, X, y, feature_array):
        """split the data into two parts

        Args:
            X (numpy.array): Design Matrix
            y (numpy.array): Response Vector
            feature_array (numpy.array): array of feature ids used for training the split tree

        Returns:
            (sample_array_left, sample_array_right): sample_array will be splitted into two parts, left and right
        """
        n, p = X.shape

        best_gain = 0
        best_split_point = 0
        best_feature_id = -1
        for feature_id in feature_array:
            cur_gain, cur_split_point = self.find_best_split(
                X[:, feature_id], y)
            if cur_gain > best_gain - self.eps:
                best_gain = cur_gain
                best_split_point = cur_split_point
                best_feature_id = feature_id

        assert(best_feature_id != -1)

        x = X[:, best_feature_id]
        left_index = x < best_split_point
        right_index = x >= best_split_point

        self.split_id = best_feature_id
        self.split_val = best_split_point

        return (left_index, right_index)

    def find_best_split(self, x, y):
        """find the (approximately) best split value for a single feature

        Args:
            x (numpy.array): Numeric Feature Vector
            y (numpy.array): Response Vector

        Returns:
            (best_gain, best_split_point): the best information gain and the best split point
        """

        # check cornor case: all same x
        n = y.size

        if all(x == x[0]):
            return (0, amin(x) - self.eps)

        sort_index = argsort(x)
        x_sorted = x[sort_index]
        y_sorted = y[sort_index]

        # build potential split index array
        split_index_array = array([i for i in range(1, n)
                                   if x_sorted[i] != x_sorted[i - 1]
                                   and y_sorted[i] != y_sorted[i - 1]])

        # split_index_array = linspace(
        #     0, y.size, num=min(5, ceil(n / 5)), endpoint=False, dtype='int')
        # split_index_array = split_index_array[1:]

        best_split_index = 0
        best_gain = 0
        h_x = self.cur_entropy

        for split_index in split_index_array:
            left_entropy = self.entropy(y_sorted[:split_index])
            right_entropy = self.entropy(y_sorted[split_index:])
            h_xy = (split_index * left_entropy +
                    (n - split_index) * right_entropy) / n
            cur_gain = h_x - h_xy

            if cur_gain > best_gain:
                best_gain = cur_gain
                best_split_index = split_index

        if best_split_index != 0:
            best_split_point = (x_sorted[best_split_index] +
                                x_sorted[best_split_index - 1]) / 2
        else:
            best_split_point = x_sorted[best_split_index] - self.eps

        return (best_gain, best_split_point)

    def entropy(self, y):
        """compute the entropy for a label distribution

        Args:
            y (numpy.array): array of labels

        Returns:
            float: entropy of this label distribution
        """
        n = y.size
        if n <= 1:
            return 0

        labels, counts = unique(y, return_counts=True)

        if counts.size <= 1:
            return 0

        probs = counts / n
        entropy = -sum([p * log(p, 2) for p in probs])
        return entropy

    def show(self):
        """recursively print the tree in a formatted way
        """
        print("depth: ", self.depth, "split_id: ", self.split_id,
              "split_val: ", self.split_val, "predict_label: ",
              self.predict_label)
        if self.left_node is not None:
            self.left_node.show()
        if self.right_node is not None:
            self.right_node.show()


# data = genfromtxt('../res/ionoshpere3.txt', delimiter=',')
# n, p = data.shape
# X = data[:, 0:(p - 1)]
# y = asarray(data[:, p - 1], dtype='int')
# n, p = X.shape
# sample_array = arange(n)
# num_features = p
# tree = Tree_Node(X, y, num_features, sample_array, 2)
# tree.show()
