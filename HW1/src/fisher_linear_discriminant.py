# Fisher's linear discriminant analysis

import numpy as np
import scipy as sp
from numpy import mat

from data_preprocessor import Data_Preprocessor
from classifier import Classifier


class Fisher_Projection (Classifier):
    def __init__(self, X, y):
        self.data_preprocessor = Data_Preprocessor(X)
        X = self.data_preprocessor.predict(X)
        y = np.copy(y)
        self.y_vals = np.unique(y)
        self.weight = self.calc_fisher_weight_vector(X, y)

    def predict(self, X):
        X = self.data_preprocessor.predict(X)
        return np.dot(X, self.weight)

    def validate(self):
        pass

    def calc_between_class_variance(self, X, y):
        num_obs, num_features = X.shape
        mu_all = np.mean(X, axis=0)
        between_class_variance = np.zeros((num_features, num_features))
        for k in range(0, self.y_vals.size):
            index = y == self.y_vals[k]
            X_sub = X[index, :]
            mu = np.mean(X_sub, axis=0)
            between_class_variance += X_sub.shape[0] * np.outer(mu - mu_all, mu - mu_all)
        return between_class_variance

    def calc_within_class_variance(self, X, y):
        num_obs, num_features = X.shape
        within_class_var = np.zeros((num_features, num_features))
        for k in range(0, self.y_vals.size):
            index = y == self.y_vals[k]
            X_sub = X[index, :]
            within_class_var += X_sub.shape[0] * np.cov(X_sub, rowvar=0, bias=1)
        # add an identity matrix to the variance matrix to fix its condition
        within_class_var += np.identity(num_features)
        return within_class_var

    def calc_fisher_weight_vector(self, X, y):
        between_class_variance = self.calc_between_class_variance(X, y)
        within_class_variance = self.calc_within_class_variance(X, y)
        tmp_matrix = mat(np.linalg.inv(within_class_variance)) * mat(between_class_variance)
        w, v = np.linalg.eig(tmp_matrix)
        # print(w[0:(self.y_vals.size - 1)])
        return v[:, 0:(self.y_vals.size - 1)].real


# print("Reading data now ...")
# data_MNIST = np.load('data_MNIST.npy')
# print("Complete reading data")
# X = data_MNIST[:, 1:]
# y = data_MNIST[:, 0].astype(int)
# model = Fisher_Projection(X, y)
# prediction = model.predict(X)
# np.savetxt('X_projected.txt', prediction)

# X = np.genfromtxt('dataset/iris_X.csv', delimiter = ",", skip_header = 1)
# y = np.genfromtxt('dataset/iris_y.csv', delimiter = ",", skip_header = 1)
# print(X)
# print(y)
# remove_zero_variance_features(X)
