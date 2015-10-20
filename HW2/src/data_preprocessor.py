# This file contains classes and functions for data preprocessing
import numpy as np


class Data_Preprocessor:
    def __init__(self, X, if_add_intercept=True):
        X = np.copy(X)
        self.num_obs, self.num_features = X.shape
        self.if_add_intercept = if_add_intercept
        X = self.remove_zero_variance_features(X)
        X = self.standardize(X)

    def remove_zero_variance_features(self, X):
        # return new X and index of valid columns
        std_array = np.std(X, axis=0)
        self.valid_col_index = std_array > 1e-03
        return X[:, self.valid_col_index]

    def standardize(self, X):
        self.mean_array = np.mean(X, axis=0)
        self.std_array = np.std(X, axis=0, ddof=1)
        return (X - self.mean_array) / self.std_array

    def add_intercept(self, X):
        return np.concatenate((np.ones((self.num_obs, 1)), X), axis=1)

    def predict(self, X):
        assert (X.shape[1] == self.num_features)
        X = X[:, self.valid_col_index]
        X = (X - self.mean_array) / self.std_array
        if self.if_add_intercept:
            X = self.add_intercept(X)
        return X
