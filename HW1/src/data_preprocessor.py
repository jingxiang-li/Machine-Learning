# This file contains classes and functions for data preprocessing
import numpy as np


class Data_Preprocessor:
    def __init__(self, X, if_remove_zero_variance=True, if_standarize=True):
        X = np.copy(X)
        self.if_remove_zero_variance = if_remove_zero_variance
        self.if_standarize = if_standarize
        self.num_features = X.shape[1]
        if if_remove_zero_variance:
            X = self.remove_zero_variance_features(X)
        if if_standarize:
            X = self.standardize(X)

    def remove_zero_variance_features(self, X):
        # return new X and index of valid columns
        std_array = np.std(X, axis=0)
        self.valid_col_index = std_array > 1e-02
        return X[:, self.valid_col_index]

    def standardize(self, X):
        self.mean_array = np.mean(X, axis=0)
        self.std_array = np.std(X, axis=0, ddof=1)
        return (X - self.mean_array) / self.std_array

    def predict(self, X):
        assert (X.shape[1] == self.num_features)
        if self.if_remove_zero_variance:
            X = X[:, self.valid_col_index]
        if self.if_standarize:
            X = (X - self.mean_array) / self.std_array
        return X
