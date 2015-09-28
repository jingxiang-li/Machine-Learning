import numpy as np
from math import log
import scipy as sp
from scipy.stats import multivariate_normal
from data_preprocessor import Data_Preprocessor
from classifier import Classifier
from fisher_linear_discriminant import Fisher_Projection


class Multivariate_Gaussian (Classifier):
    def __init__(self, X, y):
        self.data_preprocessor = Fisher_Projection(X, y)
        X = self.data_preprocessor.predict(X)
        y = np.copy(y)
        self.y_vals = np.unique(y)
        self.mu = self.calc_mu(X, y)
        self.sigma = self.calc_sigma(X, y)
        self.prior = self.calc_prior(y)

    def predict(self, X_new, output=0):
        X_new = self.data_preprocessor.predict(X_new)
        predicted_score = self.predict_score(X_new)
        predicted_class = self.predict_class(predicted_score)
        return predicted_class

    def validate(self, X_new, y_new, output=0):
        X_new = self.data_preprocessor.predict(X_new)
        predicted_score = self.predict_score(X_new)
        predicted_class = self.predict_class(predicted_score)
        prediction_error = self.calc_predict_error(predicted_class, y_new)
        return prediction_error

    def calc_mu(self, X, y):
        m = self.y_vals.size  # number of classes
        p = X.shape[1]  # number of features
        mu = np.zeros((m, p))
        for k in range(0, m):
            index = y == self.y_vals[k]
            X_sub = X[index, :]
            mu[k, :] = np.mean(X_sub, axis=0)
        return mu

    def calc_sigma(self, X, y):
        m = self.y_vals.size  # number of classes
        n = X.shape[0]  # number of observations
        p = X.shape[1]  # number of features
        sigma = np.zeros((m, p, p))
        for k in range(0, m):
            index = y == self.y_vals[k]
            X_sub = X[index, :]
            sigma[k, :, :] = np.cov(X_sub, bias=1, rowvar=0)
        return sigma

    def calc_prior(self, y):
        prior = [np.sum(y == y_val) / y.size for y_val in self.y_vals]
        return prior

    def predict_score(self, X):
        m = self.mu.shape[0]  # number of classes
        n, p = X.shape
        ans = np.zeros((n, m))
        for k in range(0, m):
            mu_k = self.mu[k, :]
            sigma_k = self.sigma[k, :, :]
            ans[:, k] = multivariate_normal.logpdf(X, mu_k, sigma_k)
        log_prior = [log(p) for p in self.prior]
        ans += log_prior
        return ans

    def predict_class(self, predicted_score):
        max_indicator = np.argmax(predicted_score, axis=1)
        return np.array([self.y_vals[i] for i in max_indicator])

    def calc_predict_error(self, predicted_class, y):
        predicted_indicator = np.array([predicted_class[i] == y[i] for i in range(0, y.size)])
        return 1 - np.sum(predicted_indicator) / y.size

# print("Reading data now ...")
# data_MNIST = np.load('data_MNIST.npy')
# print("Complete reading data")
# X = data_MNIST[:, 1:]
# y = data_MNIST[:, 0].astype(int)
# model = Multivariate_Gaussian(X, y)
# print(model.predict(X))
