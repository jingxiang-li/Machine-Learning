import numpy as np
from numpy import mat
from scipy.optimize import fmin_bfgs as fmin
from scipy.special import expit as sigmoid
from math import log
from math import exp
from data_preprocessor import Data_Preprocessor
from classifier import Classifier


class Logistic_Regression (Classifier):
    def __init__(self, X, y, lambda_):
        # preprocess X
        self.data_preprocessor = Data_Preprocessor(X)  # Data_Preprocessor will copy X
        X = self.data_preprocessor.predict(X)
        X = self.add_intercept(X)
        # preprocess y
        y = np.copy(y)
        self.y_vals = np.unique(y)
        # check number of classes here
        assert self.y_vals.size == 2
        y[y == self.y_vals[0]] = -1
        y[y == self.y_vals[1]] = +1
        # train the model
        self.weight = self.lr_train(X, y, lambda_)

    def predict(self, X, output=0):
        X = self.data_preprocessor.predict(X)
        X = self.add_intercept(X)
        predicted_score = self.predict_score(X)
        predicted_class = self.predict_class(predicted_score)
        return predicted_class

    def validate(self, X, y, output=0):
        X = self.data_preprocessor.predict(X)
        X = self.add_intercept(X)
        predicted_score = self.predict_score(X)
        predicted_class = self.predict_class(predicted_score)
        prediction_error = self.calc_predict_error(predicted_class, y)
        return prediction_error

    def add_intercept(self, X):
        num_obs = X.shape[0]
        X_new = np.concatenate((np.ones((num_obs, 1)), X), axis=1)
        return X_new

    def lr_loss(self, w, X, y, lambda_):
        # y must be in {-1, +1}
        # the first column of X should be all 1
        num_obs, num_features = X.shape
        loss = 0
        grad = np.zeros((1, num_features))
        H = - y * np.dot(X, w)
        H = [h if h > 10 else log(1 + exp(h)) for h in H]
        loss -= np.sum(H)
        loss -= lambda_ / 2 * np.dot(w[1:], w[1:])
        return -loss

    def lr_gradient(self, w, X, y, lambda_):
        # y must be in {-1, +1}
        num_obs, num_features = X.shape
        grad = np.zeros((1, num_features))
        grad += mat((1 - sigmoid(y * np.dot(X, w))) * y) * mat(X)
        # do not regularize intercep
        grad -= lambda_ * np.concatenate(([0], w[1:]))
        return -grad[0]

    def grad_check(self, w, X, y, lambda_):
        num_obs, num_features = X.shape
        grad0 = lr_gradient(w, X, y, lambda_)
        print(grad0)
        eps = 1e-05
        grad1 = np.zeros_like(grad0)
        for i in range(0, num_features):
            delta = np.zeros_like(w)
            delta[i] = eps
            grad1[i] = (lr_loss(w + delta, X, y, lambda_) - lr_loss(w - delta, X, y, lambda_)) / 2 / eps
        print(np.linalg.norm(grad1 - grad0) / np.linalg.norm(grad0))

    def lr_train(self, X, y, lambda_):
        # random initialization
        num_obs, num_features = X.shape
        w = (np.random.rand(num_features) - 0.5) * 2
        lr_fmin_result = fmin(f=self.lr_loss, x0=w, fprime=self.lr_gradient, args=(X, y, lambda_), maxiter=50, disp=False)
        return lr_fmin_result

    def predict_score(self, X):
        return mat(X) * mat(self.weight).T

    def predict_class(self, predicted_score):
        return [self.y_vals[0] if s < 0 else self.y_vals[1] for s in predicted_score]

    def calc_predict_error(self, predicted_class, y):
        predicted_indicator = np.array([predicted_class[i] == y[i] for i in range(0, y.size)])
        return 1 - np.sum(predicted_indicator) / y.size
