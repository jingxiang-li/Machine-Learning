import numpy as np
import scipy as sp
from numpy import mat
from classifier import Classifier
from data_preprocessor import Data_Preprocessor


class Least_Square_Discriminant (Classifier):
    def __init__(self, X, y):
        # initialize and train the classifier
        # Data_Preprocessor will copy X
        self.data_preprocessor = Data_Preprocessor(X)
        X = self.data_preprocessor.predict(X)
        self.y_vals = np.unique(y)
        y_recode = self.recode_y_to_bit_vector(y)
        self.weight = self.calc_weight(X, y_recode)

    def predict(self, X_new, output=0):
        # used for making prediction
        X_new = self.data_preprocessor.predict(X_new)
        predicted_score = self.predict_score(X_new, self.weight)
        predicted_class = self.predict_class(predicted_score, self.y_vals)
        return predicted_class

    def validate(self, X_new, y_new, output=0):
        # used for validating the prediction performance
        X_new = self.data_preprocessor.predict(X_new)
        predicted_score = self.predict_score(X_new, self.weight)
        predicted_class = self.predict_class(predicted_score, self.y_vals)
        prediction_error = self.calc_predict_error(predicted_class, y_new)
        return prediction_error

    def recode_y_to_bit_vector(self, y):
        y_vals = self.y_vals
        y_new = np.zeros((y.size, y_vals.size))
        for i in range(0, y.size):
            y_new[i, np.argmax(y_vals == y[i])] = 1
        return y_new

    def calc_weight(self, X, y):
        p = X.shape[1]
        # Here we add an identity matrix to X'X to fix the condition
        return np.linalg.inv(mat(X.T) * mat(X) + np.identity(p)) * mat(X.T) * mat(y)

    def predict_score(self, X, weight):
        return mat(X) * mat(weight)

    def predict_class(self, score, y_vals):
        max_indicator = np.argmax(score, axis=1)
        return np.array([y_vals[i][0] for i in max_indicator])

    def calc_predict_error(self, predicted_class, y):
        predicted_indicator = np.array([predicted_class[i] == y[i] for i in range(0, y.size)])
        return 1 - np.sum(predicted_indicator) / y.size


# print("Reading data now ...")
# data_MNIST = np.load('data_MNIST.npy')
# print("Complete reading data")
# X = data_MNIST[:, 1:]
# y = data_MNIST[:, 0].astype(int)
# ls = Least_Square_Discriminant(X, y)
# error_rate = ls.validate(X, y)
# prediction = ls.predict(X)
