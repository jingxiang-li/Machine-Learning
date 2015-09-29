import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import Logistic_Regression
from naive_bayes import Naive_Bayes
from cross_validation import fancy_cv
from random import seed


def logisticRegression__(data, num_splits=100, train_percent=np.array([5, 10, 15, 20, 25, 30])):
    X = data[:, 1:]
    y = data[:, 0].astype(int)
    test_error = fancy_cv(X, y, Logistic_Regression, num_splits, train_percent, [1])
    test_error_mean = np.mean(test_error, axis=0)
    test_error_std = np.std(test_error, axis=0, ddof=1)
    return (test_error_mean, test_error_std)


def naiveBayesGaussian__(data, num_splits=100, train_percent=np.array([5, 10, 15, 20, 25, 30])):
    X = data[:, 1:]
    y = data[:, 0].astype(int)
    test_error = fancy_cv(X, y, Naive_Bayes, num_splits, train_percent)
    test_error_mean = np.mean(test_error, axis=0)
    test_error_std = np.std(test_error, axis=0, ddof=1)
    return (test_error_mean, test_error_std)

seed(5525)
np.random.seed(5525)
data = np.genfromtxt('../dataset/spam.csv', delimiter=",", skip_header=0)
logreg_error_mean, logreg_error_std = logisticRegression__(data)
nb_error_mean, nb_error_std = naiveBayesGaussian__(data)
train_percent = np.array([5, 10, 15, 20, 25, 30])

plt.xlim([4.5, 30.5])
plt.ylim([0.05, 0.29])
plt.errorbar(train_percent, logreg_error_mean, yerr=logreg_error_std * 1.96, label='logistic regression', fmt='-o', capthick=2)
plt.errorbar(train_percent, nb_error_mean, yerr=nb_error_std * 1.96, label='naive bayes', fmt='--o', color='r', capthick=2)
plt.legend()
plt.ylabel('Test Error Rate')
plt.xlabel('Training Percent')
plt.title('Logistic Regression V.S. Naive Bayes', fontsize=18, verticalalignment='bottom')
plt.savefig('../tex/figure/nb_vs_logreg.pdf', bbox_inches='tight', dpi=600, transparent=True, figsize=(8, 6))
