import numpy as np
from numpy import genfromtxt
from random import seed
import matplotlib.pyplot as plt
from svm_sgd import SVM_SGD


def draw_plot(axis, k):
    for i in range(5):
        model = SVM_SGD(X, y, 1, k)
        axis.plot(model.loglist)


data_MNIST = genfromtxt('../res/MNIST-13.csv', delimiter=',')
X = data_MNIST[:, 1:]
y = data_MNIST[:, 0]

seed(5525)
np.random.seed(5525)

f, axarr = plt.subplots(3, 2, figsize=(8, 8))
axarr[0, 0].set_title('k = 1')
draw_plot(axarr[0, 0], 1)
axarr[0, 1].set_title('k = 20')
draw_plot(axarr[0, 1], 20)
axarr[1, 0].set_title('k = 100')
draw_plot(axarr[1, 0], 100)
axarr[1, 1].set_title('k = 200')
draw_plot(axarr[1, 1], 200)
axarr[2, 0].set_title('k = 2000')
draw_plot(axarr[2, 0], 2000)
plt.tight_layout()
plt.savefig('../tex/figure/sgd.pdf', transparent=True, dpi=600)
plt.show()
