import numpy as np
from numpy import genfromtxt
from random import seed
import matplotlib.pyplot as plt
from svm_smo import SVM_SMO

data_MNIST = genfromtxt('../res/MNIST-13.csv', delimiter=',')
X = data_MNIST[:, 1:]
y = data_MNIST[:, 0]

log_list = []

seed(5525)
np.random.seed(5525)

for i in range(5):
    print(i)
    model = SVM_SMO(X, y, 1, 2000)
    log_list.append(model.loglist)

plt.plot(log_list[0], '-')
plt.plot(log_list[1], '-')
plt.plot(log_list[2], '-')
plt.plot(log_list[3], '-')
plt.plot(log_list[4], '-')
plt.tight_layout()
# plt.savefig('../tex/figure/smo.pdf', transparent=True, dpi=600)
plt.show()
