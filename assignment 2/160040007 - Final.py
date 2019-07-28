#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cvxopt.solvers
import pandas as pd


# In[2]:


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


# In[3]:


class SVM(object):

    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), 'd')
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))


# In[4]:


def split_train(X1, y1, X2, y2):
    X1_train = X1[:80]
    y1_train = y1[:80]
    X2_train = X2[:80]
    y2_train = y2[:80]
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))
    return X_train, y_train


# In[5]:


def split_test(X1, y1, X2, y2):
    X1_test = X1[80:]
    y1_test = y1[80:]
    X2_test = X2[80:]
    y2_test = y2[80:]
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    return X_test, y_test


# In[ ]:





# In[6]:


# Preparing Data
data = pd.read_csv('creditcard.csv', sep=',', decimal='.')
data_arr = data.iloc[:, :].values

test1 = data.loc[data['Class'] == 1]
test0 = data.loc[data['Class'] == 0]

temp1 = test1.sample(n=100)
temp0 = test0.sample(n=100)

X1 = temp1.iloc[:, 0:-1].values
y1 = temp1.iloc[:, -1].values
X2 = temp0.iloc[:, 0:-1].values
y2 = temp0.iloc[:, -1].values

y1[y1 == 0] = -1
y2[y2 == 0] = -1

X_train, y_train = split_train(X1, y1, X2, y2)
X_test, y_test = split_test(X1, y1, X2, y2)


# In[7]:


clf = SVM()
clf.fit(X_train, y_train)


# In[8]:


y_predict = clf.predict(X_test)
correct = np.sum(y_predict == y_test)
print("%d of %d predictions correct" % (correct, len(y_predict)))

