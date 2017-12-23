import scipy.io as sio
from hmgcp_g2 import hmgcp
import numpy as np
import lemo.support as S
import time

p = sio.loadmat('/Users/debbie/Documents/netlib/25FV47.mat')

A = p['A'].toarray()
b = p['b'].reshape(-1)
c = p['c'].reshape(-1)
(m, n) = A.shape

Q = 2*np.eye(n, n)


def f_(xx, yy):
    return c + Q.dot(xx) + - np.dot(A.T, yy)
def c_(xx, yy):
    return A.dot(xx) - b + 0*sum(yy)

def j_f(xx, yy):
    return np.column_stack((Q, -A.T))

def j_c(xx, yy):
    return np.column_stack((A, np.zeros((m, m))))

s = time.time()
(x, y) = hmgcp(m, n, f_, c_, j_f, j_c, toler=1e-6)


e = time.time()
print(e-s)
print(np.dot(c.T, x) + 0.5*np.dot(x.T, Q.dot(x)))