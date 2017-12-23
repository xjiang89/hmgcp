import scipy.io as sio
import numpy as np
import time
import os
import scipy.sparse as sp
from hmgcp_general_sp import hmgcp

"""list names of all files in directory `netlib`. """

ls = os.listdir('/Users/debbie/Documents/netlib')
ls = ls[0:]

fname = input("Please input file name:\n")
for name in ls:
    if not name == fname:
        continue
    p = sio.loadmat('/Users/debbie/Documents/netlib/' + name)
    print(name+'\n')
    A = p['A']
    b = p['b'].reshape(-1)
    c = p['c'].reshape(-1)
    (m, n) = A.shape

    Q = sp.csr_matrix(2*np.eye(n, n))
    Q[2, 1] = -1
    Q[1, 2] = -1
    Q[n-1, n-2] = -1
    Q[n-2, n-1] = -1


    def f_(xx, yy):
        return c + Q.dot(xx) + - A.transpose().dot(yy)
    def c_(xx, yy):
        return A.dot(xx) - b

    def j_f(xx, yy):
        # return S.concat((Q, -A.T), dim=1).to_numpy()
        return sp.hstack((Q, -A.transpose())).toarray()

    def j_c(xx, yy):
        return sp.hstack((A, np.zeros((m, m)))).toarray()
        # return S.concat((A, S.variable(np.zeros((m, m)))), dim=1).to_numpy()

    x0 = np.random.rand(n)
    y0 = np.random.rand(m)
    s = time.time()
    (x, y) = hmgcp(x0, y0, f_, c_, j_f, j_c, toler=1e-8)
    e = time.time()
    print('time:', e-s)
    obj = np.dot(c.T, x) + 0.5*np.dot(x.T, Q.dot(x))
    print('=========================================================')
    print(name, '||1e-8|||', obj, '||', e-s)