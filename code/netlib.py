import scipy.io as sio
from hmgcp_general_sp import hmgcp
import numpy as np
import lemo.support as S
import time
import os
import scipy.sparse as sp

ls = os.listdir('/Users/debbie/Documents/netlib')
ls = ls[0:]
for name in ls:
    if not name=='PILOT.mat':
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

# n=n+1
# x = S.variable(np.ones(n))
# y = S.variable(np.zeros(m))
# x = (n/sum(x)) * x
# xx = x[0:n-1]/x[n-1]
# yy = y/x[n-1]
# print(np.linalg.norm(f_(xx, yy).to_numpy(), 2))