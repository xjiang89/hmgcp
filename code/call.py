#example showing how to call hmgcp
from hmgcp_general import hmgcp
from fun_def_qcqp import *
import numpy as np
import time

def f(xx, yy):
    n1 = P0.shape[0]
    x  = yy[:n1]
    y  = xx
    y_b= yy[n1:]

    # for bug
    z = S.variable(np.zeros(y.shape[0]))
    f1 = -(0.5*S.dot(S.dot(x.T, P1), x)\
           + S.dot(q1.T, x) + r1) + 0*sum(y)
    f2 = -(0.5*S.dot(S.dot(x.T, P2), x)\
           + S.dot(q2.T, x) + r2) + 0*sum(y)
    f  = S.concat((f1, f2), dim=0)
    return f

def c(xx, yy):
    n1  = P0.shape[0]
    x   = yy[0:n1]
    y   = xx
    y_b = yy[n1:]

    c1  = S.dot(P0, x) + q0\
          + y[0]*(S.dot(P1, x)+q1)\
          + y[1]*(S.dot(P2, x)+q2)\
          - S.dot(A.T, y_b)
    c2 = S.dot(A, x) - b

    c   = S.concat((c1, c2), dim=0)
    return c

def jf(xx, yy):
    xx = S.variable(xx)
    yy = S.variable(yy)

    n1  = P0.shape[0]
    m1  = 2
    m2  = A.shape[0]

    x   = yy[0:n1]
    y   = xx
    y_b = yy[n1:]

    Gf1  = S.variable(np.zeros((m1, m1)))

    Gf21 = -(S.dot(P1, x)+q1)
    Gf21 = S.variable(Gf21.to_numpy()\
                      .reshape((Gf21.shape[0], 1))).T
    Gf22 = -(S.dot(P2, x)+q2)
    Gf22 = S.variable(Gf22.to_numpy()\
                      .reshape((Gf22.shape[0], 1))).T

    Gf2 = S.concat((Gf21, Gf22), dim=0)
    Gf3  = S.variable(np.zeros((m1, m2)))
    Gf   = S.concat((Gf1, Gf2), dim=1)
    Gf   = S.concat((Gf, Gf3), dim=1)
    return Gf.to_numpy()

def jc(xx, yy):
    xx = S.variable(xx)
    yy = S.variable(yy)

    n1  = P0.shape[0]
    m1  = 2
    m2  = A.shape[0]

    x   = yy[0:n1]
    y   = xx
    y_b = yy[n1:]

    Gc1  = S.variable(np.zeros((m2+n1, m1)))

    Gc11 = S.dot(P1, x)+q1
    Gc11 = S.variable(Gc11.to_numpy().\
                      reshape((Gc11.shape[0], 1)))
    Gc21 = S.dot(P2, x) + q2
    Gc21 = S.variable(Gc21.to_numpy().\
                      reshape((Gc21.shape[0], 1)))

    Gc1[0:n1, :] = S.concat((Gc11, Gc21), dim=1)
    Gc21 = P0 + y[0]*P1 + y[1]*P2
    Gc22 = A
    Gc2  = S.concat((Gc21, Gc22), dim=0)
    Gc3  = S.variable(np.zeros((m2+n1, m2)))
    Gc3[0:n1, :] = -A.T

    Gc   = S.concat((Gc1, Gc2), dim=1)
    Gc   = S.concat((Gc, Gc3), dim=1)
    return Gc.to_numpy()

m = 110

n = 2

(x1, y1) = hmgcp(m, n, f, c, jf, jc)

s = time.time()
(x2, y2) = hmgcp(m, n, f, c)
e = time.time()
print(e-s)
print('norm(x1-x2, 2)', np.linalg.norm(x1-x2, 2))