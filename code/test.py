import scipy.sparse.linalg as la
import lemo.support as S
import numpy as np
A = np.random.rand(5, 5)
b = np.random.rand(5)
A_ = np.dot(A.T, A)
b_ = np.dot(A.T, b)
dx = la.cg(A_, b_)
dx = dx[0]
print(dx)
print(b)
print(A.dot(dx))