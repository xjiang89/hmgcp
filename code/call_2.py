# example showing how to call hmgcp
from hmgcp_general import hmgcp
from fun_def_qp import *
import numpy as np

def f(x, y):
    return S.dot(Q, x) + sum(y)
def c(x, y):
    return sum(x) - 1

m = 1

n = 3


(x2, y2) = hmgcp(m, n, f, c)