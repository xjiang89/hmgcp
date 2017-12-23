from fun_qcqp_test import *
from jacobian_qp import *
from numpy import *
import random as rd
from scipy.sparse import *
from scipy import *
import time
import pprint

#set parameters
toler = 1e-6
gamma = .5
alpha = 0.7

n = 3
n = n+1
m = 1
ee = ones(n)
e2 = random.randn(n)
x = ee
# x = 0.9*ee + 0.1*e2

x = (n/sum(x)) * x
s = ee
y = zeros(m)
mu = 1
iter = 0
start = time.clock()
start1 = start
while mu >= toler and iter <= 100:
    print(iter)
    iter += 1
    #
    # Get monotone function values and jacobians
    #
    xx = x[0:n-1]/x[n-1]
    yy = y/x[n-1]

    (f, c) = ingredient(xx, yy, 1)
    (Gf, Gc) = ingredient(xx, yy)
    #
    # Form the homoginized residuals and jacobians
    #
    r1 = s[0:n-1]-x[n-1]*f
    r2 = -x[n-1]*c
    r3 = s[n-1]+dot(x[0:n-1].T, f)+dot(y.T, c)
    r3 = array([r3])
    r  = concatenate((r1, r2, r3))

    MM0   = row_stack((Gf, Gc))
    MM1   = concatenate((f, c))
    MM2   = concatenate((xx, yy))
    MM_b2 = MM1 - dot(MM0, MM2)
    MM_b3 = -MM1.T - dot(MM2.T, MM0)
    MM_b4 = array([dot(dot(MM2.T, MM0), MM2)])
    MM    =  row_stack((column_stack((MM0, MM_b2)), concatenate((MM_b3, MM_b4))))
#     #
#     # Solving one Newton step with the augmented system
#     #
#     # XX = lil_matrix(eye(n+m, n+m))
#     # SS = lil_matrix(zeros([n+m, n+m]))
#     # XX[0:n-1, 0:n-1] = coo_matrix((x[0:n-1, 0], (range(n-1), range(n-1))))
#     # SS[0:n-1, 0:n-1] = coo_matrix((s[0:n-1, 0], (range(n-1), range(n-1))))
#     # XX[n+m-1, n+m-1] = x[n-1, 0]
#     # SS[n+m-1, n+m-1] = s[n-1, 0]
    XX = eye(n+m, n+m)
    SS = zeros([n+m, n+m])
    XX[0:n-1, 0:n-1] = np.diag(x[0:n-1])
    SS[0:n-1, 0:n-1] = np.diag(s[0:n-1])
    XX[n+m-1, n+m-1] = x[n-1]
    SS[n+m-1, n+m-1] = s[n-1]
    #
    # Compute the feasibility residual
    #
    rr1 = -x[0:n-1]*s[0:n-1]+(gamma*mu)*ee[0:n-1]
    rr2 = array([-x[n-1]*s[n-1]+gamma*mu])
    rr3  = concatenate((rr1, zeros(m), rr2))
    rr4 =  (1-gamma)*(XX.dot(r))
    rr = rr3+rr4
#     #
#     # Linear system solve
#     #
#
    w = np.diag(np.random.rand(n+m))
    dx = dot(linalg.inv(XX.dot(MM)+SS)+toler*w, rr)
    #
    # Construct primal and dual steps
    dy = dx[n-1:n+m-1]
    ds = zeros(n)
    ds[0:n-1] = dot(MM[0:n-1, :], dx)-(1-gamma)*r[0:n-1]
    ds[n-1] = dot(MM[n+m-1, :], dx)-(1-gamma)*r[n+m-1]
    dx = concatenate((dx[0:n-1], np.array([dx[n+m-1]])))
    #
    # Choose step-size
    #
    nora = min(concatenate((dx/x, ds/s)))
    nora = abs(alpha/nora)
    #
    # Update iterates
    #
    x = x + nora*dx
    s = s + nora*ds
    y = y + nora*dy
    #
    # Combination of linear and non-linear update the dual of top s
    #
    (f, c1) = ingredient(x[0:n-1]/x[n-1], y/x[n-1], 1)
    s[0:n-1] = 0.3*s[0:n-1]\
                   + 0.7*amax(column_stack((s[0:n-1],(x[n-1]*f))), axis=1)
    # Rescale the intermidiate solutions (ok since homogeneous)
    nora = (sum(x)+sum(s))/(2*n)
    (x, s, y) = (x, s, y)/nora
    # Recompute duality gap
    mu = dot(x.T, s)/n

#
# Output solution or infeasibility cerificate
#
n = n-1
tau = x[n]
kappa = s[n]
if kappa < tau:
    x = x[0:n]/tau
    y = y/tau
    (f, c1) = ingredient(x, y, 1)
    s = mat(amax(column_stack((zeros([size(x, 0), 1]), f)))).T
    disp('Find a complementarity solution')
else:
    x = x[0:n]
    (f, c1) = tau*ingredient(x/tau, y/tau, 1)
    s = mat(amax(column_stack((zeros([size(x, 0), 1]), f)))).T
    disp('The problem is near-infeasible or unattainable')

elapsed = time.clock()-start1
print('norm(x,2): ', np.linalg.norm(x, 2))
print('time used: ', elapsed)
print('iteration: ', iter)
print('obj value: ', double(0.5*dot(x.T, Q).dot(x)))
