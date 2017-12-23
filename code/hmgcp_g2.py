from scipy.sparse import *
from scipy import *
import time
import lemo.support as S
import numpy as np

def hmgcp(m, n, func_f, func_c, jac_f, jac_c, toler=1e-8, gamma=.5, alpha=0.7):
    n = n+1

    ee = ones(n)
    e2 = random.randn(n)
    x = ee

    x = (n/sum(x)) * x
    s = ee
    y = zeros(m)
    mu = 1
    iter = 0
    while mu >= toler and iter <= 100:
        print(iter)
        iter += 1
        #
        # Get monotone function values and jacobians
        #
        xx = x[0:n-1]/x[n-1]
        yy = y/x[n-1]

        (f, c) = (func_f(xx, yy), func_c(xx, yy))

        (Gf, Gc) = (jac_f(xx, yy), jac_c(xx, yy))

        #
        # Form the homoginized residuals and jacobians
        #
        rs1 = s[0:n-1]-x[n-1]*f
        rs2 = -x[n-1]*c
        rs3 = s[n-1]+dot(x[0:n-1].T, f)+dot(y.T, c)
        rs3 = array([rs3])
        r  = concatenate((rs1, rs2, rs3))

        MM0   = row_stack((Gf, Gc))
        MM1   = concatenate((f, c))
        MM2   = concatenate((xx, yy))
        MM_b2 = MM1 - dot(MM0, MM2)
        MM_b3 = -MM1.T - dot(MM2.T, MM0)
        MM_b4 = array([dot(dot(MM2.T, MM0), MM2)])
        MM    =  row_stack((column_stack((MM0, MM_b2)), concatenate((MM_b3, MM_b4))))

        # Solving one Newton step with the augmented system

        #41.91s
        XX = csr_matrix((list(x[0:n-1])+m*[1, ]+[x[n-1]], list(range(n+m)), list(range(n+m+1))), shape=(m+n, m+n))
        SS = csr_matrix((list(s[0:n-1])+m*[0, ]+[s[n-1]], list(range(n+m)), list(range(n+m+1))), shape=(m+n, m+n))

        #42.49s
        # XX = csr_matrix((list(x[0:n-1])+(m+1)*[1, ], list(range(n+m)), list(range(n+m+1))), shape=(m+n, m+n))
        # SS = csr_matrix((s[0:n-1], list(range(n-1)), list(range(n))+(m+1)*[n-1]), shape=(m+n, m+n))
        # XX[n+m-1, n+m-1] = x[n-1]
        # SS[n+m-1, n+m-1] = s[n-1]


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
        A = np.asarray(XX.dot(MM) + SS)
        dx = np.dot(np.linalg.inv(A), rr)
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
        f = func_f(x[0:n-1]/x[n-1], y/x[n-1])
        s[0:n-1] = 0.3*s[0:n-1]\
                       + 0.7*amax(column_stack((s[0:n-1], (x[n-1]*f))), axis=1)
        # Rescale the intermidiate solutions (ok since homogeneous)
        nora = (sum(x)+sum(s))/(2*n)
        (x, s, y) = (x, s, y)/nora
        # Recompute duality gap
        mu = dot(x.T, s)/n
        print(mu)

    #
    # Output solution or infeasibility cerificate
    #
    n = n-1
    tau = x[n]
    kappa = s[n]
    if kappa < tau:
        x = x[0:n]/tau
        y = y/tau
        f = func_f(x, y)
        s = mat(amax(column_stack((zeros([size(x, 0), 1]), f)))).T
        disp('Find a complementarity solution')
    else:
        x = x[0:n]
        f = tau * func_f(x/tau, y/tau)
        s = mat(amax(column_stack((zeros([size(x, 0), 1]), f)))).T
        disp('The problem is near-infeasible or unattainable')
    return x, y