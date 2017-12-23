from scipy import *
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la

def hmgcp(x0, y0,
          func_f, func_c,
          fprime=None, cprime=None,
          args_f=(), args_c=(), toler=1e-8, gamma=.5, alpha=0.7, maxiter=None):
    """
    Complete a monotone complementarity programming function using homogeneous algorithm.

    Parameters
    ----------
    func_f : callable, ``f(x, y, *args)``
        Objective function to be minimized.  Here `x` and `y` must be a 1-D array of
        the variables that are to be changed in the search for a minimum, and
        `args` are the other (fixed) parameters of `f`.
    x0 : ndarray, optional
        A user-supplied initial estimate of `xopt`, the optimal value of `x`.
        It must be a 1-D array of values.
    y0 : ndarray, optional
        A user-supplied initial estimate of `yopt`, the optimal value of `y`.
        It must be a 1-D array of values.
    fprime : callable, ``fprime(x, y, *args)``, optional
        A function that returns the gradient of `f` at `x`. Here `x`, `y` and `args`
        are as described above for `f`. The returned value must be a 1-D array.
        Defaults to None, in which case the gradient is approximated
        numerically (see `epsilon`, below).
    cprime : callable, ``cprime(x, y, *args)``, optional
        A function that returns the gradient of `c` at `x`. Here `x`, `y` and `args`
        are as described above for `c`. The returned value must be a 1-D array.
        Defaults to None, in which case the gradient is approximated
        numerically (see `epsilon`, below).
    args : tuple, optional
        Parameter values passed to `f` and `fprime`. Must be supplied whenever
        additional fixed parameters are needed to completely specify the
        functions `f` and `fprime`.
    toler : float, optional
        Stop when the norm of the gradient is less than `gtol`.
    gamma : float, optional
        Parameter gamma, which can be interpreted as a targeted reduction
        factor in the infeasibility and complementarity.
    alpha : float, optional
        Step size to use.
    maxiter : int, optional
        Maximum number of iterations to perform. Default is ``200 * len(x0)``.

    full_output : bool, optional
        If True, return `fopt`, `func_calls`, `grad_calls`, and `warnflag` in
        addition to `xopt`.  See the Returns section below for additional
        information on optional return values.
    disp : bool, optional
        If True, return a convergence message, followed by `xopt`.
    retall : bool, optional
        If True, add to the returned values the results of each iteration.
    callback : callable, optional
        An optional user-supplied function, called after each iteration.
        Called as ``callback(xk)``, where ``xk`` is the current value of `x0`.

    Returns
    -------
    xopt : ndarray
        Parameters which minimize f, i.e. ``f(xopt) == fopt``.
    fopt : float, optional
        Minimum value found, f(xopt).  Only returned if `full_output` is True.
    func_calls : int, optional
        The number of function_calls made.  Only returned if `full_output`
        is True.
    grad_calls : int, optional
        The number of gradient calls made. Only returned if `full_output` is
        True.
    warnflag : int, optional
        Integer value with warning status, only returned if `full_output` is
        True.

        0 : Success.

        1 : The maximum number of iterations was exceeded.

        2 : Gradient and/or function calls were not changing.  May indicate
            that precision was lost, i.e., the routine did not converge.

    allvecs : list of ndarray, optional
        List of arrays, containing the results at each iteration.
        Only returned if `retall` is True.

    References
    ----------
    .. [1] Wright & Nocedal, "Numerical Optimization", 1999, pp. 120-122.

    Examples
    --------
    Example 1: seek the minimum value of the expression
    ``a*u**2 + b*u*v + c*v**2 + d*u + e*v + f`` for given values
    of the parameters and an initial guess ``(u, v) = (0, 0)``.

    >>> args = (2, 3, 7, 8, 9, 10)  # parameter values
    >>> def f(x, *args):
    ...     u, v = x
    ...     a, b, c, d, e, f = args
    ...     return a*u**2 + b*u*v + c*v**2 + d*u + e*v + f
    >>> def gradf(x, *args):
    ...     u, v = x
    ...     a, b, c, d, e, f = args
    ...     gu = 2*a*u + b*v + d     # u-component of the gradient
    ...     gv = b*u + 2*c*v + e     # v-component of the gradient
    ...     return np.asarray((gu, gv))
    >>> x0 = np.asarray((0, 0))  # Initial guess.
    >>> from scipy import optimize
    >>> res1 = optimize.fmin_cg(f, x0, fprime=gradf, args=args)
    Optimization terminated successfully.
             Current function value: 1.617021
             Iterations: 4
             Function evaluations: 8
             Gradient evaluations: 8
    >>> res1
    array([-1.80851064, -0.25531915])

    """

    if fprime is None:
        pass
    else:
        myfprime = fprime

    if fprime is None:
        pass
    else:
        mycprime = cprime

    n = len(x0)
    n = n+1
    m = len(y0)

    ee = np.ones(n)
    x = np.append(x0, 1)
    x = (n/sum(x)) * x
    y = np.zeros(m)
    s = ee

    mu = 1
    k = 0

    if maxiter is None:
        maxiter = 200*len(x)

    while mu >= toler and k <= maxiter:
        print(k)
        k += 1
        #
        # Get monotone function values and jacobians
        #
        xx = x[0:n-1]/x[n-1]
        yy = y/x[n-1]

        (f, c) = (func_f(xx, yy), func_c(xx, yy))
        (Gf, Gc) = (myfprime(xx, yy), mycprime(xx, yy))
        #
        # Form the homoginized residuals and jacobians
        #
        rs1 = s[0:n-1]-x[n-1]*f
        rs2 = -x[n-1]*c
        rs3 = s[n-1]+np.dot(x[0:n-1].T, f)+np.dot(y.T, c)
        rs3 = array([rs3])
        r  = concatenate((rs1, rs2, rs3))
        # print(linalg.norm(r, 2))

        MM0   = np.row_stack((Gf, Gc))
        MM1   = np.concatenate((f, c))
        MM2   = np.concatenate((xx, yy))
        MM_b2 = MM1 - np.dot(MM0, MM2)
        MM_b3 = -MM1.T - np.dot(MM2.T, MM0)
        MM_b4 = array([np.dot(np.dot(MM2.T, MM0), MM2)])
        MM    =  np.row_stack((np.column_stack((MM0, MM_b2)), np.concatenate((MM_b3, MM_b4))))

        # Solving one Newton step with the augmented system

        #41.91s
        XX = sp.csr_matrix((list(x[0:n-1])+m*[1, ]+[x[n-1]], list(range(n+m)), list(range(n+m+1))), shape=(m+n, m+n))
        SS = sp.csr_matrix((list(s[0:n-1])+m*[0, ]+[s[n-1]], list(range(n+m)), list(range(n+m+1))), shape=(m+n, m+n))
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
        # (A, b) = (XX.dot(MM)+SS, rr)
        # A = np.asarray(A)
        # dx = np.linalg.solve(A, b)

        (A, b) = (XX.dot(sp.csr_matrix(MM))+SS, rr)
        dx = la.spsolve(A, b)
        #
        # Construct primal and dual steps
        dy = dx[n-1:n+m-1]
        ds = np.zeros(n)
        ds[0:n-1] = np.dot(MM[0:n-1, :], dx)-(1-gamma)*r[0:n-1]
        ds[n-1] = np.dot(MM[n+m-1, :], dx)-(1-gamma)*r[n+m-1]
        dx = np.concatenate((dx[0:n-1], np.array([dx[n+m-1]])))
        #
        # Choose step-size
        #
        nora = min(np.concatenate((dx/x, ds/s)))
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