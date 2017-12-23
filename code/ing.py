import lemo.support as S
import numpy as np
def ingredient(x, y, f, c, flag=0):
    if isinstance(x, np.ndarray):
        x = S.variable(x)
    if isinstance(y, np.ndarray):
        y = S.variable(y)

    def fx(xx):
        return f(xx, y)

    def fy(yy):
        return f(x, yy)

    def cx(xx):
        return c(xx, y)

    def cy(yy):
        return c(x, yy)

    if flag == 1:
        return f(x, y).to_numpy()
    elif flag == 2:
        return c(x, y).to_numpy()
    elif flag == 3:
        Gf1 = S.compute_jacobian(fx, x).to_numpy()
        Gf2 = S.compute_jacobian(fy, y).to_numpy()
        return np.hstack((Gf1, Gf2))
    elif flag == 4:
        Gc1 = S.compute_jacobian(cx, x).to_numpy()
        Gc2 = S.compute_jacobian(cy, y).to_numpy()
        return np.hstack((Gc1, Gc2))