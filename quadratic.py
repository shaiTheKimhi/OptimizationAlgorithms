import numpy as np
eps = 0.00001  # regularization


# f(x) = 1/2(x'Qx)
def func(Q):
    def f(x):
        return 0.5*(x.T @ Q @ x)
    return f


# Gf(x) = 1/2(Qx+Q'x)
def gradient(Q):
    def gf(x):
        return 0.5*((Q @ x)+(Q.T @ x))
    return gf


def hessian(Q):
    def hf(x):
        return 0.5*(Q + Q.T)

    return hf


