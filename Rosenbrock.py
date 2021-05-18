import numpy as np
eps = 0.00001  # regularization


def func():

    def f(x):
        y = 0
        for i in range(x.shape[0]-1):
            y += (1-x[i])**2+100*((x[i+1]-(x[i]**2))**2)
        return y

    return f


def gradient():

    def gf(x):
        n = x.shape[0]
        g = np.zeros(n)
        g[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - (x[0] ** 2))
        for i in range(1,n-1):
            g[i] = -2*(1-x[i]) - 400*x[i]*(x[i+1]-(x[i]**2)) + 200*(x[i]-(x[i-1]**2))
        g[n-1] = 200*(x[n-1]-(x[n-2]**2))
        return g

    return gf


def hessian():

    def hf(x):
        n = x.shape[0]
        h = np.zeros((n, n))
        h[0, 0] = 2 + 1200 * x[0]**2 * -400*x[1]
        h[0, 1] = -400*x[0]
        h[n-1, n-1] = 200
        h[n-1, n-2] = -400*x[n-2]
        for i in range(1, n-1):
            h[i, i] = 202 + 1200 * (x[i] ** 2) - 400 * x[i + 1]
            h[i, i - 1] = -400*x[i-1]
            h[i, i + 1] = -400*x[i]
        return h

    return hf


# def exact_line_search(Q):
#     def els(x, d):  # d- conjugate gradient direction x-current parameters
#         Qd = Q @ d
#         return -(x.T @ Qd) / (
#                     (d.T @ Qd) + np.ones(tuple(x.shape)) * eps)  # returns learning rate which yields optimal value
#
#     return els
