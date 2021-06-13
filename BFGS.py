from mcholmz import modifiedChol
import numpy as np


def inexact_line_search(x, dir, func, grad):
    sigma = 0.25
    beta = 0.5
    a0 = 1
    f0 = func(x)
    # s, phi1 = ls.scalar_search_armijo(func, x, d, f0, alpha0=1)
    c = (grad(x).T @ dir)  # =phi(0)
    return wolf(x, dir, sigma, 1, a0, func, grad, f0, beta, c)


def wolf(x, d, sigma, b, a0, f, g, f0, cb, c):
    alpha = a0 * b
    bound = (sigma * alpha * c) + f0
    # bound = (g(x) @ d)*a*b
    x_ad = x + alpha * d
    phi = f(x_ad)
    g2 = g(x_ad).T @ d
    g1 = 0.9 * c

    if bound >= phi and g2 >= g1:  # 0.9 is c2 this is wolfe curvature condition
        # if bound >= phi and g(x + alpha * d) @ d >= 0.9 * c:  # 0.9 is c2 this is wolfe curvature condition
        # if s * bound >= phi:
        return alpha
    return wolf(x, d, sigma, b * cb, a0, f, g, f0, cb, c)


def update(Bk, pk, qk):
    pk = pk.reshape(pk.shape[0], 1)
    qk = qk.reshape(pk.shape[0], 1)
    # Broyden family
    sk = np.matmul(Bk, qk)
    tk = sk.T @ qk
    miuk = pk.T @ qk
    vk = (1 / miuk) * pk - (1 / tk) * sk

    norm = np.linalg.norm
    return Bk + (1 / miuk) * pk @ pk.T - (1 / tk) * sk @ sk.T + tk * vk @ vk.T  # return Bk+1

    # Secant equation
    # v = pk -  Bk @ qk
    # return Bk - (v @ v.T)/(v @ qk) #B(k+1)


def BFGS(func, gradient, start_point, learn_rate, exact_ls=True, stop_criteria=10 ** -5):
    zero = 10 ** -10
    x = np.array(start_point)

    i = 0

    B = np.eye(x.shape[0])  # B0 initialization
    g = -gradient(x)
    tod = np.array([start_point])

    prev_x = x
    prev_g = g

    while np.linalg.norm(-g) > stop_criteria:
        i += 1
        # we want to solve Hdk=-g
        # 1. find L and d such that H+e=Ldiag(d)L' and LDL'dk=-g
        # 2. solve Ly=-g where y=DL'dk
        # 3. solve Dz=y where z=L'dk
        # 4. solve L'dk=z
        # L, D, e = modifiedChol(np.linalg.inv(B))
        # D = np.squeeze(D)
        # y = np.linalg.solve(L, g)  # forward_substitution(L,g)
        # z = (y / D)
        # dk = np.linalg.solve(L.T, z)  # backward_substitution(L,g)
        dk = B @ g
        if exact_ls is True:
            a = learn_rate(x, dk)
        else:
            a = learn_rate(x, dk, func, gradient)
        prev_x = np.array(x)  # keep old x and g for p and q accordingly
        prev_g = np.array(g)

        x += dk * a  # calc new x and g
        tod = np.append(tod, [x], axis=0)
        g = -gradient(x)

        # print(f"{i}:{x}")

        B = update(B, x - prev_x, -(g - prev_g))  # update approx inv hessian

    return i, tod  # returns optimal x
