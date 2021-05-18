import numpy as np
stop_criteria = 10**-5


def exact_line_search(Q):
    def els(x, d):  # d- conjugate gradient direction x-current parameters
        Qd = Q @ d
        # returns learning rate which yields optimal value
        # no need regularization due to guarantee ||d|| > 10**-5
        return -(x.T @ Qd) / (d.T @ Qd)
        #return -(x.T @ Qd) / f(d)

    return els


def inexact_line_search(x, d, func):
    s = 0.25
    b = 0.5
    a0 = 1
    f0 = func(x)
    return armijo(x, d, s, b, a0, func, f0)


def armijo(x, d, s, b, a, f, f0):
    bound = d.T@(-d)*a*b
    fi = (f(x + a*b*d) - f0)
    # print(f'{bound}, {fi}, {s*bound} : {x + a*b*d}')
    if (s*bound >= fi and bound <= fi):
        return a*b
    return armijo(x, d, s, b*0.5, a, f, f0)


# gradient- function that calculates gradient, num_epochs- number of optimization epochs
def descent(func, gradient, start_point, learn_rate, exact_ls):
    # learn rate is a function receiving x and d and returning learning rate
    x = np.array(start_point)
    i = 0
    # Trace of Descent
    tod = np.array([start_point])
    g = -gradient(x)
    while np.linalg.norm(g) > stop_criteria:
        i += 1  # i- number of epochs to convergence
        # print(f'{x}->{gradient(x)}')
        if exact_ls is True:
            a = learn_rate(x, g)
        else:
            a = learn_rate(x, g, func)
        x += g * a
        g = -gradient(x)
        tod = np.append(tod, [x], axis=0)

    return i, tod  # returns optimal x
