import numpy as np
stop_criteria = 10**-5


def inexact_line_search(x, d, func):
    s = 0.25
    b = 0.5
    a0 = 1
    f0 = func(x)
    return armijo(x, d, s, b, a0, func, f0)

def armijo(x, d, s, b, a, f, f0):
    bound = d.T@(-d)*a*b
    fi = (f(x + a*b*d) - f0)
    #print(f'{bound}, {fi}, {s*bound} : {x + a*b*d}')
    if (s*bound >= fi and bound <= fi):
        return a*b
    return armijo(x, d, s, b*0.5, a, f, f0)

# gradient- function that calculates gradient, num_epochs- number of optimization epochs
def descent(gradient, learn_rate, start_point, func):
    # learn rate is a function receiving x and d and returning learning rate
    x = np.array(start_point)
    i = 0

    g = -gradient(x)
    while np.linalg.norm(g) > stop_criteria:
        i += 1  # i- number of epochs to convergence
        # print(f'{x}->{gradient(x)}')
        g = -gradient(x)
        n = learn_rate(np.array(x), g, func)
        x += g * n

    return [i] + list(x)  # returns optimal x
