import numpy as np
stop_criteria = 10**-5


# gradient- function that calculates gradient, num_epochs- number of optimization epochs
def descent(gradient, learn_rate, start_point):
    # learn rate is a function receiving x and d and returning learning rate
    x = np.array(start_point)
    i = 0

    g = -gradient(x)
    while np.linalg.norm(g) > stop_criteria:
        i += 1  # i- number of epochs to convergence
        # print(f'{x}->{gradient(x)}')
        x += g * learn_rate(x, g)  # might need to send conjugate gradient and not gradient itself
        g = -gradient(x)
    return [i] + list(x)  # returns optimal x
