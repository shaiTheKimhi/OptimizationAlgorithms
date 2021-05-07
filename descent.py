import numpy as np
stop_criteria = 10**-5

def descent(gradient, learn_rate, start_point): #gradient- function that calculates gradient, num_epochs- number of optimization epochs
    #learn rate is a function recieving x and d and returning learning rate
    x = np.array(start_point)
    i = 0
    while True:
        i += 1 #i- number of epochs to convergence
        #print(f'{x}->{gradient(x)}')
        g = -gradient(x)
        if np.linalg.norm(g) <= stop_criterea:
            return [i] + list(x)
        x += g * learn_rate(x,g)  #might need to send conjugate gradient and not gradient itself
    return [i] + list(x) #retuns optimal x