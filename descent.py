import numpy as np
def descent(gradient, learn_rate, num_epochs, start_point): #gradient- function that calculates gradient, num_epochs- number of optimization epochs
    #learn rate is a function recieving x and d and returning learning rate
    x = np.array(start_point)
    for i in range(num_epochs):
        #print(f'{x}->{gradient(x)}')
        g = -gradient(x)
        x -= g * learn_rate(x,g)  #might need to send conjugate gradient and not gradient itself
    return x #retuns optimal x