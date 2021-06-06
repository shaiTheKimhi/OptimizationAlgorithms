import math
import numpy as np

def test_func(x):  # 1.3.5
    return x[1]*math.exp(-(x[1]**2 + x[2]**2))


def activation(x):  # 1.3.6
    out = math.tanh(x)
    d = 1-math.tanh(x)**2
    return out, d

def activation_grad(x):
    val = np.exp(x)
    return 4/((val + 1/val)**2)

def Loss(estimated, truth): #estimated is F(x;W)- the output of the model, truth is y the given labels (both numpy arrays)
    return (np.sum(estimated - truth)**2)/n
'''
def loss(pred, y):  # 1.3.7
    l = (pred-y)**2
    dl_df = l
    return dl_df
'''
