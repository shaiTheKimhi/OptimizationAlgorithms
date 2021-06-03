import math


def test_func(x):  # 1.3.5
    return x[1]*math.exp(-(x[1]**2 + x[2]**2))


def activation(x):  # 1.3.6
    out = math.tanh(x)
    d = 1-math.tanh(x)**2
    return out, d


def loss(pred, y):  # 1.3.7
    l = (pred-y)**2
    dl_df = l
    return dl_df

