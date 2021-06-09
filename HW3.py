import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def test_func(x):  # 1.3.5
    return x[1] * math.exp(-(x[1] ** 2 + x[2] ** 2))


def activation(x):  # 1.3.6
    out = math.tanh(x)
    d = 1 - (math.tanh(x) ** 2)
    return out, d


def activation_grad(x):  # 1.3.6
    val = np.exp(x)
    return 4 / ((val + 1 / val) ** 2)


def loss(estimated, truth, n):  # estimated is F(x;W)- the output of the model, truth= y the given labels
    return (np.sum((estimated - truth) ** 2)) / n


def loss_grad(pred, y):  # 1.3.7
    return 2 * np.norm(pred - y)


def layer_eval(x, w, b):
    x_pre_act = w.T @ x + b
    x_next = np.array()
    d_act = np.array()
    for s in x_pre_act:
        x_n, d = activation(s)
        np.append(x_next, x_n)
        np.append(d_act, d)

    return x_next, d_act


def forward_pass(x_s, W, b):
    x1, d1 = layer_eval(x_s, W[1, :], b[1, :])
    x2, d2 = layer_eval(x1, W[2, :], b[2, :])
    F = W[3, :].T @ x2 + b[3, :]
    return x1, d1, x2, d2, F


def eval_loss_grad(x, y):
    W = 0
    b = 0
    x1, d1, x2, d2, F = forward_pass(x, W, b)
    lg = loss_grad(F, y)
    dldx = W @ np.diag(activation_grad(d2)) @ lg
    dldw = x1 @ lg.T @ np.diag(activation_grad(d2))
    dldb = np.diag(activation_grad(d2)) @ lg.T
    return 1


def gen_training(n):  # 1.3.10
    return np.random.rand(n, 2) * 4 - 2


def gen_test(n):  # 1.3.11
    return np.random.rand(n, 2) * 4 - 2


# 1.3.12
def init_params(): #incomplite
    b = [np.zeros([4]), np.zeros([3]), np.zeros([1])]
    W = [np.random.rand(2, 4)/np.sqrt(4), np.random.rand(4, 3)/np.sqrt(3), np.random.rand(3, 1)]
    return 0


# 1.3.13
def data_vis(f,data,b): #incomplite
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(-2, 2, 0.2)
    Y = np.arange(-2, 2, 0.2)
    X, Y = np.meshgrid(X, Y)


    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
# 1.3.14
train_set = gen_training(500)
train_set = gen_test(200)
