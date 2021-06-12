import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import BFGS as bfgs


def test_func(x):  # 1.3.5
    return x[0, :] * np.exp(-(x[0, :] ** 2 + x[1, :] ** 2))


def activation(x, der=False):  # 1.3.6
    if der is False:
        return np.tanh(x)
    else:
        return 1 - (np.tanh(x) ** 2)


def loss_grad(pred, y):  # 1.3.7
    return 2 * (pred - y)


def loss(out, truth, n):  # estimated is F(x;W)- the output of the model, truth= y the given labels
    return (np.sum((out - truth) ** 2)) / n


def gen_training(n):  # 1.3.10
    x = np.random.rand(2, n) * 4 - 2
    y = test_func(x)
    return x, y


def gen_test(n):  # 1.3.11
    return np.random.rand(2, n) * 4 - 2


# 1.3.13
def data_vis(self, f, x, y, b):  # incomplete
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(-2, 2, 0.2)
    Y = np.arange(-2, 2, 0.2)
    X, Y = np.meshgrid(X, Y)
    Z = f(X,Y)
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    if b is True:
        ax.scatter(x, y, f(x, y))

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

class NN:
    numLayers = 2
    weights = []

    def __init__(self):
        # 1.3.12
        self.weights = [np.row_stack([np.random.rand(2, 4) / np.sqrt(4), np.zeros([4])]),
                        np.row_stack([np.random.rand(4, 3) / np.sqrt(3), np.zeros([3])]),
                        np.row_stack([np.random.rand(3, 1) / np.sqrt(1), np.zeros([1])])]
        self.layer_input = []
        self.layer_output = []

    def w_vec(self):
        return np.concatenate((np.reshape(self.weights[0], -1),
                               np.reshape(self.weights[1], -1),
                               np.reshape(self.weights[2], -1)))

    # 1.3.8
    def forward_pass(self, x_s, weights):
        self.layer_input = []
        self.layer_output = []

        num_samples = x_s.shape[1]

        x_s_h = np.concatenate([x_s, np.ones([1, num_samples])])
        layer_in = weights[0].T.dot(x_s_h)
        self.layer_input.append(layer_in)
        act = activation(layer_in)
        self.layer_output.append(act)

        x1_h = np.concatenate([self.layer_output[-1], np.ones([1, num_samples])])
        layer_in = weights[1].T.dot(x1_h)
        self.layer_input.append(layer_in)

        act = activation(layer_in)
        # act = act[:, np.newaxis]
        self.layer_output.append(act)

        x2_h = np.concatenate([self.layer_output[-1], np.ones([1, num_samples])])
        layer_in = weights[2].T.dot(x2_h)
        self.layer_input.append(layer_in)
        self.layer_output.append(layer_in)
        return layer_in

    # 1.3.8
    def eval_grad_of_loss_single(self, x, y):
        # 1
        x = x[:, np.newaxis]
        F = self.forward_pass(x)
        # 2
        lg = loss_grad(F, y)

        # 3
        dldx2 = self.weights[2] @ lg
        dldx2 = dldx2[0:-1]

        dldw2 = self.layer_output[1] @ lg.T
        dldw2 = np.concatenate([dldw2, lg])


        # 4  ## x=prev layer output
        dldx1 = self.weights[1] @ np.diag(np.squeeze(activation(self.layer_input[1], True))) @ dldx2
        dldx1 = dldx1[0:-1]

        dldw1 = self.layer_output[0] @ dldx2.T @ np.diag(np.squeeze(activation(self.layer_input[1], True)))
        dldw1 = np.concatenate([dldw1, dldx2.T])

        # dldx0 = self.weights[0] @ np.diag(activation(self.layer_output[0], True)) @ dldx1
        dldw0 = x @ dldx1.T @ np.diag(np.squeeze(activation(self.layer_input[0], True)))
        dldw0 = np.concatenate([dldw0, dldx1.T])

        return np.concatenate((np.reshape(dldw2, -1),
                               np.reshape(dldw1, -1),
                               np.reshape(dldw0, -1)))

    # 1.3.9
    def eval_grad_of_loss_all(self, x, y):
        num_samples = x.shape[1]
        grad = 0
        for i in range(num_samples):
            single_loss = self.eval_grad_of_loss_single(x[:, i], y[i])
            grad += single_loss * ((self.forward_pass(np.expand_dims(x[:, i], axis=1)) - y[i]) ** 2) / num_samples

        return grad

# 1.3.14
nn = NN()
train_x, train_y = gen_training(500)
train_set = gen_test(200)

for i in range(4):
    e = 10 ** -(i + 1)
    w0 = nn.w_vec()
    grad = nn.eval_grad_of_loss_all(train_x, train_y)
    # func -  feed forward and calculate loss for a given W where x and y are constants
    # gradient - calculate for a given W where x and y are constants
    # desc = bfgs.BFGS(func, nn.eval_grad_of_loss_all(train_x, train_y), w0, bfgs.inexact_line_search, False, e)[1]
