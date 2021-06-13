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


def gen_training(n):  # 1.3.10
    x = np.random.rand(2, n) * 4 - 2
    y = test_func(x)
    return x, y


def gen_test(n):  # 1.3.11
    return np.random.rand(2, n) * 4 - 2




# 1.3.12
def init_weights():
    return [np.row_stack([np.random.rand(2, 4) / np.sqrt(4), np.zeros([4])]),
                    np.row_stack([np.random.rand(4, 3) / np.sqrt(3), np.zeros([3])]),
                    np.row_stack([np.random.rand(3, 1) / np.sqrt(1), np.zeros([1])])]


def w_mat2vec(weights):
    return np.concatenate((np.reshape(weights[0], -1),
                           np.reshape(weights[1], -1),
                           np.reshape(weights[2], -1)))


def w_vec2mat(weights):
    return [np.reshape(weights[0:(3*4)], (3, 4)),
            np.reshape(weights[(3*4):(3*4)+(5*3)], (5, 3)),
            np.reshape(weights[-4:], (4, 1))]


class NN:
    numLayers = 2
    layer_input = []
    layer_output = []

    # 1.3.8
    def forward_pass(self, x_s):
        def fp(weights):
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
            return weights[2].T.dot(x2_h)
        return fp

    def forward_pass_const_w(self, weights):
        def fp(x_s):
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
            return weights[2].T.dot(x2_h)
        return fp

    # 1.3.8
    def eval_grad_of_loss_single(self, x, y, weights):
        # 1
        # x = x[:, np.newaxis]
        F = self.forward_pass(x)(weights)
        # 2
        lg = loss_grad(F, y)

        # 3
        dldx2 = weights[2] @ lg
        dldx2 = dldx2[0:-1]

        dldw2 = self.layer_output[1] @ lg.T
        dldw2 = np.concatenate([dldw2, lg])


        # 4  ## x=prev layer output
        act_der_diag_1 = np.diag(np.squeeze(activation(self.layer_input[1], True)))
        dldx1 = weights[1] @ act_der_diag_1 @ dldx2
        dldx1 = dldx1[0:-1]

        dldw1 = self.layer_output[0] @ dldx2.T @ act_der_diag_1
        dldw1 = np.concatenate([dldw1, (act_der_diag_1 @ dldx2).T])

        act_der_diag_0 = np.diag(np.squeeze(activation(self.layer_input[0], True)))
        # dldx0 = weights[0] @act_der_diag_0 @ dldx1
        dldw0 = x @ dldx1.T @ act_der_diag_0
        dldw0 = np.concatenate([dldw0, (act_der_diag_0 @ dldx1).T])
        dldw = [dldw0, dldw1, dldw2]
        return w_mat2vec(dldw)

    # 1.3.9
    def eval_grad_of_loss_all(self, x, y):
        def grad_all(weights):
            n = x.shape[1]
            mean_grad = 0
            w_mat = w_vec2mat(weights)
            for i in range(n):
                x_2dim = np.expand_dims(x[:, i], axis=1)
                single_loss_w = self.eval_grad_of_loss_single(x_2dim, y[i], w_mat)
                mean_grad += single_loss_w*((self.forward_pass(x_2dim)(w_mat) - y[i]) ** 2)
                # mean_grad += single_loss * self.loss_func(np.expand_dims(x[:, i], axis=1), y[i], num_samples)(w_mat)
            return mean_grad.T/n

        return grad_all

    def loss_func(self, x, y, n):
        def lf(weights):  # estimated is F(x;W)- the output of the model, truth= y the given labels
            w_mat = w_vec2mat(weights)
            return np.sum((self.forward_pass(x)(w_mat) - y[i]) ** 2) / n
        return lf

# 1.3.13
def data_vis(f, x=[], y=[], b=False):  # incomplete
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(-2, 2, 0.2)
    Y = np.arange(-2, 2, 0.2)
    X, Y = np.meshgrid(X, Y)
    Z = f(np.row_stack([np.reshape(X, -1), np.reshape(Y, -1)]))
    # Plot the surface.
    surf = ax.plot_surface(X, Y, np.reshape(Z, (20, 20)), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    if b is True:
        ax.scatter(x[0, :], x[1, :], y)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

# 1.3.14
nn = NN()
train_x, train_y = gen_training(500)
test_set_x = gen_test(200)

for i in range(4):
    e = 10 ** -(i + 1)
    print('epsilon: ', e)
    w0 = w_mat2vec(init_weights())
    grad_fun = nn.eval_grad_of_loss_all(train_x, train_y)
    lf = nn.loss_func(train_x, train_y, 500)
    # test_grad = grad_fun(w0)
    # test_loss = lf(w0)
    # func -  feed forward and calculate loss for a given W where x and y are constants
    # gradient - calculate for a given W where x and y are constants
    w_trail = bfgs.BFGS(lf, grad_fun, w0[:, np.newaxis], bfgs.inexact_line_search, False, e)[1]
    w_opt = w_trail[-1, :]
    w_opt_mat = w_vec2mat(w_opt)

    # b1
    fp_w = nn.forward_pass_const_w(w_opt_mat)
    data_vis(fp_w)
    # b2
    y_test = nn.forward_pass(test_set_x)(w_opt_mat)
    data_vis(test_func, test_set_x, y_test, b=True)
    # print('descent: ', desc)
