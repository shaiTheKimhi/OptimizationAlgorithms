import numpy as np
eps = 0.00001  #regularization
#f(x) = 1/2(x'Qx)
def func(Q):
    def f(x):
        return 0.5*(x.T @ Q @ x)
    return f

#Gf(x) = 1/2(Qx+Q'x)
def gradient(Q):
    def gf(x):
        return 0.5*(Q @ x+Q.T @ x) 
    return gf

def hessian(Q):
    def hf(x):
        return 0.5(Q + Q.T)
    return hf
    
def exact_line_search(Q): 
    def els(x ,d): #d- conjugate gradient direction x-current parameters
        Qd = Q @ d
        return -(x.T @ Qd) / ((d.T @ Qd) + np.ones(tuple(x.shape))*eps)#returns learning rate which yields optimal value
    return els
