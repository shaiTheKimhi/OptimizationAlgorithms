import numpy as np
from mcholmz import *

# def newton_method(f,x,search_type,step_size):
def newton_method(gradient,hessian, learn_rate, start_point)
    x = np.array(start_point)
    stop_criterea = 10 ** -5
    i = 0

    # if search_type=='exact':
    g= -gradient(x)
    while np.linalg.norm(g) > stop_criterea:
        i += 1
        # we want to solve Hdk=-g
        # 1. find L and d such that H+e=Ldiag(d)L'
        # 2. solve Ly=g where y=DL'dk
        # 3. solve Dz=y where z=L'dk
        # 4. solve L'dk=z
        L, d, e=modifiedChol(hessian(x))
        D=np.diag(d)
        y= forward_substitution(L,g)
        z = D/y;
        dk = backward_substitution(L.T, z);

        x += dk * learn_rate(x,g)
        g = -gradient(x)

    return [i] + list(x) #retuns optimal x

    # else:
    #     while err > 0.001:
    #         dk=grad(f,prev_x);
    #         step_size =
    #         next_x=prev_x-step_size*dk
    #         d_err=(f(next_x)-f(prev_x))/err
    #         err=(f(next_x)-f(prev_x))

def forward_substitution(L,b):

    return 0

def backward_substitution(L,b):

    return 0