from mcholmz import *
# from scipy.optimize import line_search
# import quadratic as qd
# from scipy.optimize import newton


# def newton_method(f,x,search_type,step_size):
def newton_method(func, gradient, hessian, start_point, learn_rate, exact_ls=True):
    x = np.array(start_point)
    stop_criteria = 10 ** -5
    i = 0

    # if search_type=='exact':
    g = -gradient(x)

    while np.linalg.norm(g) > stop_criteria:
        i += 1
        # we want to solve Hdk=-g
        # 1. find L and d such that H+e=Ldiag(d)L' and LDL'dk=-g
        # 2. solve Ly=-g where y=DL'dk
        # 3. solve Dz=y where z=L'dk
        # 4. solve L'dk=z
        L, D, e = modifiedChol(hessian(x))
        D = np.squeeze(D)
        y = np.linalg.solve(L, g)  # forward_substitution(L,g)
        z = (y/D)
        dk = np.linalg.solve(L.T, z)  # backward_substitution(L,g)
        # dk2 = np.linalg.inv(hessian(x)) @ g
        # Q1 = np.array([[10, 0], [0, 1]])
        # a = line_search(qd.func(Q1), gradient, x, dk)
        if exact_ls is True:
            a = learn_rate(x, dk)
        else:
            a = learn_rate(x, dk, func)
        x += dk * a
        g = -gradient(x)

    return [i] + list(x)  # returns optimal x
