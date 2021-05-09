import numpy as np
import descent as dc
import newton_method as nm
import quadratic as qd
from scipy.optimize import newton

Q1 = np.array([[10, 0], [0, 1]])
Q2 = np.array([[3, 0], [0, 3]])

gradQ1d = qd.gradient(Q1)
hessianQ1d = qd.hessian(Q1)
exact_line_searchQ1 = qd.exact_line_search(Q1)

# x0 = newton(qd.func(Q2),  np.array([-.2, -2]), fprime=None, args=(), tol=10 ** -5, maxiter=55000, fprime2=None)
# print("x0:", x0)
print("case_1_GD:", dc.descent(gradQ1d, exact_line_searchQ1, np.array([-.2, -2])))
print("case_1_NM:", nm.newton_method(gradQ1d, hessianQ1d, exact_line_searchQ1, np.array([-.2, -2])))


print("case_3_GD:", dc.descent(gradQ1d, exact_line_searchQ1, np.array([-2.0, 0.0])))
print("case_3_NM:", nm.newton_method(gradQ1d, hessianQ1d, exact_line_searchQ1, np.array([-2.0, 0.0])))

gradQ2d = qd.gradient(Q2)
hessianQ2d = qd.hessian(Q2)
exact_line_searchQ2 = qd.exact_line_search(Q2)
print("case_2_GD:", dc.descent(gradQ2d, exact_line_searchQ2, np.array([-.2, -2])))
print("case_2_NM:", nm.newton_method(gradQ2d, hessianQ2d, exact_line_searchQ2, np.array([-.2, -2])))



# 2.7 Find the Minimum of the Rosenbrock Function
# f*=0, plot f(x_k)-f* for GD and NM
