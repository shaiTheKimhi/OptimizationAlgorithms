import numpy as np
import descent as dc
import newton_method as nm
import quadratic as qd
import Rosenbrock as rb
from scipy.optimize import newton

Q1 = np.array([[10, 0], [0, 1]])
Q2 = np.array([[3, 0], [0, 3]])

fQ1 = qd.func(Q1)
gradQ1d = qd.gradient(Q1)
hessianQ1d = qd.hessian(Q1)
exact_line_searchQ1 = dc.exact_line_search(Q1)

fQ2 = qd.func(Q2)
gradQ2d = qd.gradient(Q2)
hessianQ2d = qd.hessian(Q2)
exact_line_searchQ2 = dc.exact_line_search(Q2)

x1_2 = np.array([-0.2, -2])
x3 = np.array([-2, 0.0])

print("case_1_GD:", dc.descent(fQ1, gradQ1d, x1_2, exact_line_searchQ1, exact_ls=True))
print("case_1_NM:", nm.newton_method(fQ1, gradQ1d, hessianQ1d, x1_2, exact_line_searchQ1, exact_ls=True))

print("case_2_GD:", dc.descent(fQ2, gradQ2d, x1_2, exact_line_searchQ2, exact_ls=True))
print("case_2_NM:", nm.newton_method(fQ2, gradQ2d, hessianQ2d, x1_2, exact_line_searchQ2, exact_ls=True))

print("case_3_GD:", dc.descent(fQ1, gradQ1d, x3, exact_line_searchQ1, exact_ls=True))
print("case_3_NM:", nm.newton_method(fQ1, gradQ1d, hessianQ1d, x3, exact_line_searchQ1, exact_ls=True))

print("case_4_GD:", dc.descent(fQ1, gradQ1d, x1_2, dc.inexact_line_search, exact_ls=False))
print("case_4_NM:", nm.newton_method(fQ1, gradQ1d, hessianQ1d, x1_2, dc.inexact_line_search, exact_ls=False))

print("case_5_GD:", dc.descent(fQ2, gradQ2d, x1_2, dc.inexact_line_search, exact_ls=False))
print("case_5_NM:", nm.newton_method(fQ2, gradQ2d, hessianQ2d, x1_2, dc.inexact_line_search, exact_ls=False))

print("case_6_GD:", dc.descent(fQ1, gradQ1d, x3, dc.inexact_line_search, exact_ls=False))
print("case_6_NM:", nm.newton_method(fQ1, gradQ1d, hessianQ1d, x3, dc.inexact_line_search, exact_ls=False))


# print(dc.descent(gradQ1d, dc.inexact_line_search, np.array([-.2, -2]), qd.func(Q1)))


# 2.7 Find the Minimum of the Rosenbrock Function
# f*=0, plot f(x_k)-f* for GD and NM
x0 = np.zeros(10)
print("Rosenbrock_GD:", dc.descent(rb.func(), rb.gradient(), x0, dc.inexact_line_search, False))
print("Rosenbrock_NM:", nm.newton_method(rb.func(), rb.gradient(), rb.hessian(), x0, dc.inexact_line_search, False))












