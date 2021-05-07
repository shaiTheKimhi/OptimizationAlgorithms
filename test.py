import numpy as np
import descent as dc 
import Newton_Method as nm
import quadratic as qd

Q1 = np.array([[10,0],[0,1]])
Q2 = np.array([[3,0],[0,3]])

gradQd = qd.gradient(Q1)
hessianQd = qd.hessian(Q1)
exact_line_search = qd.exact_line_search(Q1)

# print(dc.descent(gradQd, exact_line_search, np.array([-.2,-2])))
print(nm.newton_method(gradQd,hessianQd, exact_line_search, np.array([-.2,-2])))


# res = (dc.descent(gradQd, exact_line_search, np.array([-2.0,0.0])))
res = (nm.newton_method(gradQd,hessianQd, exact_line_search, np.array([-2.0,0.0])))
grad = qd.gradient(Q2)
exact_line_search = qd.exact_line_search(Q2)
# print(dc.descent(gradQd, exact_line_search, np.array([-.2,-2])))
print(nm.newton_method(grad,hessianQd, exact_line_search, np.array([-.2,-2])))
print(res)


