import numpy as np
import descent as dc 
import quadratic as qd

stop_critirea = 10**-5
Q1 = np.array([[10,0],[0,1]])
Q2 = np.array([[3,0],[0,3]])

grad = qd.gradient(Q1)
exact_line_search = qd.exact_line_search(Q1)

num_epochs = int(input())
print(dc.descent(grad, exact_line_search, num_epochs, np.array([-.2,-2])))


res = (dc.descent(grad, exact_line_search, num_epochs, np.array([-2.0,0.0])))
grad = qd.gradient(Q2)
exact_line_search = qd.exact_line_search(Q2)
print(dc.descent(grad, exact_line_search, num_epochs, np.array([-.2,-2])))

print(res)


