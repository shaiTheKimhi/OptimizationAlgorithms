import numpy as np
import descent as dc
import newton_method as nm
import quadratic as qd
import Rosenbrock as rb
import matplotlib.pyplot as plt
from scipy.optimize import newton

def graph(x,y,z, desc, title):
    fig = plt.figure(figsize=(10.0,5.0))
    gs = fig.add_gridspec(1, 2)
    ax = fig.add_subplot(gs[0], projection="3d")
    ax.plot_surface(x, y, z, color='blue', rstride=1, cstride=1, lw=0.5)
    ax.contour(x, y, z, 10, lw=3, cmap="autumn_r", linestyles="solid")
    ax.contour(x, y, z, 10, lw=3, cmap="autumn_r", linestyles="solid", offset=-1)
    ax.plot(desc[0], desc[1], zs=0, zdir='z', marker='o', markersize=5)
    ax.set_title(title + " (3D)")
    #plt.show()


    ax = fig.add_subplot(gs[1])
    ax.contour(x, y, z)
    ax.plot(desc[0], desc[1], marker='o', markersize=5)
    ax.set_title(title + " (2D)")
    plt.show()



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

x = np.outer(np.linspace(-2,2,30), np.ones(30))
y = x.copy().T

x1 = x.reshape(900,)
y1 = y.reshape(900,)
d = np.array([x1,y1])




#plot setting 1 -GD:
z = np.array([fQ1(v) for v in d.T])
z = z.reshape(30,30)
desc = dc.descent(fQ1, gradQ1d, x1_2, exact_line_searchQ1, exact_ls=True)[1].T

graph(x, y, z, desc,"Setting 1 -GD exact line search")
#setting 1- NM:
desc = nm.newton_method(fQ1, gradQ1d, hessianQ1d, x1_2, exact_line_searchQ1, exact_ls=True)[1].T

graph(x, y, z, desc, "Setting 1 -NM exact line search")

#setting 2- GD
z = np.array([fQ2(v) for v in d.T])
z = z.reshape(30,30)
desc = dc.descent(fQ2, gradQ2d, x1_2, exact_line_searchQ2, exact_ls=True)[1].T

graph(x, y, z, desc, "Setting 2 -GD exact line search")

#setting 2-NM
desc = nm.newton_method(fQ2, gradQ2d, hessianQ2d, x1_2, exact_line_searchQ2, exact_ls=True)[1].T

graph(x, y, z, desc, "Setting 2 -NM exact line search")

#setting 3-GD
z = np.array([fQ1(v) for v in d.T])
z = z.reshape(30,30)
desc = dc.descent(fQ1, gradQ1d, x3, exact_line_searchQ1, exact_ls=True)[1].T

graph(x, y, z, desc, "Setting 3 -GD exact line search")

#setting 3-NM
desc = nm.newton_method(fQ1, gradQ1d, hessianQ1d, x3, exact_line_searchQ1, exact_ls=True)[1].T

graph(x, y, z, desc, "Setting 3 -NM exact line search")

#setting 4-GD
desc = dc.descent(fQ1, gradQ1d, x1_2, dc.inexact_line_search, exact_ls=False)[1].T

graph(x, y, z, desc, "Setting 4 -GD inexact line search")

#setting 4-NM
desc = nm.newton_method(fQ1, gradQ1d, hessianQ1d, x1_2, dc.inexact_line_search, exact_ls=False)[1].T

graph(x, y, z, desc, "Setting 4 -NM inexact line search")

#setting 5-GD
z = np.array([fQ2(v) for v in d.T])
z = z.reshape(30,30)
desc = dc.descent(fQ2, gradQ2d, x1_2, dc.inexact_line_search, exact_ls=False)[1].T

graph(x, y, z, desc, "Setting 5 -GD inexact line search")

#setting 5-NM
desc = nm.newton_method(fQ2, gradQ2d, hessianQ2d, x1_2, dc.inexact_line_search, exact_ls=False)[1].T

graph(x, y, z, desc, "Setting 5 -NM inexact line search")

#setting 6-GD
z = np.array([fQ1(v) for v in d.T])
z = z.reshape(30,30)
desc = dc.descent(fQ1, gradQ1d, x3, dc.inexact_line_search, exact_ls=False)[1].T

graph(x, y, z, desc, "Setting 6 -GD inexact line search")

#setting 6-NM
desc = nm.newton_method(fQ1, gradQ1d, hessianQ1d, x3, dc.inexact_line_search, exact_ls=False)[1].T

graph(x, y, z, desc, "Setting 6 -NM inexact line search")



#TODO: make plots for rosenbrook function


# print(dc.descent(gradQ1d, dc.inexact_line_search, np.array([-.2, -2]), qd.func(Q1)))


# 2.7 Find the Minimum of the Rosenbrock Function
# f*=0, plot f(x_k)-f* for GD and NM
#GD:
x0 = np.zeros(10)
f = rb.func()
desc = dc.descent(rb.func(), rb.gradient(), x0, dc.inexact_line_search, False)[1]
trail = f(desc.T)
f_opt = trail[-1] #we might preffer using analytical optimal f for rosebrock
f_opt = f_opt*np.ones(trail.shape[0])

k = np.arange(trail.shape[0])

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(k, trail-f_opt)
plt.yscale("log")

plt.show()

#NM:
desc = nm.newton_method(rb.func(), rb.gradient(), rb.hessian(), x0, dc.inexact_line_search, False)[1]
trail = f(desc.T)
f_opt = trail[-1] #we might preffer using analytical optimal f for rosebrock
f_opt = f_opt*np.ones(trail.shape[0])

k = np.arange(trail.shape[0])

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(k, trail-f_opt)
plt.yscale("log")

plt.show()
#print("Rosenbrock_GD:", desc)
#print("Rosenbrock_NM:", )