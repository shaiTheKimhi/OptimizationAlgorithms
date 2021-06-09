import numpy as np
import descent as dc
import newton_method as nm
import quadratic as qd
import Rosenbrock as rb
import matplotlib.pyplot as plt
import BFGS as bfgs

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

#BFGS:
x0 = np.zeros(10)
f = rb.func()
desc = bfgs.BFGS(rb.func(), rb.gradient(), x0, bfgs.inexact_line_search, False)[1]
print(desc[-1])

trail = f(desc.T)
f_opt = trail[-1] #we might preffer using analytical optimal f for rosebrock
f_opt = f_opt*np.ones(trail.shape[0])

k = np.arange(trail.shape[0])

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(k, trail-f_opt)
plt.yscale("log")

plt.show()