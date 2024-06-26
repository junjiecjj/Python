










# Bk4_Ch3_01.py

import matplotlib.pyplot as plt
import numpy as np

p_values = [0.05, 0.2, 0.5, 1, 1.5, 2, 4, 8, np.inf]

x1 = np.linspace(-2.5, 2.5, num=101);
x2 = x1;

xx1, xx2 = np.meshgrid(x1,x2)

fig, axes = plt.subplots(ncols=3,nrows=3,
                         figsize=(12, 12))

for p, ax in zip(p_values, axes.flat):

    if np.isinf(p):
        zz = np.maximum(np.abs(xx1),np.abs(xx2))
    else:
        zz = ((np.abs((xx1))**p) + (np.abs((xx2))**p))**(1./p)

    # plot contour of Lp
    ax.contourf(xx1, xx2, zz, 20, cmap='RdYlBu_r')

    # plot contour of Lp = 1
    ax.contour (xx1, xx2, zz, [1], colors='k', linewidths = 2)

    # decorations

    ax.axhline(y=0, color='k', linewidth = 0.25)
    ax.axvline(x=0, color='k', linewidth = 0.25)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('p = ' + str(p))
    ax.set_aspect('equal', adjustable='box')

plt.show()











# Bk4_Ch3_02.py

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

u = [0,0,4, 3]
v = [0,0,-2,4]
u_bis = [4,3,v[2],v[3]]
w = [0,0,2,7]

fig, ax = plt.subplots()

plt.quiver([u[0], u_bis[0], w[0]],
           [u[1], u_bis[1], w[1]],
           [u[2], u_bis[2], w[2]],
           [u[3], u_bis[3], w[3]],
           angles='xy', scale_units='xy',
           scale=1, color=sns.color_palette())

plt.axvline(x=0, color='grey')
plt.axhline(y=0, color='grey')

plt.text(3, 1, r'$||\vec{u}||_2$',
         color=sns.color_palette()[0], size=12,
         ha='center',va='center')

plt.text(3, 6, r'$||\vec{v}||_2$',
         color=sns.color_palette()[1], size=12,
         ha='center',va='center')

plt.text(0, 4, r'$||\vec{u}+\vec{v}||_2$',
         color=sns.color_palette()[2], size=12,
         ha='center',va='center')

plt.ylabel('$x_2$')
plt.xlabel('$x_1$')
plt.axis('scaled')
ax.set_xticks(np.arange(-2,8 + 1))
ax.set_yticks(np.arange(-2,8 + 1))
ax.set_xlim(-2, 8)
ax.set_ylim(-2, 8)
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

# reference: Essential Math for Data Science




















































































































































































































































































