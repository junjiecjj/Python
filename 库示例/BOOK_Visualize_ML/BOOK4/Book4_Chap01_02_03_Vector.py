




# Bk4_Ch2_11.py

import numpy as np
a = np.array([-2, 1, 1])
b = np.array([1, -2, -1])
# a = [-2, 1, 1]
# b = [1, -2, -1]

# calculate cross product of row vectors
a_cross_b = np.cross(a, b)
print(f"a_cross_b = {a_cross_b}")

a_col = np.array([[-2], [1], [1]])
b_col = np.array([[1], [-2], [-1]])

# calculate cross product of column vectors
a_cross_b_col = np.cross(a_col, b_col, axis=0)
print(f"a_cross_b_col = {a_cross_b_col}")


#%% Bk4_Ch2_12.py

import numpy as np
a = np.array([-2, 1, 1])
b = np.array([1, -2, -1])
# a = [-2, 1, 1]
# b = [1, -2, -1]


# calculate element-wise product of row vectors
a_times_b = np.multiply(a, b)
a_times_b_2 = a*b

a_col = np.array([[-2], [1], [1]])
b_col = np.array([[1], [-2], [-1]])

# calculate element-wise product of column vectors
a_times_b_col = np.multiply(a_col, b_col)
a_times_b_col_2 = a_col*b_col



#%% Bk4_Ch2_13.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_heatmap(x,title):

    fig, ax = plt.subplots()
    ax = sns.heatmap(x,
                     cmap='RdYlBu_r',
                     cbar_kws={"orientation": "horizontal"}, vmin=-1, vmax=1)
    ax.set_aspect("equal")
    plt.title(title)

a = np.array([[0.5],[-0.7],[1],[0.25],[-0.6],[-1]])
b = np.array([[-0.8],[0.5],[-0.6],[0.9]])

a_outer_b = np.outer(a, b)
a_outer_a = np.outer(a, a)
b_outer_b = np.outer(b, b)

# Visualizations
plot_heatmap(a,'a')

plot_heatmap(b,'b')

plot_heatmap(a_outer_b,'a outer b')

plot_heatmap(a_outer_a,'a outer a')

plot_heatmap(b_outer_b,'b outer b')





# Bk4_Ch3_01.py

import matplotlib.pyplot as plt
import numpy as np

p_values = [0.05, 0.2, 0.5, 1, 1.5, 2, 4, 8, np.inf]

x1 = np.linspace(-2.5, 2.5, num=101);
x2 = x1;

xx1, xx2 = np.meshgrid(x1,x2)

fig, axes = plt.subplots(ncols=3,nrows=3, figsize=(12, 12))
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


fig, axes = plt.subplots(projection='3d', ncols=3, nrows=3, figsize=(12, 12))

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




















































































































































































































































































