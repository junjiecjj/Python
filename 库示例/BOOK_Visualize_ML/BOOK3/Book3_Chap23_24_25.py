


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Bk3_Ch23_1.py
# Bk3_Ch24_1

import numpy as np

A = np.array([[1,1],
              [2,4]])

b = np.array([[35],
              [94]])

A_inv = np.linalg.inv(A)

x = A_inv@b
print(x)

x_ = np.linalg.solve(A,b)
print(x_)

from sympy import *
x1, x2 = symbols(['x1', 'x2'])
sol = solve([x1 + x2 - 35, 2*x1 + 4*x2 - 94], [x1, x2])
print(sol)

from sympy.solvers.solveset import linsolve
sol_ = linsolve([x1 + x2 - 35, 2*x1 + 4*x2 - 94], [x1, x2])
print(sol_)


# Bk3_Ch23_2

import numpy as np
import matplotlib.pyplot as plt

def draw_vector(vector, RBG):
    array = np.array([[0, 0, vector[0], vector[1]]], dtype=object)
    X, Y, U, V = zip(*array)
    plt.quiver(X, Y, U, V,angles='xy', scale_units='xy',scale=1,color = RBG)

x1 = np.arange(-25, 25 + 1, step=1);
x2 = np.arange(-25, 25 + 1, step=1);

XX1,XX2 = np.meshgrid(x1,x2);

X = np.column_stack((XX1.ravel(),XX2.ravel()))

A = np.matrix([[1, 1],
               [2, 4]]);

Z = X@A.T;

ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

### base: e1 and e2
fig, ax = plt.subplots()
plt.plot(XX1, XX2, color = [0.8,0.8,0.8])
plt.plot(XX1.T, XX2.T, color = [0.8,0.8,0.8])


a1 = A[:,0].tolist()
a2 = A[:,1].tolist()
b  =  [3, 8]

draw_vector(a1, np.array([0,112,192])/255)
draw_vector(a2, np.array([255,0,0])/255)
draw_vector(b,  np.array([255,125,255])/255)

plt.xlabel('$e_1$')
plt.ylabel('$e_2$')

plt.axis('scaled')
ax.set_xlim([0, 8])
ax.set_ylim([0, 8])

plt.xticks(np.arange(0, 8 + 1, step=2))
plt.yticks(np.arange(0, 8 + 1, step=2))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

### base: a1 and a2
fig, ax = plt.subplots()

plt.plot(ZZ1,ZZ2, color = [0.8,0.8,0.8])
plt.plot(ZZ1.T,ZZ2.T, color = [0.8,0.8,0.8])

draw_vector(a1, np.array([0,112,192])/255)
draw_vector(a2, np.array([255,0,0])/255)
draw_vector(b,  np.array([255,125,255])/255)

plt.axis('scaled')
ax.set_xlim([0, 8])
ax.set_ylim([0, 8])
plt.xticks(np.arange(0, 8 + 1, step=2))
plt.yticks(np.arange(0, 8 + 1, step=2))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)



# Bk3_Ch23_3
import numpy as np
import matplotlib.pyplot as plt

def draw_vector(vector,RBG):
    array = np.array([[0, 0, vector[0], vector[1]]])
    X, Y, U, V = zip(*array)
    plt.quiver(X, Y, U, V,angles='xy', scale_units='xy',scale=1,color = RBG)

x1 = np.arange(-25, 25 + 1, step=1);
x2 = np.arange(-25, 25 + 1, step=1);

XX1,XX2 = np.meshgrid(x1,x2);
X = np.column_stack((XX1.ravel(),XX2.ravel()))

R = np.matrix('1,3; 2,1');

Z = X@R.T;

ZZ1 = Z[:,0].reshape((len(x1), len(x2)))
ZZ2 = Z[:,1].reshape((len(x1), len(x2)))

###
fig, ax = plt.subplots()

plt.plot(XX1,XX2,color = [0.8,0.8,0.8])
plt.plot(XX1.T,XX2.T,color = [0.8,0.8,0.8])

draw_vector([1,2],np.array([0,112,192])/255)
draw_vector([3,1],np.array([255,0,0])/255)

draw_vector([10,10],'#FF99FF')


plt.xlabel('$x_1$ (number of chickens)')
plt.ylabel('$x_2$ (number of rabbits)')

plt.axis('scaled')
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])

plt.xticks(np.arange(0, 10 + 1, step=2))
plt.yticks(np.arange(0, 10 + 1, step=2))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

###
fig, ax = plt.subplots()

plt.plot(ZZ1,ZZ2,color = [0.8,0.8,0.8])
plt.plot(ZZ1.T,ZZ2.T,color = [0.8,0.8,0.8])

draw_vector([1,2],np.array([0,112,192])/255)
draw_vector([3,1],np.array([255,0,0])/255)

draw_vector([10,10],'#FF99FF')


plt.xlabel('$z_1$ (combo A)')
plt.ylabel('$z_2$ (combo B)')

plt.axis('scaled')
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])

plt.xticks(())
plt.yticks(())

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# Bk3_Ch23_4
import numpy as np

A = np.array([[1,1],
              [1,1],
              [2,4],
              [2,4]])

b = np.array([[30],
              [35],
              [90],
              [110]])

x = np.linalg.inv(A.T@A)@A.T@b

print(x)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Bk3_Ch24
# Bk3_Ch24_1

from sympy.abc import a
import numpy as np
import matplotlib.pyplot as plt

def fig_decor(ax):
    plt.xlabel('$x$ (number of chickens)')
    plt.ylabel('$y$ (number of rabbits)')

    plt.axis('scaled')
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 80])

    plt.xticks(np.arange(0, 120 + 1, step=10))
    plt.yticks(np.arange(0, 80 + 1,  step=10))

    plt.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color=[0.8, 0.8, 0.8])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

# generate data
num_chickens = np.array([32, 110, 71, 79, 45, 20, 56, 55, 87, 68, 87, 63, 31, 88])
num_rabbits  = np.array([22, 53, 39, 40, 25, 15, 34, 34, 52 , 41, 43, 33, 24, 52])

# scatter plot
fig, ax = plt.subplots()
plt.scatter(num_chickens, num_rabbits)
fig_decor(ax)

# generate f(a), sum of squared errors (SSE), symbolic
from sympy import *
y_pred = a*num_chickens
f_a_SSE = np.sum((num_rabbits - y_pred)**2)
f_a_SSE = simplify(f_a_SSE)
print(f_a_SSE)

# plot f(a) versus a
a_array = np.linspace(0,1,51)
f_a_SSE_fcn = lambdify(a, f_a_SSE)
SSE_array = f_a_SSE_fcn(a_array)

# first-order differential
df_da_SSE = diff(f_a_SSE, a)
print(df_da_SSE)

# solution of a
a_star_only = solve(df_da_SSE, a)
print(a_star_only)
a_star_only = a_star_only[0].evalf()
SSE_min = f_a_SSE_fcn(a_star_only)
fig, ax = plt.subplots()

plt.plot(a_array, SSE_array)
plt.axvline(x=a_star_only, linestyle = '--')
plt.plot(a_star_only, SSE_min, 'rx', markersize = 16)

plt.xlabel('a, slope')
plt.ylabel('f(a), sum of squared errors, SSE')
ax.set_xlim([a_array.min(), a_array.max()])
ax.set_ylim([0, SSE_array.max()])

ax.grid(linestyle=':', linewidth='0.5', color=[0.8, 0.8, 0.8])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


#  y = ax model
def plot_square(x,y1,y2):
    if y2 > y1:
        temp = y2;
        y2 = y1;
        y1 = temp;
    d = y1 - y2;
    plt.fill(np.vstack((x, x + d, x + d, x)), np.vstack((y2, y2, y1, y1)), facecolor='b', edgecolor='none', alpha = 0.3)

x_array = np.linspace(0,150,10)[:, None]
y_pred = a_star_only*x_array

fig, ax = plt.subplots()
plt.plot(x_array, y_pred, color = 'r')
plt.scatter(num_chickens, num_rabbits)
num_rabbits_predicted = a_star_only*num_chickens
plt.plot(np.vstack((num_chickens,num_chickens)), np.vstack((num_rabbits, num_rabbits_predicted)), color = np.array([255,182,0])/255)

for i in range(0,len(num_rabbits_predicted)):
    plot_square(num_chickens[i],num_rabbits[i],num_rabbits_predicted[i]);

fig_decor(ax)

#%% Bk3_Ch24_2

import numpy as np
import matplotlib.pyplot as plt

def fig_decor(ax):
    plt.xlabel('$x$ (number of chickens)')
    plt.ylabel('$y$ (number of rabbits)')

    plt.axis('scaled')
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 80])

    plt.xticks(np.arange(0, 120 + 1, step=10))
    plt.yticks(np.arange(0, 80 + 1,  step=10))

    plt.minorticks_on()
    ax.grid(which='minor', linestyle=':',
            linewidth='0.5', color=[0.8, 0.8, 0.8])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

num_chickens = np.array([32, 110, 71, 79, 45, 20, 56, 55, 87, 68, 87, 63, 31, 88])
num_rabbits  = np.array([22, 53, 39, 40, 25, 15, 34, 34, 52, 41, 43, 33, 24, 52])

#generate f(a, b), sum of squared errors (SSE), symbolic
from sympy.abc import a, b
from sympy import *

y_pred = a*num_chickens + b
f_ab_SSE = np.sum((num_rabbits - y_pred)**2)
f_ab_SSE = simplify(f_ab_SSE)
print(f_ab_SSE)

# plot f(a) versus a
a_array = np.linspace(0,0.9,40)
b_array = np.linspace(-20,36,40)

aa,bb = np.meshgrid(a_array,b_array)
f_ab_SSE_fcn = lambdify((a,b), f_ab_SSE)
SSE_matrix = f_ab_SSE_fcn(aa, bb)
# SSE_matrix = SSE_matrix.evalf()

# first-order partial differential
df_da_SSE = diff(f_ab_SSE, a)
print(df_da_SSE)

df_db_SSE = diff(f_ab_SSE, b)
print(df_db_SSE)

# solution of (a,b)
sol = solve([df_da_SSE, df_db_SSE], [a, b])
print(sol)

a_star = sol[a]
b_star = sol[b]

a_star = a_star.evalf()
b_star = b_star.evalf()

print(a_star)
print(b_star)

SSE_min = f_ab_SSE_fcn(a_star,b_star)
print(SSE_min)

#############
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.plot_wireframe(aa,bb, SSE_matrix, color = [0.5,0.5,0.5], linewidth = 0.25)
plt.plot(a_star, b_star, SSE_min, marker = 'x', markersize = 12)
colorbar = ax.contour(aa,bb, SSE_matrix,30, cmap = 'RdYlBu_r')
fig.colorbar(colorbar, ax=ax)
ax.set_proj_type('ortho')
ax.set_xlabel('$a$, slope')
ax.set_ylabel('$b$, intercept')
ax.set_zlabel('$Sum of squared errors')
plt.tight_layout()
ax.set_xlim(aa.min(), aa.max())
ax.set_ylim(bb.min(), bb.max())
ax.view_init(azim=-135, elev=30)
ax.grid(False)
plt.show()

#############
fig, ax = plt.subplots()
colorbar = ax.contourf(aa,bb, SSE_matrix, 30, cmap='RdYlBu_r')
fig.colorbar(colorbar, ax=ax)
plt.plot(a_star, b_star, marker = 'x', markersize = 12)
ax.set_xlim(aa.min(), aa.max())
ax.set_ylim(bb.min(), bb.max())
ax.set_xlabel('$a$, slope')
ax.set_ylabel('$b$, intercept')
# plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# y = ax + b model
def plot_square(x,y1,y2):
    if y2 > y1:
        temp = y2;
        y2 = y1;
        y1 = temp;
    d = y1 - y2;
    plt.fill(np.vstack((x, x + d, x + d, x)), np.vstack((y2, y2, y1, y1)), facecolor='b', edgecolor='none', alpha = 0.3)

x_array = np.linspace(0,150,10)[:, None]
y_pred = a_star*x_array + b_star
fig, ax = plt.subplots()
plt.plot(x_array, y_pred, color = 'r')
plt.scatter(num_chickens, num_rabbits)
num_rabbits_predicted = a_star*num_chickens + b_star

plt.plot(np.vstack((num_chickens,num_chickens)), np.vstack((num_rabbits, num_rabbits_predicted)), color = np.array([255,182,0])/255)
plt.plot(num_chickens, num_rabbits_predicted, linestyle = 'None', marker = 'x', markerfacecolor = 'darkorange', markeredgecolor = 'darkorange', markersize = 10)
for i in range(0,len(num_rabbits_predicted)):
    plot_square(num_chickens[i],num_rabbits[i],num_rabbits_predicted[i]);

fig_decor(ax)

#%% Bk3_Ch24_3

import numpy as np
import matplotlib.pyplot as plt

def fig_decor(ax):
    plt.xlabel('$x$ (number of chickens)')
    plt.ylabel('$y$ (number of rabbits)')

    plt.axis('scaled')
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 80])

    plt.xticks(np.arange(0, 120 + 1, step=10))
    plt.yticks(np.arange(0, 80 + 1,  step=10))

    plt.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color=[0.8, 0.8, 0.8])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    return

num_chickens = np.array([32, 110, 71, 79, 45, 20, 56, 55, 87, 68, 87, 63, 31, 88])
num_rabbits  = np.array([22, 53, 39, 40, 25, 15, 34, 34, 52, 41, 43, 33, 24, 52])

# scatter plot
fig, ax = plt.subplots()
plt.scatter(num_chickens, num_rabbits)
fig_decor(ax)

#  proportional function, y = ax
x_array = np.linspace(0,150,10)[:, None]

x = num_chickens[:, None]
y = num_rabbits[:, None]

a_star_only = np.linalg.inv(x.T@x)@x.T@y
y_pred = a_star_only*x_array
fig, ax = plt.subplots()
plt.plot(x_array, y_pred, color = 'r')
plt.scatter(num_chickens, num_rabbits)
fig_decor(ax)


#  linear function, y = ax + b
X = np.hstack((np.ones_like(x), x))
sol = np.linalg.inv(X.T@X)@X.T@y

a_star_ = sol[0]
b_star_ = sol[1]


a_star, b_star = np.polyfit(num_chickens, num_rabbits, 1)
y_pred = a_star*x_array + b_star

fig, ax = plt.subplots()

plt.plot(x_array, y_pred, color = 'r')
plt.scatter(num_chickens, num_rabbits)

fig_decor(ax)


#%% Bk3_Ch24_4

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def fig_decor(ax):

    plt.xlabel('$x$ (number of chickens)')
    plt.ylabel('$y$ (number of rabbits)')

    plt.axis('scaled')
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 80])

    plt.xticks(np.arange(0, 120 + 1, step=10))
    plt.yticks(np.arange(0, 80 + 1,  step=10))
    plt.minorticks_on()
    ax.grid(which='minor', linestyle=':',
            linewidth='0.5', color=[0.8, 0.8, 0.8])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

num_chickens = np.array([32, 110, 71, 79, 45, 20, 56, 55, 87, 68, 87, 63, 31, 88])
num_rabbits  = np.array([22, 53, 39, 40, 25, 15, 34, 34, 52, 41, 43, 33, 24, 52])

sigma_X = num_chickens.std(ddof = 1)
sigma_Y = num_rabbits.std(ddof = 1)
rho_XY = np.corrcoef(num_chickens, num_rabbits)[1][0]
mean_X = num_chickens.mean()
mean_Y = num_rabbits.mean()

a = rho_XY*sigma_Y/sigma_X
b = -a*mean_X + mean_Y

print('=== Slope, a ===')
print(a)
print('=== Intercept, b ===')
print(b)

x_array = np.linspace(0,120,20)
fig, ax = plt.subplots()
sns.regplot(x=num_chickens, y=num_rabbits, ax = ax,  truncate=False,
            line_kws={"color": "red"});

plt.plot(mean_X,mean_Y, marker = 'x', markerfacecolor = 'r',
         markersize = 12)
fig_decor(ax)

#  use sklearn

from sklearn.linear_model import LinearRegression

x = num_chickens.reshape((-1, 1))
y = num_rabbits

model = LinearRegression().fit(x, y)
print('Slope, a:', model.coef_)
print('Intercept, b:', model.intercept_)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Bk3_Ch25
# Bk3_Ch25_1

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# transition matrix
T = np.matrix([[0.7, 0.2],
               [0.3, 0.8]])

# pi(i), state vector
pi_i = np.matrix([[0.6],
                  [0.4]])

#%% pi(i) >>> pi(i + 1): pi(i + 1) = T@pi(i)

pi_i_1 = T@pi_i

fig, axes = plt.subplots(1, 5, figsize=(12, 3))

all_max = 1
all_min = 0

plt.sca(axes[0])
ax = sns.heatmap(T,cmap='RdYlBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"}, annot = True,fmt=".3f")
ax.set_aspect("equal")
plt.title('$T$')
plt.yticks(rotation=0)

plt.sca(axes[1])
plt.title('$@$')
plt.axis('off')

plt.sca(axes[2])
ax = sns.heatmap(pi_i,cmap='RdYlBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"}, annot = True,fmt=".3f")
ax.set_aspect("equal")
plt.title('$\pi_{i}$')
plt.yticks(rotation=0)

plt.sca(axes[3])
plt.title('$=$')
plt.axis('off')

plt.sca(axes[4])
ax = sns.heatmap(pi_i_1,cmap='RdYlBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"}, annot = True,fmt=".3f")
ax.set_aspect("equal")
plt.title('$\pi_{i+1}$')
plt.yticks(rotation=0)

#%% pi(i) >>> pi(i + 2): pi(i + 2) = T^2@pi(i)

pi_i_2 = T@T@pi_i

fig, axes = plt.subplots(1, 7, figsize=(12, 3))

plt.sca(axes[0])
ax = sns.heatmap(T,cmap='RdYlBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"}, annot = True,fmt=".3f")
ax.set_aspect("equal")
plt.title('$T$')
plt.yticks(rotation=0)

plt.sca(axes[1])
plt.title('$@$')
plt.axis('off')

plt.sca(axes[2])
ax = sns.heatmap(T,cmap='RdYlBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"}, annot = True,fmt=".3f")
ax.set_aspect("equal")
plt.title('$T$')
plt.yticks(rotation=0)

plt.sca(axes[3])
plt.title('$@$')
plt.axis('off')

plt.sca(axes[4])
ax = sns.heatmap(pi_i,cmap='RdYlBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"}, annot = True,fmt=".3f")
ax.set_aspect("equal")
plt.title('$\pi_{i}$')
plt.yticks(rotation=0)

plt.sca(axes[5])
plt.title('$=$')
plt.axis('off')

plt.sca(axes[6])
ax = sns.heatmap(pi_i_2,cmap='RdYlBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"}, annot = True,fmt=".3f")
ax.set_aspect("equal")
plt.title('$\pi_{i+2}$')
plt.yticks(rotation=0)


# Bk3_Ch25_2

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

T = np.matrix([[0.7, 0.2],
               [0.3, 0.8]])

pi_i = np.matrix([[0.6],
                  [0.4]])

all_max = 1
all_min = 0

pi_array = np.vstack((np.linspace(1,0,11),1 - np.linspace(1,0,11)))
pi_array=np.matrix(pi_array)
num_steps = 12

for ini in np.arange(0,np.shape(pi_array)[1]):
    pi = pi_array[:,ini]
    fig, axes = plt.subplots(1, num_steps + 1, figsize=(12, 3))
    for i in np.arange(0,num_steps + 1):
        plt.sca(axes[i])
        ax = sns.heatmap(pi,cmap='RdYlBu_r',vmax = all_max,vmin = all_min, annot = True,fmt=".3f",cbar=False, xticklabels=False, yticklabels=False)
        ax.set_aspect("equal")
        plt.title('$\pi(' + str(i) + ')$')
        ax.tick_params(left=False, bottom=False)
        pi = T@pi

# Bk3_Ch25_3

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def draw_vector(vector,RBG, ax):
    ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color = RBG)

T = np.matrix([[0.7, 0.2],
               [0.3, 0.8]])
all_max = 1
all_min = 0

x1 = np.linspace(-1.1, 1.1, num=201)
x2 = x1
xx1, xx2 = np.meshgrid(x1,x2)
zz = ((np.abs((xx1))**2) + (np.abs((xx2))**2))**(1./2)

pi_array = np.vstack((np.linspace(1,0,11),1 - np.linspace(1,0,11)))
pi_array=np.matrix(pi_array)
num_steps = 12

colors = plt.cm.rainbow(np.linspace(0,1,num_steps + 1))

for ini in np.arange(0,np.shape(pi_array)[1]):
    pi = pi_array[:,ini]
    fig, ax = plt.subplots(figsize=(10, 10))
    # plot a reference line
    plt.plot(x1,1-x1,color = 'k', linestyle = '--')

    # plot a unit circle as reference
    plt.contour(xx1, xx2, zz, levels = [1], colors='k', linestyles = ['--'])

    for i in np.arange(0,num_steps + 1):
        # plot normalized vector
        draw_vector(pi/np.linalg.norm(pi), colors[i], ax)
        # plot original vector
        draw_vector(pi, colors[i], ax)
        ax.tick_params(left=False, bottom=False)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        # plt.axis('off')
        ax.axvline(x = 0, color = 'k')
        ax.axhline(y = 0, color = 'k')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(color = [0.8,0.8,0.8])
        plt.xticks(np.linspace(-1,1,21))
        plt.yticks(np.linspace(-1,1,21))

        pi = T@pi
        # update pi




















































































































































































































































