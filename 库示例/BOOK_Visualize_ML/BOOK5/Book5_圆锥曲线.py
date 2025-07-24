
"""

与圆锥曲线相关的章节：
Book03_08, Book03_09, Book04_20, Book05_10

"""


#%% Bk3_Ch8_01

from matplotlib import pyplot as plt
from sympy import plot_implicit, symbols, Eq
x1, x2 = symbols('x1 x2')

def plot_curve(Eq_sym):
    h_plot = plot_implicit(Eq_sym, (x1, -2.5, 2.5), (x2, -2.5, 2.5), xlabel = r'$\it{x_1}$', ylabel = r'$\it{x_2}$')
    h_plot.show()

#%% plot ellipses

plt.close('all')

# major axis on x1
Eq_sym = Eq(x1**2 + x2**2, 1)
plot_curve(Eq_sym)

# major axis on x1
Eq_sym = Eq(x1**2/4 + x2**2, 1)
plot_curve(Eq_sym)

# major axis on x2
Eq_sym = Eq(x1**2 + x2**2/4, 1)
plot_curve(Eq_sym)

# major axis on x1 with center (h,k)
Eq_sym = Eq((x1-0.5)**2/4 + (x2-0.5)**2, 1)
plot_curve(Eq_sym)

# major axis on x2 with center (h,k)
Eq_sym = Eq((x1-0.5)**2 + (x2-0.5)**2/4, 1)
plot_curve(Eq_sym)

# major axis rotated counter clockwise by pi/4, 逆时针旋转 θ = 45° = π/4 获得
Eq_sym = Eq(5*x1**2/8 -3*x1*x2/4 + 5*x2**2/8, 1)
plot_curve(Eq_sym)

# major axis rotated counter clockwise by 3*pi/4, 逆时针旋转 θ = 135° = 3π/4 获
Eq_sym = Eq(5*x1**2/8 +3*x1*x2/4 + 5*x2**2/8, 1)
plot_curve(Eq_sym)



#%% Bk3_Ch9_01

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

x = np.linspace(-4, 4, num = 201)
y = np.linspace(-4, 4, num = 201)
m = 1
n = 1.5
xx, yy = np.meshgrid(x,y);
e_array = np.linspace(0, 3,num = 11)
fig, ax = plt.subplots(figsize=(8, 8))
colors = plt.cm.RdYlBu(np.linspace(0,1,len(e_array)))
for i in range(0,len(e_array)):
    e = e_array[i]
    ellipse = yy**2 - (e**2 - 1)*xx**2 - 2*xx;
    color_code = colors[i,:].tolist()
    plt.contour(xx, yy, ellipse, levels = [0], colors = [color_code])

plt.axvline(x = 0, color = 'k', linestyle = '-')
plt.axhline(y = 0, color = 'k', linestyle = '-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()
plt.close()

#%% Bk3_Ch9_02

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

a = 2
b = 1

x = np.linspace(-4,4,num = 201)
y = np.linspace(-4,4,num = 201)

xx,yy = np.meshgrid(x,y);
# k_array = np.linspace(0, 2, num = 11)
# k_array = np.linspace(0, 1, num = 21)
# k_array = np.linspace(0, -2, num = 21)
k_array = np.linspace(0,  2, num = 21)
fig, ax = plt.subplots(figsize=(8, 8))
colors = plt.cm.RdYlBu(np.linspace(0,1,len(k_array)))
for i in range(0,len(k_array)):
    k = k_array[i]
    ellipse = (xx/a)**2 + (yy/b)**2 - 2*k*(xx/a)*(yy/b);
    color_code = colors[i,:].tolist()
    plt.contour(xx,yy,ellipse,levels = [1], colors = [color_code])

plt.axvline(x = 0, color = 'k', linestyle = '-')
plt.axhline(y = 0, color = 'k', linestyle = '-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()
plt.close()

#%% Bk3_Ch9_03

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

x = np.linspace(-4, 4, num = 201)
y = np.linspace(-4, 4, num = 201)
m = 1
n = 1.5

xx,yy = np.meshgrid(x, y);
rho_array = np.linspace(-0.95, 0, num = 3)
fig, ax = plt.subplots(figsize=(8, 8))
# Create a Rectangle patch
rect = patches.Rectangle((-m, -n), 2*m, 2*n, linewidth = 0.25, edgecolor='k', linestyle = '--', facecolor = 'none')
# Add the patch to the Axes
ax.add_patch(rect)
colors = plt.cm.hsv(np.linspace(0,1,len(rho_array)))
for i in range(0,len(rho_array)):
    rho = rho_array[i]
    ellipse = ((xx/m)**2 - 2*rho*(xx/m)*(yy/n) + (yy/n)**2)/(1 - rho**2);
    color_code = colors[i,:].tolist()
    ax.contour(xx, yy, ellipse, levels = [1], colors = [color_code])
    ax.plot(m, rho * n, marker = 'x', color = 'r', ms = 12)
    ax.plot(rho * m, n, marker = 'x', color = 'b', ms = 12)
    ax.plot(-m, -rho * n, marker = 'x', color = 'g', ms = 12)
    ax.plot(-rho * m, -n, marker = 'x', color = 'c', ms = 12)

plt.axvline(x = 0, color = 'k', linestyle = '-')
plt.axhline(y = 0, color = 'k', linestyle = '-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.show()
plt.close()

#%% Bk3_Ch9_04, 超椭圆：和范数有关
import matplotlib.pyplot as plt
import numpy as np

a = 1;
b = 1;

p = [0.5, 1, 2, 3]
q = p

pp, qq = np.meshgrid(p, q)
pp = pp.flatten()
qq = qq.flatten()

x1 = np.linspace(-2, 2, num=11);
x2 = x1;

xx1, xx2 = np.meshgrid(x1,x2)
fig, axes = plt.subplots(ncols=4, nrows=4,  figsize=(12, 12))
for p, q, ax in zip(pp, qq, axes.flat):
    if np.isinf(p):
        zz = np.maximum(np.abs(xx1/a), np.abs(xx2/b))
    else:
        zz = ((np.abs(xx1/a)**p) + (np.abs(xx2/b)**q)) **(1./q)
    # plot contour of Lp
    ax.contourf(xx1, xx2, zz, 20, cmap='RdYlBu_r')
    # plot contour of Lp = 1
    ax.contour(xx1, xx2, zz, [1], colors='k', linewidths = 2)
    # decorations

    ax.axhline(y=0, color='k', linewidth = 0.25)
    ax.axvline(x=0, color='k', linewidth = 0.25)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title('p = ' + str(p) + 'q = ' + str(q))
    ax.set_aspect('equal', adjustable='box')
plt.show()
plt.close()


#%% Bk4_Ch20_01.py, 图 3. 通过单位圆获得几个不同的旋转椭圆
import numpy as np
import matplotlib.pyplot as plt

alphas = np.linspace(0, 2*np.pi, 100)
# unit circle
r = np.sqrt(1.0)
z1 = r*np.cos(alphas)
z2 = r*np.sin(alphas)
Z = np.array([z1, z2]).T # data of unit circle
# scale
S = np.array([[2, 0],
              [0, 0.5]])
thetas = np.array([0, 30, 45, 60, 90, 120])
for theta in thetas:
    # rotate
    print('==== Rotate ====')
    print(theta)
    theta = theta/180*np.pi
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    # translate
    c = np.array([2, 1])
    X = Z@S@R.T + c;

    Q = R@np.linalg.inv(S)@np.linalg.inv(S)@R.T
    # print('==== Q ====')
    # print(Q)
    LAMBDA, V = np.linalg.eig(Q)
    # print('==== LAMBDA ====')
    # print(LAMBDA)
    # print('==== V ====')
    # print(V)

    x1 = X[:,0]
    x2 = X[:,1]

    fig, ax = plt.subplots(1)
    ax.plot(z1, z2, 'b') # plot the unit circle
    ax.plot(x1, x2, 'r') # plot the transformed shape
    ax.plot(c[0],c[1],'xk') # plot the center

    ax.quiver(0, 0, 1, 0, color = 'b', angles='xy', scale_units='xy', scale=1)
    ax.quiver(0, 0, 0, 1, color = 'b', angles='xy', scale_units='xy', scale=1)
    ax.quiver(0, 0, -1, 0, color = 'b', angles='xy', scale_units='xy', scale=1)
    ax.quiver(0, 0, 0, -1, color = 'b', angles='xy', scale_units='xy', scale=1)

    ax.quiver(0, 0, c[0], c[1],color = 'k', angles='xy', scale_units='xy', scale=1)

    ax.quiver(c[0], c[1], V[0,0]/np.sqrt(LAMBDA[0]), V[1,0]/np.sqrt(LAMBDA[0]), color = 'r', angles='xy', scale_units='xy', scale=1)
    ax.quiver(c[0], c[1], V[0,1]/np.sqrt(LAMBDA[1]), V[1,1]/np.sqrt(LAMBDA[1]), color = 'r', angles='xy', scale_units='xy', scale=1)
    ax.quiver(c[0], c[1], -V[0,0]/np.sqrt(LAMBDA[0]), -V[1,0]/np.sqrt(LAMBDA[0]), color = 'r', angles='xy', scale_units='xy', scale=1)
    ax.quiver(c[0], c[1], -V[0,1]/np.sqrt(LAMBDA[1]), -V[1,1]/np.sqrt(LAMBDA[1]), color = 'r', angles='xy', scale_units='xy', scale=1)

    plt.axvline(x=0, color= 'k', zorder=0)
    plt.axhline(y=0, color= 'k', zorder=0)

    ax.set_aspect(1)
    plt.xlim(-2,4)
    plt.ylim(-2,4)
    plt.grid(linestyle='--')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.show()
plt.close()


#%% Bk4_Ch20_02.py, 图 5. 通过单位双曲线旋转得到的一系列双曲线
import numpy as np
import matplotlib.pyplot as plt

alphas = np.linspace(0, 2*np.pi, 100)

# unit circle
r = np.sqrt(1.0)

z1 = r*1/np.cos(alphas)  # 参数方程
z2 = r*np.tan(alphas)

Z = np.array([z1, z2]).T # data of unit circle

# scale
S = np.array([[1, 0],
              [0, 1]])
thetas = np.array([0, 30, 45, 60, 90, 120])
for theta in thetas:
    # rotate
    # print('==== Rotate ====')
    # print(theta)
    theta = theta/180*np.pi
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    X = Z@S@R.T
    x1 = X[:,0]
    x2 = X[:,1]
    fig, ax = plt.subplots(1)
    ax.plot(z1, z2, 'b') # plot the unit circle
    ax.plot(x1, x2, 'r') # plot the transformed shape

    plt.axvline(x=0, color= 'k', zorder=0)
    plt.axhline(y=0, color= 'k', zorder=0)
    ax.set_aspect(1)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.grid(linestyle='--')

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.show()

#%% Bk4_Ch20_03.py, 图 7. 椭圆切线分布
import numpy as np
import matplotlib.pyplot as plt

a = 1.5
b = 1

x1 = np.linspace(-3,3,200)
x2 = np.linspace(-3,3,200)
xx1,xx2 = np.meshgrid(x1, x2)
theta = np.arange(0, 2*np.pi, 0.1)
fig, ax = plt.subplots(figsize=(10, 10))
theta_array = np.linspace(0, 2*np.pi, 100)
ax.plot(a*np.cos(theta), b*np.sin(theta),color = 'k', lw = 4)
# plt.show()
colors = plt.cm.RdYlBu(np.linspace(0,1,len(theta_array)))
for i in range(len(theta_array)):
    theta = theta_array[i]
    p1 = a*np.cos(theta)
    p2 = b*np.sin(theta)
    tangent = p1*xx1/a**2 + p2*xx2/b**2 - p1**2/a**2 - p2**2/b**2
    colors_i = colors[int(i),:]
    ax.contour(xx1, xx2, tangent, levels = [0], colors = [colors_i])

plt.axis('scaled')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axvline(x=0,color = 'k')
ax.axhline(y=0,color = 'k')
plt.show()


#%%  图 11. 椭圆法线分布
import numpy as np
import matplotlib.pyplot as plt

a = 1.5
b = 1

x1 = np.linspace(-3,3,200)
x2 = np.linspace(-3,3,200)
xx1,xx2 = np.meshgrid(x1, x2)
theta = np.arange(0, 2*np.pi, 0.1)
fig, ax = plt.subplots(figsize=(10, 10))
theta_array = np.linspace(0, 2*np.pi, 100)
ax.plot(a*np.cos(theta), b*np.sin(theta), color = 'k', lw = 4)
# plt.show()
colors = plt.cm.RdYlBu(np.linspace(0,1,len(theta_array)))
for i in range(len(theta_array)):
    theta = theta_array[i]
    p1 = a*np.cos(theta)
    p2 = b*np.sin(theta)
    tangent = p2*xx1/b**2 - p1*xx2/a**2 - p1*p2/b**2 + p1*p2/a**2
    colors_i = colors[int(i),:]
    ax.contour(xx1, xx2, tangent, levels = [0], colors = [colors_i])

plt.axis('scaled')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axvline(x=0,color = 'k')
ax.axhline(y=0,color = 'k')
plt.show()

#%% Bk4_Ch20_03.py, 图 8. 单位圆切线分布
import numpy as np
import matplotlib.pyplot as plt

r = 1.5

x1 = np.linspace(-3,3,200)
x2 = np.linspace(-3,3,200)
xx1,xx2 = np.meshgrid(x1,x2)
theta = np.arange(0, 2*np.pi, 0.1)
fig, ax = plt.subplots(figsize=(10, 10))
theta_array = np.linspace(0, 2*np.pi, 100)
ax.plot(r*np.cos(theta), r*np.sin(theta), color = 'k', lw = 5)
colors = plt.cm.RdYlBu(np.linspace(0,1,len(theta_array)))
for i in range(len(theta_array)):
    theta = theta_array[i]
    p1 = r*np.cos(theta)
    p2 = r*np.sin(theta)
    tangent = p1*xx1 + p2*xx2  - r**2
    colors_i = colors[int(i),:]
    ax.contour(xx1, xx2, tangent, levels = [0], colors = [colors_i])

plt.axis('scaled')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axvline(x=0,color = 'k')
ax.axhline(y=0,color = 'k')
plt.show()



#%% Bk4_Ch20_03.py, 图 10. 双曲线左右两侧切线分布
import numpy as np
import matplotlib.pyplot as plt

a = 1.5
b = 1

x1 = np.linspace(-3, 3, 200)
x2 = np.linspace(-3, 3, 200)
xx1,xx2 = np.meshgrid(x1, x2)
theta = np.arange(0, 2*np.pi, 0.1)
fig, ax = plt.subplots(figsize=(10, 10))
theta_array = np.linspace(0, 2*np.pi, 100)
ax.plot(a/np.cos(theta), b*np.tan(theta), color = 'k', lw = 5)
# plt.show()
colors = plt.cm.RdYlBu(np.linspace(0,1,len(theta_array)))
for i in range(len(theta_array)):
    theta = theta_array[i]
    p1 = a/np.cos(theta)
    p2 = b*np.tan(theta)
    tangent = p1*xx1/a**2 - p2*xx2/b**2 - p1**2/a**2 + p2**2/b**2
    colors_i = colors[int(i),:]
    ax.contour(xx1, xx2, tangent, levels = [0], colors = [colors_i])

plt.axis('scaled')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axvline(x=0,color = 'k')
ax.axhline(y=0,color = 'k')
plt.show()































































































































































































































































































































































































































































































































































































