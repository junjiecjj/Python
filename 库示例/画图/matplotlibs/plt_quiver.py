#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:45:10 2023

@author: jack

这里是一些常用的图：

柱状图：pyplot.bar

直方图：pyplot.barh

水平直方图：pyplot.broken_barh

等高线图：pyplot.contour

误差线：pyplot.errorbar

柱形图：pyplot.hist

水平柱状图：pyplot.hist2d

饼状图：pyplot.pie

量场图：pyplot.quiver

散点图：pyplot.scatter


"""

import numpy as np
import matplotlib.pyplot as plt

n = 8

# 二维网格坐标
X, Y = np.mgrid[0:n, 0:n]

# U,V 定义方向
U = X + 1
V = Y + 1

# C 定义颜色
C = X + Y

plt.quiver(X, Y, U, V, C)
plt.show()






##================================================================================================
X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
U, V = np.cos(X), np.sin(Y)
fig, ax = plt.subplots()

ax.set_title("pivot='mid'; every third arrow; units='inches'")
Q = ax.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3], units='inches', pivot='mid', color='g')
qk = ax.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
ax.scatter(X[::3, ::3], Y[::3, ::3], color='r', s=5)
plt.show()


##================================================================================================
X = np.arange(-10, 10, 1)
Y = np.arange(-10, 10, 1)
# meshgrid 生成网格，此处生成两个 shape = (20,20) 的 ndarray, 详见参考资料2,3
U, V = np.meshgrid(X, Y)
C = np.sin(U)
fig, ax = plt.subplots()
# 绘制箭头
q = ax.quiver(X, Y, U, V, C)
# 该函数绘制一个箭头标签在 (X, Y) 处， 长度为U, 详见参考资料4
ax.quiverkey(q, X=0.3, Y=1.1, U=10,
             label='Quiver key, length = 10', labelpos='E')
plt.show()



##================================================================================================
X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
U, V = np.cos(X), np.sin(Y)
fig, ax = plt.subplots()

ax.set_title("pivot='mid'; every third arrow; units='inches'")
Q = ax.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3], units='inches', pivot='mid', color='g')
qk = ax.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
ax.scatter(X[::3, ::3], Y[::3, ::3], color='r', s=5)
plt.show()




##================================================================================================
fig, ax = plt.subplots()
# 以水平轴按照 angles 参数逆时针旋转得到箭头方向， units='xy' 指出了箭头长度计算方法
ax.quiver((0, 0), (0, 0), (1, 0), (1, 3), angles=[60, 300], units='xy', scale=1, color='r')
plt.axis('equal')
plt.xticks(range(-5, 6))
plt.yticks(range(-5, 6))
plt.grid()
plt.show()



##================================================================================================
X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
U, V = np.cos(X), np.sin(Y)
fig, ax = plt.subplots()
# M 为颜色矩阵
M = np.hypot(U, V)
ax.set_title("pivot='tip'; scales with x view")
Q = ax.quiver(X, Y, U, V, M, units='xy', scale=1 / 0.15, pivot='tip')
qk = ax.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                  coordinates='figure')
ax.scatter(X, Y, color='r', s=1)
plt.show()




##================================================================================================


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


x, y, z = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))

vx = np.sin(x) * np.cos(y)
vy = np.sin(y) * np.cos(z)
vz = np.sin(z) * np.cos(x)

vec_field = np.stack([vx, vy, vz], axis=-1)



curl = np.gradient(vec_field)

curl_x = curl[2][:, :, :, 1] - curl[1][:, :, :, 2]
curl_y = curl[0][:, :, :, 2] - curl[2][:, :, :, 0]
curl_z = curl[1][:, :, :, 0] - curl[0][:, :, :, 1]




fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')

Ax, Ay, Az = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))

ax.quiver(Ax, Ay, Az, curl_x, curl_y, curl_z, length=3, arrow_length_ratio=0.3, pivot='middle', color=cm.jet(curl_z))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)


filepath2 = '/home/jack/snap/'
out_fig = plt.gcf()
# out_fig .savefig(filepath2+'smooth.eps', format='eps',  bbox_inches = 'tight')
#out_fig .savefig(filepath2+'hh.png',format='png',dpi=1000, bbox_inches = 'tight')
plt.show()











import sympy
import numpy as np
from sympy.functions import exp

#define symbolic vars, function
x1,x2 = sympy.symbols('x1 x2')

f_x = x1*exp(-(x1**2 + x2**2))

print(f_x)

#take the gradient symbolically
grad_f = [sympy.diff(f_x,var) for var in (x1,x2)]
print(grad_f)

f_x_fcn = sympy.lambdify([x1,x2],f_x)

#turn into a bivariate lambda for numpy
grad_fcn = sympy.lambdify([x1,x2],grad_f)

import matplotlib.pyplot as plt

xx1, xx2 = np.meshgrid(np.linspace(-2,2,40),np.linspace(-2,2,40))

# coarse mesh
xx1_, xx2_ = np.meshgrid(np.linspace(-2,2,20),np.linspace(-2,2,20))
V = grad_fcn(xx1_,xx2_)


ff_x = f_x_fcn(xx1,xx2)

color_array = np.sqrt(V[0]**2 + V[1]**2)

# 3D visualization
ax = plt.figure().add_subplot(projection='3d')
ax.plot_wireframe(xx1, xx2, ff_x, rstride=1,
                  cstride=1, color = [0.5,0.5,0.5],
                  linewidth = 0.2)
ax.contour3D(xx1, xx2, ff_x, 20, cmap = 'RdBu_r')

ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.zaxis.set_ticks([])
plt.xlim(-2,2)
plt.ylim(-2,2)
ax.view_init(30, -125)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1,x_2)$')
plt.tight_layout()
plt.show()

# 2D visualization
fig, ax = plt.subplots()

plt.contourf(xx1, xx2, ff_x,20, cmap = 'RdBu_r')

plt.quiver (xx1_, xx2_, V[0], V[1],
            angles='xy', scale_units='xy',
            edgecolor='none', facecolor= 'k')

plt.show()
ax.set_aspect('equal')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
plt.tight_layout()








































































































































































