#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:07:13 2024

@author: jack
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
# plt.rcParams['font.size'] = 18               # 设置全局字体大小
# plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
# plt.rcParams['axes.linewidth'] = 1
# plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
# plt.rcParams['xtick.labelsize'] = 16         # 设置 x 轴刻度字体大小
# plt.rcParams['ytick.labelsize'] = 16         # 设置 y 轴刻度字体大小
# plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
# plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# # plt.rcParams['figure.dpi'] = 300           # 每英寸点数
# plt.rcParams['lines.linestyle'] = '-'
# plt.rcParams['lines.linewidth'] = 2          # 线条宽度
# plt.rcParams['lines.color'] = 'blue'
# plt.rcParams['lines.markersize'] = 6         # 标记大小
# # plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
# plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
# plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
# plt.rcParams['axes.spines.left'] = 1
# plt.rcParams['axes.spines.left'] = 1
# plt.rcParams['legend.fontsize'] = 18
# plt.rcParams['legend.labelspacing'] = 0.2

#%% Bk_2_Ch18_01 各种<参数方程>曲面, 参见Bk_2_Ch18_01，Book2_chap23
import numpy as np
import matplotlib.pyplot as plt

# 正球体
u = np.linspace(0, np.pi, 101)
v = np.linspace(0, 2 * np.pi, 101)

x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot_surface(x, y, z, rstride=2, cstride=2, cmap = 'RdYlBu', edgecolors='k')
ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))
# fig.savefig('Figures/正球体.svg', format='svg')
ax.set_title('正球体')
plt.show()


# 游泳圈
n = 100

theta = np.linspace(0, 2.*np.pi, n)
phi = np.linspace(0, 2.*np.pi, n)
theta, phi = np.meshgrid(theta, phi)
c, a = 2, 1
x = (c + a*np.cos(theta)) * np.cos(phi)
y = (c + a*np.cos(theta)) * np.sin(phi)
z = a * np.sin(theta)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot_surface(x, y, z, rstride=2, cstride=2, cmap = 'RdYlBu', edgecolors='k')
ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
ax.set_zlim(-3,3)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))
ax.set_title('游泳圈')
# fig.savefig('Figures/游泳圈.svg', format='svg')
plt.show()

#  “花篮”
dphi, dtheta = np.pi / 250.0, np.pi / 250.0
[phi, theta] = np.mgrid[0:np.pi + dphi * 1.5:dphi, 0:2 * np.pi + dtheta * 1.5:dtheta]
m0 = 4; m1 = 3; m2 = 2; m3 = 3;
m4 = 6; m5 = 2; m6 = 6; m7 = 4;

# 参数方程
r = (np.sin(m0 * phi) ** m1 + np.cos(m2 * phi) ** m3 + np.sin(m4 * theta) ** m5 + np.cos(m6 * theta) ** m7)
x = r * np.sin(phi) * np.cos(theta)
y = r * np.cos(phi)
z = r * np.sin(phi) * np.sin(theta)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot_surface(x, y, z, rstride=3, cstride=3, cmap = 'RdYlBu', edgecolors='k')
ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))
ax.set_title('花篮')
# fig.savefig('Figures/花篮.svg', format='svg')
plt.show()


#  “螺旋”
s = np.linspace(0, 2 * np.pi, 240)
t = np.linspace(0, np.pi, 240)
tGrid, sGrid = np.meshgrid(s, t)

r = 2 + np.sin(7 * sGrid + 5 * tGrid)  # r = 2 + sin(7s+5t)
x = r * np.cos(sGrid) * np.sin(tGrid)  # x = r*cos(s)*sin(t)
y = r * np.sin(sGrid) * np.sin(tGrid)  # y = r*sin(s)*sin(t)
z = r * np.cos(tGrid)                  # z = r*cos(t)


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot_surface(x, y, z, rstride=3, cstride=3, cmap = 'RdYlBu', edgecolors='k')
ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))
ax.set_title('“螺旋”')
plt.show()


#%%  Bk_2_Ch18_02 # 三维柱状图
import matplotlib.pyplot as plt
import numpy as np
from sympy.abc import x, y
from sympy import *

## 创建数据
f_xy = exp(- x**2 - y**2);
f_xy_fcn = lambdify([x,y],f_xy)
a = -2; b = 1
c = -1; d = 2
x_array_fine = np.linspace(a,b,300)
y_array_fine = np.linspace(c,d,300)

xx_fine,yy_fine = np.meshgrid(x_array_fine,y_array_fine)
zz_fine = f_xy_fcn(xx_fine, yy_fine)

## 可视化
num_array = [5,20]
for num in num_array:
    x_array = np.linspace(a,b - (b - a)/num,num)
    y_array = np.linspace(c,d - (d - c)/num,num)
    xx,yy = np.meshgrid(x_array,y_array)

    xx_array = xx.ravel()
    yy_array = yy.ravel()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    zz_array = np.zeros_like(yy_array)

    dx = np.ones_like(yy_array)/num*(b - a)
    dy = np.ones_like(yy_array)/num*(d - c)
    dz = f_xy_fcn(xx_array, yy_array)

    # 三维树状图
    ax.bar3d(xx_array, yy_array, zz_array,
             dx, dy, dz, shade=False,
             color = '#DEEAF6',
             edgecolor = '#B2B2B2')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim((a,b))
    ax.set_ylim((c,d))
    ax.set_zlim((0,zz_fine.max()))
    ax.grid(False)
    ax.view_init(azim=-135, elev=30)
    ax.set_proj_type('ortho')
    # fig.savefig('Figures/' + str(num) + '.svg', format='svg')

#%% Bk_2_Ch18_03 三角网格, 参考 Bk_2_Ch32_06, BK_2_Ch10_07, Bk_2_Ch18_03
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# 正球体，规则三角网格
u = np.linspace(0, np.pi, 26)
v = np.linspace(0, 2 * np.pi, 26)
u, v = np.meshgrid(u, v)
u, v = u.flatten(), v.flatten()

r = 1
# z轴坐标网格数据
z = r*np.cos(u)

# x轴坐标网格数据
x = r*np.sin(u)*np.cos(v)

# y轴坐标网格数据
y = r*np.sin(u)*np.sin(v)

# 三角网格化
tri = mtri.Triangulation(u, v)
# 基于u和v，相当于theta-phi平面的网格点的三角化

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.RdYlBu, edgecolor = 'k')
ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))
ax.set_title('正球体 不规则三角网格')
# fig.savefig('Figures/正球体.svg', format='svg')
plt.show()


# 游泳圈，规则三角网格
n = 21

theta = np.linspace(0, 2.*np.pi, n)
phi = np.linspace(0, 2.*np.pi, n)
theta, phi = np.meshgrid(theta, phi)
theta = theta.flatten()
phi = phi.flatten()

c, a = 2, 1
x = (c + a*np.cos(theta)) * np.cos(phi)
y = (c + a*np.cos(theta)) * np.sin(phi)
z = a * np.sin(theta)

# 三角网格化
tri = mtri.Triangulation(theta, phi)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.RdYlBu, edgecolor = 'k')
ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
ax.set_zlim(-3,3)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))
# fig.savefig('Figures/游泳圈.svg', format='svg')
ax.set_title('游泳圈 不规则三角网格')
plt.show()


# 莫比乌斯环，规则三角网格
u = np.linspace(0, 2.0 * np.pi, endpoint=True, num=75)
v = np.linspace(-0.5, 0.5, endpoint=True, num=15)
u, v = np.meshgrid(u, v)
u, v = u.flatten(), v.flatten()

x = (1 + 0.5 * v * np.cos(u / 2.0)) * np.cos(u)
y = (1 + 0.5 * v * np.cos(u / 2.0)) * np.sin(u)
z = 0.5 * v * np.sin(u / 2.0)

# 三角网格化
tri = mtri.Triangulation(u, v)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.RdYlBu, edgecolor = 'k')
ax.set_zlim(-0.6, 0.6)
ax.view_init(azim=-30, elev=45)
ax.set_proj_type('ortho')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))
# fig.savefig('Figures/莫比乌斯环.svg', format='svg')
ax.set_title('莫比乌斯环 不规则三角网格')
plt.show()


# 正球体，不规则三角网格
u = np.random.uniform(low=0.0, high=np.pi, size=1000)
v = np.random.uniform(low=0.0, high=2*np.pi, size=1000)

r = 1
# z轴坐标网格数据
z = r*np.cos(u)

# x轴坐标网格数据
x = r*np.sin(u)*np.cos(v)

# y轴坐标网格数据
y = r*np.sin(u)*np.sin(v)

# 三角网格化
tri = mtri.Triangulation(u, v)
# 基于u和v，相当于theta-phi平面的网格点的三角化

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.RdYlBu, edgecolor = 'k')
ax.view_init(azim=-120, elev=30)
# ax.view_init(azim=-30, elev=45)
ax.set_proj_type('ortho')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))
ax.set_title('正球体，不规则三角网格')
# fig.savefig('Figures/正球体，不规则三角网格.svg', format='svg')
plt.show()

#%% Bk_2_Ch18_04 用voxels绘制几何体
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

filled = np.array([
    [[1, 0, 1], [0, 0, 1], [0, 1, 0]],
    [[0, 1, 1], [1, 0, 0], [1, 0, 1]],
    [[1, 1, 0], [1, 1, 1], [0, 0, 0]]
])

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.set_proj_type('ortho')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlim(0,3)
ax.set_ylim(0,3)
ax.set_zlim(0,3)
ax.set_box_aspect(aspect = (1,1,1))
ax.voxels(np.ones((3, 3, 3)), facecolors='#1f77b430', edgecolors='gray')
# fig.savefig('Figures/voxels_A.svg', format='svg')

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')
ax.set_proj_type('ortho')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlim(0,3)
ax.set_ylim(0,3)
ax.set_zlim(0,3)
ax.set_box_aspect(aspect = (1,1,1))
ax.voxels(filled, facecolors='#1f77b430', edgecolors='gray')
# fig.savefig('Figures/voxels_B.svg', format='svg')
plt.show()

#%%  Bk_2_Ch18_05 # 用voxels展示色彩空间
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def midpoints(x):
    sl = ()
    for _ in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x
## RGB色彩空间
r, g, b = np.indices((17, 17, 17)) / 16.0
rc = midpoints(r)
gc = midpoints(g)
bc = midpoints(b)

# define a sphere about [0.5, 0.5, 0.5]
sphere = (rc - 0.5)**2 + (gc - 0.5)**2 + (bc - 0.5)**2 < 0.5**2

# combine the color components
colors = np.zeros(sphere.shape + (3,))
colors[..., 0] = rc
colors[..., 1] = gc
colors[..., 2] = bc

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')
ax.voxels(r, g, b, sphere,
          facecolors=colors,
          edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
          linewidth=0.5)

ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
# ax.set_zlim(-3,3)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))
# fig.savefig('Figures/RGB色块.svg', format='svg')
ax.set_title('RGB色块')
plt.show()


## CMYK空间
r, theta, z = np.mgrid[0:1:21j, 0:np.pi*2:25j, -0.5:0.5:21j]
x = r*np.cos(theta)
y = r*np.sin(theta)

rc, thetac, zc = midpoints(r), midpoints(theta), midpoints(z)

# define a wobbly torus about [0.7, *, 0]
sphere = (rc - 0.7)**2 + (zc + 0.2*np.cos(thetac*2))**2 < 0.2**2

# combine the color components
hsv = np.zeros(sphere.shape + (3,))
hsv[..., 0] = thetac / (np.pi*2)
hsv[..., 1] = rc
hsv[..., 2] = zc + 0.5
colors = mpl.colors.hsv_to_rgb(hsv)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')

ax.voxels(x, y, z, sphere,
          facecolors=colors,
          edgecolors=np.clip(2*colors - 0.5, 0, 1),  # brighter
          linewidth=0.5)
ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
# ax.set_zlim(-3,3)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect(aspect = (1,1,1))
ax.set_title('CMYK色块')
# fig.savefig('Figures/CMYK色块.svg', format='svg')
plt.show()

#%% Bk_2_Ch18_06 # 可视化单位球体几何转换 , 参见 Bk_2_Ch18_06, BK_2_Ch22_09, Bk_2_Ch27_03

"""
关于三维球等几个形状的说明：

(一)如果只是为了画三维图形，则可以利用$\theta, \phi,r$生成三个二维矩阵X,Y,Z，再画图surface(x,y,z)即可。这时候还可以对x,y,z变换，得到另一组x,y,z画图。这时候也是《参数方程》的方法。参见Bk_2_Ch27_03

(二)如果是 利用二维矩阵$X,Y$和一个z值，生成一组X,Y,z对应的二维矩阵Z，则Z实则是z的等高线，因为Z = fn(X, Y, z) = X^2 + Y^2 + z^2 - 1 , ax.contour(X, Y, Z+z, [z], zdir='z', linewidths = 0.25, colors = '#0066FF', linestyles = 'solid')相当于 X^2 + Y^2 + z^2 - 1 + z = z,则–> X^2 + Y^2 + z^2 = 1，这就是固定z时的等高线。这时实际上是用《隐函数》的方式展示三维几个体。参见 Bk_2_Ch18_06, BK_2_Ch22_09

"""

# 导入包
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, lambdify, expand, simplify


def plot_implicit(fn, X_plot, Y_plot, Z_plot, ax, bbox, filename):
    # 等高线的起止范围
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    ax.set_proj_type('ortho')
    # 绘制三条参考线
    k = 1.5
    ax.plot((xmin * k, xmax * k), (0, 0), (0, 0), 'k', lw = 0.1)
    ax.plot((0, 0), (ymin * k, ymax * k), (0, 0), 'k', lw = 0.1)
    ax.plot((0, 0), (0, 0), (zmin * k, zmax * k), 'k', lw = 0.1)
    # 等高线的分辨率
    A = np.linspace(xmin, xmax, 500)
    # 产生网格数据
    A1, A2 = np.meshgrid(A, A)
    # 等高线的分割位置
    B = np.linspace(xmin, xmax, 40)
    # 绘制 XY 平面等高线
    if X_plot == True:
        for z in B:
            X, Y = A1, A2
            Z = fn(X, Y, z)
            cset = ax.contour(X, Y, Z+z, [z], zdir='z', linewidths = 0.25, colors = '#0066FF', linestyles = 'solid')
    # 绘制 XZ 平面等高线
    if Y_plot == True:
        for y in B:
            X,Z = A1,A2
            Y = fn(X, y, Z)
            cset = ax.contour(X, Y+y, Z, [y], zdir='y', linewidths = 0.25, colors = '#88DD66', linestyles = 'solid')
    # 绘制 YZ 平面等高线
    if Z_plot == True:
        for x in B:
            Y,Z = A1,A2
            X = fn(x, Y, Z)
            cset = ax.contour(X+x, Y, Z, [x], zdir='x', linewidths = 0.25, colors = '#FF6600', linestyles = 'solid')
    ax.quiver(0, 0, 0, xmax, 0, 0, length = 1, color = 'r', normalize=False, arrow_length_ratio = .07, linestyles = 'solid', linewidths = 0.25)
    ax.quiver(0, 0, 0, 0, ymax, 0, length = 1, color = 'g', normalize=False, arrow_length_ratio = .07, linestyles = 'solid', linewidths = 0.25)
    ax.quiver(0, 0, 0, 0, 0, zmax, length = 1, color = 'b', normalize=False, arrow_length_ratio = .07, linestyles = 'solid', linewidths = 0.25)
    # ax.set_zlim(zmin * k,zmax * k)
    # ax.set_xlim(xmin * k,xmax * k)
    # ax.set_ylim(ymin * k,ymax * k)

    ax.set_box_aspect([1,1,1])
    ax.view_init(azim=60, elev=30)
    ax.axis('off')
    # fig.savefig('Figures/' + filename + '.svg', format='svg')
    plt.show()
    plt.close()

##>>>>>>>> 单位球
Identity = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
A = Identity
def unit(X, Y, Z):
    x1, x2, x3 = symbols('x1 x2 x3')
    # 符号向量
    x = np.array([[x1, x2, x3]]).T
    # 二次型
    f = x.T @ np.linalg.pinv(A @ A.T) @ x
    f = f[0][0]
    fcn = lambdify([x1, x2, x3], f)
    ff_xyz = fcn(X, Y, Z)
    return ff_xyz - 1

fig = plt.figure(figsize=(6,6), constrained_layout = True)
ax = fig.add_subplot(111, projection='3d')
plot_implicit(unit, True, True, True, ax, (-1, 1), '单位球')

##>>>>>>>> 缩放
A = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 3]])
def scale_z_rotate_x(X, Y, Z):
    x1, x2, x3 = symbols('x1 x2 x3')
    # 符号向量
    x = np.array([[x1, x2, x3]]).T
    # 二次型
    f = x.T @ np.linalg.pinv(A @ A.T) @ x
    f = f[0][0]
    fcn = lambdify([x1, x2, x3], f)
    ff_xyz = fcn(X, Y, Z)
    return ff_xyz - 1
fig = plt.figure(figsize=(10, 10), constrained_layout = True)
ax = fig.add_subplot(111, projection='3d')
plot_implicit(scale_z_rotate_x, True, True, True, ax, (-4,4), '缩放')

##>>>>>>>> 缩放 --- x旋转
S_z = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 3]])
alpha = np.deg2rad(30)
R_x = np.array([[1, 0,              0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha),  np.cos(alpha)]])
A = R_x @ S_z
def scale_z_rotate_x(X, Y, Z):
    x1, x2, x3 = symbols('x1 x2 x3')
    # 符号向量
    x = np.array([[x1, x2, x3]]).T
    # 二次型
    f = x.T @ np.linalg.pinv(A @ A.T) @ x
    f = f[0][0]
    fcn = lambdify([x1, x2, x3], f)
    ff_xyz = fcn(X, Y, Z)
    return ff_xyz - 1
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
plot_implicit(scale_z_rotate_x, True, True, True, ax, (-4, 4), '缩放 --- x旋转')

##>>>>>>>> 缩放 --- xy旋转
S_z = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 3]])
alpha = np.deg2rad(30)
R_x = np.array([[1, 0,              0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha),  np.cos(alpha)]])
beta = np.deg2rad(45)
# 从 y 正方向看
R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                [0,            1, 0],
                [-np.sin(beta),0, np.cos(beta)]])
A = R_y @ R_x @ S_z
def scale_z_rotate_x_y(X, Y, Z):
    x1, x2, x3 = symbols('x1 x2 x3')
    # 符号向量
    x = np.array([[x1, x2, x3]]).T
    # 二次型
    f = x.T @ np.linalg.pinv(A @ A.T) @ x
    f = f[0][0]
    fcn = lambdify([x1, x2, x3], f)
    ff_xyz = fcn(X, Y, Z)
    return ff_xyz - 1
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
plot_implicit(scale_z_rotate_x_y, True, True, True, ax, (-4, 4), '缩放 --- xy旋转')

##>>>>>>>> 缩放 --- xyz旋转
S_z = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 3]])
alpha = np.deg2rad(30)
R_x = np.array([[1, 0,              0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha),  np.cos(alpha)]])
beta = np.deg2rad(45)
# 从 y 正方向看
R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                [0,            1, 0],
                [-np.sin(beta),0, np.cos(beta)]])
gamma = np.deg2rad(60)
R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                [np.sin(gamma),  np.cos(gamma), 0],
                [0,              0,             1]])
A = R_z @ R_y @ R_x @ S_z
def scale_z_rotate_x_y_z(X, Y, Z):
    x1, x2, x3 = symbols('x1 x2 x3')
    # 符号向量
    x = np.array([[x1, x2, x3]]).T
    # 二次型
    f = x.T @ np.linalg.pinv(A @ A.T) @ x
    f = f[0][0]
    fcn = lambdify([x1, x2, x3], f)
    ff_xyz = fcn(X, Y, Z)
    return ff_xyz - 1
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
plot_implicit(scale_z_rotate_x_y_z, True, True, True, ax, (-4, 4), '缩放 --- xyz旋转')



#%% Bk_2_Ch18_07 # 可视化椭球
# 导入包
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

### 定义函数
def plot_implicit_V(fn,V, azimuth, elevation, xlabel, ylabel, zlabel, bbox=(-2.5,2.5)):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    width = bbox[1]
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(111, projection='3d', proj_type = 'ortho')
    A = np.linspace(xmin, xmax, 200)
    B = np.linspace(xmin, xmax, 25)
    A1,A2 = np.meshgrid(A,A)
    for z in B:
        X,Y = A1,A2
        Z = fn(X,Y,z)
        ax.contour(X, Y, Z+z, [z], zdir='z', colors = [[0.5, 0.5, 0.5]], )
    for y in B:
        X,Z = A1,A2
        Y = fn(X,y,Z)
        ax.contour(X, Y+y, Z, [y], zdir='y', colors = [[0.5, 0.5, 0.5]],  )
    for x in B:
        Y,Z = A1,A2
        X = fn(x,Y,Z)
        ax.contour(X+x, Y, Z, [x], zdir='x', colors = [[0.5, 0.5, 0.5]], )
    colors = ['b', 'r', 'g']
    # 增加三个箭头，表示椭球的三个主轴
    for i in np.arange(0,3):
        vector = V[:,i]
        v = np.array([vector[0],vector[1],vector[2]])
        vlength=np.linalg.norm(v)
        ax.quiver(0,0,0,vector[0],vector[1],vector[2], length=vlength, color = colors[i])
    x, y, z = np.array([[-width,0,0],[0,-width,0],[0,0,-width]])
    u, v, w = np.array([[2*width,0,0],[0,2*width,0],[0,0,2*width]])
    ax.quiver(x,y,z,u,v,w, arrow_length_ratio=0, color="black")

    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(azim = azimuth, elev = elevation)
    ax.set_box_aspect([1,1,1])
    ax.set_proj_type('ortho')
    # fig.savefig('Figures/几何体投影_azimuth_' + str(azimuth) + '_elevation_' + str(elevation) + '.svg', format='svg')

### 处理数据
# 导入鸢尾花数据
iris_sns = sns.load_dataset("iris")
# A copy from Seaborn
iris = load_iris()
# A copy from Sklearn
X = iris.data
y = iris.target
feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$',
                 'Petal length, $X_3$','Petal width, $X_4$']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)
SIGMA_all = X_df.cov().to_numpy()
# 计算协方差矩阵
SIGMA = SIGMA_all[0:3,0:3]
# 提取前三个特征的协方差矩阵
def cov_ellipsoid(X, Y, Z):
    x1, x2, x3 = symbols('x1 x2 x3')
    # 符号向量
    x = np.array([[x1, x2, x3]]).T
    # 二次型
    f = x.T @ np.linalg.pinv(SIGMA) @ x
    f = f[0][0]
    fcn = lambdify([x1, x2, x3], f)
    ff_xyz = fcn(X, Y, Z)
    return ff_xyz - 1
Lambda,V = np.linalg.eig(SIGMA)
plot_implicit_V(cov_ellipsoid, V,  -145, 30, 'x1','x2','x3', bbox=(-2.,2.))

plot_implicit_V(cov_ellipsoid, V, -90, 90,  'x1','x2','x3', bbox=(-2.,2.)) # x1-x2

plot_implicit_V(cov_ellipsoid, V, -90, 0, 'x1','x2','x3',  bbox=(-2.,2.)) # x1-x3

plot_implicit_V(cov_ellipsoid, V, 0, 0, 'x1','x2','x3', bbox=(-2.,2.)) # x2-x3



### 正椭球  利用特征值分解得到正椭球的主轴长度
Lambda,V = np.linalg.eig(SIGMA)

SIGMA_Z = np.diag(Lambda)
def cov_ellipsoid_orthg(X, Y, Z):
    x1, x2, x3 = symbols('x1 x2 x3')
    # 符号向量
    x = np.array([[x1, x2, x3]]).T
    # 二次型
    f = x.T @ np.linalg.pinv(SIGMA_Z) @ x
    f = f[0][0]
    fcn = lambdify([x1, x2, x3], f)
    ff_xyz = fcn(X, Y, Z)
    return ff_xyz - 1
plot_implicit_V(cov_ellipsoid_orthg, np.identity(3),  -145, 30, 'z1','z2','z3', bbox=(-2.,2.))
# np.identity(3)，产生 3 * 3 单位矩阵，这是旋转后正椭球的主轴所在方向
plot_implicit_V(cov_ellipsoid_orthg, np.identity(3), -90, 90, 'z1','z2','z3', bbox=(-2.,2.)) # z1-z2

plot_implicit_V(cov_ellipsoid_orthg, np.identity(3), -90, 0, 'z1','z2','z3', bbox=(-2.,2.)) # z1-z3

plot_implicit_V(cov_ellipsoid_orthg, np.identity(3), 0, 0, 'z1','z2','z3', bbox=(-2.,2.)) # z2-z3


#%% Bk_2_Ch18_09 # Plotly绘制三角网格
import plotly.figure_factory as ff
import numpy as np
from scipy.spatial import Delaunay

## 圆球
u = np.linspace(0, np.pi, 26)
v = np.linspace(0, 2 * np.pi, 26)
u, v = np.meshgrid(u, v)
u, v = u.flatten(), v.flatten()

r = 1
# z轴坐标网格数据
z = r*np.cos(u)

# x轴坐标网格数据
x = r*np.sin(u)*np.cos(v)

# y轴坐标网格数据
y = r*np.sin(u)*np.sin(v)

points2D = np.vstack([u,v]).T
tri = Delaunay(points2D)
simplices = tri.simplices

surfacecolor = x**2 + 2*y**2 + 3*z**2
fig = ff.create_trisurf(x=x, y=y, z=z, simplices=simplices, aspectratio=dict(x=1, y=1, z=1))
fig.update_layout(autosize=False, width=500, height=500, margin=dict(l=65, r=50, b=65, t=90))
fig.show()

## 游泳圈
u = np.linspace(0, 2*np.pi, 20)
v = np.linspace(0, 2*np.pi, 20)
u,v = np.meshgrid(u,v)
u = u.flatten()
v = v.flatten()

x = (3 + (np.cos(v)))*np.cos(u)
y = (3 + (np.cos(v)))*np.sin(u)
z = np.sin(v)

points2D = np.vstack([u,v]).T
tri = Delaunay(points2D)
simplices = tri.simplices

fig = ff.create_trisurf(x=x, y=y, z=z, simplices=simplices, aspectratio=dict(x=1, y=1, z=0.3))
fig.update_layout(autosize=False, width=500, height=500, margin=dict(l=65, r=50, b=65, t=90))
fig.show()


import plotly.figure_factory as ff
import numpy as np
from scipy.spatial import Delaunay

u = np.linspace(0, 2*np.pi, 24)
v = np.linspace(-1, 1, 8)
u,v = np.meshgrid(u,v)
u = u.flatten()
v = v.flatten()

tp = 1 + 0.5*v*np.cos(u/2.)
x = tp*np.cos(u)
y = tp*np.sin(u)
z = 0.5*v*np.sin(u/2.)

points2D = np.vstack([u,v]).T
tri = Delaunay(points2D)
simplices = tri.simplices
fig = ff.create_trisurf(x=x, y=y, z=z, colormap="Portland", simplices=simplices, title="Mobius Band")
fig.show()



#%% Bk_2_Ch18_10 # Plotly绘制三维体积图
import plotly.graph_objects as go
import numpy as np

x = np.linspace(-2,2,21)
y = np.linspace(-2,2,21)
z = np.linspace(-2,2,21)

X, Y, Z = np.meshgrid(x,y,z)

fff  = X**2 + 2*Y**2 + 3*Z**2

fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=fff.flatten(),
    opacity=0.18,
    surface_count=11,
    colorscale='RdYlBu'
    ))
fig.update_layout(autosize=False, width=500, height=500, margin=dict(l=65, r=50, b=65, t=90))
fig.show()


#%% Bk_2_Ch18_11
# Plotly绘制三维等高线
import plotly.graph_objects as go
import numpy as np

x = np.linspace(-2,2,21)
y = np.linspace(-2,2,21)
z = np.linspace(-2,2,21)

X, Y, Z = np.meshgrid(x,y,z)

fff  = X**2 + 2*Y**2 + 3*Z**2

fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=fff.flatten(),
    # isomin=10,
    # isomax=50,
    surface_count=8,
    colorscale='RdYlBu',
    caps=dict(x_show=False, y_show=False)
    ))

fig.update_layout(autosize=False, width=500, height=500, margin=dict(l=65, r=50, b=65, t=90))

fig.show()

























































































































































































































































































































































































































































































































































































































































































































































































































