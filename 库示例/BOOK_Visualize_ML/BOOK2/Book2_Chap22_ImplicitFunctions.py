#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:05:06 2024

@author: jack
 Chapter 22 隐函数 | Book 2《可视之美》


"""

#%% BK_2_Ch22_01 绘制线段
# 导入包
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-4, 4, num = 1001)
y = np.linspace(-4, 4, num = 1001)

xx, yy = np.meshgrid(x, y);

# 绘制 x + y = c
fig, ax = plt.subplots(figsize=(5, 5), constrained_layout = True)
levels = np.arange(-6, 6 + 1)
CS = plt.contour(xx, yy, xx + yy,
            levels = levels,
            cmap = 'rainbow',
            inline = True)

ax.clabel(CS, inline=True, fontsize=10)

ax.axvline(x = 0, color = 'k', linestyle = '-')
ax.axhline(y = 0, color = 'k', linestyle = '-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid()

# fig.savefig('Figures/直线，1.svg', format='svg')
plt.show()

#%% BK_2_Ch22_02 绘制抛物线
# 导入包
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-4,4,num = 1001)
y = np.linspace(-4,4,num = 1001)

xx,yy = np.meshgrid(x,y);
## 绘制 x**2 + y = c
fig, ax = plt.subplots(figsize=(5, 5))
levels = np.arange(-4,4 + 1)
CS = plt.contour(xx, yy, xx**2 + yy, levels = levels, cmap = 'rainbow', inline = True)
ax.clabel(CS, inline=True, fontsize=10)
ax.axvline(x = 0, color = 'k', linestyle = '-')
ax.axhline(y = 0, color = 'k', linestyle = '-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid()
# fig.savefig('Figures/抛物线，1.svg', format='svg')

## 绘制 x**2 - y = c
fig, ax = plt.subplots(figsize=(5, 5))
levels = np.arange(-4,4 + 1)
CS = plt.contour(xx, yy, xx**2 - yy, levels = levels, cmap = 'rainbow', inline = True)
ax.clabel(CS, inline=True, fontsize=10)

ax.axvline(x = 0, color = 'k', linestyle = '-')
ax.axhline(y = 0, color = 'k', linestyle = '-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid()

# fig.savefig('Figures/抛物线，2.svg', format='svg')

## 绘制 x + y**2 = c
fig, ax = plt.subplots(figsize=(5, 5))
levels = np.arange(-4,4 + 1)
CS = plt.contour(xx, yy, xx + yy**2, levels = levels, cmap = 'rainbow', inline = True)

ax.clabel(CS, inline=True, fontsize=10)
ax.axvline(x = 0, color = 'k', linestyle = '-')
ax.axhline(y = 0, color = 'k', linestyle = '-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid()
# fig.savefig('Figures/抛物线，3.svg', format='svg')

# 绘制 x - y**2 = c
fig, ax = plt.subplots(figsize=(5, 5), constrained_layout = True)
levels = np.arange(-4, 3 + 1)
CS = plt.contour(xx, yy, xx - yy**2, levels = levels, cmap = 'rainbow', inline = True)

ax.clabel(CS, inline=True, fontsize=10)

ax.axvline(x = 0, color = 'k', linestyle = '-')
ax.axhline(y = 0, color = 'k', linestyle = '-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-4,4])
ax.set_ylim([-4,4])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid()

# fig.savefig('Figures/抛物线，4.svg', format='svg')
plt.show()


#%% BK_2_Ch22_03 离心率可视化一组圆锥曲线

# 导入包
import matplotlib.pyplot as plt
import numpy as np

# 产生数据
x = np.linspace(-4, 4, num = 1001)
y = np.linspace(-4, 4, num = 1001)

xx,yy = np.meshgrid(x, y);

# 一组离心率取值
e_array = np.linspace(0, 3, num = 51)

# 离心率绘制椭圆
# 𝑦2−(𝑒2−1)𝑥2−2𝑥=0, 其中， 𝑒 为离心率

fig, ax = plt.subplots(figsize=(5, 5), constrained_layout = True)

colors = plt.cm.rainbow(np.linspace(0,1,len(e_array)))
# 利用色谱生成一组渐变色，颜色数量和 e_array 一致

for i in range(0,len(e_array)):
    e = e_array[i]
    ellipse = yy**2 - (e**2 - 1)*xx**2 - 2*xx;
    color_code = colors[i,:].tolist()
    plt.contour(xx, yy, ellipse, levels = [0], colors = [color_code])

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

# fig.savefig('Figures/圆锥曲线，随离心率变化.svg', format='svg')
plt.show()

#%% BK_2_Ch22_04 # 椭圆切线
# 导入包
import matplotlib.pyplot as plt
import numpy as np

### 产生数据
a = 1.5
b = 1

x1 = np.linspace(-3,3,200)
x2 = np.linspace(-3,3,200)
xx1,xx2 = np.meshgrid(x1,x2)

fig, ax = plt.subplots(figsize=(5, 5))
theta_array = np.linspace(0,2*np.pi,100)
ax.plot(a*np.cos(b*np.sin(theta_array)),b*np.sin(b*np.sin(theta_array)),color = 'k')
# 利用参数方程绘制椭圆
colors = plt.cm.hsv(np.linspace(0,1,len(theta_array)))
for i in range(len(theta_array)):
    theta = theta_array[i]
    p1 = a*np.cos(theta)
    p2 = b*np.sin(theta)
    # 椭圆上某一点 P 坐标 (p1, p2)
    tangent = p1*xx1/a**2 + p2*xx2/b**2 - p1**2/a**2 - p2**2/b**2
    # P点切线
    colors_i = colors[int(i),:]
    ax.contour(xx1, xx2, tangent, levels = [0], colors = [colors_i])
ax.axis('scaled')
ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axvline(x=0,color = 'k')
ax.axhline(y=0,color = 'k')
# fig.savefig('Figures/椭圆切线.svg', format='svg')


#%% BK_2_Ch22_05 # 和给定矩形相切的一组椭圆
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

### 产生数据
x = np.linspace(-4, 4, num = 1001)
y = np.linspace(-4, 4, num = 1001)
m = 1.5
n = 1.5
xx, yy = np.meshgrid(x, y);
rho_array = np.linspace(-0.95, 0.95, num = 50)

### 可视化
# $\frac{x^2}{m} - 2\rho \frac{xy}{mn}+ \frac{y^2}{n} = 1-\rho^2$
fig, ax = plt.subplots(figsize=(5, 5))
# 矩形位置、形状信息
rect = patches.Rectangle((-m, -n), 2*m, 2*n, linewidth = 0.25, edgecolor='k', linestyle = '--', facecolor = 'none')
ax.add_patch(rect)
# 绘制矩形
colors = plt.cm.rainbow(np.linspace(0,1,len(rho_array)))
for i in range(0,len(rho_array)):
    rho = rho_array[i]
    ellipse = ((xx/m)**2 - 2*rho*(xx/m)*(yy/n) + (yy/n)**2)/(1 - rho**2);
    color_code = colors[i,:].tolist()
    plt.contour(xx, yy, ellipse, levels = [1], colors = [color_code], linewidths = 0.25)

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

# fig.savefig('Figures/和给定矩形相切的一组椭圆.svg', format='svg')


#%% BK_2_Ch22_06 # 和给定椭圆相切的一组矩形
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.pyplot import cm

### 定义函数
def Mahal_d(Mu, Sigma, x):
    x_demeaned = x - Mu
    inv_covmat = np.linalg.inv(Sigma)
    left = np.dot(x_demeaned, inv_covmat)
    mahal = np.dot(left, x_demeaned.T)
    return np.sqrt(mahal).diagonal()

### 产生数据
x1 = np.linspace(-5,5,201)
x2 = np.linspace(-5,5,201)
xx1,xx2 = np.meshgrid(x1,x2)
x_array = np.vstack([xx1.ravel(),xx2.ravel()]).T
Mu = np.array([[0],
               [0]]).T
Sigma = np.array([[1,0.5],
                  [0.5,1]])
total_variance = np.diag(Sigma).sum()
x_array = np.vstack([xx1.ravel(),xx2.ravel()]).T
Mu = np.array([[0],
               [0]]).T

d_array = Mahal_d(Mu, Sigma, x_array)
# 计算网格散点的马氏距离
d_array = d_array.reshape(xx1.shape)
# 让马氏距离数据形状等同于 xx1

### 分析椭圆
Lambdas, V_sigma = np.linalg.eig(Sigma)
# 利用特征值分解获得椭圆
alpha = np.arctan(V_sigma[1,0]/V_sigma[0,0])
major_semi = np.sqrt(Lambdas[0])
minor_semi = np.sqrt(Lambdas[1])
# 椭圆的半长轴、半短轴长度
theta_array = np.linspace(0, np.pi/2, 90)

contour_x = (major_semi*np.cos(theta_array)*np.cos(alpha) - minor_semi*np.sin(theta_array)*np.sin(alpha))
contour_y = (major_semi*np.cos(theta_array)*np.sin(alpha) + minor_semi*np.sin(theta_array)*np.cos(alpha))
# 旋转椭圆的极坐标 (contour_x, contour_y)

### 可视化
fig, ax = plt.subplots(figsize=(5, 5))
ax.contour(xx1, xx2, d_array, levels = [1], colors = 'k')
# 绘制马氏距离为 1 的椭圆
contour_array = np.column_stack((contour_x,contour_y))
# 构造椭圆上点的数组
inv_covmat = np.linalg.inv(Sigma)
# 计算Sigma的逆矩阵
step_size = 2
# 每隔一个点画一个矩形

loop_array = np.arange(0, contour_array.shape[0], step_size)
colors = cm.rainbow(np.linspace(0, 1, len(loop_array))) # rainbow
for idx, c_idx in zip(loop_array, colors):
    x_idx = contour_array[idx,:].reshape(-1,1)
    v_idx = inv_covmat @ x_idx
    v_idx = v_idx/np.linalg.norm(v_idx)
    # print(np.linalg.norm(v_idx))
    theta = np.arctan(v_idx[1]/v_idx[0])
    theta = theta*180/np.pi
    # 矩形的旋转角度
    d1_idx_sq = v_idx.T @ Sigma @ v_idx
    d1_idx = np.sqrt(d1_idx_sq)
    d2_idx_sq = total_variance - d1_idx_sq
    d2_idx = np.sqrt(d2_idx_sq)
    rect = Rectangle([-d1_idx, -d2_idx] , # 矩形的位置
                     width = 2*d1_idx,  # 矩形的宽
                     height = 2*d2_idx, # 矩形的长
                     edgecolor = c_idx,facecolor="none",
                     transform=Affine2D().rotate_deg_around(*(0,0), theta)+ax.transData)
    # 矩形仿射变换
    ax.add_patch(rect)

ax.set_xlim(-2,2)
ax.set_ylim(-2,2)

ax.axhline(0, color = 'k')
ax.axvline(0, color = 'k')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_aspect('equal', adjustable='box')
ax.axis('off')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# fig.savefig('Figures/和给定椭圆相切的一组矩形.svg', format='svg')


#%% BK_2_Ch22_07 # 一组椭圆，长半轴平方、短半轴平方之和为定值

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sympy
import os

x1,x2 = sympy.symbols('x1 x2')
x = np.array([[x1,x2]]).T

xx1, xx2 = np.meshgrid(np.linspace(-2,2,201),np.linspace(-2,2,201))
sum_a_sq_b_sq = 2

step_size = 0.05
a_sq_array = np.arange(step_size,sum_a_sq_b_sq,step = step_size)
# a_sq_array
# 2D visualization

colors = plt.cm.rainbow(np.linspace(0,1,len(a_sq_array), endpoint = True))
fig, ax = plt.subplots()
for idx, a_sq_idx in enumerate(a_sq_array):
    b_sq_idx = sum_a_sq_b_sq - a_sq_idx
    SIGMA = np.array([[a_sq_idx, 0], [0, b_sq_idx]])
    f_x = x.T@np.linalg.inv(SIGMA)@x
    f_x = f_x[0][0]
    f_x_fcn = sympy.lambdify([x1,x2],f_x)
    ff_x = f_x_fcn(xx1,xx2)
    plt.contour(xx1, xx2, ff_x, levels = [1], colors = [colors[idx,:]])

ax.set_aspect('equal')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.axis('off')
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')
plt.tight_layout()
# fig.savefig('Figures/长半轴平方、短半轴平方之和为定值，正椭圆.svg', format='svg')
plt.show()

step_size = 0.05
a_sq_array = np.arange(step_size,sum_a_sq_b_sq,step = step_size)
theta_size = np.pi/4

theta_array = np.arange(theta_size, np.pi, theta_size)
# theta_array

# 2D visualization
colors = plt.cm.rainbow(np.linspace(0,1,len(a_sq_array), endpoint = True))
fig, ax = plt.subplots()
for idx, a_sq_idx in enumerate(a_sq_array):
    for theta_idx in theta_array:
        b_sq_idx = sum_a_sq_b_sq - a_sq_idx
        ab_cos_theta = np.cos(theta_idx) * np.sqrt(a_sq_idx) * np.sqrt(b_sq_idx)
        SIGMA = np.array([[a_sq_idx, ab_cos_theta], [ab_cos_theta, b_sq_idx]])
        f_x = x.T@np.linalg.inv(SIGMA)@x
        f_x = f_x[0][0]
        f_x_fcn = sympy.lambdify([x1,x2],f_x)
        ff_x = f_x_fcn(xx1,xx2)
        plt.contour(xx1, xx2, ff_x, levels = [1], colors = [colors[idx,:]])
ax.set_aspect('equal')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.axis('off')
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')
plt.tight_layout()
# fig.savefig('Figures/长半轴平方、短半轴平方之和为定值，旋转椭圆.svg', format='svg')
plt.show()

#%% BK_2_Ch22_08 # 星形曲线Astroid
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

x = np.linspace(-2,2,num = 1001)
y = np.linspace(-2,2,num = 1001)

xx,yy = np.meshgrid(x,y);
c_array = np.linspace(0, 1,num = 31)
fig, ax = plt.subplots(figsize=(5, 5))

colors = plt.cm.rainbow(np.linspace(0,1,len(c_array)))
for i in range(0,len(c_array)):
    c_i = c_array[i]
    ellipse = (xx/c_i)**2 + (yy/(1-c_i))**2
    color_code = colors[i,:].tolist()
    plt.contour(xx, yy, ellipse, levels = [1], colors = [color_code], linewidths = 0.25)
# plt.axvline(x = 0, color = 'k', linestyle = '-')
# plt.axhline(y = 0, color = 'k', linestyle = '-')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.axis('off')
fig.savefig('1.svg')
# fig.savefig('Figures/和给定矩形相切的一组椭圆.svg', format='svg')

def visualize(loc_x,loc_y):
    for i in range(0,len(c_array)):
        c_i = c_array[i]
        ellipse = ((xx - loc_x)/c_i)**2 + ((yy - loc_y)/(1-c_i))**2
        plt.contour(xx, yy, ellipse, levels = [1], colors = '0.5', linewidths = 0.25)

fig, ax = plt.subplots(figsize=(8, 8))
loc_x = np.arange(-2,3)
loc_y = loc_x
loc_xx,loc_yy = np.meshgrid(loc_x,loc_y)
for loc_x,loc_y in zip(loc_xx.ravel(),loc_yy.ravel()):
    visualize(loc_x,loc_y)
# # plt.axvline(x = 0, color = 'k', linestyle = '-')
# # plt.axhline(y = 0, color = 'k', linestyle = '-')
# ax.set_xticks([])
# ax.set_yticks([])
# # ax.set_xlim([-2,2])
# # ax.set_ylim([-2,2])
ax.axis('off')
# fig.savefig('2.svg')

#%% BK_2_Ch22_09 用等高线绘制几何体, f = f(x, y, z)
# 导入包
import matplotlib.pyplot as plt
import numpy as np

# 0. 可视化隐函数
def plot_implicit(fn, X_plot, Y_plot, Z_plot, ax, bbox):
    # 等高线的起止范围
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    ax.set_proj_type('ortho')
    # 绘制三条参考线
    k = 1.5
    ax.plot((xmin * k, xmax * k), (0, 0), (0, 0), 'k')
    ax.plot((0, 0), (ymin * k, ymax * k), (0, 0), 'k')
    ax.plot((0, 0), (0, 0), (zmin * k, zmax * k), 'k')
    # 等高线的分辨率
    A = np.linspace(xmin, xmax, 500)
    # 产生网格数据
    A1, A2 = np.meshgrid(A, A)
    # 等高线的分割位置
    B = np.linspace(xmin, xmax, 20)

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
            Y = fn(X,y,Z)
            cset = ax.contour(X, Y+y, Z, [y], zdir='y', linewidths = 0.25, colors = '#88DD66', linestyles = 'solid')
    # 绘制 YZ 平面等高线
    if Z_plot == True:
        for x in B:
            Y, Z = A1, A2
            X = fn(x,Y,Z)
            cset = ax.contour(X+x, Y, Z, [x], zdir='x', linewidths = 0.25, colors = '#FF6600', linestyles = 'solid')
    ax.set_zlim(zmin * k, zmax * k)
    ax.set_xlim(xmin * k, xmax * k)
    ax.set_ylim(ymin * k, ymax * k)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(azim=-120, elev=30)
    ax.axis('off')
    # plt.show()
    return

def visualize_four_ways(fn, title, bbox=(-2.5, 2.5)):
    fig = plt.figure(figsize=(20, 8), constrained_layout = True)

    ax = fig.add_subplot(1, 4, 1, projection='3d')
    plot_implicit(fn, True, False, False, ax, bbox)

    ax = fig.add_subplot(1, 4, 2, projection='3d')
    plot_implicit(fn, False, True, False, ax, bbox)

    ax = fig.add_subplot(1, 4, 3, projection='3d')
    plot_implicit(fn, False, False, True, ax, bbox)

    ax = fig.add_subplot(1, 4, 4, projection='3d')
    plot_implicit(fn, True, True, True, ax, bbox)

    # fig.savefig('Figures/' + title + '.svg', format='svg')
    plt.show()
    return

# 1. 单位球
def unit_sphere(x,y,z):
    return x**2 + y**2 + z**2 - 1
visualize_four_ways(unit_sphere, '单位球', bbox = (-1,1))

# 2. 椭球
# Ellipsoid
def Ellipsoid(x,y,z):
    a = 1
    b = 2
    c = 1
    return x**2/a**2 + y**2/b**2 + z**2/c**2 - 1
visualize_four_ways(Ellipsoid, '椭球', bbox = (-2,2))

# 3. 双曲抛物面
# 双曲抛物面是一个二次曲面，其形状像一个双曲面和抛物面的组合。
# 𝑥2𝑎2−𝑦2𝑏2−𝑧=0
# Hyperbolic_paraboloid
def Hyperbolic_paraboloid(x,y,z):
    a = 1
    b = 1
    return x**2/a**2 - y**2/b**2 - z
visualize_four_ways(Hyperbolic_paraboloid, '双曲抛物面', bbox = (-2,2))

# 4. 旋转双曲抛物面:𝑥𝑦−𝑧=0
# Hyperbolic_paraboloid, rotated
def Hyperbolic_paraboloid_rotated(x,y,z):
    return x*y - z
visualize_four_ways(Hyperbolic_paraboloid_rotated, '旋转双曲抛物面', bbox = (-2,2))

# 5A. 正圆抛物面，开口朝上
# 𝑥2+𝑦2−𝑧−2=0
#  Circular paraboloid
def circular_paraboloid(x,y,z):
    return x**2 + y**2 - 2 - z
visualize_four_ways(circular_paraboloid, '正圆抛物面，开口朝上', bbox = (-2,2))

# 5B. 正圆抛物面，开口朝下
# 𝑥2+𝑦2+𝑧−2=0
#  Circular paraboloid
def circular_paraboloid(x,y,z):
    return x**2 + y**2 - 2 + z
visualize_four_ways(circular_paraboloid, '正圆抛物面，开口朝下', bbox = (-2,2))

# 5C. 正圆抛物面，x轴
# 𝑦2+𝑧2−𝑥−2=0
#  Circular paraboloid
def circular_paraboloid(x,y,z):
    return y**2 + z**2 - 2 - x
visualize_four_ways(circular_paraboloid, '正圆抛物面，开口沿x轴', bbox = (-2,2))

# 5C. 正圆抛物面，y轴
# 𝑥2+𝑧2−𝑦−2=0
#  Circular paraboloid
def circular_paraboloid(x,y,z):
    return x**2 + z**2 - 2 - y
visualize_four_ways(circular_paraboloid, '正圆抛物面，开口沿y轴', bbox = (-2,2))

# 6A. 单叶双曲面，z轴
# 𝑥2+𝑦2−𝑧2−2=0
#  Hyperboloid of revolution of one sheet (special case of hyperboloid of one sheet)
def Hyperboloid_1_sheet(x,y,z):
    return x**2 + y**2 - z**2 - 2
visualize_four_ways(Hyperboloid_1_sheet, '单叶双曲面，z轴', bbox = (-4,4))

# 6B. 单叶双曲面，y轴
# 𝑥2−𝑦2+𝑧2−2=0
#  Hyperboloid of revolution of one sheet (special case of hyperboloid of one sheet)
def Hyperboloid_1_sheet(x,y,z):
    return x**2 - y**2 + z**2 - 2
visualize_four_ways(Hyperboloid_1_sheet, '单叶双曲面，y轴', bbox = (-4,4))

# 6C. 单叶双曲面，x轴
# −𝑥2+𝑦2+𝑧2−2=0
#  Hyperboloid of revolution of one sheet (special case of hyperboloid of one sheet)
def Hyperboloid_1_sheet(x,y,z):
    return - x**2 + y**2 + z**2 - 2
visualize_four_ways(Hyperboloid_1_sheet, '单叶双曲面，x轴', bbox = (-4,4))

# 7A. 双叶双曲面，z轴
# 𝑥2+𝑦2−𝑧2+1=0
#  Hyperboloid of revolution of two sheets
def Hyperboloid_2_sheets(x,y,z):
    return x**2 + y**2 - z**2 + 1

visualize_four_ways(Hyperboloid_2_sheets, '双叶双曲面，z轴', bbox = (-4,4))

# 7B. 双叶双曲面，y轴
# 𝑥2−𝑦2+𝑧2+2=0
#  Hyperboloid of revolution of two sheets
def Hyperboloid_2_sheets(x,y,z):
    return x**2 - y**2 + z**2 + 2
visualize_four_ways(Hyperboloid_2_sheets, '双叶双曲面，y轴', bbox = (-4,4))

# 7C. 双叶双曲面，x轴
# −𝑥2+𝑦2+𝑧2+1=0
#  Hyperboloid of revolution of two sheets
def Hyperboloid_2_sheets(x,y,z):
    return - x**2 + y**2 + z**2 + 1
visualize_four_ways(Hyperboloid_2_sheets, '双叶双曲面，x轴', bbox = (-4,4))

# 8A. 圆锥面，z轴
# 𝑥2+𝑦2−𝑧2=0
#    Circular cone
def Circular_cone(x,y,z):
    return x**2 + y**2 - z**2
visualize_four_ways(Circular_cone, '圆锥面', bbox = (-4, 4))

# 8B. 圆锥面，y轴
# 𝑥2−𝑦2+𝑧2=0
#    Circular cone
def Circular_cone(x,y,z):
    return x**2 - y**2 + z**2

visualize_four_ways(Circular_cone, '圆锥面_y_轴', bbox = (-4, 4))

# 8C. 圆锥面，x轴
# −𝑥2+𝑦2+𝑧2=0
#    Circular cone
def Circular_cone(x,y,z):
    return -x**2 + y**2 + z**2
visualize_four_ways(Circular_cone, '圆锥面_x_轴', bbox = (-4, 4))

# 9A. 圆柱面，z轴
# 𝑥2+𝑦2−1=0
#    Circular cylinder
def Circular_cylinder(x,y,z):
    return x**2 + y**2 - 1
visualize_four_ways(Circular_cylinder, '圆柱面，z轴', bbox = (-1,1))


# 9B. 圆柱面，y轴
# 𝑥2+𝑧2−1=0
#    Circular cylinder
def Circular_cylinder(x,y,z):
    return x**2 + z**2 - 1
visualize_four_ways(Circular_cylinder, '圆柱面，y轴', bbox = (-1,1))

#    Circular cylinder
def Circular_cylinder(x,y,z):
    return x**2 + z**2 - 1
visualize_four_ways(Circular_cylinder, '圆柱面，y轴', bbox = (-1,1))

# 9C. 圆柱面，x轴
# 𝑦2+𝑧2−1=0
#    Circular cylinder
def Circular_cylinder(x,y,z):
    return y**2 + z**2 - 1
visualize_four_ways(Circular_cylinder, '圆柱面，x轴', bbox = (-1,1))
#    Circular cylinder
def Circular_cylinder(x,y,z):
    return y**2 + z**2 - 1
visualize_four_ways(Circular_cylinder, '圆柱面，x轴', bbox = (-1,1))

# 10. 古尔萨特结
def Tanglecube(x,y,z):
    a,b,c = 0.0,-5.0,11.8
    return x**4+y**4+z**4+a*(x**2+y**2+z**2)**2+b*(x**2+y**2+z**2)+c
visualize_four_ways(Tanglecube, '古尔萨特结')

# 11. 心形
# (𝑥2+9/4𝑦2+𝑧2−1)3−𝑥2𝑧3−9/80𝑦2𝑧3=0
def heart(x,y,z):
    return (x**2 + 9/4*y**2 + z**2 - 1)**3 - x**2*z**3 - 9/80 * y**2 * z**3
visualize_four_ways(heart, '心形', (-1.2,1.2))


# 12. 环面
# 参考： https://en.wikipedia.org/wiki/Implicit_surface

# (𝑥2+𝑦2+𝑧2+𝑅2−𝑎2)2−4𝑅2(𝑥2+𝑧2)=0
def Torus(x,y,z):
    R = 2.5
    a = 0.8
    return (x**2 + y**2 + z**2 + R**2 - a**2)**2 - 4*R**2*(x**2 + z**2)
visualize_four_ways(Torus, '环面', (-3,3))

# 范数
def vector_norm(x,y,z):
    p = 0.6
    # 非范数。Lp范数，p >=1
    return (np.abs(x)**p + np.abs(y)**p + np.abs(z)**p)**(1/p) - 1
visualize_four_ways(vector_norm, 'norm_0.6', bbox = (-1,1))

def vector_norm(x,y,z):
    p = 1
    return (np.abs(x)**p + np.abs(y)**p + np.abs(z)**p)**(1/p) - 1
visualize_four_ways(vector_norm, 'norm_1', bbox = (-1,1))

def vector_norm(x,y,z):
    p = 1.5
    return (np.abs(x)**p + np.abs(y)**p + np.abs(z)**p)**(1/p) - 1
visualize_four_ways(vector_norm, 'norm_1.5', bbox = (-1,1))

def vector_norm(x,y,z):
    p = 2
    return (np.abs(x)**p + np.abs(y)**p + np.abs(z)**p)**(1/p) - 1
visualize_four_ways(vector_norm, 'norm_2', bbox = (-1,1))

def vector_norm(x,y,z):
    p = 3
    return (np.abs(x)**p + np.abs(y)**p + np.abs(z)**p)**(1/p) - 1
visualize_four_ways(vector_norm, 'norm_3', bbox = (-1,1))


def vector_norm(x,y,z):
    p = 8
    return (np.abs(x)**p + np.abs(y)**p + np.abs(z)**p)**(1/p) - 1
visualize_four_ways(vector_norm, 'norm_8', bbox = (-1,1))










































































































































































































































































































