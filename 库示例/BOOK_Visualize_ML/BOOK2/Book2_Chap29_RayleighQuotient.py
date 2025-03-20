#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:07:45 2024

@author: jack

 Chapter 29 瑞利商 | Book 2《可视之美》

"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)

fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

#%% 把瑞利商看成是二元函数

# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex, simplify,symbols

from matplotlib import cm
# 导入色谱模块

import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# p = plt.rcParams
# # p["font.sans-serif"] = ["Roboto"]
# p["font.weight"] = "light"
# p["ytick.minor.visible"] = False
# p["xtick.minor.visible"] = False
# p["axes.grid"] = True
# p["grid.color"] = "0.5"
# p["grid.linewidth"] = 0.5

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 8  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 12  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 12  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [12, 8] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 1     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 12


def mesh(num = 100):
    # 偶数避免零点
    # number of mesh grids
    x_array = np.linspace(-2,2,num)
    y_array = np.linspace(-2,2,num)
    xx,yy = np.meshgrid(x_array,y_array)

    return xx, yy

x1, x2 = symbols('x1 x2')
# 自定义函数计算二元瑞利商
def Rayleigh_Q(Q, xx1, xx2):
    x = np.array([[x1], [x2]])

    # 瑞利商，符号式
    f_x1x2 = x.T @ Q @ x/(x.T @ x)
    # 将符号函数表达式转换为Python函数
    f_x1x2_fcn = lambdify([x1, x2], f_x1x2[0][0])
    # 计算二元函数函数值
    ff = f_x1x2_fcn(xx1, xx2)
    return ff, simplify(f_x1x2[0][0])


def visualize(Q, title):
    xx1, xx2 = mesh(num = 200)
    ff, f_x1x2 = Rayleigh_Q(Q, xx1, xx2)

    levels = np.linspace(-2,2,41)

    # 特征值分解
    _, V = np.linalg.eig(Q)
    v1 = V[:,0]
    v2 = V[:,1]

    ### 单位圆坐标
    theta_array = np.linspace(0, 2*np.pi, 100)
    x1_circle = np.cos(theta_array)
    x2_circle = np.sin(theta_array)

    fig = plt.figure(figsize=(8,4))

    ax = fig.add_subplot(1, 2, 1)
    ax.contourf(xx1, xx2, ff, levels = levels, cmap='RdYlBu_r')
    ax.plot(x1_circle, x2_circle, color = 'k')
    # 绘制向量 v1
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color = 'k')

    # 绘制向量 v2
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color = 'k')

    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_wireframe(xx1, xx2, ff, color = [0.5,0.5,0.5], rstride=10, cstride=10, linewidth = 0.25)

    CS = ax.contour(xx1, xx2, ff, cmap = 'RdYlBu_r', levels = levels)
    fig.colorbar(CS, ax=ax, shrink=0.8)
    f_circle, _ = Rayleigh_Q(Q, x1_circle, x2_circle)
    ax.plot(x1_circle, x2_circle, f_circle, color = 'k')

    ax.set_proj_type('ortho')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1,x_2)$')

    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([])
    # ax.view_init(azim=-120, elev=30)
    ax.view_init(azim=-120, elev=60)
    font1 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
    plt.suptitle(title, fontproperties=font1)
    plt.tight_layout()
    ax.grid(False)
    plt.show()
    # fig.savefig('Figures/' + title + '.svg', format='svg')

    return f_x1x2

Q = np.array([[1, 0],
              [0, 2]])
print(np.linalg.eig(Q))
f_x1x2 = visualize(Q, '开口朝上正椭圆面')

# Q = np.array([[1.5,0.5],
#               [0.5,1.5]])
# print(np.linalg.eig(Q))
# f_x1x2 = visualize(Q, '开口朝上旋转椭圆面')


# Q = np.array([[-1,0],
#               [0,-2]])
# print(np.linalg.eig(Q))
# f_x1x2 = visualize(Q, '开口朝下正椭圆面')



# Q = np.array([[-1.5,-0.5],
#               [-0.5,-1.5]])
# print(np.linalg.eig(Q))
# f_x1x2 = visualize(Q, '开口朝下旋转椭圆面')


# Q = np.array([[1,-1],
#               [-1,1]])
# print(np.linalg.eig(Q))
# f_x1x2 = visualize(Q, '旋转山谷')


Q = np.array([[-1,1],
              [1,-1]])
print(np.linalg.eig(Q))
f_x1x2 = visualize(Q, '旋转山脊')


Q = np.array([[1,0],
              [0,-1]])
print(np.linalg.eig(Q))
f_x1x2 = visualize(Q, '双曲面')

Q = np.array([[0,2],
              [2,0]])
print(np.linalg.eig(Q))
f_x1x2 = visualize(Q, '旋转双曲面')


#%% 极坐标视角看瑞利商
# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex, simplify,symbols,cos,sin,trigsimp

from matplotlib import cm
# 导入色谱模块

import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = False
p["xtick.minor.visible"] = False
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

def mesh(num = 100):
    # 偶数避免零点
    # number of mesh grids
    x_array = np.linspace(-2, 2, num)
    y_array = np.linspace(-2, 2, num)
    xx, yy = np.meshgrid(x_array, y_array)
    return xx, yy

theta, r = symbols('theta r')
# 自定义函数计算二元瑞利商
def Rayleigh_Q(Q, theta_array):
    x = np.array([[cos(theta) * r],
                  [sin(theta) * r]])
    # 瑞利商，符号式
    f_theta = x.T @ Q @ x/(x.T @ x)
    # 将符号函数表达式转换为Python函数
    f_theta_fcn = lambdify(theta, f_theta[0][0])
    # 计算二元函数函数值
    f_array = f_theta_fcn(theta_array)
    return f_array, trigsimp(f_theta[0][0])

def visualize(Q, title):
    theta_array = np.linspace(0, 2*np.pi, 100)
    f_array, f_x1x2 = Rayleigh_Q(Q, theta_array)

    ### 单位圆坐标
    x1_circle = np.cos(theta_array)
    x2_circle = np.sin(theta_array)

    fig = plt.figure(figsize=(8,4))

    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(x1_circle, x2_circle, c = f_array, cmap = 'RdYlBu_r')
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(theta_array, f_array)

    lambdas,_ = np.linalg.eig(Q)
    lambda_1 = max(lambdas)
    lambda_2 = min(lambdas)
    ax.axhspan(ymin = lambda_2, ymax = lambda_1, color = '#DEEAF6')
    ax.set_xlim(theta_array.min(), theta_array.max())
    xticks = [0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi]
    xtick_labels = ['0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$']
    plt.xticks(xticks, xtick_labels)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('Rayleigh quotient')
    font1 = FontProperties(fname=fontpath+"simsun.ttf", size=22)
    plt.suptitle(title, fontproperties=font1)
    plt.tight_layout()
    ax.grid(True)
    plt.show()
    # fig.savefig('Figures/' + title + '.svg', format='svg')

    return f_x1x2

# Q = np.array([[1,0],
#               [0,2]])
# print(np.linalg.eig(Q))
# f_x1x2 = visualize(Q, '开口朝上正椭圆面_极坐标')

# Q = np.array([[1.5,0.5],
#               [0.5,1.5]])
# print(np.linalg.eig(Q))
# f_x1x2 = visualize(Q, '开口朝上旋转椭圆面_极坐标')

# Q = np.array([[-1,0],
#               [0,-2]])
# print(np.linalg.eig(Q))
# f_x1x2 = visualize(Q, '开口朝下正椭圆面_极坐标')

# Q = np.array([[-1.5,-0.5],
#               [-0.5,-1.5]])
# print(np.linalg.eig(Q))
# f_x1x2 = visualize(Q, '开口朝下旋转椭圆面_极坐标')

# Q = np.array([[1,-1],
#               [-1,1]])
# print(np.linalg.eig(Q))
# f_x1x2 = visualize(Q, '旋转山谷_极坐标')

# Q = np.array([[-1,1],
#               [1,-1]])
# print(np.linalg.eig(Q))
# f_x1x2 = visualize(Q, '旋转山脊_极坐标')

Q = np.array([[1,0],
              [0,-1]])
print(np.linalg.eig(Q))
f_x1x2 = visualize(Q, '双曲面_极坐标')

# Q = np.array([[0,2],
#               [2,0]])
# print(np.linalg.eig(Q))
# f_x1x2 = visualize(Q, '旋转双曲面_极坐标')




#%% 单位球上看瑞利商
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# 0. 球坐标
# 设置步数
intervals = 50
ntheta = intervals
nphi = 2*intervals

# 单位球，球坐标
# theta取值范围为 [0, pi]
theta = np.linspace(0, np.pi*1, ntheta+1)
# phi取值范围为 [0, 2*pi]
phi   = np.linspace(0, np.pi*2, nphi+1)

# 单位球半径
r = 1

### 球坐标转化为三维直角坐标
# 第一种方法
# z轴坐标网格数据
Z = np.outer(r*np.cos(theta), np.ones(nphi+1))
# x轴坐标网格数据
X = np.outer(r*np.sin(theta), np.cos(phi))
# y轴坐标网格数据
Y = np.outer(r*np.sin(theta), np.sin(phi))

# 第二种方法
pp_, tt_ = np.meshgrid(phi, theta)
# z轴坐标网格数据
Z = r*np.cos(tt_)
# x轴坐标网格数据
X = r*np.sin(tt_)*np.cos(pp_)
# y轴坐标网格数据
Y = r*np.sin(tt_)*np.sin(pp_)

############# 1
fig = plt.figure(figsize = (20, 20))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, linewidth=0.25)
# 单独绘制经纬线
# surf = ax.plot_wireframe(X, Y, Z, rstride=0, cstride=2,
#                linewidth=0.25)

# surf = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=0,
#                linewidth=0.25)
surf.set_facecolor((0,0,0,0)) #利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
k = 1.5
# three reference lines
ax.plot((-k, k), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k, k), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k, k), 'k')
ax.axis('off')
ax.set_xlim((-k, k))
ax.set_ylim((-k, k))
ax.set_zlim((-k, k))
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-155, elev=35)
ax.grid(False)
# fig.savefig('Figures/单位球.svg', format='svg')
plt.show()

############# 2
fig = plt.figure(figsize = (20, 20))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color = '0.68',linewidth=0.25)

surf.set_facecolor((0,0,0,0))

ax.scatter(X[::2,::2],
           Y[::2,::2],
           Z[::2,::2],
           s = 3.8)
ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
k = 1.5
# three reference lines
ax.plot((-k, k), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k, k), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k, k), 'k')
ax.axis('off')
ax.set_xlim((-k, k))
ax.set_ylim((-k, k))
ax.set_zlim((-k, k))
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-155, elev=35)
ax.grid(False)
# fig.savefig('Figures/单位球面上点.svg', format='svg')
plt.show()

################## 1. 计算瑞利商
# 每一行代表一个三维直角坐标系坐标点
# 所有坐标点都在单位球面上
Points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

# 定义矩阵Q
Q = np.array([[1, 0.5, 1],
              [0.5, 2, -0.2],
              [1, -0.2, 1]])
# 计算 xT @ Q @ x
Rayleigh_Q = np.diag(Points @ Q @ Points.T)
#
Rayleigh_Q_ = np.reshape(Rayleigh_Q, X.shape)

## 1  小彩灯
fig = plt.figure(figsize = (12, 12), constrained_layout = True)
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color = '0.68',linewidth=0.25)

surf.set_facecolor((0, 0, 0, 0))

ax.scatter(X[::2,::2], Y[::2,::2], Z[::2,::2], c = Rayleigh_Q_[::2,::2], cmap = 'hsv', s = 15)
ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
k = 1.5
# three reference lines
ax.plot((-k, k), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k, k), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k, k), 'k')
ax.axis('off')
ax.set_xlim((-np.max(np.abs(X)), np.max(np.abs(X))))
ax.set_ylim((-np.max(np.abs(Y)), np.max(np.abs(Y))))
ax.set_zlim((-np.max(np.abs(Z)), np.max(np.abs(Z))))
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-155, elev=35)
ax.grid(False)
# fig.savefig('Figures/单位球面上点 + 渲染.svg', format='svg')
plt.show()


## 2. 填充
norm_plt = plt.Normalize(Rayleigh_Q_.min(), Rayleigh_Q_.max())
colors = cm.RdYlBu_r(norm_plt(Rayleigh_Q_))

fig = plt.figure(figsize = (20, 20))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,  linewidth=0.25, shade=False)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
k = 1.5
# three reference lines
ax.plot((-k, k), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k, k), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k, k), 'k')
ax.axis('off')
ax.set_xlim((-k, k))
ax.set_ylim((-k, k))
ax.set_zlim((-k, k))
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-130, elev=30)
ax.grid(False)
# fig.savefig('Figures/瑞利商，填充.svg', format='svg')
plt.show()

## 3. 只有网格
fig = plt.figure(figsize = (20, 20))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,  linewidth=0.25, shade=False)

surf.set_facecolor((0,0,0,0))  # 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.plot((-k, k), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k, k), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k, k), 'k')
ax.axis('off')
k = 1.5
ax.set_xlim((-k, k))
ax.set_ylim((-k, k))
ax.set_zlim((-k, k))
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-130, elev=30)
ax.grid(False)
# fig.savefig('Figures/瑞利商，网格.svg', format='svg')
plt.show()


## 将瑞利商球面展开
fig, ax = plt.subplots(figsize = (12, 5))
c = ax.scatter(pp_,
               tt_,
               s = 1.8)
ax.set_ylim(tt_.min(), tt_.max())
ax.set_xlim(pp_.min(), pp_.max())
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$', rotation=0)
ax.set_xticks(np.linspace(0, 2*np.pi, 5))
ax.set_xticklabels(['0',
                    r'$\pi/2 (90^\circ)$',
                    r'$\pi (180^\circ)$',
                    r'$3\pi/2 (270^\circ)$',
                    r'$2\pi (360^\circ)$'])

ax.set_yticks(np.linspace(0, np.pi, 3))
ax.set_yticklabels(['0',
                    r'$\pi/2 (90^\circ)$',
                    r'$\pi (180^\circ)$'])
ax.invert_yaxis()
plt.axis('scaled')
plt.grid()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# fig.savefig('Figures/角度网格散点.svg', format='svg')
plt.show()

## θ、φ 网格数据,将球体展开成平面
fig, ax = plt.subplots()
c = ax.scatter(pp_,
               tt_,
               c = Rayleigh_Q_,
               cmap='RdYlBu_r', s = 1.8)

ax.set_ylim(tt_.min(), tt_.max())
ax.set_xlim(pp_.min(), pp_.max())
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$', rotation=0)
ax.set_xticks(np.linspace(0, 2*np.pi, 5))
ax.set_xticklabels(['0',
                    r'$\pi/2 (90^\circ)$',
                    r'$\pi (180^\circ)$',
                    r'$3\pi/2 (270^\circ)$',
                    r'$2\pi (360^\circ)$'])

ax.set_yticks(np.linspace(0, np.pi, 3))
ax.set_yticklabels(['0',
                    r'$\pi/2 (90^\circ)$',
                    r'$\pi (180^\circ)$'])
ax.invert_yaxis()
plt.axis('scaled')
plt.grid()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# fig.savefig('Figures/角度网格散点 + 渲染.svg', format='svg')
plt.show()


## θ、φ 网格数据,将球体展开成平面
fig, ax = plt.subplots()
# 利用 Pcolormesh()方法在二维平面上绘制一个伪彩色网格。
c = ax.pcolormesh(pp_,
                  tt_,
                  Rayleigh_Q_,
                  cmap='RdYlBu_r', shading='auto',
                  vmin=Rayleigh_Q_.min(),
                  vmax=Rayleigh_Q_.max())
fig.colorbar(c, ax=ax, shrink = 0.58)


ax.set_xlim(tt_.min(), tt_.max())
ax.set_ylim(pp_.min(), pp_.max())
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$', rotation=0)
ax.set_xticks(np.linspace(0, 2*np.pi, 5))
ax.set_xticklabels(['0',
                    r'$\pi/2 (90^\circ)$',
                    r'$\pi (180^\circ)$',
                    r'$3\pi/2 (270^\circ)$',
                    r'$2\pi (360^\circ)$'])

ax.set_yticks(np.linspace(0, np.pi, 3))
ax.set_yticklabels(['0',
                    r'$\pi/2 (90^\circ)$',
                    r'$\pi (180^\circ)$'])
ax.invert_yaxis()
plt.axis('scaled')
plt.grid()
plt.show()

## 3 填充等高线图+等高线
fig, ax = plt.subplots()
levels = np.linspace(Rayleigh_Q_.min(),Rayleigh_Q_.max(),18)

colorbar = ax.contourf(pp_, tt_, Rayleigh_Q_, levels = levels, cmap='RdYlBu_r')
ax.contour(pp_,tt_, Rayleigh_Q_, levels = levels, colors = 'w')

fig.colorbar(colorbar, ax=ax, shrink = 0.58)
ax.set_ylim(tt_.min(), tt_.max())
ax.set_xlim(pp_.min(), pp_.max())
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$', rotation=0)
ax.set_xticks(np.linspace(0, 2*np.pi, 5))
ax.set_xticklabels(['0',
                    r'$\pi/2 (90^\circ)$',
                    r'$\pi (180^\circ)$',
                    r'$3\pi/2 (270^\circ)$',
                    r'$2\pi (360^\circ)$'])

ax.set_yticks(np.linspace(0, np.pi, 3))
ax.set_yticklabels(['0',
                    r'$\pi/2 (90^\circ)$',
                    r'$\pi (180^\circ)$'])
ax.invert_yaxis()
plt.axis('scaled')
plt.grid()
plt.show()


## 4 等高线
fig, ax = plt.subplots()

colorbar = ax.contour(pp_,tt_, Rayleigh_Q_, levels = levels, cmap='RdYlBu_r')

fig.colorbar(colorbar, ax=ax, shrink = 0.58)
ax.set_ylim(tt_.min(), tt_.max())
ax.set_xlim(pp_.min(), pp_.max())
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$', rotation=0)
ax.set_xticks(np.linspace(0, 2*np.pi, 5))
ax.set_xticklabels(['0',
                    r'$\pi/2 (90^\circ)$',
                    r'$\pi (180^\circ)$',
                    r'$3\pi/2 (270^\circ)$',
                    r'$2\pi (360^\circ)$'])

ax.set_yticks(np.linspace(0, np.pi, 3))
ax.set_yticklabels(['0',
                    r'$\pi/2 (90^\circ)$',
                    r'$\pi (180^\circ)$'])
ax.invert_yaxis()
plt.axis('scaled')
plt.grid()
plt.show()

##  5 用三维等高线 + 网格曲面展示三元瑞利商
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(pp_, tt_, Rayleigh_Q_, colors = '0.8')
ax.contour(pp_,tt_, Rayleigh_Q_, levels = levels, cmap='RdYlBu_r')

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式
ax.set_ylim(tt_.min(), tt_.max())
ax.set_xlim(pp_.min(), pp_.max())
ax.set_xlabel(r'$\phi$')
ax.set_ylabel(r'$\theta$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
# ax.set_box_aspect([1,1,1])
ax.view_init(azim=-145, elev=75)
ax.grid(False)
# fig.savefig('Figures/瑞利商，网格.svg', format='svg')
# fig.savefig('瑞利商，网格.svg', format='svg')

#%% 不同三元瑞利商

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import plotly.graph_objects as go

# import os
# # 如果文件夹不存在，创建文件夹
# if not os.path.isdir("Figures"):
#     os.makedirs("Figures")
# 设置步数
intervals = 50
ntheta = intervals
nphi = 2*intervals

# 单位球，球坐标
# theta取值范围为 [0, pi]
theta = np.linspace(0, np.pi*1, ntheta+1)
# phi取值范围为 [0, 2*pi]
phi   = np.linspace(0, np.pi*2, nphi+1)

# 单位球半径
r = 1

# 球坐标转化为三维直角坐标
pp_,tt_ = np.meshgrid(phi,theta)

# z轴坐标网格数据
Z = r*np.cos(tt_)

# x轴坐标网格数据
X = r*np.sin(tt_)*np.cos(pp_)

# y轴坐标网格数据
Y = r*np.sin(tt_)*np.sin(pp_)
# 每一行代表一个三维直角坐标系坐标点
# 所有坐标点都在单位球面上
Points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

def visualize_Q(Q, title):
    Rayleigh_Q = np.diag(Points @ Q @ Points.T)
    Rayleigh_Q_ = np.reshape(Rayleigh_Q,X.shape)
    print('Rayleigh_Q_.min()')
    print(Rayleigh_Q_.min())

    print('Rayleigh_Q_.max()')
    print(Rayleigh_Q_.max())

    # norm_plt = plt.Normalize(Rayleigh_Q_.min(), Rayleigh_Q_.max())
    norm_plt = plt.Normalize(-3, 3)
    colors = cm.RdYlBu_r(norm_plt(Rayleigh_Q_))

    fig = plt.figure(figsize = (8,4))
    ax = fig.add_subplot(121, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,
                   linewidth=0.25, shade=False)

    # surf.set_facecolor((0,0,0,0))

    ax.set_proj_type('ortho')
    # 另外一种设定正交投影的方式

    ax.set_xlabel('$\it{x_1}$')
    ax.set_ylabel('$\it{x_2}$')
    ax.set_zlabel('$\it{x_3}$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    k = 1.5
    # three reference lines
    ax.plot((-k, k), (0, 0), (0, 0), 'k')
    ax.plot((0, 0), (-k, k), (0, 0), 'k')
    ax.plot((0, 0), (0, 0), (-k, k), 'k')
    ax.axis('off')
    ax.set_xlim((-k, k))
    ax.set_ylim((-k, k))
    ax.set_zlim((-k, k))
    ax.set_box_aspect([1,1,1])
    ax.view_init(azim=-130, elev=25)
    ax.grid(False)


    ax = fig.add_subplot(122)
    levels = np.linspace(-3,3,25)

    colorbar = ax.contourf(pp_, tt_, Rayleigh_Q_, levels = levels, cmap='RdYlBu_r')
    ax.contour(pp_,tt_, Rayleigh_Q_, levels = levels, colors = 'w')

    fig.colorbar(colorbar, ax=ax, shrink = 0.38)
    ax.set_ylim(tt_.min(), tt_.max())
    ax.set_xlim(pp_.min(), pp_.max())
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\theta$', rotation=0)
    ax.set_xticks(np.linspace(0, 2*np.pi, 3))
    ax.set_xticklabels(['0',
                        r'$\pi/2 (90^\circ)$',
                        r'$\pi (180^\circ)$'])

    ax.set_yticks(np.linspace(0, np.pi, 3))
    ax.set_yticklabels(['0',
                        r'$\pi/2 (90^\circ)$',
                        r'$\pi (180^\circ)$'])
    ax.invert_yaxis()
    plt.axis('scaled')
    # fig.savefig(title + '.svg', format='svg')
    plt.show()
    return


def plotly_Q(Q):
    Rayleigh_Q = np.diag(Points @ Q @ Points.T)
    Rayleigh_Q_ = np.reshape(Rayleigh_Q,X.shape)
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z,
                                     surfacecolor=Rayleigh_Q_,
                                     cmax = 3,cmin = -3,
                                     colorscale='RdYlBu_r')])

    fig.update_layout(
    autosize=False,
    width =800,
    height=600,
    margin=dict(l=65, r=50, b=65, t=90))
    fig.layout.scene.camera.projection.type = "orthographic"
    fig.show()
    return


# 定义矩阵Q # 正定
Q1 = np.array([[1, 0, 0],
              [0, 2, 0],
              [0, 0, 3]])
visualize_Q(Q1,'Q1')

# 定义矩阵Q # 半正定
Q2 = np.array([[1, 0, 0],
              [0, 0, 0],
              [0, 0, 3]])
visualize_Q(Q2,'Q2')

# 定义矩阵Q, 负定
Q3 = np.array([[-1, 0, 0],
              [0, -2, 0],
              [0, 0, -3]])
visualize_Q(Q3,'Q3')

# 定义矩阵Q，半负定
Q4 = np.array([[-1, 0, 0],
              [0, 0, 0],
              [0, 0, -3]])
visualize_Q(Q4,'Q4')

# 定义矩阵Q，不定
Q5 = np.array([[0, 1, 0],
              [1, 0, 0],
              [0, 0, 0]])
visualize_Q(Q5,'Q5')

# 定义矩阵Q, 不定
Q6 = np.array([[1, 0, 0],
              [0, -2, 0],
              [0, 0, 3]])
visualize_Q(Q6,'Q6')

#%% 单位球面上的瑞利商等高线
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# 设置步数
intervals = 50
ntheta = intervals
nphi = 2*intervals

# 单位球，球坐标
# theta取值范围为 [0, pi]
theta = np.linspace(0, np.pi*1, ntheta+1)
# phi取值范围为 [0, 2*pi]
phi   = np.linspace(0, np.pi*2, nphi+1)

# 单位球半径
r = 1

# 球坐标转化为三维直角坐标
# z轴坐标网格数据
Z = np.outer(r*np.cos(theta), np.ones(nphi+1))
# x轴坐标网格数据
X = np.outer(r*np.sin(theta), np.cos(phi))
# y轴坐标网格数据
Y = np.outer(r*np.sin(theta), np.sin(phi))
pp_, tt_ = np.meshgrid(phi, theta)

# 每一行代表一个三维直角坐标系坐标点
# 所有坐标点都在单位球面上
Points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

# 定义矩阵Q
Q = np.array([[1, 0.5, 1],
              [0.5, 2, -0.2],
              [1, -0.2, 1]])
Rayleigh_Q = np.diag(Points @ Q @ Points.T)
Rayleigh_Q_ = np.reshape(Rayleigh_Q, X.shape)

## 1 展开成平面
fig, ax = plt.subplots()
levels = np.linspace(Rayleigh_Q_.min(), Rayleigh_Q_.max(), 12)

colorbar = ax.contour(pp_,tt_, Rayleigh_Q_, levels = levels, cmap='RdYlBu_r')

fig.colorbar(colorbar, ax=ax)
ax.set_xlim(tt_.min(), tt_.max())
ax.set_ylim(pp_.min(), pp_.max())
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_xlabel('$phi$')
ax.set_ylabel('$theta$')
plt.axis('scaled')
# fig.savefig('Figures/对x1偏导.svg', format='svg')
plt.show()

## 2
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
all_contours = ax.contour(pp_,tt_, Rayleigh_Q_, levels = levels, cmap='RdYlBu_r')
# 提取等高线
ax.cla()
# 擦去等高线这个“艺术家”
plt.show()

## 3 球面颜色映射 + 单色等高线
fig = plt.figure(figsize = (12, 12))
ax = fig.add_subplot(111, projection='3d')

norm_plt = plt.Normalize(Rayleigh_Q_.min(), Rayleigh_Q_.max())
colors = cm.RdYlBu_r(norm_plt(Rayleigh_Q_))
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,  linewidth=0.25, shade=False)
# surf.set_facecolor((0,0,0,0)) ##

for level_idx, ctr_idx in zip(all_contours.levels, all_contours.allsegs):
    for i in range(0,len(ctr_idx)):
        phi_i, theta_i = ctr_idx[i][:,0],ctr_idx[i][:,1]
        # 单位球半径
        r = 1
        # 球坐标转化为三维直角坐标
        # z轴坐标网格数据
        Z_i = r*np.cos(theta_i)
        # x轴坐标网格数据
        X_i = r*np.sin(theta_i)*np.cos(phi_i)
        # y轴坐标网格数据
        Y_i = r*np.sin(theta_i)*np.sin(phi_i)
        # 绘制映射结果
        ax.plot(X_i, Y_i, Z_i, color = 'w',  linewidth = 1, zorder = 1e10)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
k = 1.5
# three reference lines
ax.plot((-k, k), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k, k), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k, k), 'k')
ax.axis('off')
ax.set_xlim((-k, k))
ax.set_ylim((-k, k))
ax.set_zlim((-k, k))
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-130, elev=30)
ax.grid(False)
# fig.savefig('瑞利商，球面颜色映射 + 单色等高线.svg', format='svg')
plt.show()

## 4 球面颜色映射 + 彩色等高线
from matplotlib.colors import Normalize
norm = Normalize(vmin = all_contours.levels.min(), vmax = all_contours.levels.max(),  clip = True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlBu_r)

fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color = [0.5, 0.5, 0.5], linewidth=0.5)
surf.set_facecolor((0,0,0,0))

for level_idx, ctr_idx in zip(all_contours.levels,  all_contours.allsegs):
    for i in range(0,len(ctr_idx)):
        phi_i,theta_i = ctr_idx[i][:,0],ctr_idx[i][:,1]
        # 单位球半径
        r = 1
        # 球坐标转化为三维直角坐标
        # z轴坐标网格数据
        Z_i = r*np.cos(theta_i)
        # x轴坐标网格数据
        X_i = r*np.sin(theta_i)*np.cos(phi_i)
        # y轴坐标网格数据
        Y_i = r*np.sin(theta_i)*np.sin(phi_i)
        # 绘制映射结果
        ax.plot(X_i, Y_i, Z_i,  color = mapper.to_rgba(level_idx),  linewidth = 3)

ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('$\it{x_3}$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
k = 1.5
# three reference lines
ax.plot((-k, k), (0, 0), (0, 0), 'k')
ax.plot((0, 0), (-k, k), (0, 0), 'k')
ax.plot((0, 0), (0, 0), (-k, k), 'k')
ax.axis('off')
ax.set_xlim((-k, k))
ax.set_ylim((-k, k))
ax.set_zlim((-k, k))
ax.set_box_aspect([1,1,1])
ax.view_init(azim=-130, elev=30)
ax.grid(False)
# fig.savefig('瑞利商，球面颜色映射 + 彩色等高线.svg', format='svg')
plt.show()

plt.close('all')
































































































































































































































































































































































































































































































































































