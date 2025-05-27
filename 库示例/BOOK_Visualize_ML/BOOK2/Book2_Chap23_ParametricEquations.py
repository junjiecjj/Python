#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:05:24 2024

@author: jack
 Chapter 23 参数方程 | Book 2《可视之美》


"""


#%% BK_2_Ch23_01 心形线
import numpy as np
import matplotlib.pyplot as plt

p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5

a = 1
phi_array = np.linspace(0,2*np.pi,181)
x_array = 2*a*(1 - np.cos(phi_array)) * np.cos(phi_array)
y_array = 2*a*(1 - np.cos(phi_array)) * np.sin(phi_array)

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(x_array,y_array)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_xticks(np.arange(-5,6))
ax.set_yticks(np.arange(-5,6))
ax.axvline(x = 0, c = 'k')
ax.axhline(y = 0, c = 'k')
ax.grid(True)
# ax.axis('off')
# fig.savefig('心形线，参数方程，1.svg') # png
plt.show()


a = 1
phi_array = np.linspace(0,2*np.pi,181)
x_array = 2*a*(1 + np.cos(phi_array)) * np.cos(phi_array)
y_array = 2*a*(1 + np.cos(phi_array)) * np.sin(phi_array)

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(x_array, y_array)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_xticks(np.arange(-5,6))
ax.set_yticks(np.arange(-5,6))
ax.axvline(x = 0, c = 'k')
ax.axhline(y = 0, c = 'k')
ax.grid(True)
# ax.axis('off')
# fig.savefig('心形线，参数方程，2.svg') # png
plt.show()


##############
a = 1
phi_array = np.linspace(0,2*np.pi,181)
x_array = 2*a*(1 + np.sin(phi_array)) * np.cos(phi_array)
y_array = 2*a*(1 + np.sin(phi_array)) * np.sin(phi_array)

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(x_array,y_array)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_xticks(np.arange(-5,6))
ax.set_yticks(np.arange(-5,6))
ax.axvline(x = 0, c = 'k')
ax.axhline(y = 0, c = 'k')
ax.grid(True)
# ax.axis('off')
# fig.savefig('心形线，参数方程，3.svg') # png
plt.show()



#%% BK_2_Ch23_02 利萨茹曲线
import matplotlib.pyplot as plt
import numpy as np

k = 0
fig = plt.figure(figsize = (12,12),constrained_layout=True)
gspec = fig.add_gridspec(9, 9)
nrows, ncols = gspec.get_geometry()
axs = np.array([[fig.add_subplot(gspec[i, j]) for j in range(ncols)] for i in range(nrows)])
t = np.linspace(0, 4, 1000)

for i in range(nrows):
    for j in range(ncols):
        nx = i + 1
        ny = j + 1
        x_traj = np.cos(2*np.pi*nx*t)
        y_traj = np.cos(2*np.pi*ny*t + k*np.pi/4/nx)
        axs[i, j].plot(x_traj, y_traj)
        axs[i, j].set_aspect('equal', 'box')
        axs[i, j].axis('off')
# fig.savefig('Figures/利萨茹曲线，k = 0.svg', format='svg')
plt.show()


k = 2
fig = plt.figure(figsize = (12,12),constrained_layout=True)
gspec = fig.add_gridspec(9, 9)
nrows, ncols = gspec.get_geometry()
axs = np.array([[fig.add_subplot(gspec[i, j]) for j in range(ncols)] for i in range(nrows)])
t = np.linspace(0, 4, 1000)

for i in range(nrows):
    for j in range(ncols):
        nx = i + 1
        ny = j + 1
        x_traj = np.cos(2*np.pi*nx*t)
        y_traj = np.cos(2*np.pi*ny*t + k*np.pi/4/nx)
        axs[i, j].plot(x_traj, y_traj)
        axs[i, j].set_aspect('equal', 'box')
        axs[i, j].axis('off')

# fig.savefig('Figures/利萨茹曲线，k = 2.svg', format='svg')
plt.show()


fig = plt.figure(figsize = (12,12),constrained_layout=True)
gspec = fig.add_gridspec(9, 9)
nrows, ncols = gspec.get_geometry()
axs = np.array([[fig.add_subplot(gspec[i, j]) for j in range(ncols)] for i in range(nrows)])
t = np.linspace(0, 4, 1000)
nx_array = [1, 2, 3, 3, 4, 4, 5, 5, 5]
ny_array = [1, 1, 1, 2, 1, 3, 1, 2, 3]
for i in range(nrows):
    nx = nx_array[i]
    ny = ny_array[i]
    for j in range(ncols):
        k = j
        x_traj = np.cos(2*np.pi*nx*t)
        y_traj = np.cos(2*np.pi*ny*t + k*np.pi/4/nx)

        axs[i, j].plot(x_traj, y_traj)
        axs[i, j].set_aspect('equal', 'box')
        axs[i, j].axis('off')
        # axs[i, j].set_title('nx = ' + str(nx) + '; ny = ' + str(ny) +
        #                     '; k = ' + str(k), fontsize = 6)
# fig.savefig('Figures/利萨茹曲线，k变化.svg', format='svg')
plt.show()


#%% BK_2_Ch23_03 各种参数方程曲面, 参见Bk_2_Ch18_01，Book2_chap23

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import plotly.graph_objects as go

def visualize(X, Y, Z, title):
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, linewidth=0.25)
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
    ax.set_xlim((-k, k))
    ax.set_ylim((-k, k))
    ax.set_zlim((-k, k))
    ax.axis('off')
    ax.set_box_aspect([1,1,1])
    ax.view_init(azim=-130, elev=25)
    ax.grid(False)
    # plt.savefig(title + '.svg')
    plt.show()
    return

def plotly_visualize(X,Y,Z):
    data = go.Surface(x=X, y=Y, z=Z, colorscale = 'Plotly3')
    fig = go.Figure(data=[data])

    fig.update_layout(autosize=False, width=500, height=500, margin=dict(l=65, r=50, b=65, t=90))
    fig.show()
    return

################ 上下半球面
# 设置步数
intervals = 50
ntheta = intervals
nphi = 2*intervals

# 单位球，球坐标
# theta取值范围为 [0, pi]
theta = np.linspace(0, np.pi/2, ntheta+1)
# phi取值范围为 [0, 2*pi]
phi   = np.linspace(0, np.pi*2, nphi+1)

# 单位球半径
r = 1
# 球坐标转化为三维直角坐标
pp_, tt_ = np.meshgrid(phi, theta)
# z轴坐标网格数据
Z = r*np.cos(tt_)
# x轴坐标网格数据
X = r*np.sin(tt_)*np.cos(pp_)
# y轴坐标网格数据
Y = r*np.sin(tt_)*np.sin(pp_)

visualize(X,Y,Z,'上半球')


# 设置步数
intervals = 50
ntheta = intervals
nphi = 2*intervals
# 单位球，球坐标
# theta取值范围为 [0, pi]
theta = np.linspace(np.pi/2, np.pi, ntheta+1)
# phi取值范围为 [0, 2*pi]
phi   = np.linspace(0, np.pi*2, nphi+1)
# 单位球半径
r = 1
# 球坐标转化为三维直角坐标
pp_, tt_ = np.meshgrid(phi, theta)
# z轴坐标网格数据
Z = r*np.cos(tt_)
# x轴坐标网格数据
X = r*np.sin(tt_)*np.cos(pp_)
# y轴坐标网格数据
Y = r*np.sin(tt_)*np.sin(pp_)
visualize(X,Y,Z,'下半球')
################ 球面
# 设置步数
intervals = 50
ntheta = intervals
nphi = 2*intervals

# 单位球，球坐标
# theta取值范围为 [0, pi]
theta = np.linspace(0, np.pi, ntheta+1)
# phi取值范围为 [0, 2*pi]
phi   = np.linspace(0, np.pi*2, nphi+1)

# 单位球半径
r = 1

# 球坐标转化为三维直角坐标
pp_,tt_ = np.meshgrid(phi, theta)

# z轴坐标网格数据
Z = r*np.cos(tt_)

# x轴坐标网格数据
X = r*np.sin(tt_)*np.cos(pp_)

# y轴坐标网格数据
Y = r*np.sin(tt_)*np.sin(pp_)

visualize(X,Y,Z,'下半球')


################ 左半球面
# 设置步数
intervals = 50
ntheta = intervals
nphi = 2*intervals

# 单位球，球坐标
# theta取值范围为 [0, pi]
theta = np.linspace(0, np.pi*1, ntheta+1)
# phi取值范围为 [0, 2*pi]
phi   = np.linspace(0, np.pi, nphi+1)

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

visualize(X,Y,Z,'左半球面')

# 设置步数
intervals = 50
ntheta = intervals
nphi = 2*intervals

# 单位球，球坐标
# theta取值范围为 [0, pi]
theta = np.linspace(0, np.pi*1, ntheta+1)
# phi取值范围为 [0, 2*pi]
phi   = np.linspace(np.pi, 2*np.pi, nphi+1)

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

visualize(X,Y,Z,'右半球面')


################ 圆柱
# 设置步数
intervals = 50
ntheta = intervals
nphi = 2*intervals

# 单位球，球坐标
# theta取值范围为 [0, pi]
theta = np.linspace(0, np.pi*1, ntheta+1)
# phi取值范围为 [0, 2*pi]
phi   = np.linspace(0, 2*np.pi, nphi+1)

# 单位球半径
r = 1

# 球坐标转化为三维直角坐标
pp_,tt_ = np.meshgrid(phi,theta)

# z轴坐标网格数据
Z = np.linspace(-1, 1, intervals + 1)*np.ones_like(tt_).T
Z = Z.T

# x轴坐标网格数据
X = r*np.cos(pp_)

# y轴坐标网格数据
Y = r*np.sin(pp_)

visualize(X,Y,Z,'圆柱')

##################### 椎体
# 设置步数
intervals = 50
ntheta = intervals
nphi = 2*intervals

# 单位球，球坐标
# theta取值范围为 [0, pi]
theta = np.linspace(0, np.pi*1, ntheta+1)
# phi取值范围为 [0, 2*pi]
phi   = np.linspace(0, 2*np.pi, nphi+1)

# 单位球半径
r = 1

# 球坐标转化为三维直角坐标
pp_,tt_ = np.meshgrid(phi,theta)

# z轴坐标网格数据
Z = np.linspace(0,1,intervals + 1)*np.ones_like(tt_).T
Z = Z.T

# x轴坐标网格数据
X = np.linspace(0,1,intervals + 1)*np.cos(pp_).T
X = X.T
# y轴坐标网格数据
Y = np.linspace(0,1,intervals + 1)*np.sin(pp_).T
Y = Y.T
visualize(X,Y,Z,'椎体，开口朝上')


# 设置步数
intervals = 50
ntheta = intervals
nphi = 2*intervals

# 单位球，球坐标
# theta取值范围为 [0, pi]
theta = np.linspace(0, np.pi*1, ntheta+1)
# phi取值范围为 [0, 2*pi]
phi   = np.linspace(0, 2*np.pi, nphi+1)

# 单位球半径
r = 1

# 球坐标转化为三维直角坐标
pp_,tt_ = np.meshgrid(phi,theta)

# z轴坐标网格数据
Z = np.linspace(-1,0,intervals + 1)*np.ones_like(tt_).T
Z = Z.T

# x轴坐标网格数据
X = np.linspace(-1,0,intervals + 1)*np.cos(pp_).T
X = X.T
# y轴坐标网格数据
Y = np.linspace(-1,0,intervals + 1)*np.sin(pp_).T
Y = Y.T
visualize(X,Y,Z,'椎体，开口朝下')


################# 对顶圆锥
# 设置步数
intervals = 50
ntheta = intervals
nphi = 2*intervals

# 单位球，球坐标
# theta取值范围为 [0, pi]
theta = np.linspace(0, np.pi*1, ntheta+1)
# phi取值范围为 [0, 2*pi]
phi   = np.linspace(0, 2*np.pi, nphi+1)

# 单位球半径
r = 1

# 球坐标转化为三维直角坐标
pp_,tt_ = np.meshgrid(phi,theta)

# z轴坐标网格数据
Z = np.linspace(-1,1,intervals + 1)*np.ones_like(tt_).T
Z = Z.T

# x轴坐标网格数据
X = np.linspace(-1,1,intervals + 1)*np.cos(pp_).T
X = X.T
# y轴坐标网格数据
Y = np.linspace(-1,1,intervals + 1)*np.sin(pp_).T
Y = Y.T
visualize(X,Y,Z,'对顶圆锥')


################ 椭球
u = np.linspace(0, np.pi, 101)
v = np.linspace(0, 2 * np.pi, 101)

x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))

S_xyz = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 2]])
position = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
position_trans = position @ S_xyz.T

x_trans = position_trans[:,0].reshape(x.shape)
y_trans = position_trans[:,1].reshape(x.shape)
z_trans = position_trans[:,2].reshape(x.shape)

visualize(x_trans,y_trans,z_trans,'椭球')

################### 救生圈
# 设置步数
intervals = 50
ntheta = intervals
nphi = 2*intervals

# 单位球，球坐标
# theta取值范围为 [0, pi]
theta = np.linspace(0, 2*np.pi*1, ntheta+1)
# phi取值范围为 [0, 2*pi]
phi   = np.linspace(0, 2*np.pi, nphi+1)
r = 0.5
R = 2

# 球坐标转化为三维直角坐标
pp_, tt_ = np.meshgrid(phi, theta)
X = (R + r*np.cos(tt_))*np.cos(pp_)
Y = (R + r*np.cos(tt_))*np.sin(pp_)
Z = r*np.sin(tt_)

visualize(X,Y,Z,'游泳圈')
















































































































































































































































































































































































































































































































