#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 16:22:49 2024

@author: jack
"""



#%% 1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # Colormaps
from scipy.stats import norm


X = np.arange(-3, 3, 0.05)
mu = 0
sigma = 1

# 图 9. 标准正态分布 z 和 PDF 的对应关系
x_selected = np.linspace(-2.8, 2.8, num=29)
colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(x_selected)))
colors = cm.RdYlBu_r(np.linspace(0, 1, len(x_selected))) # (29, 4)
f_x = norm.pdf(X, loc = mu, scale = sigma)

fig, ax = plt.subplots()
plt.plot(X,f_x)
for i in np.linspace(0,len(x_selected)-1,len(x_selected)):
    x_selected_i = x_selected[int(i)]
    x_PDF = norm.pdf(x_selected_i)
    plt.vlines(x = x_selected_i, ymin = 0, ymax = x_PDF, color = colors[int(i)])
    plt.plot(x_selected_i, x_PDF, marker = 'x', color = colors[int(i)], markersize = 12)

ax.set_xlim(-3, 3)
ax.set_ylim(0,  0.5)
ax.set_xlabel('z')
ax.set_ylabel('PDF, $f_{Z}(z)$')


#%% 1
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def mesh_square(x1_0, x2_0,r, num):
    # generate mesh
    rr = np.linspace(-r,r,num)
    xx1,xx2 = np.meshgrid(rr,rr);
    xx1 = xx1 + x1_0;
    xx2 = xx2 + x2_0;
    return xx1, xx2

def plot_surf(xx1,xx2,ff,caption):
    norm_plt = plt.Normalize(ff.min(), ff.max())
    colors = cm.coolwarm(norm_plt(ff)) # (30, 30, 4)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
    surf = ax.plot_surface(xx1, xx2, ff, facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    # z_lim = [ff.min(),ff.max()]
    # ax.plot3D([0,0],[0,0],z_lim,'k')
    plt.show()

    plt.tight_layout()
    ax.set_xlabel('$\it{x_1}$')
    ax.set_ylabel('$\it{x_2}$')
    ax.set_zlabel('$\it{f}$($\it{x_1}$,$\it{x_2}$)')
    ax.set_title(caption)

    ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "10"

def plot_contourf(xx1, xx2, ff, caption):
    fig, ax = plt.subplots()
    cntr2 = ax.contourf(xx1,xx2,ff, levels = 15, cmap="RdBu_r")

    fig.colorbar(cntr2, ax=ax)

    ax.set_xlabel('$\it{x_1}$')
    ax.set_ylabel('$\it{x_2}$')

    ax.set_title(caption)
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    plt.show()
# initialization

x1_0  = 0;  # center of the mesh
x2_0  = 0;  # center of the mesh
r     = 2;  # radius of the mesh
num   = 30; # number of mesh grids
xx1, xx2 = mesh_square(x1_0, x2_0, r, num); # generate mesh

#  Visualizations
plt.close('all')

# f(x1,x2) = -x1
ff = -xx1;
caption = '$\it{f} = -\it{x_1}$';
plot_surf(xx1,xx2,ff,caption)
plot_contourf (xx1,xx2,ff,caption)


#  f(x1,x2) = x2
ff = xx2;
caption = '$\it{f} = \it{x_2}$';
plot_surf (xx1,xx2,ff,caption)
plot_contourf (xx1,xx2,ff,caption)

#  f(x1,x2) = x1 + x2
ff = xx1 + xx2;
caption = '$\it{f} = \it{x_1} + \it{x_2}$';
plot_surf (xx1,xx2,ff,caption)
plot_contourf (xx1,xx2,ff,caption)

#  f(x1,x2) = -x1 + x2
ff = -xx1 + xx2;
caption = '$\it{f} = -\it{x_1} + \it{x_2}$';
plot_surf (xx1,xx2,ff,caption)
plot_contourf (xx1,xx2,ff,caption)





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 2 瑞利商

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
Z = np.outer(r*np.cos(theta), np.ones(nphi+1)) #  (51, 101)
# x轴坐标网格数据
X = np.outer(r*np.sin(theta), np.cos(phi)) #  (51, 101)
# y轴坐标网格数据
Y = np.outer(r*np.sin(theta), np.sin(phi)) #  (51, 101)

# 第二种方法
pp_, tt_ = np.meshgrid(phi, theta)
# z轴坐标网格数据
Z = r*np.cos(tt_) #  (51, 101)
# x轴坐标网格数据
X = r*np.sin(tt_)*np.cos(pp_) #  (51, 101)
# y轴坐标网格数据
Y = r*np.sin(tt_)*np.sin(pp_) #  (51, 101)

################## 1. 计算瑞利商
# 每一行代表一个三维直角坐标系坐标点
# 所有坐标点都在单位球面上
Points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

# 定义矩阵Q
Q = np.array([[1, 0.5, 1], [0.5, 2, -0.2], [1, -0.2, 1]])
# 计算 xT @ Q @ x
Rayleigh_Q = np.diag(Points @ Q @ Points.T) # (5151,)
#
Rayleigh_Q_ = np.reshape(Rayleigh_Q, X.shape) #  (51, 101)

######## 1  小彩灯:根据瑞丽熵映射球面颜色
fig = plt.figure(figsize = (20, 20))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color = '0.68',linewidth=0.25)
surf.set_facecolor((0, 0, 0, 0))
ax.scatter(X[::2,::2], Y[::2,::2], Z[::2,::2], c = Rayleigh_Q_[::2,::2], cmap = 'hsv', s = 15)
# 另外一种设定正交投影的方式
ax.set_proj_type('ortho')
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
# fig.savefig('Figures/单位球面上点 + 渲染.svg', format='svg')
plt.show()

######## 1  小彩灯:根据Z值映射球面颜色
fig = plt.figure(figsize = (20, 20))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color = '0.68', linewidth=0.25)
surf.set_facecolor((0, 0, 0, 0))
ax.scatter(X[::2,::2], Y[::2,::2], Z[::2,::2], c = Z[::2,::2], cmap = 'hsv', s = 15)
# 另外一种设定正交投影的方式
ax.set_proj_type('ortho')
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
# fig.savefig('Figures/单位球面上点 + 渲染.svg', format='svg')
plt.show()

######## 2. 填充
norm_plt = plt.Normalize(Rayleigh_Q_.min(), Rayleigh_Q_.max())
colors = cm.RdYlBu_r(norm_plt(Rayleigh_Q_)) #  (51, 101, 4)

fig = plt.figure(figsize = (20, 20))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,  linewidth=0.25, shade=False)
# 另外一种设定正交投影的方式
ax.set_proj_type('ortho')
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

######## 3. 只有网格
fig = plt.figure(figsize = (20, 20))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,  linewidth=0.25, shade=False)
surf.set_facecolor((0,0,0,0))  # 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。
# 另外一种设定正交投影的方式
ax.set_proj_type('ortho')
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 3
def visualize_1D(array, title):
    fig, ax = plt.subplots()
    colors = cm.RdYlBu_r(np.linspace(0,1,len(array)))
    for idx in range(len(array)):
        circle_idx = plt.Circle((idx, 0), 0.5, facecolor=colors[idx], edgecolor = 'w')
        ax.add_patch(circle_idx)
        ax.text(idx, 0, s = str(array[idx]), horizontalalignment = 'center', verticalalignment = 'center')

    ax.set_xlim(-0.6, 0.6 + len(array))
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    # fig.savefig('Figures/' + title + '.svg', format='svg')


# ### 生成1D数组
a_1D_array = np.arange(-7, 7 + 1)
a_1D_array
visualize_1D(a_1D_array, 'aa')

#%%=======================
# 1 表面图（Surface plots）
#======================
# 导入包
import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import lambdify, diff, exp, latex
from sympy.abc import x, y
# 导入符号变量
import os

from matplotlib import cm
# 导入色谱模块
# 1. 定义函数¶
num = 301; # number of mesh grids
x_array = np.linspace(-3,3,num)
y_array = np.linspace(-3,3,num)
xx,yy = np.meshgrid(x_array,y_array)

# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x,y],f_xy)
# 将符号函数表达式转换为Python函数
ff = f_xy_fcn(xx, yy)

############ 1 用plot_surface() 绘制二元函数曲面
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (12,12))
ax.set_proj_type('ortho')
#  正交投影模式

# 使用 RdYlBu 色谱
# 请大家试着调用其他色谱
surf = ax.plot_surface(xx,yy, ff,
                cmap=cm.RdYlBu,
                antialiased = False, # 反锯齿
                rstride=2, cstride=2,
                linewidth = 1,
                edgecolors = [0.5,0.5,0.5],
                shade = False
                ) # 删除阴影 shade = False
surf.set_facecolor((0,0,0,0)) # 网格面填充为空, 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。
fig.colorbar(surf, shrink=0.5, aspect=20)
# 设定横纵轴标签
ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('$\it{f}$($\it{x_1}$,$\it{x_2}$)')

# 设定横、纵轴取值范围
ax.set_xlim(x_array.min(), x_array.max())
ax.set_ylim(y_array.min(), y_array.max())

# 设定观察视角
ax.view_init(azim=-135, elev=30)

# 删除网格
ax.grid(False)

# 修改字体、字号
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"

plt.show()

############ 2 翻转色谱
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (12,12))
ax.set_proj_type('ortho')
surf = ax.plot_surface(xx,yy,ff, cmap='RdYlBu_r', linewidth=2, antialiased=False)

ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('$\it{f}$($\it{x_1}$,$\it{x_2}$)')

ax.set_xlim(x_array.min(), x_array.max())
ax.set_ylim(y_array.min(), y_array.max())

ax.view_init(azim=-135, elev=30)

ax.grid(False)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"

fig.colorbar(surf, shrink=0.5, aspect=20)
plt.show()

############ 3 只保留网格线, 同样使用 plot_surface()，不同的是只保留彩色网格
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (12,12))
ax.set_proj_type('ortho')
norm_plt = plt.Normalize(ff.min(), ff.max())
colors = cm.RdYlBu_r(norm_plt(ff))
# colors = cm.Blues_r(norm_plt(ff))

surf = ax.plot_surface(xx,yy,ff,
                       # cmap=cm.hsv,
                       facecolors = colors,
                       rstride = 2,
                       cstride = 2,
                       # edgecolors = [0.5,0.5,0.5],
                       linewidth = 1, # 线宽
                       shade = False) # 删除阴影
surf.set_facecolor((0,0,0,0)) # 网格面填充为空, 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。

# 三维等高线
# colorbar = ax.contour(xx,yy, ff, 20,  cmap = 'hsv')
colorbar = ax.contour3D(xx,yy, ff, 20,  cmap = 'hsv')
fig.colorbar(colorbar, ax = ax, shrink=0.5, aspect=20)

# 二维等高线
ax.contour(xx, yy, ff, zdir='z', offset= ff.min(), levels = 20, linewidths = 2, cmap = "hsv")  # 生成z方向投影，投到x-y平面

ax.set_proj_type('ortho')
ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('$\it{f}$($\it{x_1}$,$\it{x_2}$)')

ax.set_xlim(x_array.min(), x_array.max())
ax.set_ylim(y_array.min(), y_array.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "10"
# fig.savefig('Figures/只保留网格线.svg', format='svg')
plt.show()


############## 4 plot_wireframe() 绘制网格曲面 + 三维等高线
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (12,12))

ax.plot_wireframe(xx,yy, ff, color = [0.5,0.5,0.5], linewidth = 0.25)

# 三维等高线
# colorbar = ax.contour(xx,yy, ff,20,  cmap = 'RdYlBu_r')
# 三维等高线
colorbar = ax.contour(xx,yy, ff, 20,  cmap = 'hsv')
# fig.colorbar(colorbar, ax = ax, shrink=0.5, aspect=20)

# 二维等高线
ax.contour(xx, yy, ff, zdir='z', offset= ff.min(), levels = 20, linewidths = 2, cmap = "hsv")  # 生成z方向投影，投到x-y平面

fig.colorbar(colorbar, ax=ax, shrink=0.5, aspect=20)
ax.set_proj_type('ortho')

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # 3D坐标区的背景设置为白色
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.set_xlabel('$\it{x_1}$')
ax.set_ylabel('$\it{x_2}$')
ax.set_zlabel('$\it{f}$($\it{x_1}$,$\it{x_2}$)')

ax.set_xlim(x_array.min(), x_array.max())
ax.set_ylim(y_array.min(), y_array.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
plt.show()


############## 5. 绘制网格化散点
num = 70 # number of mesh grids
x_array = np.linspace(-3,3,num)
y_array = np.linspace(-3,3,num)
xx,yy = np.meshgrid(x_array,y_array)

# 用 sympy 库定义 MATLAB二元函数 peaks()
f_xy =  3*(1-x)**2*exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*exp(-x**2-y**2) - 1/3*exp(-(x+1)**2 - y**2)

f_xy_fcn = lambdify([x,y],f_xy)
# 将符号函数表达式转换为Python函数
ff = f_xy_fcn(xx, yy)
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (12,12))

ax.plot_wireframe(xx,yy, ff, color = [0.6,0.6,0.6], rstride=5, cstride=5, linewidth = 0.25)
ax.scatter(xx, yy, ff, c = ff, s = 10, cmap = 'RdYlBu_r')
ax.set_proj_type('ortho')
# 另外一种设定正交投影的方式

ax.set_xlabel('$\it{x}$')
ax.set_ylabel('$\it{y}$')
ax.set_zlabel('$\it{f}$($\it{x}$,$\it{y}$)')

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

ax.view_init(azim=-135, elev=30)
ax.grid(False)
# fig.savefig('Figures/增加网格散点.svg', format='svg')
plt.show()


################## 6  用冷暖色表示函数的不同高度取值
fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (12, 12))

surf = ax.plot_surface(xx,yy, ff,
                cmap=cm.RdYlBu_r,
                rstride=2, cstride=2,
                linewidth = 1,
                edgecolors = [0.5,0.5,0.5],
                shade = False
                ) # 删除阴影 shade = False
# surf.set_facecolor((0,0,0,0)) # 网格面填充为空, 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。

ax.set_proj_type('ortho')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f(x,y)$')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.view_init(azim=-135, elev=30)
plt.tight_layout()
ax.grid(False)
plt.show()




























































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































