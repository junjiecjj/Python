#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:59:34 2024
https://geek-docs.com/matplotlib/matplotlib-pyplot/matplotlib-pyplot-streamplot-in-python.html#google_vignette
@author: jack
"""


#%% Bk_2_Ch17_01 二维箭头图

# 导入包
import matplotlib.pyplot as plt
import numpy as np
### 定义向量
a = np.array([[-2],
              [5]])

b = np.array([[5],
              [-1]])
### 向量加法
c = a + b
fig, ax = plt.subplots(figsize=(5, 5))
# 绘制向量 a
ax.quiver(0, 0, a[0], a[1],  angles='xy', scale_units='xy', scale=1,  color = '#92D050')
ax.text(-3, 2.5, '$a (-2, 5)$', fontsize = 10)
# 绘制向量 b
ax.quiver(0, 0, b[0], b[1],  angles='xy', scale_units='xy', scale=1, color = '#FFC000')
ax.text(1.5, -1.5, '$b (5, -1)$', fontsize = 10)
# 绘制向量 c
ax.quiver(0, 0, c[0], c[1],  angles='xy', scale_units='xy', scale=1,  color = '#0099FF')
ax.text(2, 2, '$c (3, 4)$', fontsize = 10)
# 绘制 a, c 终点连线
ax.plot([a[0], c[0]],  [a[1], c[1]],  c = 'k', ls = '--')
# 绘制 b, c 终点连线
ax.plot([b[0], c[0]], [b[1], c[1]], c = 'k', ls = '--')

# 添加阴影填充
fill_color = np.array([219,238,243])/255

X = np.array([[0, 0],
              [-2, 5],
              [3, 4],
              [5, -1],
              [0, 0]])

plt.fill(X[:,0], X[:,1], color = fill_color, edgecolor = None, alpha = 0.5)

# 增加横轴、纵轴线
ax.axhline(y = 0, c = 'k', lw = 0.25)
ax.axvline(x = 0, c = 'k', lw = 0.25)

ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_xticks(np.arange(-5,6))
ax.set_yticks(np.arange(-5,6))
ax.set_aspect('equal', adjustable='box')
ax.grid(c = [0.8, 0.8, 0.8], zorder = 1e3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_xticklabels([])
ax.set_yticklabels([])
plt.show()
# fig.savefig('Figures/向量加法，第一方法.svg', format='svg')


fig, ax = plt.subplots(figsize=(5, 5))
# 绘制向量 a
ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1, color = '#92D050')
ax.text(-3, 2.5, '$a (-2, 5)$', fontsize = 10)
# 绘制向量 b
ax.quiver(a[0], a[1], b[0], b[1], angles='xy', scale_units='xy', scale=1, color = '#FFC000')
ax.text(1, 5, '$b (5, -1)$', fontsize = 10)
# 绘制向量 c
ax.quiver(0, 0, c[0], c[1], angles='xy', scale_units='xy', scale=1, color = '#0099FF')
ax.text(2, 2, '$c (3, 4)$', fontsize = 10)
# 添加阴影填充
fill_color = np.array([219,238,243])/255
X = np.array([[0, 0],
              [-2, 5],
              [3, 4],
              [0, 0]])

plt.fill(X[:,0], X[:,1],
     color = fill_color,
     edgecolor = None,
     alpha = 0.5)

# 增加横轴、纵轴线
ax.axhline(y = 0, c = 'k', lw = 0.25)
ax.axvline(x = 0, c = 'k', lw = 0.25)

ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_xticks(np.arange(-5,6))
ax.set_yticks(np.arange(-5,6))
ax.set_aspect('equal', adjustable='box')
ax.grid(c = [0.8, 0.8, 0.8], zorder = 1e3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_xticklabels([])
ax.set_yticklabels([])
plt.show()
# fig.savefig('Figures/向量加法，第二方法.svg', format='svg')


### 向量长度
fig, ax = plt.subplots(figsize=(5, 5))

# 绘制向量 c
ax.quiver(0, 0, c[0], c[1], angles='xy', scale_units='xy', scale=1, color = '#0099FF')
ax.text(2, 2, '$c (3, 4)$', fontsize = 10)
ax.text(0.5, 2, '$\sqrt{3^2+4^2} = 5$', fontsize = 10, rotation = np.arctan(4/3)*180/np.pi)

# 增加横轴、纵轴线
ax.axhline(y = 0, c = 'k', lw = 0.25)
ax.axvline(x = 0, c = 'k', lw = 0.25)

ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_xticks(np.arange(-5,6))
ax.set_yticks(np.arange(-5,6))
ax.set_aspect('equal', adjustable='box')
ax.grid(c = [0.8, 0.8, 0.8], zorder = 1e3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_xticklabels([])
ax.set_yticklabels([])
plt.show()
# fig.savefig('Figures/向量长度.svg', format='svg')


### 向量减法
d = a - b
fig, ax = plt.subplots(figsize=(5, 5))
# 绘制向量 a
ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1, color = '#92D050')
ax.text(-3, 2.5, '$a (-2, 5)$', fontsize = 10)
# 绘制向量 b
ax.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1, color = '#FFC000')
ax.text(1.5, -1.5, '$b (5, -1)$', fontsize = 10)
# 绘制向量 d
ax.quiver(b[0], b[1], d[0], d[1], angles='xy', scale_units='xy', scale=1, color = '#000000')
ax.text(2, 2, '$d = a - b, (-7, 6)$', fontsize = 10)

# 添加阴影填充
fill_color = np.array([219,238,243])/255
X = np.array([[0, 0],
              [-2, 5],
              [5, -1],
              [0, 0]])

plt.fill(X[:,0], X[:,1], color = fill_color, edgecolor = None, alpha = 0.5)
# 增加横轴、纵轴线
ax.axhline(y = 0, c = 'k', lw = 0.25)
ax.axvline(x = 0, c = 'k', lw = 0.25)

ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_xticks(np.arange(-5,6))
ax.set_yticks(np.arange(-5,6))
ax.set_aspect('equal', adjustable='box')
ax.grid(c = [0.8, 0.8, 0.8], zorder = 1e3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_xticklabels([])
ax.set_yticklabels([])
plt.show()
# fig.savefig('Figures/向量减法.svg', format='svg')

### 标量乘向量
fig, ax = plt.subplots(figsize=(5, 5))
# 绘制向量 c
ax.quiver(0, 0, c[0], c[1], angles='xy', scale_units='xy', scale=1, color = '#0099FF')
ax.text(3, 3, '$c (3, 4)$', fontsize = 10)
# 绘制向量 c * k
k = -1
ax.quiver(0, 0, k*c[0], k*c[1], angles='xy', scale_units='xy', scale=1, color = '#0070C0')
ax.text(-2, -4, '$-c (-3, -4)$', fontsize = 10)
# 绘制向量 c * k
k = 0.5
ax.quiver(0, 0, k*c[0], k*c[1], angles='xy', scale_units='xy', scale=1, color = '#00477B')
ax.text(1.5, 1, '$0.5c (1.5, 2)$', fontsize = 10)
# 增加横轴、纵轴线
ax.axhline(y = 0, c = 'k', lw = 0.25)
ax.axvline(x = 0, c = 'k', lw = 0.25)

ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_xticks(np.arange(-5,6))
ax.set_yticks(np.arange(-5,6))
ax.set_aspect('equal', adjustable='box')
ax.grid(c = [0.8, 0.8, 0.8], zorder = 1e3)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_xticklabels([])
ax.set_yticklabels([])
plt.show()
# fig.savefig('Figures/标量乘法.svg', format='svg')



#%% Bk_2_Ch17_02 三维箭头图
import numpy as np
import matplotlib.pyplot as plt

## 定义向量
a = np.array([1, 2, 4])
b = np.array([4, 3, 1])
## 向量加法
c = a + b
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# 绘制向量 a
ax.quiver(0, 0, 0, a[0], a[1], a[2],
          color = '#92D050',
          normalize=False,
          arrow_length_ratio = .07,
          linestyles = 'solid',
          linewidths = 1)
# 绘制向量 b
ax.quiver(0, 0, 0, b[0], b[1], b[2],
          color = '#FFC000',
          normalize=False,
          arrow_length_ratio = .07,
          linestyles = 'solid',
          linewidths = 1)
# 绘制向量 c
ax.quiver(0, 0, 0, c[0], c[1], c[2],
          color = '#0099FF',
          normalize=False,
          arrow_length_ratio = .07,
          linestyles = 'solid',
          linewidths = 1)
ax.plot([a[0],c[0]], [a[1],c[1]], [a[2],c[2]],
        ls = '--',
        lw = 1,
        c = 'k')
ax.plot([b[0],c[0]], [b[1],c[1]], [b[2],c[2]],
        ls = '--',
        lw = 1,
        c = 'k')
ax.set_proj_type('ortho')
ax.view_init(azim=30, elev=30)
ax.set_box_aspect([1,1,1])

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_zlim(0, 5)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# fig.savefig('Figures/三维向量加法.svg', format='svg')

## 投影到平面
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# 绘制向量 c
ax.quiver(0, 0, 0,  c[0], c[1], c[2],
          color = '#003F6C',
          normalize=False,
          arrow_length_ratio = .07,
          linestyles = 'solid',
          linewidths = 1)
# 绘制向量 c，投影到xy
ax.quiver(0, 0, 0,  c[0], c[1], c[2] * 0,
          color = '#0099FF',
          normalize=False,
          arrow_length_ratio = .07,
          linestyles = 'solid',
          linewidths = 1)
# 绘制向量 c，投影到xz
ax.quiver(0, 0, 0,  c[0], c[1] * 0, c[2],
          color = '#0099FF',
          normalize=False,
          arrow_length_ratio = .07,
          linestyles = 'solid',
          linewidths = 1)
# 绘制向量 c，投影到yz
ax.quiver(0, 0, 0,  c[0] * 0, c[1], c[2],
          color = '#0099FF',
          normalize=False,
          arrow_length_ratio = .07,
          linestyles = 'solid',
          linewidths = 1)

# 绘制投影线
ax.plot([c[0],c[0]], [c[1],c[1]], [c[2],c[2] * 0],
        ls = '--',
        lw = 1,
        c = 'k')
ax.plot([c[0],c[0] * 0], [c[1],c[1]], [c[2],c[2]],
        ls = '--',
        lw = 1,
        c = 'k')
ax.plot([c[0],c[0]], [c[1],c[1] * 0], [c[2],c[2]],
        ls = '--',
        lw = 1,
        c = 'k')

ax.set_proj_type('ortho')
ax.view_init(azim=30, elev=30)
ax.set_box_aspect([1,1,1])

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_zlim(0, 5)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# fig.savefig('Figures/三维向量，投影到平面.svg', format='svg')

## 投影到轴
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# 绘制向量 c
ax.quiver(0, 0, 0, c[0], c[1], c[2],
          color = '#003F6C',
          normalize=False,
          arrow_length_ratio = .07,
          linestyles = 'solid',
          linewidths = 1)
# 绘制向量 c，投影到x
ax.quiver(0, 0, 0, c[0], c[1] * 0, c[2] * 0,
          color = '#0099FF',
          normalize=False,
          arrow_length_ratio = .07,
          linestyles = 'solid',
          linewidths = 1)
# 绘制向量 c，投影到y
ax.quiver(0, 0, 0, c[0] * 0, c[1], c[2] * 0,
          color = '#0099FF',
          normalize=False,
          arrow_length_ratio = .07,
          linestyles = 'solid',
          linewidths = 1)
# 绘制向量 c，投影到z
ax.quiver(0, 0, 0, c[0] * 0, c[1] * 0, c[2],
          color = '#0099FF',
          normalize=False,
          arrow_length_ratio = .07,
          linestyles = 'solid',
          linewidths = 1)
# 绘制投影线
ax.plot([c[0],c[0]], [c[1],c[1] * 0], [c[2],c[2] * 0],
        ls = '--',
        lw = 1,
        c = 'k')
ax.plot([c[0],c[0] * 0], [c[1],c[1]], [c[2],c[2] * 0],
        ls = '--',
        lw = 1,
        c = 'k')
ax.plot([c[0],c[0] * 0], [c[1],c[1] * 0], [c[2],c[2]],
        ls = '--',
        lw = 1,
        c = 'k')
# 绘制装饰框线
ax.plot([c[0],c[0]],
        [c[1],c[1]],
        [c[2],c[2] * 0],
        ls = '--',
        lw = 1,
        c = 'k')
ax.plot([c[0],c[0] * 0], [c[1],c[1]], [c[2],c[2]],
        ls = '--',
        lw = 1,
        c = 'k')
ax.plot([c[0],c[0]], [c[1],c[1] * 0], [c[2],c[2]],
        ls = '--',
        lw = 1,
        c = 'k')
ax.set_proj_type('ortho')
ax.view_init(azim=30, elev=30)
ax.set_box_aspect([1,1,1])

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_zlim(0, 5)

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
# fig.savefig('Figures/三维向量，投影到轴.svg', format='svg')

## RGB单位向量
fig = plt.figure(figsize = (12,12), constrained_layout=True)
angle_array = np.linspace(0, 180, 13)
num_grid = len(angle_array)
gspec = fig.add_gridspec(num_grid, num_grid)
nrows, ncols = gspec.get_geometry()

axs = np.array([[fig.add_subplot(gspec[i, j], projection='3d')  for j in range(ncols)]   for i in range(nrows)])

for i in range(nrows):
    elev = angle_array[i]
    for j in range(ncols):
        azim = angle_array[j]
        axs[i, j].plot(0, 0, 0, marker = '.', c = 'k', markersize = 12)
        # 红色箭头，代表x轴
        axs[i, j].quiver(0, 0, 0, 1, 0, 0,
                  color = 'r',
                  normalize=False,
                  arrow_length_ratio = .18,
                  linestyles = 'solid',
                  linewidths = 1)
        # 绿色箭头，代表y轴
        axs[i, j].quiver(0, 0, 0, 0, 1, 0,
                  color = 'g',
                  normalize=False,
                  arrow_length_ratio = .18,
                  linestyles = 'solid',
                  linewidths = 1)
        # 蓝色箭头，代表z轴
        axs[i, j].quiver(0, 0, 0, 0, 0, 1,
                  color = 'b',
                  normalize=False,
                  arrow_length_ratio = .18,
                  linestyles = 'solid',
                  linewidths = 1)
        axs[i, j].set_proj_type('ortho')
        axs[i, j].grid('off')
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].set_zticks([])
        axs[i, j].set_xlim(0,1)
        axs[i, j].set_ylim(0,1)
        axs[i, j].set_zlim(0,1)
        axs[i, j].view_init(elev=elev, azim=azim)
# fig.savefig('子图，RGB单位向量多视角.svg', format='svg')
plt.show()


#%% Bk_2_Ch17_03 可视化特征向量
import numpy as np
import matplotlib.pyplot as plt
### 产生数据
A = np.array([[1.25,  -0.75],
              [-0.75, 1.25]])

xx1, xx2 = np.meshgrid(np.linspace(-8, 8, 9), np.linspace(-8, 8, 9))
num_vecs = np.prod(xx1.shape);

thetas = np.linspace(0, 2*np.pi, num_vecs)

thetas = np.reshape(thetas, (-1, 9))
thetas = np.flipud(thetas);

uu = np.cos(thetas);
vv = np.sin(thetas);

# 矩阵乘法
V = np.array([uu.flatten(),vv.flatten()]).T;
W = V@A;
# 矩阵A线性映射

uu_new = np.reshape(W[:,0],(-1, 9));
vv_new = np.reshape(W[:,1],(-1, 9));

fig, ax = plt.subplots(figsize = (6,6))
# 绘制线性映射之前的向量
ax.quiver(xx1,xx2, # 向量始点位置坐标，网格化数据
          uu,vv,   # 两个方向的投影量
          angles='xy', scale_units='xy',
          scale=0.8, # 稍微放大
          width = 0.0025, # 宽度，默认0.005
          edgecolor='none', facecolor= 'b')

# 绘制线性映射之后的向量
ax.quiver(xx1,xx2,uu_new,vv_new,
          angles='xy', scale_units='xy',
          scale=0.8,
          width = 0.0025,
          edgecolor='none', facecolor= 'r')

plt.axis('scaled')
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xticks(np.linspace(-10,10,11));
ax.set_yticks(np.linspace(-10,10,11));

ax.set_xticklabels([])
ax.set_yticklabels([])
# fig.savefig('Figures/特征向量.svg', format='svg')

fig, ax = plt.subplots(figsize = (6,6))
import matplotlib
cm = matplotlib.cm.rainbow
norm = matplotlib.colors.Normalize()
ax.quiver(xx1*0,xx2*0,uu,vv,
          angles='xy', scale_units='xy',scale=1,
          width = 0.0025,
          edgecolor='none',
          facecolor=cm(norm(range(len(xx1.ravel())))))
ax.quiver(xx1*0,xx2*0,uu_new,vv_new,
          angles='xy', scale_units='xy',scale=1,
          width = 0.0025,
          edgecolor='none',
          facecolor=cm(norm(range(len(xx1.ravel())))))
plt.axis('scaled')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_xticks(np.linspace(-2,2,5));
ax.set_yticks(np.linspace(-2,2,5));

ax.set_xticklabels([])
ax.set_yticklabels([])
# fig.savefig('Figures/特征向量，单位圆.svg', format='svg')

#%%  Bk_2_Ch17_04 # 可视化几何变换
import sympy
import numpy as np
import matplotlib.pyplot as plt
# 创建网格坐标数据
xx1_, xx2_ = np.meshgrid(np.linspace(-2,2,18), np.linspace(-2,2,18))
# 构造复数
zz_ = xx1_ + xx2_ * 1j
# 计算辐角
zz_angle_ = np.angle(zz_)
# 全零矩阵
zeros = np.zeros_like(xx1_)
fig, ax = plt.subplots(figsize = (5,5))
plt.quiver (zeros, zeros, # 向量场起点
            xx1_, xx2_,   # 横纵轴分量
            zz_angle_,    # 颜色映射依据
            angles='xy', scale_units='xy', scale = 1,
            edgecolor='none', alpha=0.8, cmap = 'hsv')

ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.axhline(y = 0, c = 'k')
ax.axvline(x = 0, c = 'k')
ax.set_xticks(np.arange(-2,3))
ax.set_yticks(np.arange(-2,3))
# plt.grid()
ax.axis('off')
# fig.savefig('Figures/用辐角大小给箭头着色，原图.svg', format='svg')

#>>>>>>>>>>>>>>>>>>
A = np.array([[2,  0],
              [0, 1]])
V = np.array([xx1_.flatten(),xx2_.flatten()]).T;
W = V@A;
uu_new = np.reshape(W[:,0],xx1_.shape);
vv_new = np.reshape(W[:,1],xx1_.shape);
fig, ax = plt.subplots(figsize = (5,5))
plt.quiver (zeros, zeros, uu_new, vv_new, zz_angle_,
            angles='xy', scale_units='xy', scale = 1,
            edgecolor='none', alpha=0.8, cmap = 'hsv')
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.axhline(y = 0, c = 'k')
ax.axvline(x = 0, c = 'k')
# plt.grid()
ax.axis('off')
# fig.savefig('Figures/用辐角大小给箭头着色，缩放.svg', format='svg')

#>>>>>>>>>>>>>>>>>>
theta = np.pi/3
A = np.array([[np.cos(theta),  -np.sin(theta)],
              [np.sin(theta),   np.cos(theta)]])
V = np.array([xx1_.flatten(),xx2_.flatten()]).T;
W = V@A.T;

uu_new = np.reshape(W[:,0],xx1_.shape);
vv_new = np.reshape(W[:,1],xx1_.shape);

fig, ax = plt.subplots(figsize = (5,5))
plt.quiver (zeros, zeros, uu_new, vv_new, zz_angle_,
            angles='xy', scale_units='xy', scale = 1,
            edgecolor='none', alpha=0.8, cmap = 'hsv')
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.axhline(y = 0, c = 'k')
ax.axvline(x = 0, c = 'k')
# plt.grid()
ax.axis('off')
# fig.savefig('Figures/用辐角大小给箭头着色，旋转.svg', format='svg')

#>>>>>>>>>>>>>>>>>>
A = np.array([[1.25,  -0.75],
              [-0.75, 1.25]])
V = np.array([xx1_.flatten(),xx2_.flatten()]).T;
W = V@A;
uu_new = np.reshape(W[:,0],xx1_.shape);
vv_new = np.reshape(W[:,1],xx1_.shape);
fig, ax = plt.subplots(figsize = (5,5))
plt.quiver (zeros, zeros, uu_new, vv_new, zz_angle_,
            angles='xy', scale_units='xy', scale = 1,
            edgecolor='none', alpha=0.8, cmap = 'hsv')
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)
ax.axhline(y = 0, c = 'k')
ax.axvline(x = 0, c = 'k')
# plt.grid()
ax.axis('off')
# fig.savefig('Figures/用辐角大小给箭头着色，旋转 + 缩放.svg', format='svg')




#%% Bk_2_Ch17_05 平面等高线 + 梯度

import numpy as np
import matplotlib.pyplot as plt
import sympy
import numpy as np
from sympy.functions import exp

############# 定义符号函数
# 定义符号变量
x1,x2 = sympy.symbols('x1 x2')

# 定义符号二元函数
f_x = x1*exp(-(x1**2 + x2**2))

# 将符号函数转换为Python函数
f_x_fcn = sympy.lambdify([x1, x2], f_x)

# 计算梯度
grad_f = [sympy.diff(f_x, var) for var in (x1,x2)]

# 将符号梯度转化为Python函数
grad_fcn = sympy.lambdify([x1, x2],grad_f)

# 产生数据, 细腻颗粒度
x1_array = np.linspace(-2, 2, 401)
x2_array = np.linspace(-2, 2, 401)
xx1, xx2 = np.meshgrid(x1_array,x2_array)
ff_x  = f_x_fcn(xx1, xx2)
# 粗糙颗粒度
x1_array_ = np.linspace(-2, 2, 21)
x2_array_ = np.linspace(-2, 2, 21)
xx1_, xx2_ = np.meshgrid(x1_array_,x2_array_)
V = grad_fcn(xx1_, xx2_)
ff_x_ = f_x_fcn(xx1_, xx2_)

# 平面向量场
fig, ax = plt.subplots(figsize=(6,6))
# 用颗粒度高的数据绘制等高线
ax.contour(xx1, xx2, ff_x, 20, cmap = 'RdYlBu_r')

# 用颗粒度低的数据绘制向量场
ax.quiver(xx1_, xx2_, V[0], V[1],
          width = 0.0025,
          color = 'k')

ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
plt.xlim(-2,2)
plt.ylim(-2,2)

ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
plt.show()
# fig.savefig('Figures/平面向量场.svg', format='svg')


# import plotly.figure_factory as ff

# fig = ff.create_quiver(xx1_, xx2_,
#                        V[0], V[1],
#                        scale=0.38,
#                        arrow_scale=.28,
#                        line_width=1)

# fig.update_layout(autosize=False,
#                   width=500, height=500)
# fig.show()


#%% Bk_2_Ch17_06 三维向量场
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, simplify, expand, lambdify, diff

def fcn_n_grdnt(A, xxx1, xxx2, xxx3):

    x1,x2,x3 = symbols('x1 x2 x3')
    # 符号向量
    x = np.array([[x1,x2,x3]]).T
    # 二次型
    f_x = x.T@A@x
    f_x = f_x[0][0]
    print(simplify(expand(f_x)))

    # 计算梯度，符号
    grad_f = [diff(f_x,var) for var in (x1,x2,x3)]

    # 计算三元函数值 f(x1,x2,x3)
    f_x_fcn = lambdify([x1,x2,x3],f_x)
    ff_x = f_x_fcn(xxx1,xxx2,xxx3)

    # 梯度函数
    grad_fcn = lambdify([x1,x2,x3],grad_f)

    # 计算梯度
    V = grad_fcn(xxx1,xxx2,xxx3)

    # 修复梯度值
    if isinstance(V[0], int):
        V[0] = np.zeros_like(xxx1)

    if isinstance(V[1], int):
        V[1] = np.zeros_like(xxx1)

    if isinstance(V[2], int):
        V[2] = np.zeros_like(xxx1)

    return ff_x, V


# 创建数据
x1_array = np.linspace(-5,5,11)
x2_array = np.linspace(-5,5,11)
x3_array = np.linspace(-5,5,11)

xxx1, xxx2, xxx3 = np.meshgrid(x1_array, x2_array, x3_array)

A = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

# 计算矩阵秩
print(np.linalg.matrix_rank(A))

# 计算三元函数值和梯度
f3_array, V = fcn_n_grdnt(A,xxx1,xxx2,xxx3)

# 可视化
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')
ax.quiver(xxx1.ravel(), xxx2.ravel(), xxx3.ravel(),
          V[0].ravel(), V[1].ravel(), V[2].ravel(),
          colors = 'b',
          edgecolors='face',
          arrow_length_ratio = 0,
          length=0.8, normalize=True)

ax.set_proj_type('ortho')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
# plt.savefig('test.svg')
plt.show()



#%% Bk_2_Ch17_07 水流图
import matplotlib.pyplot as plt
import numpy as np
# import os

x = np.arange(-3, 3, 0.3)
y = np.arange(-3, 3, 0.3)
xx,yy = np.meshgrid(x,y)

Fx  = np.cos(xx + 2*yy)
Fy  = np.sin(xx - 2*yy)

color_array = np.sqrt(Fx**2 + Fy**2)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

# plotting the vectors
# ax.quiver(x,y,Fx,Fy)
ax.streamplot(xx, yy, Fx, Fy,
              density = 2,
              arrowstyle = 'fancy')
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
# fig.savefig('Figures/水流图.svg', format='svg')
plt.show()


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

# plotting the vectors
# ax.quiver(x,y,Fx,Fy)
ax.streamplot(xx, yy,
              Fx, Fy,
              color = color_array,
              cmap = 'RdYlBu',
              density = 2,
              arrowstyle = 'fancy')
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
# fig.savefig('Figures/水流图，渲染.svg', format='svg')
plt.show()



fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

# 比较水流图和向量场
ax.quiver(x, y, Fx,Fy)
ax.streamplot(x, y, Fx, Fy,
              density = 2,
              arrowstyle = 'fancy')
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
# fig.savefig('Figures/水流图 + 箭头图.svg', format='svg')
plt.show()
























