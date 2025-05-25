#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 18:04:51 2024

@author: jack
"""

# 导入包
#%% # 平面两点连线
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

iris_sns = sns.load_dataset("iris")
data = iris_sns[["sepal_length", "sepal_width"]].to_numpy()

### 和原点连线
Origin = data*0

fig, ax = plt.subplots(figsize=(5,3))

plt.plot(Origin[0,0], Origin[0,1], color = 'r', marker = 'x', markersize = 20)

plt.plot(([i for (i,j) in Origin], [i for (i,j) in data]),
         ([j for (i,j) in Origin], [j for (i,j) in data]),
         c = [0.6,0.6,0.6], alpha = 0.5)
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width")
ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_xticks(np.arange(0, 8 + 1, step=1))
ax.set_yticks(np.arange(0, 6 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 0, upper = 8)
ax.set_ybound(lower = 0, upper = 6)

### 质心
Centroid = data*0 + data.mean(axis = 0)
fig, ax = plt.subplots(figsize=(5,3))
plt.plot(Centroid[0,0], Centroid[0,1], color = 'r', marker = 'x', markersize = 20)
plt.plot(([i for (i,j) in Centroid], [i for (i,j) in data]),
         ([j for (i,j) in Centroid], [j for (i,j) in data]),
         c = [0.6, 0.6, 0.6], alpha = 0.5)
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width")

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_xticks(np.arange(0, 8 + 1, step=1))
ax.set_yticks(np.arange(0, 6 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 0, upper = 8)
ax.set_ybound(lower = 0, upper = 6)

### 簇质心
fig, ax = plt.subplots(figsize=(5,3))

ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width", hue = 'species')

for label in iris_sns.species.unique().tolist():
    data_label = iris_sns.loc[iris_sns.species == label, ['sepal_length', 'sepal_width']]
    data_label = data_label.to_numpy()

    centroid_cluster = data_label*0 + data_label.mean(axis = 0)
    plt.plot(centroid_cluster[0,0], centroid_cluster[0,1],
             color = 'r', marker = 'x', markersize = 20)

    plt.plot(([i for (i,j) in centroid_cluster], [i for (i,j) in data_label]),
             ([j for (i,j) in centroid_cluster], [j for (i,j) in data_label]),
             c=[0.6,0.6,0.6],
             alpha = 0.5)

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_xticks(np.arange(0, 8 + 1, step=1))
ax.set_yticks(np.arange(0, 6 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 0, upper = 8)
ax.set_ybound(lower = 0, upper = 6)

### 任意两点连线
import itertools
fig, ax = plt.subplots(figsize=(5,3))

plt.plot(*zip(*itertools.chain.from_iterable(itertools.combinations(data, 2))),
    color = [0.7,0.7,0.7],
    alpha = 0.5)

ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width")

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_xticks(np.arange(0, 8 + 1, step=1))
ax.set_yticks(np.arange(0, 6 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 0, upper = 8)
ax.set_ybound(lower = 0, upper = 6)

### 水平方向
x_axis = np.copy(data)
x_axis[:,1] = 0

fig, ax = plt.subplots(figsize=(5,3))
plt.plot(x_axis[:,0], x_axis[:,1], marker = 'x', c='r', linestyle = 'None')
plt.plot(([i for (i,j) in x_axis], [i for (i,j) in data]), ([j for (i,j) in x_axis], [j for (i,j) in data]), c=[0.6,0.6,0.6], alpha = 0.5)
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width")

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_xticks(np.arange(0, 8 + 1, step=1))
ax.set_yticks(np.arange(0, 6 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 0, upper = 8)
ax.set_ybound(lower = 0, upper = 6)

### 竖直方向
y_axis = np.copy(data)
y_axis[:,0] = 0

fig, ax = plt.subplots(figsize=(5,3))
plt.plot(y_axis[:,0],y_axis[:,1], marker = 'x', c='r', linestyle = 'None')
plt.plot(([i for (i,j) in y_axis], [i for (i,j) in data]), ([j for (i,j) in y_axis], [j for (i,j) in data]), c=[0.6,0.6,0.6], alpha = 0.5)
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width")

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_xticks(np.arange(0, 8 + 1, step=1))
ax.set_yticks(np.arange(0, 6 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 0, upper = 8)
ax.set_ybound(lower = 0, upper = 6)

### 沿y朝任意直线投影
proj_along_x = np.copy(data)
# 函数 y = 0.5x + 1
proj_along_x[:,1] = 0.3*proj_along_x[:,0] + 1
x_array = np.linspace(0,8,10)
y_array = 0.3*x_array + 1

fig, ax = plt.subplots(figsize=(5,3))

plt.plot(proj_along_x[:,0],proj_along_x[:,1],
         marker = 'x',
         c='r', linestyle = 'None')
plt.plot(x_array, y_array, 'k')

plt.plot(([i for (i,j) in proj_along_x], [i for (i,j) in data]),
         ([j for (i,j) in proj_along_x], [j for (i,j) in data]),
         c=[0.6,0.6,0.6], alpha = 0.5)

ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width")

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_xticks(np.arange(0, 8 + 1, step=1))
ax.set_yticks(np.arange(0, 6 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 0, upper = 8)
ax.set_ybound(lower = 0, upper = 6)


### 沿 x 朝任意直线投影
proj_along_y = np.copy(data)
# 函数 y = 2x + 1
# 反函数 x = (y - 1)/2

proj_along_y[:,0] = (proj_along_y[:,1] - 1)/2
x_array = np.linspace(0,8,10)
y_array = 2*x_array + 1

fig, ax = plt.subplots(figsize=(5,3))

plt.plot(proj_along_y[:,0],proj_along_y[:,1],
         marker = 'x',
         c='r', linestyle = 'None')
plt.plot(x_array, y_array, 'k')

plt.plot(([i for (i,j) in proj_along_y], [i for (i,j) in data]),
         ([j for (i,j) in proj_along_y], [j for (i,j) in data]),
         c=[0.6,0.6,0.6], alpha = 0.5)

ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width")

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_xticks(np.arange(0, 8 + 1, step=1))
ax.set_yticks(np.arange(0, 6 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 0, upper = 8)
ax.set_ybound(lower = 0, upper = 6)

### 朝过原点直线正交投影
theta = 15
print('====================')
print('theta = ' + str(theta) + ' degrees')
theta = theta*np.pi/180

v = np.array([[np.cos(theta)],
              [np.sin(theta)]])
T = v@v.T
data_projected = data@T
y_array = np.tan(theta)*x_array

fig, ax = plt.subplots(figsize=(5,3))

plt.plot(data_projected[:,0],data_projected[:,1],
         marker = 'x',
         c='r', linestyle = 'None')

plt.plot(x_array, y_array, 'k')

plt.plot(([i for (i,j) in data_projected], [i for (i,j) in data]),
         ([j for (i,j) in data_projected], [j for (i,j) in data]),
         c=[0.6,0.6,0.6], alpha = 0.5)

ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width")

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_xticks(np.arange(0, 8 + 1, step=1))
ax.set_yticks(np.arange(0, 6 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 0, upper = 8)
ax.set_ybound(lower = 0, upper = 6)


#%% # 不同方式展示二元欧氏距离
import numpy as np
import matplotlib.pyplot as plt
import os
from sympy import init_printing, symbols, diff, lambdify, expand, simplify, sqrt
init_printing("mathjax")

## 二元函数
x1_array = np.linspace(-2, 2, 20)
x2_array = np.linspace(-2, 2, 20)

xx1, xx2 = np.meshgrid(x1_array, x2_array)

def fcn_n_grdnt(A, xx1, xx2):
    x1,x2 = symbols('x1 x2')
    x = np.array([[x1,x2]]).T
    f_x = x.T@A@x
    f_x = f_x[0][0]
    f_x = sqrt(f_x)
    # print(simplify(expand(f_x)))

    #take the gradient symbolically
    grad_f = [diff(f_x,var) for var in (x1,x2)]

    f_x_fcn = lambdify([x1,x2],f_x)

    ff_x = f_x_fcn(xx1,xx2)

    #turn into a bivariate lambda for numpy
    grad_fcn = lambdify([x1,x2],grad_f)

    xx1_ = xx1[::20,::20]
    xx2_ = xx2[::20,::20]

    V = grad_fcn(xx1_,xx2_)

    if isinstance(V[1], int):
        V[1] = np.zeros_like(V[0])

    elif isinstance(V[0], int):
        V[0] = np.zeros_like(V[1])

    return ff_x, V

A = np.array([[1, 0],
              [0, 1]])
f2_array, gradient_array = fcn_n_grdnt(A, xx1, xx2)
### 网格面
fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.plot_wireframe(xx1, xx2, f2_array, rstride = 10, cstride = 10, color = [0.8,0.8,0.8], linewidth = 0.25)
ax.contour(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')

# ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
# ax.set_zlabel('$f(x_1,x_2)$')
ax.set_proj_type('ortho')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(azim=-120, elev=30)
ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max());
ax.set_ylim(xx2.min(), xx2.max())
plt.tight_layout()

ax = fig.add_subplot(1, 3, 2)
ax.contour(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')
# ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max());
ax.set_ylim(xx2.min(), xx2.max())
plt.tight_layout()

ax = fig.add_subplot(1, 3, 3)
ax.contourf(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')
xx1_ = xx1[::20,::20]
xx2_ = xx2[::20,::20]
plt.quiver(xx1_, xx2_, gradient_array[0], gradient_array[1], angles='xy', scale_units='xy', edgecolor='none', alpha=0.8)
# ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max());
ax.set_ylim(xx2.min(), xx2.max())
plt.tight_layout()

#%% # 三维两点连线
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
import os

### 导入数据
iris_sns = sns.load_dataset("iris")
x1=iris_sns['sepal_length']
x2=iris_sns['sepal_width']
x3=iris_sns['petal_length']
### 散点和投影点连线
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x1, x2, x3)
ax.scatter(x1, x2, zs = 1, zdir = 'z', marker = 'x')
AA = np.stack([x1.to_numpy(),x2.to_numpy(),x3.to_numpy()]).T
BB = np.stack([x1.to_numpy(),x2.to_numpy(),x3.to_numpy()*0 + 1]).T
for A_idx, B_idx in zip(AA,BB):
    ax.plot([A_idx[0],B_idx[0]],
            [A_idx[1],B_idx[1]],
            [A_idx[2],B_idx[2]],
            color = [0.8, 0.8, 0.8])

# ax.plot(([i for (i,j,k) in AA], [i for (i,j,k) in BB]), ([j for (i,j,k) in AA], [j for (i,j,k) in BB]),  ([k for (i,j,k) in AA], [k for (i,j,k) in BB]), c=[0.6,0.6,0.6],  )

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_zlabel('Petal length, $x_3$ (cm)')
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)
ax.set_box_aspect([1,1,1])
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.grid([])

### 散点和质心连线
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x1, x2, x3)
# 质心
MU = [x1.mean(),x2.mean(),x3.mean()]
ax.scatter(MU[0], MU[1], MU[2], marker = 'x', s = 50)
for A_idx in AA:
    ax.plot([A_idx[0], x1.mean()],
            [A_idx[1], x2.mean()],
            [A_idx[2], x3.mean()],
            color = [0.8, 0.8, 0.8])
ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Sepal width, $x_2$ (cm)')
ax.set_zlabel('Petal length, $x_3$ (cm)')
ax.set_proj_type('ortho')
ax.view_init(azim=-135, elev=30)
ax.set_box_aspect([1,1,1])
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlim(4, 8)
ax.set_ylim(2, 5)
ax.set_zlim(1, 7)
ax.grid([])

#%% # 用切豆腐的方法展示三元欧氏距离
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from sympy import init_printing
from sympy import symbols, simplify, expand, sqrt, lambdify

## 三元函数
x1_array = np.linspace(-2,2,101)
x2_array = np.linspace(-2,2,101)
x3_array = np.linspace(-2,2,101)
xxx1, xxx2, xxx3 = np.meshgrid(x1_array, x2_array, x3_array)
# (101, 101, 101)
# 定义三元二次型
def Q3(A,xxx1,xxx2,xxx3):
    x1,x2,x3 = symbols('x1 x2 x3')
    x = np.array([[x1,x2,x3]]).T

    f_x = x.T@A@x
    f_x = sqrt(f_x[0][0])
    # print(simplify(expand(f_x[0][0])))
    f_x_fcn = lambdify([x1,x2,x3],f_x)
    qqq = f_x_fcn(xxx1,xxx2,xxx3)
    return qqq

A = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
print(np.linalg.matrix_rank(A))
f3_array = np.sqrt(xxx1**2 + xxx2**2 + xxx3**2)
f3_array.shape  # (101, 101, 101)

### Plotly Volume
### 外立面
# 设定统一等高线分层
levels = np.linspace(0,4,21)
# 定义等高线高度
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
# 绘制三维等高线，填充
ax.contourf(xxx1[:, :, -1],
            xxx2[:, :, -1],
            f3_array[:, :, -1],
            levels = levels,
            zdir='z', offset=xxx3.max(),
            cmap = 'RdYlBu_r') # RdYlBu_r
ax.contour(xxx1[:, :, -1],
            xxx2[:, :, -1],
            f3_array[:, :, -1],
            levels = levels,
            zdir='z', offset=xxx3.max(),
            linewidths = 0.25,
            colors = '1')
ax.contourf(xxx1[0, :, :],
            f3_array[0, :, :],
            xxx3[0, :, :],
            levels = levels,
            zdir='y',
            cmap = 'RdYlBu_r',
            offset=xxx2.min())
ax.contour(xxx1[0, :, :],
            f3_array[0, :, :],
            xxx3[0, :, :],
            levels = levels,
            zdir='y',
            colors = '1',
            linewidths = 0.25,
            offset=xxx2.min())
CS = ax.contourf(f3_array[:, 0, :],
            xxx2[:, 0, :],
            xxx3[:, 0, :],
            levels = levels,
            cmap = 'RdYlBu_r',
            zdir='x',
            offset=xxx1.min())
ax.contour(f3_array[:, 0, :],
            xxx2[:, 0, :],
            xxx3[:, 0, :],
            levels = levels,
            zdir='x',
            colors = '1',
            linewidths = 0.25,
            offset=xxx1.min())
fig.colorbar(CS, ticks = np.linspace(np.floor(f3_array.min()),np.ceil(f3_array.max()), int(np.ceil(f3_array.max()) - np.floor(f3_array.min())) + 1))
# Set limits of the plot from coord limits
xmin, xmax = xxx1.min(), xxx1.max()
ymin, ymax = xxx2.min(), xxx2.max()
zmin, zmax = xxx3.min(), xxx3.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
# 绘制框线
edges_kw = dict(color='0.6', linewidth=1, zorder=1e5)
# zorder 控制呈现 artist 的先后顺序
# zorder 越小，artist 置于越底层
# zorder 赋值很大的数，这样确保 zorder 置于最顶层
ax.plot([xmin, xmax], [ymin, ymin], [zmax, zmax], **edges_kw)
ax.plot([xmin, xmin], [ymin, ymax], [zmax, zmax], **edges_kw)
ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **edges_kw)
# ax.set_xticks([-1,0,1])
# ax.set_yticks([-1,0,1])
# ax.set_zticks([-1,0,1])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.view_init(azim=-120, elev=30)
ax.set_proj_type('ortho')
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')
# ax.set_zlabel('$x_3$')
ax.set_box_aspect((1, 1, 1))
plt.show()
plt.close('all')


### 将等高线展开，沿x3
fig = plt.figure(figsize=(6, 36))
for fig_idx, idx in enumerate(np.arange(0, len(x3_array), 25)):
    ax = fig.add_subplot(len(np.arange(0, len(x3_array), 25)), 1, fig_idx + 1, projection = '3d')
    x3_idx = x3_array[idx]
    ax.contourf(xxx1[:, :, idx],
                xxx2[:, :, idx],
                f3_array[:, :, idx],
                levels = levels,
                zdir='z',
                offset=x3_idx,
                cmap = 'RdYlBu_r')
    ax.contour(xxx1[:, :, idx],
                xxx2[:, :, idx],
                f3_array[:, :, idx],
                levels = levels,
                zdir = 'z',
                offset = x3_idx,
                linewidths = 0.25,
                colors = '1')
    ax.plot([xmin, xmin], [ymin, ymax], [x3_idx, x3_idx], **edges_kw)
    ax.plot([xmax, xmin], [ymin, ymin], [x3_idx, x3_idx], **edges_kw)
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
    # 绘制框线
    edges_kw = dict(color='0.5', linewidth=1, zorder=1e3)
    ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **edges_kw)
    ax.plot([xmin, xmin], [ymin, ymax], [zmax, zmax], **edges_kw)
    ax.plot([xmax, xmin], [ymin, ymin], [zmax, zmax], **edges_kw)

    ax.view_init(azim=-125, elev=30)
    ax.set_box_aspect(None)
    ax.set_proj_type('ortho')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$x_3$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)

### 将等高线展开，沿x2
fig = plt.figure(figsize=(6, 36))
for fig_idx,idx in enumerate(np.arange(0,len(x2_array),25)):
    ax = fig.add_subplot(len(np.arange(0,len(x2_array),25)), 1,
                         fig_idx + 1, projection='3d')
    x2_idx = x2_array[idx]
    ax.contourf(xxx1[idx, :, :],
                f3_array[idx, :, :],
                xxx3[idx, :, :],
                levels = levels,
                zdir='y',
                offset=x2_idx,
                cmap = 'RdYlBu_r')
    ax.contour(xxx1[idx, :, :],
                f3_array[idx, :, :],
                xxx3[idx, :, :],
                levels = levels,
                zdir='y',
                offset=x2_idx,
               linewidths = 0.25,
                colors = '1')
    ax.plot([xmin, xmin], [ymin, ymax], [x3_idx, x3_idx], **edges_kw)
    ax.plot([xmax, xmin], [ymin, ymin], [x3_idx, x3_idx], **edges_kw)
    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    # Plot edges
    edges_kw = dict(color='0.5', linewidth=1, zorder=1e3)
    ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **edges_kw)
    ax.plot([xmin, xmin], [ymin, ymax], [zmax, zmax], **edges_kw)
    ax.plot([xmax, xmin], [ymin, ymin], [zmax, zmax], **edges_kw)

    # Set zoom and angle view
    ax.view_init(azim=-125, elev=30)
    ax.set_box_aspect(None)
    ax.set_proj_type('ortho')
    # ax.set_xlabel('$x_1$')
    # ax.set_ylabel('$x_2$')
    # ax.set_zlabel('$x_3$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)

### 将等高线展开，沿x1
fig = plt.figure(figsize=(6, 36))
for fig_idx,idx in enumerate(np.arange(0,len(x1_array),25)):
    ax = fig.add_subplot(len(np.arange(0,len(x1_array),25)), 1, fig_idx + 1, projection='3d')
    x1_idx = x1_array[idx]
    ax.contourf(f3_array[:, idx, :],
                xxx2[:, idx, :],
                xxx3[:,idx,  :],
                levels = levels,
                zdir='x',
                offset=x1_idx,
                cmap = 'RdYlBu_r')
    ax.contour(f3_array[:, idx, :],
                xxx2[:, idx, :],
                xxx3[:,idx,  :],
                levels = levels,
                zdir='x',
                offset=x1_idx,
               linewidths = 0.25,
                colors = '1')
    ax.plot([xmin, xmin], [ymin, ymax], [x3_idx, x3_idx], **edges_kw)
    ax.plot([xmax, xmin], [ymin, ymin], [x3_idx, x3_idx], **edges_kw)

    ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

    # 绘制框线
    ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **edges_kw)
    ax.plot([xmin, xmin], [ymin, ymax], [zmax, zmax], **edges_kw)
    ax.plot([xmax, xmin], [ymin, ymin], [zmax, zmax], **edges_kw)

    ax.view_init(azim=-125, elev=30)
    ax.set_box_aspect(None)
    ax.set_proj_type('ortho')
    # ax.set_xlabel('$x_1$')
    # ax.set_ylabel('$x_2$')
    # ax.set_zlabel('$x_3$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)

#%% # 根据欧氏距离远近渲染两点连线
import matplotlib.pyplot as plt
import itertools
import numpy as np
import matplotlib as mpl

num = 50
data = np.random.uniform(0, 1,(num,2))
data = np.row_stack((data,
                     np.array([[0, 0],
                               [1, 0],
                               [0, 1],
                               [1, 1]])))
cmap = mpl.colormaps.get_cmap('RdYlBu')
fig, ax = plt.subplots()
for i, d in enumerate(itertools.combinations(data, 2)):
    distance_idx = np.sqrt((d[0][0] - d[1][0])**2 + (d[0][1] - d[1][1])**2)
    plt.plot([d[0][0],d[1][0]],
             [d[0][1],d[1][1]],
             color = cmap(distance_idx/np.sqrt(2)),
             lw = 0.2)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
ax.set_aspect('equal')
# fig.savefig('Figures/根据距离远近匹配颜色，单位正方形.svg', format='svg')
plt.show()


theta_array = np.random.uniform(0, 1,40) * 2 * np.pi
x_array = np.cos(theta_array)
y_array = np.sin(theta_array)
data = np.column_stack((x_array,y_array))
cmap = mpl.colormaps.get_cmap('RdYlBu')
fig, ax = plt.subplots()
for i, d in enumerate(itertools.combinations(data, 2)):
    distance_idx = np.sqrt((d[0][0] - d[1][0])**2 + (d[0][1] - d[1][1])**2)
    plt.plot([d[0][0],d[1][0]],
             [d[0][1],d[1][1]],
             color = cmap(distance_idx/np.sqrt(2)),
             lw = 0.2)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
ax.set_aspect('equal')
# fig.savefig('Figures/根据距离远近匹配颜色，圆盘.svg', format='svg')
plt.show()

#%% # 平面Lp范数等高线
## 创建数据
p_values = [1, 1.5, 2, 4, 8, np.inf]
# 给定不同p值
x1 = np.linspace(-2.5, 2.5, num=101);
x2 = x1;
xx1, xx2 = np.meshgrid(x1,x2)
## 自定义Lp范数函数
def Lp_norm(p):
    # 计算范数
    if np.isinf(p):
        zz = np.maximum(np.abs(xx1),np.abs(xx2))
    else:
        zz = ((np.abs((xx1))**p) + (np.abs((xx2))**p))**(1./p)
    return zz
## 可视化
fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(6, 9))
for p, ax in zip(p_values, axes.flat):
    # 计算范数
    zz = Lp_norm(p)
    # 绘制等高线
    ax.contourf(xx1, xx2, zz, 20, cmap='RdYlBu_r')
    # 绘制Lp norm = 1的等高线
    ax.contour (xx1, xx2, zz, [1], colors='k', linewidths = 2)

    # 装饰
    ax.axhline(y=0, color='k', linewidth = 0.25)
    ax.axvline(x=0, color='k', linewidth = 0.25)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel('$x_1$')
    # ax.set_ylabel('$x_2$')
    ax.set_title('p = ' + str(p))
    ax.set_aspect('equal', adjustable='box')


#%% # 平面Lp范数等高线
## 创建数据
p_values = [1, 1.5, 2, 4, 8, np.inf]
# 给定不同p值
x1 = np.linspace(-2.5, 2.5, num=101);
x2 = x1;
xx1, xx2 = np.meshgrid(x1,x2)
## 自定义Lp范数函数
def Lp_norm(p):
    # 计算范数
    if np.isinf(p):
        zz = np.maximum(np.abs(xx1),np.abs(xx2))
    else:
        zz = ((np.abs((xx1))**p) + (np.abs((xx2))**p))**(1./p)
    return zz
## 可视化
# fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(12, 18), projection = '3d')
fig = plt.figure(figsize=(12, 18), constrained_layout = True)
for i, p in enumerate(p_values):
    ax = fig.add_subplot(3, 2, i+1, projection='3d')
    # 计算范数
    zz = Lp_norm(p)

    ## 4 plot_wireframe() 绘制网格曲面 + 三维等高线

    ax.plot_wireframe(xx1, xx2, zz, color = [0.5,0.5,0.5], linewidth = 0.25)

    # 三维等高线
    # colorbar = ax.contour(xx,yy, ff,20,  cmap = 'RdYlBu_r')
    # 三维等高线
    colorbar = ax.contour(xx1, xx2, zz, 20,  cmap = 'hsv')
    # fig.colorbar(colorbar, ax = ax, shrink=0.5, aspect=20)

    # 二维等高线
    ax.contour(xx1, xx2, zz, zdir = 'z', offset= zz.min(), levels = 20, linewidths = 2, cmap = "hsv")  # 生成z方向投影，投到x-y平面

    fig.colorbar(colorbar, ax=ax, shrink=0.5, aspect=20)
    ax.set_proj_type('ortho')

    # 3D坐标区的背景设置为白色
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.set_xlabel(r'$\it{x_1}$')
    ax.set_ylabel(r'$\it{x_2}$')
    ax.set_zlabel(r'$\it{f}$($\it{x_1}$,$\it{x_2}$)')
    ax.set_title(f"p = {p}")
    ax.set_xlim(x1.min(), x1.max())
    ax.set_ylim(x2.min(), x2.max())

    ax.view_init(azim=-135, elev=30)
    ax.grid(False)
plt.show()


#%% # 三维空间Lp范数
# 导入包
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

## 自定义函数展示隐函数
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
    A = np.linspace(xmin, xmax, 100)
    # 产生网格数据
    A1,A2 = np.meshgrid(A,A)
    # 等高线的分割位置
    B = np.linspace(xmin, xmax, 20)
    # 绘制 XY 平面等高线
    if X_plot == True:
        for z in B:
            X,Y = A1,A2
            Z = fn(X,Y,z)
            cset = ax.contour(X, Y, Z+z, [z],
                              zdir='z',
                              linewidths = 0.25,
                              colors = '#0066FF',
                              linestyles = 'solid')
    # 绘制 XZ 平面等高线
    if Y_plot == True:
        for y in B:
            X,Z = A1,A2
            Y = fn(X,y,Z)
            cset = ax.contour(X, Y+y, Z, [y],
                              zdir='y',
                              linewidths = 0.25,
                              colors = '#88DD66',
                              linestyles = 'solid')
    # 绘制 YZ 平面等高线
    if Z_plot == True:
        for x in B:
            Y,Z = A1,A2
            X = fn(x,Y,Z)
            cset = ax.contour(X+x, Y, Z, [x],
                              zdir='x',
                              linewidths = 0.25,
                              colors = '#FF6600',
                              linestyles = 'solid')
    ax.set_zlim(zmin * k,zmax * k)
    ax.set_xlim(xmin * k,xmax * k)
    ax.set_ylim(ymin * k,ymax * k)
    ax.set_box_aspect([1,1,1])
    ax.view_init(azim=-120, elev=30)
    ax.axis('off')

def visualize_four_ways(fn, title, bbox=(-2.5,2.5)):
    fig = plt.figure(figsize=(12, 4), constrained_layout = True)

    ax = fig.add_subplot(1, 4, 1, projection='3d')
    plot_implicit(fn, True, False, False, ax, bbox)

    ax = fig.add_subplot(1, 4, 2, projection='3d')
    plot_implicit(fn, False, True, False, ax, bbox)

    ax = fig.add_subplot(1, 4, 3, projection='3d')
    plot_implicit(fn, False, False, True, ax, bbox)

    ax = fig.add_subplot(1, 4, 4, projection='3d')
    plot_implicit(fn, True, True, True, ax, bbox)

## 可视化
def vector_norm(x,y,z):
    p = 4
    return (np.abs(x)**p + np.abs(y)**p + np.abs(z)**p)**(1/p) - 1

visualize_four_ways(vector_norm, 'norm_1000', bbox = (-1,1))


#%% # 基于鸢尾花数据个各种距离度量等高线
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from numpy import linalg as LA

## 导入数据
iris_sns = sns.load_dataset("iris")
iris_sns = iris_sns.drop('species', axis = 1)
SIGMA = iris_sns.cov()
CORR = iris_sns.corr()
SIGMA = np.array(SIGMA)
CORR = np.array(CORR)

# plt.close('all')
SIGMA_13 = SIGMA[[0,2], :][:, [0,2]]
CORR_13  = CORR[[0,2], :][:, [0,2]]
sigma_x = iris_sns['sepal_length'].std()
sigma_y = iris_sns['petal_length'].std()
cov_xy = SIGMA_13[0,1]

mu_x = iris_sns['sepal_length'].mean()
mu_y = iris_sns['petal_length'].mean()

x = np.linspace(3, 9, num = 201)
y = np.linspace(1, 7, num = 201)
xx,yy = np.meshgrid(x,y)

## 自定函数产生旋转网格
def generate_grid(V, mu_x, mu_y):
    # grid rotation
    x1_grid = np.arange(-10, 10 + 1, step = 1);
    x2_grid = np.arange(-10, 10 + 1, step = 1);

    XX1_grid, XX2_grid = np.meshgrid(x1_grid, x2_grid);
    X_grid = np.column_stack((XX1_grid.ravel(), XX2_grid.ravel()))
    Z_grid = X_grid@V.T
    ZZ1_grid = Z_grid[:,0].reshape((len(x1_grid), len(x2_grid)))
    ZZ2_grid = Z_grid[:,1].reshape((len(x1_grid), len(x2_grid)))
    # translate centroid
    ZZ1_grid = ZZ1_grid + mu_x
    ZZ2_grid = ZZ2_grid + mu_y

    return ZZ1_grid, ZZ2_grid
##>>>>>>>>>>> 欧氏距离
I = np.array([[1, 0],
              [0, 1]])
ZZ1_grid, ZZ2_grid = generate_grid(I, mu_x, mu_y)
x_array = np.array(iris_sns["sepal_length"])
y_array = np.array(iris_sns["petal_length"])

fig, ax = plt.subplots(figsize = (6,6))
plt.plot([x_array, mu_x+x_array*0],
         [y_array, mu_y+y_array*0],
         color = [0.7, 0.7, 0.7])
ax = sns.scatterplot(data = iris_sns, x = "sepal_length", y = "petal_length")
sns.rugplot(data = iris_sns, x = "sepal_length", y = "petal_length", ax = ax)

plt.axvline(x = mu_x, linestyle = '--', color = 'r')
plt.axhline(y = mu_y, linestyle = '--', color = 'r')
plt.plot(mu_x, mu_y, color = 'k', marker = 'x', markersize = 15)

ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Petal length, $x_3$ (cm)')
ax.set_xticks([])
ax.set_yticks([])
plt.plot(ZZ1_grid, ZZ2_grid,color = [0.7,0.7,0.7])
plt.plot(ZZ1_grid.T, ZZ2_grid.T,color = [0.7,0.7,0.7])
ax.axis('scaled')
ax.set_xbound(3,9)
ax.set_ybound(1,7)
# Euclidean distance
dd = np.sqrt((xx - mu_x)**2 + (yy - mu_y)**2)
ax.contour(xx, yy, dd, levels = [1, 2, 3], colors = 'r')

##>>>>>>>>>>> L1范数
I = np.array([[1, 0],
              [0, 1]])
ZZ1_grid, ZZ2_grid = generate_grid(I, mu_x, mu_y)
x_array = np.array(iris_sns["sepal_length"])
y_array = np.array(iris_sns["petal_length"])
fig, ax = plt.subplots(figsize = (6,6))
plt.plot([x_array, mu_x+x_array*0],
         [y_array, mu_y+y_array*0],
         color = [0.7,0.7,0.7])
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="petal_length")
sns.rugplot(data=iris_sns, x="sepal_length", y="petal_length", ax = ax)
plt.axvline(x=mu_x, linestyle = '--', color = 'r')
plt.axhline(y=mu_y, linestyle = '--', color = 'r')

plt.plot(mu_x,mu_y, color = 'k', marker = 'x', markersize = 15)

# ax.set_xlabel('Sepal length, $x_1$ (cm)')
# ax.set_ylabel('Petal length, $x_3$ (cm)')

ax.set_xticks([])
ax.set_yticks([])
plt.plot(ZZ1_grid, ZZ2_grid,color = [0.7,0.7,0.7])
plt.plot(ZZ1_grid.T, ZZ2_grid.T,color = [0.7,0.7,0.7])

ax.axis('scaled')
ax.set_xbound(3,9)
ax.set_ybound(1,7)

## Euclidean distance
dd = np.abs(xx - mu_x) + np.abs((yy - mu_y))
ax.contour(xx, yy, dd, levels = [1, 2, 3], colors = 'r')

##>>>>>>>>>>> L1.5范数
I = np.array([[1, 0],
              [0, 1]])
ZZ1_grid, ZZ2_grid = generate_grid(I, mu_x, mu_y)
x_array = np.array(iris_sns["sepal_length"])
y_array = np.array(iris_sns["petal_length"])

fig, ax = plt.subplots(figsize = (6,6))
plt.plot([x_array, mu_x+x_array*0],
         [y_array, mu_y+y_array*0],
         color = [0.7,0.7,0.7])

ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="petal_length")
sns.rugplot(data=iris_sns, x="sepal_length", y="petal_length", ax = ax)
plt.axvline(x=mu_x, linestyle = '--', color = 'r')
plt.axhline(y=mu_y, linestyle = '--', color = 'r')

plt.plot(mu_x,mu_y, color = 'k', marker = 'x', markersize = 15)

# ax.set_xlabel('Sepal length, $x_1$ (cm)')
# ax.set_ylabel('Petal length, $x_3$ (cm)')
ax.set_xticks([])
ax.set_yticks([])

plt.plot(ZZ1_grid,ZZ2_grid,color = [0.7,0.7,0.7])
plt.plot(ZZ1_grid.T,ZZ2_grid.T,color = [0.7,0.7,0.7])

ax.axis('scaled')
ax.set_xbound(3,9)
ax.set_ybound(1,7)

# Euclidean distance
p = 1.5
dd = ((np.abs((xx - mu_x))**p) + (np.abs((yy - mu_y))**p))**(1./p)
ax.contour(xx,yy,dd,levels = [1, 2, 3], colors = 'r')

##>>>>>>>>>>> L4范数
I = np.array([[1, 0],
              [0, 1]])
ZZ1_grid, ZZ2_grid = generate_grid(I, mu_x, mu_y)
x_array = np.array(iris_sns["sepal_length"])
y_array = np.array(iris_sns["petal_length"])

fig, ax = plt.subplots(figsize = (6,6))
plt.plot([x_array, mu_x+x_array*0],
         [y_array, mu_y+y_array*0],
         color = [0.7,0.7,0.7])
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="petal_length")
sns.rugplot(data=iris_sns, x="sepal_length", y="petal_length", ax = ax)

plt.axvline(x=mu_x, linestyle = '--', color = 'r')
plt.axhline(y=mu_y, linestyle = '--', color = 'r')
plt.plot(mu_x,mu_y, color = 'k', marker = 'x', markersize = 15)

# ax.set_xlabel('Sepal length, $x_1$ (cm)')
# ax.set_ylabel('Petal length, $x_3$ (cm)')
ax.set_xticks([])
ax.set_yticks([])
plt.plot(ZZ1_grid,ZZ2_grid,color = [0.7,0.7,0.7])
plt.plot(ZZ1_grid.T,ZZ2_grid.T,color = [0.7,0.7,0.7])

ax.axis('scaled')
ax.set_xbound(3,9)
ax.set_ybound(1,7)

# Euclidean distance
p = 4
dd = ((np.abs((xx - mu_x))**p) + (np.abs((yy - mu_y))**p))**(1./p)
ax.contour(xx,yy,dd,levels = [1, 2, 3], colors = 'r')

##>>>>>>>>>>> L8范数
I = np.array([[1, 0],
              [0, 1]])

ZZ1_grid, ZZ2_grid = generate_grid(I, mu_x, mu_y)

x_array = np.array(iris_sns["sepal_length"])
y_array = np.array(iris_sns["petal_length"])

fig, ax = plt.subplots(figsize = (6,6))

plt.plot([x_array, mu_x+x_array*0],
         [y_array, mu_y+y_array*0],
         color = [0.7,0.7,0.7])

ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="petal_length")
sns.rugplot(data=iris_sns, x="sepal_length", y="petal_length", ax = ax)

plt.axvline(x=mu_x, linestyle = '--', color = 'r')
plt.axhline(y=mu_y, linestyle = '--', color = 'r')

plt.plot(mu_x,mu_y, color = 'k',
         marker = 'x', markersize = 15)

# ax.set_xlabel('Sepal length, $x_1$ (cm)')
# ax.set_ylabel('Petal length, $x_3$ (cm)')

ax.set_xticks([])
ax.set_yticks([])

plt.plot(ZZ1_grid,ZZ2_grid,color = [0.7,0.7,0.7])
plt.plot(ZZ1_grid.T,ZZ2_grid.T,color = [0.7,0.7,0.7])

ax.axis('scaled')
ax.set_xbound(3,9)
ax.set_ybound(1,7)

# Euclidean distance
p = 8
dd = ((np.abs((xx - mu_x))**p) + (np.abs((yy - mu_y))**p))**(1./p)
ax.contour(xx, yy, dd,levels = [1, 2, 3], colors = 'r')

##>>>>>>>>>>> L_inf范数
I = np.array([[1, 0],
              [0, 1]])
ZZ1_grid, ZZ2_grid = generate_grid(I, mu_x, mu_y)
x_array = np.array(iris_sns["sepal_length"])
y_array = np.array(iris_sns["petal_length"])
fig, ax = plt.subplots(figsize = (6,6))
plt.plot([x_array, mu_x+x_array*0],
         [y_array, mu_y+y_array*0],
         color = [0.7,0.7,0.7])

ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="petal_length")
sns.rugplot(data=iris_sns, x="sepal_length", y="petal_length", ax = ax)

plt.axvline(x=mu_x, linestyle = '--', color = 'r')
plt.axhline(y=mu_y, linestyle = '--', color = 'r')

plt.plot(mu_x,mu_y, color = 'k', marker = 'x', markersize = 15)
ax.set_xlabel('Sepal length, $x_1$ (cm)')
ax.set_ylabel('Petal length, $x_3$ (cm)')

plt.plot(ZZ1_grid,ZZ2_grid,color = [0.7,0.7,0.7])
plt.plot(ZZ1_grid.T,ZZ2_grid.T,color = [0.7,0.7,0.7])

ax.axis('scaled')
ax.set_xbound(3,9)
ax.set_ybound(1,7)

# Euclidean distance
p = 8
dd =  np.maximum(np.abs(xx - mu_x), np.abs(yy - mu_y))
ax.contour(xx,yy,dd,levels = [1, 2, 3], colors = 'r')

ax.set_xticks([])
ax.set_yticks([])

##>>>>>>>>>>> 标准欧氏距离
fig, ax = plt.subplots(figsize = (6,6))

plt.plot([x_array, mu_x+x_array*0],
         [y_array, mu_y+y_array*0],
         color = [0.7,0.7,0.7])

ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="petal_length")
sns.rugplot(data=iris_sns, x="sepal_length", y="petal_length", ax = ax)

plt.axvline(x=mu_x, linestyle = '--', color = 'r')
plt.axhline(y=mu_y, linestyle = '--', color = 'r')

plt.plot(mu_x,mu_y, color = 'k',
         marker = 'x', markersize = 15)

# ax.set_xlabel('Sepal length, $x_1$ (cm)')
# ax.set_ylabel('Petal length, $x_3$ (cm)')

ax.set_xticks([])
ax.set_yticks([])

plt.plot(ZZ1_grid,ZZ2_grid,color = [0.7,0.7,0.7])
plt.plot(ZZ1_grid.T,ZZ2_grid.T,color = [0.7,0.7,0.7])

ax.axis('scaled')
ax.set_xbound(3,9)
ax.set_ybound(1,7)
# Euclidean distance
dd = np.sqrt(((xx - mu_x)/sigma_x)**2 + ((yy - mu_y)/sigma_y)**2)
ax.contour(xx,yy,dd,levels = [1, 2, 3], colors = 'r')

##>>>>>>>>>>> 马氏距离
lambdas, V = LA.eig(SIGMA_13)
zz_maha = np.c_[xx.ravel(), yy.ravel()]
X = iris_sns.to_numpy()
X13 = np.array(X[:,[0,2]], dtype=float)
emp_cov_Xc = EmpiricalCovariance().fit(X13)
mahal_sq_Xc = emp_cov_Xc.mahalanobis(zz_maha)
mahal_sq_Xc = mahal_sq_Xc.reshape(xx.shape)
mahal_d_Xc = np.sqrt(mahal_sq_Xc)

ZZ1_grid, ZZ2_grid = generate_grid(V@np.diag(np.sqrt(lambdas)), mu_x, mu_y)
x_array = np.array(iris_sns["sepal_length"])
y_array = np.array(iris_sns["petal_length"])

fig, ax = plt.subplots(figsize = (6,6))
plt.plot([x_array, mu_x+x_array*0],
         [y_array, mu_y+y_array*0],
         color = [0.7,0.7,0.7])
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="petal_length")
sns.rugplot(data=iris_sns, x="sepal_length", y="petal_length", ax = ax)

plt.axvline(x=mu_x, linestyle = '--', color = 'r')
plt.axhline(y=mu_y, linestyle = '--', color = 'r')

plt.plot(mu_x,mu_y, color = 'k', marker = 'x', markersize = 15)

# ax.set_xlabel('Sepal length, $x_1$ (cm)')
# ax.set_ylabel('Petal length, $x_3$ (cm)')

ax.set_xticks([])
ax.set_yticks([])

plt.plot(ZZ1_grid,ZZ2_grid,color = [0.7,0.7,0.7])
plt.plot(ZZ1_grid.T,ZZ2_grid.T,color = [0.7,0.7,0.7])

ax.axis('scaled')
ax.set_xbound(3,9)
ax.set_ybound(1,7)
ax.contour(xx, yy, mahal_d_Xc,levels = [1, 2, 3], colors = 'r')









