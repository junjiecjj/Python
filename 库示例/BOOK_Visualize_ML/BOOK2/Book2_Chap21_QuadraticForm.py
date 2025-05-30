

#%% # 不同方式展示二元欧氏距离 Bk2_Ch25
import numpy as np
import matplotlib.pyplot as plt
import os
from sympy import init_printing, symbols, diff, lambdify, expand, simplify, sqrt
# init_printing("mathjax")

## 二元函数
x1_array = np.linspace(-2, 2, 201)
x2_array = np.linspace(-2, 2, 201)
xx1, xx2 = np.meshgrid(x1_array, x2_array)

def fcn_n_grdnt(A, xx1, xx2):
    x1,x2 = symbols('x1 x2')
    x = np.array([[x1, x2]]).T
    f_x = x.T@A@x
    f_x = f_x[0][0]
    # f_x = sqrt(f_x)
    # print(simplify(expand(f_x)))

    # take the gradient symbolically
    grad_f = [diff(f_x,var) for var in (x1,x2)]
    f_x_fcn = lambdify([x1, x2], f_x)
    ff_x = f_x_fcn(xx1,xx2)

    #turn into a bivariate lambda for numpy
    grad_fcn = lambdify([x1, x2], grad_f)
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
fig = plt.figure(figsize=(12, 6), constrained_layout = True)
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
# plt.tight_layout()

xx1_ = xx1[::20,::20]
xx2_ = xx2[::20,::20]
ax = fig.add_subplot(1, 3, 2)
ax.contour(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')
ax.quiver(xx1_, xx2_, gradient_array[0], gradient_array[1],  angles='xy', scale_units='xy', cmap = 'RdYlBu_r', edgecolor='none', alpha=0.8)
# ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max());
ax.set_ylim(xx2.min(), xx2.max())
# plt.tight_layout()

ax = fig.add_subplot(1, 3, 3)
ax.contourf(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')

ax.quiver(xx1_, xx2_, gradient_array[0], gradient_array[1],  angles='xy', scale_units='xy', cmap = 'RdYlBu_r', edgecolor='none', alpha=0.8)
# ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
ax.grid(False)
ax.set_xlim(xx1.min(), xx1.max());
ax.set_ylim(xx2.min(), xx2.max())
# plt.tight_layout()

#%% Bk4_Ch21
# 使用 Cholesky 分解判定矩阵是否为正定矩阵
import numpy as np

def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

A = np.array([[1,0],
              [0,2]])
print(is_pos_def(A))

# Bk4_Ch21_02.py
import sympy
import numpy as np
import matplotlib.pyplot as plt

def mesh_circ(c1, c2, r, num):
    theta = np.arange(0, 2*np.pi+np.pi/num, np.pi/num)
    r     = np.arange(0, r, r/num)
    theta, r = np.meshgrid(theta,r)
    xx1 = np.cos(theta)*r + c1
    xx2 = np.sin(theta)*r + c2

    return xx1, xx2

#define symbolic vars, function
x1, x2 = sympy.symbols('x1 x2')

A = np.array([[ 2, 0],
              [0,  1]])
x = np.array([[x1, x2]]).T
f_x = x.T@A@x
f_x = f_x[0][0]
f_x_fcn = sympy.lambdify([x1, x2], f_x)
xx1, xx2 = mesh_circ(0, 0, 4, 20)
ff_x = f_x_fcn(xx1, xx2)

#take the gradient symbolically
grad_f = [sympy.diff(f_x, var) for var in (x1, x2)]
#turn into a bivariate lambda for numpy
grad_fcn = sympy.lambdify([x1, x2], grad_f)

# coarse mesh
xx1_, xx2_ = mesh_circ(0, 0, 4, 30)
V = grad_fcn(xx1_,xx2_)
V_z = np.ones_like(V[1]);

if isinstance(V[1], int):
    V[1] = np.zeros_like(V[0])

elif isinstance(V[0], int):
    V[0] = np.zeros_like(V[1])

color_array = np.sqrt(V[0]**2 + V[1]**2)
l_3D_vectors = np.sqrt(V[0]**2 + V[1]**2 + V_z**2)

# 3D visualization
ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
ax.plot_wireframe(xx1, xx2, ff_x, rstride = 1, cstride=1, color = [0.5,0.5,0.5], linewidth = 0.2)
# ax.contour3D(xx1, xx2, ff_x, 20, cmap = 'RdYlBu_r')

ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.zaxis.set_ticks([])
plt.xlim(xx1.min(),xx1.max())
plt.ylim(xx2.min(),xx2.max())
ax.set_proj_type('ortho')
ax.view_init(30, -125)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$f(x_1,x_2)$')
plt.tight_layout()
plt.show()

color_array = np.sqrt(V[0]**2 + V[1]**2)
# 2D visualization
fig, ax = plt.subplots(figsize=(10, 10))
plt.quiver (xx1_, xx2_, -V[0], -V[1], color_array, angles='xy', scale_units='xy', edgecolor='none', alpha=0.8,cmap = 'RdYlBu_r')

plt.contour(xx1, xx2, ff_x,20, cmap = 'RdYlBu_r')
ax.set_aspect('equal')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
plt.tight_layout()
plt.show()

#%% Bk4_Ch14_03.py
import sympy
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as L

def mesh_circ(c1, c2, r, num):
    theta = np.linspace(0, 2*np.pi, num)
    r     = np.linspace(0,r, num)
    theta,r = np.meshgrid(theta,r)
    xx1 = np.cos(theta)*r + c1
    xx2 = np.sin(theta)*r + c2
    return xx1, xx2

#define symbolic vars, function
x1,x2 = sympy.symbols('x1 x2')
A = np.array([[0.5, -0.5],
              [-0.5, 0.5]])
Lambda, V = L.eig(A)
x = np.array([[x1,x2]]).T
f_x = x.T@A@x
f_x = f_x[0][0]
f_x_fcn = sympy.lambdify([x1,x2],f_x)
xx1, xx2 = mesh_circ(0, 0, 1, 50)
ff_x = f_x_fcn(xx1,xx2)
if Lambda[1] > 0:
    levels = np.linspace(0,Lambda[0],21)
else:
    levels = np.linspace(Lambda[1],Lambda[0],21)

t = np.linspace(0, np.pi*2, 100)

# 2D visualization
fig, ax = plt.subplots()
ax.plot(np.cos(t), np.sin(t), color = 'k')
cs = ax.contourf(xx1, xx2, ff_x, levels = levels, cmap = 'RdYlBu_r')
plt.show()
ax.set_aspect('equal')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
clb = fig.colorbar(cs, ax=ax)
clb.set_ticks(levels)
plt.show()
plt.close()


#  3D surface of f(x1,x2)
x1_ = np.linspace(-1.2,1.2,31)
x2_ = np.linspace(-1.2,1.2,31)

xx1_fine, xx2_fine = np.meshgrid(x1_,x2_)
ff_x_fine = f_x_fcn(xx1_fine, xx2_fine)
f_circle = f_x_fcn(np.cos(t), np.sin(t))
# 3D visualization
fig, ax = plt.subplots()
ax = plt.axes(projection='3d')
ax.plot(np.cos(t), np.sin(t), f_circle, color = 'k')
# circle projected to f(x1,x2)

ax.plot_wireframe(xx1_fine, xx2_fine, ff_x_fine, color = [0.8,0.8,0.8], linewidth = 0.25)
ax.contour(xx1_fine, xx2_fine, ff_x_fine, 15, cmap = 'RdYlBu_r')

ax.view_init(elev=30, azim=60)
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.zaxis.set_ticks([])
ax.set_xlim(xx1_fine.min(),xx1_fine.max())
ax.set_ylim(xx2_fine.min(),xx2_fine.max())
plt.tight_layout()
ax.set_proj_type('ortho')
plt.show()


#%% Bk1_Ch25_03.ipynb
# 导入包
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, expand, simplify

# 定义可视化函数
def visualize(xx1, xx2, f2_array):
    fig = plt.figure(figsize=(12, 6))
    # 左子图，三维
    ax_3D = fig.add_subplot(1, 2, 1, projection='3d')

    ax_3D.plot_wireframe(xx1, xx2, f2_array,  rstride = 5, cstride = 5,  color = [0.5, 0.5, 0.5], linewidth = 0.5)
    ax_3D.contour(xx1, xx2, f2_array, levels = 16, cmap = 'RdYlBu_r')

    ax_3D.set_xlabel(r'$x_1$')
    ax_3D.set_ylabel(r'$x_2$')
    ax_3D.set_zlabel(r'$f(x_1,x_2)$')
    ax_3D.set_proj_type('ortho')
    ax_3D.set_xticks([])
    ax_3D.set_yticks([])
    ax_3D.set_zticks([])
    ax_3D.view_init(azim=-120, elev=30)
    ax_3D.grid(False)
    ax_3D.set_xlim(xx1.min(), xx1.max());
    ax_3D.set_ylim(xx2.min(), xx2.max())

    # 右子图，平面等高线
    ax_2D = fig.add_subplot(1, 2, 2)
    ax_2D.contour(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')

    ax_2D.set_xlabel(r'$x_1$'); ax_2D.set_ylabel(r'$x_2$')
    ax_2D.set_xticks([]); ax_2D.set_yticks([])
    ax_2D.set_aspect('equal'); ax_2D.grid(False)
    ax_2D.set_xlim(xx1.min(), xx1.max());
    ax_2D.set_ylim(xx2.min(), xx2.max())
    plt.tight_layout()

# 生成数据
x1_array = np.linspace(-2, 2, 201)
x2_array = np.linspace(-2, 2, 201)
xx1, xx2 = np.meshgrid(x1_array, x2_array)

# 定义二元函数
def fcn(A, xx1, xx2):
    x1, x2 = symbols('x1 x2')
    x = np.array([[x1, x2]]).T  ## 这个技术很好用
    f_x = x.T @ A @ x
    f_x = f_x[0][0]
    print(simplify(expand(f_x)))

    f_x_fcn = lambdify([x1, x2], f_x)
    ff_x = f_x_fcn(xx1, xx2)

    return ff_x

# xx定矩阵
A = np.array([[2, 0],
              [0, 3]])

f2_array = fcn(A, xx1, xx2)
visualize(xx1, xx2, f2_array)

#%% Bk2_Ch21_01   二元二次型
import numpy as np
import matplotlib.pyplot as plt
import os
from sympy import symbols, diff, lambdify, expand, simplify

# 二元函数
def fcn_n_grdnt(A, xx1, xx2):
    x1, x2 = symbols('x1 x2')
    # 符号向量
    x = np.array([[x1,x2]]).T
    # 二次型
    f_x = x.T@A@x
    f_x = f_x[0][0]
    print(simplify(expand(f_x)))

    # 计算梯度，符号
    grad_f = [diff(f_x,var) for var in (x1,x2)]

    # 计算二元函数值 f(x1, x2)
    f_x_fcn = lambdify([x1, x2], f_x)
    ff_x = f_x_fcn(xx1, xx2)

    # 梯度函数
    grad_fcn = lambdify([x1,x2],grad_f)

    # 采样，降低颗粒度
    xx1_ = xx1[::20,::20]
    xx2_ = xx2[::20,::20]

    # 计算梯度
    V = grad_fcn(xx1_,xx2_)

    # 修复梯度值
    if isinstance(V[1], int):
        V[1] = np.zeros_like(xx1_)

    if isinstance(V[0], int):
        V[0] = np.zeros_like(xx1_)
    return ff_x, V

# 可视化函数
def visualize(xx1, xx2, f2_array, gradient_array):
    fig = plt.figure( figsize=(12, 6), constrained_layout = True)
    # 第一幅子图
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.plot_wireframe(xx1, xx2, f2_array, rstride=10, cstride=10, color = [0.8,0.8,0.8], linewidth = 0.25)
    ax.contour(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')

    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_zlabel(r'$f(x_1,x_2)$')
    ax.set_proj_type('ortho')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(azim=-120, elev=30); ax.grid(False)
    ax.set_xlim(xx1.min(), xx1.max()); ax.set_ylim(xx2.min(), xx2.max())

    # 第二幅子图
    ax = fig.add_subplot(1, 3, 2)
    ax.contour(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')

    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal'); ax.grid(False)
    ax.set_xlim(xx1.min(), xx1.max()); ax.set_ylim(xx2.min(), xx2.max())

    # 第三幅子图
    ax = fig.add_subplot(1, 3, 3)
    ax.contourf(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')
    xx1_ = xx1[::20,::20]
    xx2_ = xx2[::20,::20]
    color_array = np.sqrt(gradient_array[0]**2 + gradient_array[1]**2)
    # ax.quiver(xx1_, xx2_, gradient_array[0], gradient_array[1], color_array, angles='xy', scale_units='xy', cmap = 'RdYlBu_r', edgecolor='none', alpha=0.8)
    ax.quiver(xx1_, xx2_, gradient_array[0], gradient_array[1],  angles='xy', scale_units='xy', cmap = 'RdYlBu_r', edgecolor='none', alpha=0.8)

    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlim(xx1.min(), xx1.max());
    ax.set_ylim(xx2.min(), xx2.max())

## 生成网格化数据
x1_array = np.linspace(-2,2,201)
x2_array = np.linspace(-2,2,201)

xx1, xx2 = np.meshgrid(x1_array, x2_array)

## 正定性
A = np.array([[1, 0],
              [0, 1]])

f2_array, gradient_array = fcn_n_grdnt(A, xx1, xx2)
visualize(xx1, xx2, f2_array, gradient_array)

## 半正定
A = np.array([[1, 0],
              [0, 0]])

f2_array, gradient_array = fcn_n_grdnt(A, xx1, xx2)
visualize(xx1,xx2,f2_array,gradient_array)


## 负定
A = np.array([[-1, 0],
              [0, -1]])

f2_array, gradient_array = fcn_n_grdnt(A, xx1, xx2)
visualize(xx1,xx2,f2_array,gradient_array)


## 半负定
A = np.array([[-1, 0],
              [0,  0]])

f2_array, gradient_array = fcn_n_grdnt(A, xx1, xx2)
visualize(xx1,xx2,f2_array,gradient_array)

## 不定
A = np.array([[0, 1],
              [1, 0]])

f2_array, gradient_array = fcn_n_grdnt(A, xx1, xx2)
visualize(xx1,xx2,f2_array,gradient_array)



#%% 三元二次型 Bk2_Ch21_02, 参见切豆腐:Bk_2_Ch16_07, Bk2_Ch21_02, BK_2_Ch25_04
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sympy import symbols, simplify, expand, lambdify, diff

## 自定义函数
# 定义三元二次型
def fcn_3(A,xxx1,xxx2,xxx3):
    x1,x2,x3 = symbols('x1 x2 x3')
    x = np.array([[x1,x2,x3]]).T
    f_x = x.T@A@x
    print(simplify(expand(f_x[0][0])))
    f_x_fcn = lambdify([x1,x2,x3], f_x[0][0])
    fff = f_x_fcn(xxx1,xxx2,xxx3)
    return fff

## 创建数据
x1_array = np.linspace(-2,2,101)
x2_array = np.linspace(-2,2,101)
x3_array = np.linspace(-2,2,101)
xxx1, xxx2, xxx3 = np.meshgrid(x1_array, x2_array, x3_array)
A = np.array([[1, 0, 0],
              [0, 0, 0],
              [0, 0, -1]])
# 计算矩阵秩
print(np.linalg.matrix_rank(A))
f3_array = fcn_3(A,xxx1,xxx2,xxx3)

## 切豆腐
### 外立面
# 设定统一等高线分层
levels = np.linspace(f3_array.min(),f3_array.max(),21)
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
fig.colorbar(CS, ticks=np.linspace(np.floor(f3_array.min()),np.ceil(f3_array.max()), int(np.ceil(f3_array.max()) - np.floor(f3_array.min())) + 1))
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
# fig.savefig('Figures/一次函数，三元，3D外立面.svg', format='svg')

### 将等高线展开，沿x3
fig = plt.figure(figsize=(6, 36))
for fig_idx, idx in enumerate(np.arange(0, len(x3_array), 25)):
    ax = fig.add_subplot(len(np.arange(0, len(x3_array), 25)), 1, fig_idx + 1, projection='3d')
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
                zdir='z',
                offset=x3_idx,
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
    # ax.set_xlabel('$x_1$')
    # ax.set_ylabel('$x_2$')
    # ax.set_zlabel('$x_3$')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)
# fig.savefig('Figures/一次函数，三元，沿x3分层等高线.svg', format='svg')


### 将等高线展开，沿x2
fig = plt.figure(figsize=(6, 36))
for fig_idx,idx in enumerate(np.arange(0,len(x2_array),25)):
    ax = fig.add_subplot(len(np.arange(0,len(x2_array),25)), 1, fig_idx + 1, projection='3d')
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
# fig.savefig('Figures/一次函数，三元，沿x2分层等高线.svg', format='svg')


### 将等高线展开，沿x1
fig = plt.figure(figsize=(6, 36))
for fig_idx,idx in enumerate(np.arange(0,len(x1_array),25)):
    ax = fig.add_subplot(len(np.arange(0,len(x1_array),25)), 1,  fig_idx + 1, projection='3d')
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
# fig.savefig('Figures/一次函数，三元，沿x1分层等高线.svg', format='svg')

#%% # 用切豆腐的方法展示三元欧氏距离 Book2_chap25
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
    f_x_fcn = lambdify([x1,x2,x3], f_x)
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
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$x_3$')
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

























































































































































































































































































































































































































































