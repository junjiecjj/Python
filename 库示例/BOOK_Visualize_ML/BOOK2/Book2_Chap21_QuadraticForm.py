



# Bk1_Ch25_03.ipynb
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

    ax_3D.set_xlabel('$x_1$')
    ax_3D.set_ylabel('$x_2$')
    ax_3D.set_zlabel('$f(x_1,x_2)$')
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

    ax_2D.set_xlabel('$x_1$'); ax_2D.set_ylabel('$x_2$')
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


#%% Bk2_Ch21_01
import numpy as np
import matplotlib.pyplot as plt
import os
from sympy import symbols, diff, lambdify, expand, simplify

# 二元函数
def fcn_n_grdnt(A, xx1, xx2):
    x1,x2 = symbols('x1 x2')
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
    fig = plt.figure( figsize=(12, 6))
    # 第一幅子图
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.plot_wireframe(xx1, xx2, f2_array, rstride=10, cstride=10, color = [0.8,0.8,0.8], linewidth = 0.25)
    ax.contour(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')

    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1,x_2)$')
    ax.set_proj_type('ortho')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(azim=-120, elev=30); ax.grid(False)
    ax.set_xlim(xx1.min(), xx1.max()); ax.set_ylim(xx2.min(), xx2.max())

    # 第二幅子图
    ax = fig.add_subplot(1, 3, 2)
    ax.contour(xx1, xx2, f2_array, levels = 12, cmap = 'RdYlBu_r')

    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
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

    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xlim(xx1.min(), xx1.max());
    ax.set_ylim(xx2.min(), xx2.max())


# 生成网格化数据

x1_array = np.linspace(-2,2,201)
x2_array = np.linspace(-2,2,201)

xx1, xx2 = np.meshgrid(x1_array, x2_array)


# 正定性
A = np.array([[1, 0],
              [0, 1]])

f2_array, gradient_array = fcn_n_grdnt(A, xx1, xx2)
visualize(xx1,xx2,f2_array,gradient_array)




















































































































































































































