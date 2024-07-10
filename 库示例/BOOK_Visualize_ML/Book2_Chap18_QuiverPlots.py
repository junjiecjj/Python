#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:59:34 2024
https://geek-docs.com/matplotlib/matplotlib-pyplot/matplotlib-pyplot-streamplot-in-python.html#google_vignette
@author: jack
"""

#%% 平面等高线 + 梯度

import numpy as np
import matplotlib.pyplot as plt
import sympy
import numpy as np
from sympy.functions import exp
# import os

# # 如果文件夹不存在，创建文件夹
# if not os.path.isdir("Figures"):
#     os.makedirs("Figures")




############# 定义符号函数
# 定义符号变量
x1,x2 = sympy.symbols('x1 x2')

# 定义符号二元函数
f_x = x1*exp(-(x1**2 + x2**2))

# 将符号函数转换为Python函数
f_x_fcn = sympy.lambdify([x1,x2],f_x)

# 计算梯度
grad_f = [sympy.diff(f_x,var) for var in (x1,x2)]

# 将符号梯度转化为Python函数
grad_fcn = sympy.lambdify([x1,x2],grad_f)


# 产生数据
# 细腻颗粒度
x1_array = np.linspace(-2,2,401)
x2_array = np.linspace(-2,2,401)
xx1, xx2 = np.meshgrid(x1_array,x2_array)

# 粗糙颗粒度
x1_array_ = np.linspace(-2,2,21)
x2_array_ = np.linspace(-2,2,21)
xx1_, xx2_ = np.meshgrid(x1_array_,x2_array_)
V = grad_fcn(xx1_,xx2_)

ff_x = f_x_fcn(xx1,xx2)
ff_x_ = f_x_fcn(xx1_,xx2_)

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


#%% 三维向量场
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



#%% 水流图
import matplotlib.pyplot as plt
import numpy as np
# import os

# # 如果文件夹不存在，创建文件夹
# if not os.path.isdir("Figures"):
#     os.makedirs("Figures")


x = np.arange(-3,3,0.3)
y = np.arange(-3,3,0.3)
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
ax.quiver(x,y,Fx,Fy)
ax.streamplot(x, y, Fx, Fy,
              density = 2,
              arrowstyle = 'fancy')
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
# fig.savefig('Figures/水流图 + 箭头图.svg', format='svg')
plt.show()






































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































