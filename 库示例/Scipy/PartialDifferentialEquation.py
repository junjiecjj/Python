#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:14:35 2023

@author: jack


https://blog.csdn.net/youcans/article/details/119755450

https://blog.csdn.net/weixin_46178278/article/details/135621598
https://zhuanlan.zhihu.com/p/81488678

https://zhuanlan.zhihu.com/p/388879602

"""



#%% https://blog.csdn.net/weixin_46178278/article/details/135621598
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 定义热传导方程
def heat_equation(t, u, alpha, dx):
    du_dx2 = np.gradient(np.gradient(u, dx), dx)
    return alpha * du_dx2

# 定义初始条件和空间网格
initial_condition = np.sin(np.pi * np.linspace(0, 1, 100))
space_grid = np.linspace(0, 1, 100)

# 求解热传导方程
solution = solve_ivp(heat_equation, [0, 0.1], initial_condition, args=(0.01, space_grid), t_eval=[0, 0.02, 0.05, 0.1])

# 绘制温度分布随时间的演化
plt.figure(figsize=(10, 6))
for i in range(len(solution.t)):
    plt.plot(space_grid, solution.y[:, i], label=f't={solution.t[i]:.2f}')
plt.xlabel('空间')
plt.ylabel('温度分布')
plt.title('一维热传导方程的数值求解')
plt.legend()
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

# 定义二维波动方程
def wave_equation(t, u, c, dx, dy):
    du_dx2 = np.gradient(np.gradient(u, dx, axis=0), dx, axis=0)
    du_dy2 = np.gradient(np.gradient(u, dy, axis=1), dy, axis=1)
    return c**2 * (du_dx2 + du_dy2)

# 定义初始条件和空间网格
initial_condition = np.exp(-((np.linspace(0, 1, 50) - 0.5)**2 + (np.linspace(0, 1, 50) - 0.5)**2) / 0.1)
space_grid_x, space_grid_y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))

# 求解二维波动方程
solution = solve_ivp(wave_equation, [0, 1], initial_condition.flatten(), args=(1.0, space_grid_x, space_grid_y),
                     t_eval=np.linspace(0, 1, 50))

# 绘制振幅随时间的演化
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(len(solution.t)):
    ax.plot_surface(space_grid_x, space_grid_y, solution.y[:, i].reshape((50, 50)), cmap='viridis', alpha=0.5,
                    rstride=100, cstride=100)
ax.set_xlabel('空间 X')
ax.set_ylabel('空间 Y')
ax.set_zlabel('振幅')
ax.set_title('二维波动方程的数值求解')
plt.show()

