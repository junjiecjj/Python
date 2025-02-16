#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 00:29:16 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzAwNTkyNTUxMA==&mid=2247492360&idx=1&sn=da8dc2a1b223ae747c8bff0950f49c4c&chksm=9a189b6d4e201217a56568bf03d51ec7e1d8905be6e48e135b886161095a46b47b4f04a4e3ca&mpshare=1&scene=1&srcid=0217JO92auO8ax8hFoN7wUwY&sharer_shareinfo=a1e88b44f7c1c6126b5561aa39d4fca0&sharer_shareinfo_first=a1e88b44f7c1c6126b5561aa39d4fca0&exportkey=n_ChQIAhIQvdwtzpT40xbpSiMUuYLY6RKfAgIE97dBBAEAAAAAAAV8LAHNp0gAAAAOpnltbLcz9gKNyK89dVj0EJEDXL8zRBnV4V54krzUo3ttoFAftJl%2F56TXBy4uA6KWtr0KC%2BMTDkW8LuaHdKkpGdrfQ8ElrivrmQMG5BH5Jz0x5b%2FrZZQ5kX9zlLFkLuDbhtxGeDJlc7UBZZmfGKU%2F10neZI3FW%2FQq1WEbmNvuYHzG%2FYlU%2FQnhIsqjJwi2L71DPLVUe4zZLgFxS8sn5HXuR7u5c%2BjK%2B36al6XsyiGTmAWdk49Zm6R5DU9SjqJkejzySo4aR1y2g3DaDV3cLntrJSMrZjGoMaY4RyJMszAu9AnfHFw5c%2FrXxEmxonEMXkP38OLz%2BqidH5O6M27Fl0CIj1bW5xQVcRHZ&acctmode=0&pass_ticket=eWjxh8LXRBF0Ly5GXwxeNwiLlPbTVjoAUV6tgnqRKV3kk2xPeyGpTFkzugsk4FH4&wx_header=0#rd


"""

import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
# from mpl_toolkits.mplot3d import Axes3D

# 目标函数（复杂的非凸函数）
def black_box_function(x, y):
    return -np.sin(3*x) - x**2 + 0.7*x + np.cos(2*y) + y**2 - 0.5*y

# 定义贝叶斯优化的边界
pbounds = {'x': (-2, 2), 'y': (-2, 2)}

# 初始化贝叶斯优化器
optimizer = BayesianOptimization(
    f = black_box_function,
    pbounds = pbounds,
    random_state = 42
)

# 运行优化
optimizer.maximize(init_points = 10, n_iter = 30)

# 获取优化历史
x_vals, y_vals, target_vals = [], [], []
for res in optimizer.res:
    x_vals.append(res["params"]["x"])
    y_vals.append(res["params"]["y"])
    target_vals.append(res["target"])

# 生成虚拟数据进行可视化
x_range = np.linspace(-2, 2, 100)
y_range = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = black_box_function(X, Y)

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 目标函数等高线 + 采样点分布
ax1 = axes[0, 0]
c = ax1.contourf(X, Y, Z, levels=20, cmap='viridis')
fig.colorbar(c, ax=ax1)
ax1.scatter(x_vals, y_vals, c='red', marker='o', label='Sampled Points')
ax1.set_title('Target Function Contour with Sample Points')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()

# 2. 优化目标值的变化趋势
ax2 = axes[0, 1]
ax2.plot(range(len(target_vals)), target_vals, marker='o', linestyle='-', color='b')
ax2.set_title('Optimization Progress')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Target Value')
ax2.grid()

# 3. 采样点 x, y 的变化趋势
ax3 = axes[1, 0]
ax3.plot(range(len(x_vals)), x_vals, marker='s', linestyle='-', label='x', color='r')
ax3.plot(range(len(y_vals)), y_vals, marker='^', linestyle='-', label='y', color='g')
ax3.set_title('Sampled Parameter Values Over Iterations')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Parameter Values')
ax3.legend()
ax3.grid()

# 4. 采样点分布的 3D 视图
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.plot_surface(X, Y, Z, cmap='plasma', alpha=0.7)
ax4.scatter(x_vals, y_vals, target_vals, c='black', marker='o')
ax4.set_title('3D View of Optimization Process')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_zlabel('Target Value')

plt.tight_layout()
plt.show()

















