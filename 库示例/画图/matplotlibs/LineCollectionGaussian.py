#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 13:57:58 2024

@author: jack
"""

# 导数包
from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# 导入色谱
from scipy.stats import norm
# 导入正态分布
# 参考
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html

import os
# # 如果文件夹不存在，创建文件夹
# if not os.path.isdir("Figures"):
#     os.makedirs("Figures")

#%%. 用for循环
x_array = np.linspace(-6, 6, 200)
sigma_array = np.linspace(0.5,5,10)
# 设定标准差一系列取值

num_lines = len(sigma_array)
# 概率密度曲线条数

colors = cm.RdYlBu(np.linspace(0,1,num_lines))
# 选定色谱，并产生一系列色号

fig, ax = plt.subplots(figsize = (5,4))

for idx, sigma_idx in enumerate(sigma_array):

    pdf_idx = norm.pdf(x_array, scale = sigma_idx)
    legend_idx = '$\sigma$ = ' + str(sigma_idx)
    plt.plot(x_array, pdf_idx, color=colors[idx], label = legend_idx)
    # 依次绘制概率密度曲线

plt.legend()
# 增加图例

plt.xlim(x_array.min(),x_array.max())
plt.ylim(0,1)
plt.xlabel('x')
plt.ylabel('PDF, $f_X(x)$')
plt.show()
plt.close()

#%%. 用for循环
x_array = np.linspace(-6, 6, 200)
sigma_array = np.linspace(0.5,5,10)
# 设定标准差一系列取值

# 概率密度曲线条数
num_lines = len(sigma_array)

# 选定色谱，并产生一系列色号
colors = plt.cm.jet(np.linspace(0, 1, len(sigma_array))) # colormap

fig, ax = plt.subplots(figsize = (5,4))

for idx, sigma_idx in enumerate(sigma_array):

    pdf_idx = norm.pdf(x_array, scale = sigma_idx)
    legend_idx = '$\sigma$ = ' + str(sigma_idx)
    plt.plot(x_array, pdf_idx, color=colors[idx], label = legend_idx)
    # 依次绘制概率密度曲线

plt.legend()
# 增加图例

plt.xlim(x_array.min(),x_array.max())
plt.ylim(0,1)
plt.xlabel('x')
plt.ylabel('PDF, $f_X(x)$')
plt.show()
plt.close()





#%% 2. 用LineCollection
PDF_curves = [np.column_stack([x_array, norm.pdf(x_array, scale = sigma_idx)]) for sigma_idx in sigma_array]

fig, ax = plt.subplots(figsize = (5,4))
# LineCollection 可以看成是一系列线段的集合
lc = LineCollection(PDF_curves, cmap='rainbow', array=sigma_array, linewidth=1)
# 可以用色谱分别渲染每一条线段,这样可以得到颜色连续变化的效果
line = ax.add_collection(lc) #add to the subplot
# 添加色谱条
fig.colorbar(line, label='$\sigma$')

plt.xlim(x_array.min(), x_array.max())
plt.ylim(0,1)
plt.xlabel('x')
plt.ylabel('PDF, $f_X(x)$')
# fig.savefig('Figures/用LineCollection.svg', format='svg')
plt.show()
















































































































































































































































