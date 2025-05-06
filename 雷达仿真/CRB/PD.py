#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 14:53:53 2025

@author: jack
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18

np.random.seed(42)


#%% https://deepinout.com/numpy/numpy-questions/109_numpy_how_can_i_compute_the_null_spacekernel_x_mx_0_of_a_sparse_matrix_in_python.html#google_vignette
import numpy as np
# from scipy.sparse import random
# from scipy.sparse.linalg import svds

# 创建一个随机稀疏矩阵
M = scipy.sparse.random(6, 6, density=0.5, format='lil', random_state=0)

# 计算随机稀疏矩阵的SVD
U, s, Vt = scipy.sparse.linalg.svds(M, k=1)

# 计算空间/核
null_space = Vt.T[:, -1]

# 输出结果
print("稀疏矩阵M：\n", M.todense())
print("空间/核：\n", null_space)

#%% https://blog.51cto.com/u_16213370/11892718

import numpy as np
# from scipy.linalg import null_space

# 定义一个矩阵 A
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# 计算零空间
null_space_A = scipy.linalg.null_space(A)

print("矩阵 A 的零空间:")
print(null_space_A)

# 计算矩阵的秩
rank_A = np.linalg.matrix_rank(A)
print("矩阵 A 的秩:", rank_A)

#%% https://www.zhihu.com/question/294214797

def null(A, eps = 1e-15):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s < eps)
    null_space = np.compress(null_mask, vh, axis = 0)
    return np.transpose(null_space)

# 计算零空间
null_space_A = null(A)

print("矩阵 A 的零空间:")
print(null_space_A)





# %% Define Parameters
pi = np.pi
# Speed of light
c = 3*10**8
#  Nr Comm Receivers
Nr = 2
#  Mt Radar Transmiters
Mt = 8
#  Mr Radar Receivers
Mr = Mt
#  Radial velocity of 2000 m/s
v_r = 2000
#  Radar reference point
r_0 = 500*10**3
#  Carrier frequency 3.5GHz
f_c = 3.5*1e9  #  Angular carrier frequency
omega_c = 2*pi*f_c
lamba = (2*pi*c)/omega_c
theta = 0










































































