#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:25:56 2025

@author: jack
<Wireless communication systems in Matlab> Chap2
"""
#%%
import scipy
import numpy as np
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import commpy

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22


#%% 2.5 Generating correlated random variables

# 2.5.2 Generating multiple sequences of correlated random variables using Cholesky decomposition

C = np.array([[1,0.5,0.3],[0.5,1,0.3],[0.3,0.3,1]])

L = np.linalg.cholesky(C)
U = L.T

R = np.random.randn(100000, 3)
Rc = R@U

X = Rc[:,0]
Y = Rc[:,1]
Z = Rc[:,2]

C_hat = np.cov(Rc.T)
print("相关系数矩阵=\n", C_hat)
corr = np.corrcoef(Rc.T,Rc.T)
print("相关系数矩阵=\n", corr)


##### plot
fig, axs = plt.subplots(1, 3, figsize = (12, 4), constrained_layout = True)

# x
axs[0].scatter(X, X, color = 'b',  label = '原始波形')
axs[0].set_xlabel('X',)
axs[0].set_ylabel('X',)
lb = 'X and X, ' + r'$\rho = {:.2f}$'.format(C_hat[0,0])
axs[0].set_title(lb )
# axs[0].legend()

axs[1].scatter(X, Y, color = 'r', label = '载波信号')
axs[1].set_xlabel('X',)
axs[1].set_ylabel('Y',)
lb = 'X and Y, ' + r'$\rho = {:.2f}$'.format(C_hat[0,1])
axs[1].set_title(lb)
# axs[1].legend()

axs[2].scatter(X, Z, color = 'gray', label = '幅度调制信号')
axs[2].set_xlabel('X',)
axs[2].set_ylabel('Z',)
lb = 'X and Z, ' + r'$\rho = {:.2f}$'.format(C_hat[0,2])
axs[2].set_title(lb )
# axs[2].legend()


plt.show()
plt.close()



#%% 2.6 Generating correlated Gaussian sequences
#%% Program 2.26: Jakes ﬁlter.m: Jakes ﬁlter using spectral factorization method


























#%%









































































































































































































































































































































































































































































