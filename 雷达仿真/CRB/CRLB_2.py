#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:28:29 2025

@author: jack

https://github.com/LiZhuoRan0/CRLB-demo


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


def genSteerVector(theta_deg, N, d, lambda_c):
    n = np.arange(N)
    at = np.exp(1j * 2 * np.pi * d * np.sin(theta) * n / lambda_c)
    return at

def genPartialSteerVector(theta, N, d, lamda, flag):
    n = np.arange(N)
    if flag == 1:
        at = (1j * 2 * np.pi * d * np.cos())
    else:

    return



#%% 参数设置
N               =           64                 # 基站处天线
fc              =           100e9              #100GHz
lambda_c        =           3e8/fc
d               =           lambda_c/2
T               =           10
Nit             =           1e2
SNRdBs          =           np.arange(-10, 22, 2)
#%%
theta   = np.deg2rad(30)  #  np.pi/6

H   = genSteerVector(theta, N, d, lambda_c)
Y   = np.zeros(N, T)
A   = np.eye(N)


















