#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:07:05 2024

@author: jack

"""


import numpy as np
import scipy
import matplotlib.pyplot as plt

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
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

filepath2 = '/home/jack/snap/'
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


#%%%%%%%% CBF for Uniform Linear Array %%%%%%%%
pi = np.pi
derad = pi/180                        # 角度->弧度
N = 8                                 # 阵元个数
M = 3                                 # 信源数目
thetaTrue = [-30, 0, 60]
theta = np.deg2rad(thetaTrue)         # 待估计角度
snr = 10                              # 信噪比
K = 512                               # 快拍数

d = np.arange(0, N).reshape(-1, 1)
A = np.exp(-1j * pi * d @ np.sin(theta).reshape(1, -1) )   # 方向矢量

# 构建信号模型
S = np.random.randn(M, K)             # 信源信号，入射信号
X = A@S                                # 构造接收信号
SigPow = np.power(np.abs(X), 2).mean()
noise_pwr = SigPow/(10**(snr/10))
noise = np.sqrt(noise_pwr ) *  np.random.randn(*(X.shape))
X1 = X + noise                  # 将白色高斯噪声添加到信号中
# 计算协方差矩阵
Rxx = X1 @ X1.T.conjugate()
Thetalst = np.arange(-90, 90.1, 0.5)
angle = np.deg2rad(Thetalst)
Pcbf = np.zeros(angle.size)
for i, ang in enumerate(angle):
    a = np.exp(-1j * pi * d * np.sin(ang))
    Pcbf[i] = np.real(a.T.conjugate() @ Rxx @ a)[0,0]

Pcbf = np.abs(Pcbf) / np.abs(Pcbf).max()
Pcbf = 10 * np.log10(Pcbf)
peaks, _ =  scipy.signal.find_peaks(Pcbf, height=-2,  distance = 10)
print(f"True = {thetaTrue}\n est = {Thetalst[peaks]}")
### 画图
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
axs.plot(Thetalst, Pcbf , color = 'b', linestyle='-', lw = 3, label = "CBF", )
axs.plot(Thetalst[peaks], Pcbf[peaks], linestyle='', marker = 'o', color='r', markersize = 12)
axs.legend()

axs.set_xlabel( "DOA/(degree)", )
axs.set_ylabel('Normalized Spectrum/(dB)', )

plt.show()
plt.close()


#%%%%%%%% Capon for Uniform Linear Array %%%%%%%%
pi = np.pi
derad = pi/180                        # 角度->弧度
N = 8                                 # 阵元个数
M = 3                                 # 信源数目
thetaTrue = [-30, 0, 60]
theta = np.deg2rad(thetaTrue)         # 待估计角度
snr = 10                              # 信噪比
K = 512                               # 快拍数

d = np.arange(0, N).reshape(-1, 1)
A = np.exp(-1j * pi * d @ np.sin(theta).reshape(1, -1) )   # 方向矢量

# 构建信号模型
S = np.random.randn(M, K)             # 信源信号，入射信号
X = A@S                                # 构造接收信号
SigPow = np.power(np.abs(X), 2).mean()
noise_pwr = SigPow/(10**(snr/10))
noise = np.sqrt(noise_pwr ) *  np.random.randn(*(X.shape))
X1 = X + noise                  # 将白色高斯噪声添加到信号中
# 计算协方差矩阵
Rxx = X1 @ X1.T.conjugate()
Thetalst = np.arange(-90, 90.1, 0.5)
angle = np.deg2rad(Thetalst)
Pcapon = np.zeros(angle.size)
# for i, ang in enumerate(angle):
#     a = np.exp(-1j * pi * d * np.sin(ang))
#     Pcbf[i] = np.real(a.T.conjugate() @ Rxx @ a)[0,0]

for i, ang in enumerate(angle):
    a = np.exp(-1j * pi * d * np.sin(ang))
    Pcapon[i] = 1/np.real(a.T.conjugate() @ scipy.linalg.inv(Rxx) @ a)[0,0]

Pcapon = np.abs(Pcapon) / np.abs(Pcapon).max()
Pcapon = 10 * np.log10(Pcapon)
peaks, _ =  scipy.signal.find_peaks(Pcapon, height=-2,  distance = 10)
print(f"True = {thetaTrue}\n est = {Thetalst[peaks]}")
### 画图
fig, axs = plt.subplots(1, 1, figsize=(10, 8))
axs.plot(Thetalst, Pcapon , color = 'b', linestyle='-', lw = 3, label = "Capon", )
axs.plot(Thetalst[peaks], Pcapon[peaks], linestyle='', marker = 'o', color='r', markersize = 12)
axs.legend()

axs.set_xlabel( "DOA/(degree)", )
axs.set_ylabel('Normalized Spectrum/(dB)', )

plt.show()
plt.close()









































































































































































