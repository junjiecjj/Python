#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:41:19 2025

@author: jack

https://blog.csdn.net/weixin_43270276/article/details/124647286

https://blog.csdn.net/qq_42432868/article/details/129804574?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ECtr-4-129804574-blog-124647286.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ECtr-4-129804574-blog-124647286.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=8

https://blog.csdn.net/xhblair/article/details/137021514?spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_baidulandingword%7ECtr-7-137021514-blog-124647286.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7Ebaidujs_baidulandingword%7ECtr-7-137021514-blog-124647286.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=12

https://blog.csdn.net/weixin_42305982/article/details/133837974?spm=1001.2101.3001.6650.11&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-11-133837974-blog-124647286.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-11-133837974-blog-124647286.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=16

https://blog.csdn.net/weixin_43270276/article/details/119831879?spm=1001.2101.3001.6650.12&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-12-119831879-blog-124647286.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-12-119831879-blog-124647286.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=17

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

#%%  单目标测向系统中CRLB推导
pi = np.pi
derad = pi/180.0             # 角度->弧度
M = 4                        # 阵元个数
K = 1                        # 信源数目
N = 10                       # 快拍数
tarDOA = 30 # 5.739170477266787                  # 待估计的Angle,可以由测向信号产生
SNR = np.arange(10, 32, 2)   # 遍历信噪比范围
trail = 10000                # 尝试次数
d = np.arange(M)             # 线阵间隔
esti_music = np.zeros(trail)
Len_SNR = SNR.size
VAREsti_snr = np.zeros(Len_SNR)
VARCalc_snr = np.zeros(Len_SNR)
Thetalst = np.arange(-90, 90.1, 0.5)
angle = np.deg2rad(Thetalst)
A = np.exp(-1j * d * pi * np.sin(tarDOA*derad))                 # 方向矢量，复数形式

for i, snr in enumerate(SNR):
    print(f"  {i+1}/{SNR.size}")
    # snr_current = SNR[index_snr]
    for k in range(trail):
        S = np.random.randn(K, N)                          # 信源信号，入射信号，不相干即可，也可以用正弦替代
        X = np.outer(A, S)                                 # 构造接收信号
        sig_pow = (np.abs(X)**2).mean()
        noisevar = sig_pow * 10 ** (-snr / 10)
        noise = np.sqrt(noisevar/2) * (np.random.randn(*X.shape) + 1j * np.random.randn(*X.shape))
        X += noise
        Rxx = X @ X.T.conjugate() / N

        eigenvalues, eigvector = np.linalg.eigh(Rxx)          # 特征值分解
        idx = np.argsort(eigenvalues)                         # 将特征值排序 从小到大
        eigvector = eigvector[:, idx]
        eigvector = eigvector[:,::-1]                         # 对应特征矢量排序
        Un = eigvector[:, K:M]

        UnUnH = Un @ Un.T.conjugate()
        Pmusic = np.zeros(angle.size)
        for j, ang in enumerate(angle):
            a = np.exp(-1j * np.pi * d * np.sin(ang)).reshape(-1, 1)
            Pmusic[j] = 1/np.abs(a.T.conjugate() @ UnUnH @ a)[0,0]

        peaks = Pmusic.argmax()
        angle_est = Thetalst[peaks]
        esti_music[k] = angle_est * derad                  # 寻找最大值对应延时
    VAREsti_snr[i] = np.sum((pi * np.sin(esti_music) - pi * np.sin(tarDOA*derad))**2)/trail                  # 计算方差
SNR_Linear = 10**(SNR/10)                                                 # 将对数SNR转换成线性格式
VARCalc_snr = 6/(N*M*(M-1)*(M+1)*SNR_Linear)

fig = plt.figure(figsize = (8, 12), constrained_layout = True)
ax1 = fig.add_subplot(211, )
ax1.plot(SNR, VAREsti_snr, color = 'b', label = 'Simu')
ax1.plot(SNR, VARCalc_snr, color = 'k', label = 'Theo')
ax1.set_xlabel('SNR(dB)')
ax1.set_ylabel('CRB')
ax1.set_title(f'M = {M}, K = {K}, N = {N},')
ax1.legend()

ax2 = fig.add_subplot(212, )
ax2.plot(SNR, 10*np.log10(VAREsti_snr), color = 'b', label = 'Simu')
ax2.plot(SNR, 10*np.log10(VARCalc_snr), color = 'k', label = 'Theo')
ax2.set_xlabel('SNR(dB)')
ax2.set_ylabel('CRB')
ax2.set_title(f'M = {M}, K = {K}, N = {N}, 10log10')
ax2.legend()

plt.show()
plt.close()





























































