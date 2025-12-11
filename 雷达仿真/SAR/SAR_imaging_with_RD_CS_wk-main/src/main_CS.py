#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 21:00:03 2025

@author: jack

https://zhuanlan.zhihu.com/p/408253349
https://blog.csdn.net/weixin_41476562/article/details/136006008


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
from skimage import exposure
import warnings
warnings.filterwarnings('ignore')


# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300      # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22


# 数据读取
# 加载数据
echo1_data = loadmat('CDdata1.mat')
echo2_data = loadmat('CDdata2.mat')
para_data = loadmat('CD_run_params.mat')

# 提取回波数据（假设.mat文件中的变量名是'echo1'和'echo2'）
echo1 = echo1_data['data']  # 根据实际变量名调整
echo2 = echo2_data['data']  # 根据实际变量名调整

# 将回波拼装在一起
echo = np.vstack((echo1, echo2)).astype(np.complex128)

# 加载参数
para = para_data
Fr = para['Fr'].item()   # 距离向采样率
Fa = para['PRF'].item()  # 方位向采样率
f0 = para['f0'].item()   # 中心频率
Tr = para['Tr'].item()   # 脉冲持续时间
R0 = para['R0'].item()   # 最近点斜距
Kr = -para['Kr'].item()  # 线性调频率
c = para['c'].item()     # 光速

# 以下参数来自课本附录A
Vr = 7062       # 等效雷达速度
Ka = 1733       # 方位向调频率
f_nc = -6900    # 多普勒中心频率

# 图像填充
# 计算参数
Na, Nr = echo.shape
# 按照全尺寸对图像进行补零
pad_na = round(Na / 6)
pad_nr = round(Nr / 3)
echo = np.pad(echo, ((pad_na, pad_na), (pad_nr, pad_nr)), mode='constant')
# 计算参数
Na, Nr = echo.shape

# 轴产生
# 距离向时间轴及频率轴
tr_axis = 2 * R0 / c + (np.arange(-Nr/2, Nr/2)) / Fr  # 距离向时间轴
fr_gap = Fr / Nr
fr_axis = np.fft.fftshift(np.arange(-Nr/2, Nr/2)) * fr_gap  # 距离向频率轴

# 方位向时间轴及频率轴
ta_axis = (np.arange(-Na/2, Na/2)) / Fa  # 方位向时间轴
ta_gap = Fa / Na
fa_axis = f_nc + np.fft.fftshift(np.arange(-Na/2, Na/2)) * ta_gap  # 方位向频率轴
# 方位向对应纵轴，应该转置成列向量
ta_axis = ta_axis.reshape(-1, 1)
fa_axis = fa_axis.reshape(-1, 1)

# 第一步 相位相乘
# 方位向下变频
echo = echo * np.exp(-2j * np.pi * f_nc * ta_axis)
# 方位向傅里叶变换
echo_fft_a = np.fft.fft(echo, axis=0)
# 计算徙动参数
D_fa_Vr = np.sqrt(1 - c**2 * fa_axis**2 / (4 * Vr**2 * f0**2))  # 关于方位向频率的徙动参数矩阵
D_fnc_Vr = np.sqrt(1 - c**2 * f_nc**2 / (4 * Vr**2 * f0**2))    # 关于参考多普勒中心的徙动参数
R0_var = c * tr_axis / 2  # 随距离变化的最近点斜距

# 改变后的距离向调频率
Km = Kr / (1 - Kr * (c * R0_var * fa_axis**2 / (2 * Vr**2 * f0**3 * D_fa_Vr**3)))

# 计算变标方程
tao_new = tr_axis - 2 * R0 / (c * D_fa_Vr)  # 新的距离向时间
Ssc = np.exp(1j * np.pi * Km * (D_fnc_Vr / D_fa_Vr - 1) * (tao_new**2))  # 变标方程
# 补余RCMC中的Chirp Scaling操作
echo_s1 = echo_fft_a * Ssc

# 第二步 相位相乘
# 距离向傅里叶变换
echo_s2 = np.fft.fft(echo_s1, axis=1)
# 补偿第2项
echo_d2_mf = np.exp(1j * np.pi * D_fa_Vr * (fr_axis**2) / (Km * D_fnc_Vr))
# 补偿第4项
echo_d4_mf = np.exp(4j * np.pi / c * R0 * (1 / D_fa_Vr - 1 / D_fnc_Vr) * fr_axis)
# 参考函数相乘用于距离压缩、SRC和一致性RCMC
echo_s3 = echo_s2 * echo_d2_mf * echo_d4_mf

# 第三步 相位相乘
# 距离向逆傅里叶变换
echo_s4 = np.fft.ifft(echo_s3, axis=1)
# 方位向匹配滤波
echo_d1_mf = np.exp(4j * np.pi * R0_var * f0 / c * D_fa_Vr)  # 方位向匹配滤波器
# 变标相位矫正
echo_d3_mf = np.exp(-4j * np.pi * Km / c**2 * (1 - D_fa_Vr / D_fnc_Vr) * (R0_var / D_fa_Vr - R0 / D_fa_Vr)**2)
# 方位向逆傅里叶变换
echo_s5 = np.fft.ifft(echo_s4 * echo_d1_mf * echo_d3_mf, axis=0)

# 数据最后的矫正
# 根据实际观感，方位向做合适的循环位移
echo_s5 = np.roll(echo_s5, -3328, axis=0)
# 上下镜像
echo_s6 = np.flipud(echo_s5)
# 取模
echo_s7 = np.abs(echo_s6)

# 数据可视化
# 绘制直方图
plt.figure()
plt.hist(echo_s7.flatten(), bins=50)
plt.show()

# 根据直方图结果做饱和处理
saturation = 50
echo_s7[echo_s7 > saturation] = saturation

# 绘制处理结果热力图
plt.figure()
plt.imshow(echo_s7, extent=[tr_axis[0]*c, tr_axis[-1]*c, ta_axis[-1]*c, ta_axis[0]*c], aspect='auto')
plt.title('处理结果(CS算法)')
plt.colorbar()
plt.show()

# 绘制处理结果灰度图
# 做一些图像处理。。。
echo_res = echo_s7 / saturation

# 直方图均衡
echo_res = exposure.equalize_adapthist(echo_res, clip_limit=0.004, nbins=256)

plt.figure()
plt.imshow(echo_res, cmap='gray')
plt.axis('off')
plt.show()
