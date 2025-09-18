#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 21:01:43 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage import exposure
import warnings
warnings.filterwarnings('ignore')

# 数据读取
# 加载数据
echo1_data = loadmat('CDdata1.mat')
echo2_data = loadmat('CDdata2.mat')
para_data = loadmat('CD_run_params.mat')

# 提取回波数据（根据实际变量名调整）
echo1 = echo1_data['data']
echo2 = echo2_data['data']

# 将回波拼装在一起
echo = np.vstack((echo1, echo2)).astype(np.complex128)

# 加载参数
para = para_data
Fr = para['Fr'].item() if 'Fr' in para else para[list(para.keys())[-1]]['Fr'].item()
Fa = para['PRF'].item() if 'PRF' in para else para[list(para.keys())[-1]]['PRF'].item()
f0 = para['f0'].item() if 'f0' in para else para[list(para.keys())[-1]]['f0'].item()
Tr = para['Tr'].item() if 'Tr' in para else para[list(para.keys())[-1]]['Tr'].item()
R0 = para['R0'].item() if 'R0' in para else para[list(para.keys())[-1]]['R0'].item()
Kr = -para['Kr'].item() if 'Kr' in para else -para[list(para.keys())[-1]]['Kr'].item()
c = para['c'].item() if 'c' in para else para[list(para.keys())[-1]]['c'].item()

# 以下参数来自课本附录A
Vr = 7062       # 等效雷达速度
Ka = 1733       # 方位向调频率
f_nc = -6900    # 多普勒中心频率
lamda = c / f0  # 波长

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

# 第一步 距离压缩
# 方位向下变频
echo_s1 = echo * np.exp(-2j * np.pi * f_nc * ta_axis)
# 距离向傅里叶变换
echo_s1 = np.fft.fft(echo_s1, axis=1)
# 距离向距离压缩滤波器
echo_d1_mf = np.exp(1j * np.pi / Kr * fr_axis**2)
# 距离向匹配滤波
echo_s1 = echo_s1 * echo_d1_mf

# 第二步 方位向傅里叶变换&距离徙动矫正
# 方位向傅里叶变换
echo_s2 = np.fft.fft(echo_s1, axis=0)
# 计算徙动因子
D = lamda**2 * R0 / (8 * Vr**2) * fa_axis**2
G = np.exp(4j * np.pi / c * fr_axis * D)
# 校正
echo_s2 = echo_s2 * G
# 滚回距离多普勒域
echo_s2 = np.fft.ifft(echo_s2, axis=1)

# 第三步 方位压缩
# 方位向滤波器
echo_d3_mf = np.exp(-1j * np.pi / Ka * fa_axis**2)
# 方位向脉冲压缩
echo_s3 = echo_s2 * echo_d3_mf
# 方位向逆傅里叶变换
echo_s3 = np.fft.ifft(echo_s3, axis=0)

# 数据最后的矫正
# 根据实际观感，方位向做合适的循环位移
echo_s4 = np.roll(np.abs(echo_s3), -3193, axis=0)
# 上下镜像
echo_s4 = np.flipud(echo_s4)
echo_s5 = np.abs(echo_s4)
saturation = 50
echo_s5[echo_s5 > saturation] = saturation

# 成像
# 绘制处理结果热力图
plt.figure(figsize=(10, 8))
plt.imshow(echo_s5, extent=[tr_axis[0]*c, tr_axis[-1]*c, ta_axis[-1]*c, ta_axis[0]*c],
           aspect='auto', cmap='hot')
plt.title('处理结果(RD算法)')
plt.colorbar()
plt.xlabel('距离向')
plt.ylabel('方位向')
plt.show()

# 以灰度图显示
echo_res = echo_s5 / saturation

# 直方图均衡
echo_res = exposure.equalize_adapthist(echo_res, clip_limit=0.004, nbins=256)

plt.figure(figsize=(10, 8))
plt.imshow(echo_res, cmap='gray')
plt.axis('off')
plt.title('RD算法成像结果（直方图均衡后）')
plt.show()
