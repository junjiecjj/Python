#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 21:15:18 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
from skimage import exposure
import warnings
from tqdm import tqdm
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

# 第一步 二维傅里叶变换
# 方位向下变频
echo = echo * np.exp(-2j * np.pi * f_nc * ta_axis)
# 二维傅里叶变换
echo_s1 = np.fft.fft2(echo)

# 第二步 参考函数相乘(一致压缩)
# 生成参考函数
theta_ft_fa = (4 * np.pi * R0 / c *
               np.sqrt((f0 + fr_axis)**2 - (c**2 / (4 * Vr**2)) * fa_axis**2) +
               np.pi / Kr * fr_axis**2)
theta_ft_fa = np.exp(1j * theta_ft_fa)
# 一致压缩
echo_s2 = echo_s1 * theta_ft_fa

# 第三步 在距离域进行Stolt插值操作(补余压缩)
# 计算映射后的距离向频率
fr_new_mtx = np.sqrt((f0 + fr_axis)**2 - (c**2 / (4 * Vr**2)) * fa_axis**2) - f0

# Stolt映射
echo_s3 = np.zeros((Na, Nr), dtype=np.complex128)

# 使用tqdm显示进度条
for i in tqdm(range(Na), desc='Stolt映射中'):
    # 创建插值函数（使用三次样条插值近似MATLAB的spline）
    interp_func = interp1d(fr_axis, echo_s2[i, :], kind='cubic',
                          bounds_error=False, fill_value=0+0j)
    # 进行插值
    echo_s3[i, :] = interp_func(fr_new_mtx[i, :])

# 第四步 二维逆傅里叶变换
echo_s4 = np.fft.ifft2(echo_s3)

# 第五步 图像纠正
echo_s5 = np.roll(echo_s4, -1800, axis=1)  # 距离向循环位移
echo_s5 = np.roll(echo_s5, -3365, axis=0)  # 方位向循环位移
echo_s5 = np.flipud(echo_s5)  # 上下镜像

# 画图
saturation = 50
echo_s6 = np.abs(echo_s5)
echo_s6[echo_s6 > saturation] = saturation

plt.figure(figsize=(12, 8))
plt.imshow(echo_s6, extent=[tr_axis[0]*c, tr_axis[-1]*c, ta_axis[-1]*c, ta_axis[0]*c],
           aspect='auto', cmap='hot')
plt.title('ωk处理结果(精确版本)')
plt.colorbar()
plt.xlabel('距离向')
plt.ylabel('方位向')
plt.show()

# 绘制处理结果灰度图
echo_res = echo_s6 / saturation

# 直方图均衡
echo_res = exposure.equalize_adapthist(echo_res, clip_limit=0.004, nbins=256)

plt.figure(figsize=(12, 8))
plt.imshow(echo_res, cmap='gray')
plt.axis('off')
plt.title('ωk算法成像结果（直方图均衡后）')
plt.show()
