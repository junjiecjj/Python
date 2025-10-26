#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 12:02:02 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
# from generateDataGaussianWhite import generateDataGaussianWhite
from ca_cfar import *

# 设置随机种子以便结果可重现
np.random.seed(42)

# CA CFAR示例 + 对比CA CFAR与阈值检测性能曲线
MC_num = 1000  # Monte Carlo次数
num_unit = 200
pos_target = 100

noise_power_dB = 20
Pfa = 1e-5

# 一个示例
pos = []
tmpi = 1
echo_power_dB = noise_power_dB + 35

# 生成示例信号
while len(pos) <= 1:
    if tmpi % 1000 == 0:
        print(f"times: {tmpi}")
    signal = generateDataGaussianWhite(num_unit, [pos_target], echo_power_dB, noise_power_dB)
    pos, thres, start_cell, stop_cell = cacfar(signal, Pfa, 10, 2)
    tmpi += 1

# 绘制示例图
signal_db = 10 * np.log10(signal)
thres_db = 10 * np.log10(thres)
# 结果可视化
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
axs.plot(range(1, num_unit + 1), signal_db, 'k-', linewidth=0.5, label = '信号')
axs.plot(range(start_cell, stop_cell + 1), thres_db, 'k--', linewidth=1, label = 'CA CFAR阈值')

# 标记检测到的目标
for p in pos:
    axs.plot(p, 10 * np.log10(signal[p-1]), 'ro', markersize=10, label = '检测到目标')

axs.set_xlabel('距离单元')
axs.set_ylabel('幅度(dB)')
axs.legend()
plt.show()
plt.close()

# Monte Carlo计算性能曲线
SNR_dB = np.linspace(0, 20, 50)
detection_num = np.zeros(len(SNR_dB))
PD = []

for ii in range(len(SNR_dB)):
    print(f"SNR(dB) = {SNR_dB[ii]}")
    echo_power_dB = noise_power_dB + SNR_dB[ii]
    for mc in range(MC_num):
        signal = generateDataGaussianWhite(num_unit, [pos_target], echo_power_dB, noise_power_dB)
        pos, _, _, _ = cacfar(signal, Pfa, 10, 2)
        if pos_target in pos:
            detection_num[ii] += 1

PD = detection_num / MC_num

# 绘制性能曲线
fig, axs = plt.subplots(1, 1, figsize = (8, 6), constrained_layout = True)
axs.plot(SNR_dB, PD, 'k-', linewidth=1, label = 'CA CFAR检测')
axs.set_xlabel('SNR(dB)')
axs.set_ylabel('检测概率P_D')
axs.legend()
plt.show()
plt.close()

