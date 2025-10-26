#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 11:30:30 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
# from generateDataGaussianWhite import generateDataGaussianWhite
from ca_cfar import *
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

# 清除变量，关闭所有图形，清空命令行
plt.close('all')
np.random.seed(42)  # 设置随机种子以便结果可重现

# 生成一个例子，然后用以下算法处理
# CA CFAR, OS CFAR, SOCA CFAR, GOCA CFAR, S-CFAR, Log CFAR
# 位置在50和55处有目标，50处信噪比10dB，55处信噪比15dB
# 前100单元噪声功率20dB，后100单元噪声功率30dB

num_cell = 200
Pfa = 10**(-5)

# 生成信号
signal1 = generateDataGaussianWhite(100, [50, 55], [35, 40], 20)
signal2 = generateDataGaussianWhite(100, [], [], 30)
signal = np.concatenate([signal1, signal2])

# 将功率转换为dB
signal_db = 10 * np.log10(signal)

# 绘制原始信号
plt.figure(figsize=(12, 8))
plt.plot(range(1, num_cell + 1), signal_db, 'k-', lw = 1, )


# CA CFAR处理
position, threshold, start_cell, stop_cell = cacfar(signal, Pfa, 10, 2)
threshold_db = 10 * np.log10(threshold)
plt.plot(range(start_cell, stop_cell + 1), threshold_db,  label='CA CFAR阈值')

# OS CFAR处理
position, threshold, start_cell, stop_cell = oscfar(signal, Pfa, 10, 2, 15)
threshold_db = 10 * np.log10(threshold)
plt.plot(range(start_cell, stop_cell + 1), threshold_db, label='OS CFAR阈值')

# SOCA CFAR处理
position, threshold, start_cell, stop_cell = socacfar(signal, Pfa, 10, 2)
threshold_db = 10 * np.log10(threshold)
plt.plot(range(start_cell, stop_cell + 1), threshold_db,  label='SOCA CFAR阈值')

# GOCA CFAR处理
position, threshold, start_cell, stop_cell = gocacfar(signal, Pfa, 10, 2)
threshold_db = 10 * np.log10(threshold)
plt.plot(range(start_cell, stop_cell + 1), threshold_db, label='GOCA CFAR阈值')

plt.grid(True)
plt.legend(['信号', 'CA CFAR阈值', 'OS CFAR阈值', 'SOCA CFAR阈值', 'GOCA CFAR阈值'])
plt.xlabel('单元索引')
plt.ylabel('功率 (dB)')
plt.title('不同CFAR算法性能比较')
plt.show()







