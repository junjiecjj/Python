#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 11:03:58 2026

@author: jack
"""

"""Plot the theoretical ACF curves already stored in memory."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 22                     # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22                # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22                # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18               # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18               # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False         # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]            # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300                 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2                # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6               # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22
np.random.seed(42)

#%% 画图
x_PACF = np.arange(-N//2, N//2)
x_AACF = np.arange(-N+1, N)

fig, axs = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True, gridspec_kw={'height_ratios':[1, 1]})

#%% 上面的子图：16QAM with CP
axs[0].plot(x_PACF, 10*np.log10(TheoAvePACF_SC), color='#F65314', linestyle='--', linewidth=1, label='CP-SC')
axs[0].plot(x_PACF, 10*np.log10(TheoAvePACF_CDMA), color='#00A1F1', linestyle='--', linewidth=1, label='CP-CDMA')
axs[0].plot(x_PACF, 10*np.log10(TheoAvePACF_OTFS), color='#05C349', linestyle='-', linewidth=1, label='CP-OTFS')
axs[0].plot(x_PACF, 10*np.log10(TheoAvePACF_AFDM), color='#000000', linestyle=':', linewidth=2, label='CP-AFDM')
axs[0].plot(x_PACF, 10*np.log10(TheoAvePACF_OFDM), color='#8A2BE2', linestyle='-', linewidth=1, label='CP-OFDM')

font1 = {'family':'Times New Roman', 'style':'normal', 'size':16}
font1 = FontProperties(family='Times New Roman', style='normal', size=16)
legend1 = axs[0].legend(loc='best', borderaxespad=0, edgecolor='black', labelspacing=0.2, prop=font1)
frame1 = legend1.get_frame()
frame1.set_alpha(1)
frame1.set_facecolor('none')

bw = 2
axs[0].spines['bottom'].set_linewidth(bw)
axs[0].spines['left'].set_linewidth(bw)
axs[0].spines['right'].set_linewidth(bw)
axs[0].spines['top'].set_linewidth(bw)
# axs[0].set_ylabel(r'Ambiguity Level (dB)')
axs[0].set_title(r'16QAM with CP')
axs[0].set_xlim([-64, 64])
axs[0].set_ylim([-30, 0])
axs[0].set_xticks(np.arange(-64, 65, 16))
axs[0].set_yticks(np.arange(-30, 1, 10))
axs[0].tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=bw)
labels = axs[0].get_xticklabels()+axs[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(18) for label in labels]
axs[0].grid(linestyle=(0, (5, 10)), linewidth=0.5)

#%% 下面的子图：16QAM without CP
axs[1].plot(x_AACF, 10*np.log10(TheoAveAACF_SC), color='#F65314', linestyle='--', linewidth=1, label='SC')
axs[1].plot(x_AACF, 10*np.log10(TheoAveAACF_CDMA), color='#00A1F1', linestyle='--', linewidth=1, label='CDMA')
axs[1].plot(x_AACF, 10*np.log10(TheoAveAACF_OTFS), color='#05C349', linestyle='-', linewidth=1, label='OTFS')
axs[1].plot(x_AACF, 10*np.log10(TheoAveAACF_AFDM), color='#000000', linestyle=':', linewidth=2, label='AFDM')
axs[1].plot(x_AACF, 10*np.log10(TheoAveAACF_OFDM), color='#8A2BE2', linestyle='-', linewidth=1, label='OFDM')

legend2 = axs[1].legend(loc='best', borderaxespad=0, edgecolor='black', labelspacing=0.2, prop=font1)
frame2 = legend2.get_frame()
frame2.set_alpha(1)
frame2.set_facecolor('none')

axs[1].spines['bottom'].set_linewidth(bw)
axs[1].spines['left'].set_linewidth(bw)
axs[1].spines['right'].set_linewidth(bw)
axs[1].spines['top'].set_linewidth(bw)
axs[1].set_xlabel(r'Delay Index')
# axs[1].set_ylabel(r'Ambiguity Level (dB)')
axs[1].set_title(r'16QAM without CP')
axs[1].set_xlim([-128, 128])
axs[1].set_ylim([-40, 0])
axs[1].set_xticks(np.arange(-128, 129, 32))
axs[1].set_yticks(np.arange(-40, 1, 10))
axs[1].tick_params(direction='in', axis='both', top=True, right=True, labelsize=16, width=bw)
labels = axs[1].get_xticklabels()+axs[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[label.set_fontsize(18) for label in labels]
axs[1].grid(linestyle=(0, (5, 10)), linewidth=0.5)


fig.supylabel(r'Ambiguity Level (dB)', y=0.5, fontsize=18, fontname='Times New Roman')
plt.savefig("Fig_2_4_TIT_With_Without_CP.pdf" )
plt.show()
plt.close()





