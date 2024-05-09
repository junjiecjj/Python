#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:27:00 2024

@author: jack

https://zhuanlan.zhihu.com/p/151422677

"""

"""
矩形波单边带调制与解调的主程序代码
"""


import numpy as np
import scipy.fftpack as fftp
import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import numpy as np
import torch
from matplotlib.font_manager import FontProperties
import Filter
# mpl.rcParams['font.sans-serif'] = ['KaiTi']   # 保证正常显示中文
# mpl.rcParams['font.serif'] = ['KaiTi']        # 保证正常显示中文
# mpl.rcParams['axes.unicode_minus'] = False    # 保证负号正常显示
config = {
    "font.family": "serif",  # 使用衬线体
    "font.serif": ["SimSun"],  # 全局默认使用衬线宋体
    "font.size": 12,  # 五号，10.5磅
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",  # 设置 LaTeX 字体，stix 近似于 Times 字体
}
plt.rcParams.update(config)



dt = 0.001                                    # 时间域采样间隔
Fs = 1/dt                                     # 采样率
T = 4                                         # 矩形波的周期
nT = 10                                       # 总的周期数
nPT = int(T/dt+0.5)                           # 每个周期的样点数
n = nT*nPT                                    # 总样点数
alpha = 0.5                                   # 矩形波的占空比
t = np.arange(dt, n*dt+dt, dt)
f = np.zeros((n, 1), dtype=np.float64)
A = 1
for i in range(0, nT, 1):
    f[(i+1)*nPT-int(alpha*nPT):(i+1)*nPT] = A   # 得到矩形波


fc = 100  # 载波的频率
g1 = np.cos(2*np.pi*fc*t).reshape(n, 1)
g2 = np.sin(2*np.pi*fc*t).reshape(n, 1)
hf = fftp.hilbert(f.reshape(n, ))        # 矩形波的希尔伯特变换

s = 0.5*f*g1-0.5*hf.reshape(n, 1)*g2     # 已调信号
s_dm = s*g1                              # 解调
cutoff = 60                              # 截至频率
order = 6                                # 滤波器的阶数
s_dm = 4*Filter.butter_lowpass_filtfilt(s_dm, cutoff, Fs, order)  # 作低通滤波

f_spec = np.fft.fft(f, n, 0)             # 作傅立叶变换
freq = np.fft.fftfreq(n, dt).reshape(n, 1)  # 得到每点的频率
f_spec_amp = abs(f_spec)/(n/2)              # 对频谱的振幅作归一化
f_spec_amp[0] = f_spec_amp[0]/2
f_spec_amp[int(n/2)+1] = f_spec_amp[int(n/2)+1]/2

s_spec = np.fft.fft(s, n, 0)
s_spec_amp = abs(s_spec)/(n/2)
s_spec_amp[0] = s_spec_amp[0]/2
s_spec_amp[int(n/2)+1] = s_spec_amp[int(n/2)+1]/2

# 绘制一个周期内的矩形波
plt.figure(num=1, constrained_layout = True)
plt.plot(t[0:nPT], f[0:nPT], 'b-')
plt.xlim(0, T)
plt.ylim(-0.5, 1.5)
plt.xlabel('时间/(s)')
plt.ylabel('振幅')
plt.title('一个周期内的矩形波信号')
# plt.savefig('图1.jpg', dpi=600)
plt.show()

# 绘制已调信号
plt.figure(num=2, constrained_layout = True)
plt.plot(t[0:int(10/fc/dt+0.5)], s[0:int(10/fc/dt+0.5)], 'b-')
plt.xlim(0, 10/fc)
plt.ylim(-1.5, 1.5)
plt.xlabel('时间/(s)')
plt.ylabel('振幅')
plt.title('已调信号')
# plt.savefig('图2.jpg', dpi=600)
plt.show()

# 绘制基带信号与已调信号的频谱
plt.figure(num=3, constrained_layout = True)
plt.plot(freq, f_spec_amp, 'b-', freq, s_spec_amp, 'r-')
plt.xlim(min(freq), max(freq))
plt.legend(('基带信号的频谱', '已调信号的频谱'), loc='upper right')
plt.xlabel('频率/(Hz)')
plt.ylabel('振幅')
plt.title('基带信号和已调信号的归一化频谱')
# plt.savefig('图3.jpg', dpi=600)
plt.show()

# 绘制基带信号与解调信号
plt.figure(num=4, constrained_layout = True)
plt.plot(t[0:nPT], f[0:nPT], 'b-', t[0:nPT], s_dm[0:nPT], 'r-')
plt.legend(('基带信号', '解调信号'), loc='upper right')
plt.xlim(0, T)
plt.ylim(-0.5, 1.5)
plt.xlabel('时间/(s)')
plt.ylabel('振幅')
plt.title('基带信号和解调信号')
# plt.savefig('图4.jpg', dpi=600)
plt.show()
