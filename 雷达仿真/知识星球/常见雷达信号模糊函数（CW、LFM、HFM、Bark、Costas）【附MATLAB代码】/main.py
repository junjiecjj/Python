#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 16:25:01 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 16  # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 16  # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 16  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 16  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 16  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'  # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'  # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 16
# 完全对应MATLAB代码
T = 0.26
fs = 20000
fc = 1000
B = 1000
t = np.arange(0, T, 1/fs)

type_index = 2
type_name = ['PCW', 'LFM', 'HFM', 'Bark', 'Costas']

# 信号生成
if type_index == 1:
    x = np.exp(1j * 2 * np.pi * fc * t)
elif type_index == 2:
    k = B / T
    f0 = fc - B / 2
    x = np.exp(1j * 2 * np.pi * (f0 * t + k / 2 * t**2))
elif type_index == 3:
    f0 = fc + B / 2
    beta = B / f0 / (fc - B / 2) / T
    x = np.exp(1j * 2 * np.pi / beta * np.log(1 + beta * f0 * t))
elif type_index == 4:
    bark  = np.random.randint(0, 2, size = 100) * 2 - 1
    # [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]  # 固定序列
    Tbark = T / len(bark)
    tbark = np.arange(0, Tbark, 1/fs)
    s = np.zeros(len(bark) * len(tbark), dtype=complex)
    for i in range(len(bark)):
        start_idx = i * len(tbark)
        end_idx = (i + 1) * len(tbark)
        if bark[i] == 1:
            s[start_idx:end_idx] = np.exp(1j * 2 * np.pi * fc * tbark)
        else:
            s[start_idx:end_idx] = np.exp(1j * (2 * np.pi * fc * tbark + np.pi))
    x = np.concatenate([s, np.zeros(len(t) - len(s), dtype=complex)])
elif type_index == 5:
    costas = [2, 4, 8, 5, 10, 9, 7, 3, 6, 1]
    f = fc - B / 2 + (np.array(costas) - 1) * B / (len(costas) - 1)
    Tcostas = T / len(costas)
    tcostas = np.arange(0, Tcostas, 1/fs)
    s = np.zeros(len(costas) * len(tcostas), dtype=complex)
    for i in range(len(costas)):
        start_idx = i * len(tcostas)
        end_idx = (i + 1) * len(tcostas)
        s[start_idx:end_idx] = np.exp(1j * 2 * np.pi * f[i] * tcostas)
    x = np.concatenate([s, np.zeros(len(t) - len(s), dtype=complex)])

print(f"信号长度: {len(x)}")

# 模糊函数计算
re_fs = np.arange(0.9 * fs, 1.1 * fs + 2, 2)
alpha = re_fs / fs             # Doppler ratio, alpha = 1-2*v/c
doppler = (1 - alpha) * fc     # Doppler = 2v/c*fc = (1-alpha)*fc

print(f"alpha数量: {len(alpha)}")

# 计算N_a
min_fs = np.min(re_fs)
N_a = len(signal.resample(x, int(np.ceil(len(x) * fs / min_fs)) ))
N = N_a + len(x) - 1
afmag = np.zeros((len(alpha), N), dtype = complex)

print(f"N_a: {N_a}, N: {N}")
tic = time.time()

for i in range(len(alpha)):
    new_len = int(np.ceil(len(x) * fs / re_fs[i]))
    x_alpha = signal.resample(x, new_len)
    if len(x_alpha) < N_a:
        x_alpha = np.concatenate([x_alpha, np.zeros(N_a - len(x_alpha), dtype = complex)])
    else:
        x_alpha = x_alpha[:N_a]
    x_temp = np.conj(x_alpha[::-1])
    afmag_temp = np.convolve(x_temp, x, mode='full')

    afmag[i, :] = afmag_temp * np.sqrt(alpha[i])

toc = time.time()
print(f"计算完成，耗时: {toc - tic:.2f} 秒")
delay = (np.arange(1, N + 1) - N_a) / fs

tau = 0.2
fd = 100
indext = np.where((delay >= -tau) & (delay <= tau))[0]
indexf = np.where((doppler >= -fd) & (doppler <= fd))[0]
delay1 = delay[indext]
doppler1 = doppler[indexf]

print(f"delay1长度: {len(delay1)}, doppler1长度: {len(doppler1)}")

mag = np.abs(afmag)
mag = mag / np.max(mag)
# mag = 10 * np.log10(mag)
mag1 = mag[np.ix_(indexf, indext)]

print(f"mag1形状: {mag1.shape}")

# 阈值处理
row, col = np.where(mag1 < -100)
mag1[row, col] = -60

##>>>>>  绘图1
X, Y = np.meshgrid(doppler1, delay1)  # X形状: (len(delay1), len(doppler1))
Z = mag1.T  # Z形状: (len(delay1), len(doppler1))

fig = plt.figure(figsize = (8, 8) , constrained_layout = True)
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, rstride = 5, cstride = 5, cmap = plt.get_cmap('jet'))
ax.grid(False)
ax.set_proj_type('ortho')
ax.view_init(azim=-135,    elev = 30)
ax.set_xlabel('Doppler (Hz)')
ax.set_ylabel('Delay (sec)')
ax.set_zlabel('Level')
ax.set_title(f'WAF of {type_name[type_index-1]} Signal')
plt.show()
plt.close()


##>>>>>  绘图2 - 等高线图
levels = np.arange(0.1, 1.1, 0.1)
fig, ax = plt.subplots( figsize = (8, 8))
ax.contour(Y, X, mag1.T, levels = levels, cmap = 'rainbow')

ax.grid(True)
ax.set_xlabel('Delay (Sec)')
ax.set_ylabel('Doppler (Hz)')
ax.set_title('Contour of AF')

plt.show()
plt.close()


##>>>>>  绘图3 - 零延迟和零多普勒切面
fig, axs = plt.subplots(2, 1, figsize = (8, 8), constrained_layout = True)

zero_delay_idx = len(indext) // 2
axs[0].plot(doppler1, mag1[:, zero_delay_idx], 'b', linewidth=1.5)
axs[0].set_xlabel('Doppler (Hz)')
axs[0].set_ylabel('Amp')
axs[0].set_title('Zero Delay')
# axs[0].legend()

zero_doppler_idx = len(indexf) // 2
axs[1].plot(delay1, mag1[zero_doppler_idx, :], 'b', linewidth=1.5)
axs[1].set_xlabel('Delay (sec)')
axs[1].set_ylabel('Amp')
axs[1].set_title('Zero Doppler')
# axs[1].legend()

plt.show()
plt.close()

print("所有图形绘制完成")






































