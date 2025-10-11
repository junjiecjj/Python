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
afmag = np.zeros((len(alpha), N))

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

# 绘图1 - 修正维度问题
fig1 = plt.figure(1, figsize=(12, 8))
ax1 = fig1.add_subplot(111, projection='3d')

# MATLAB: mesh(doppler1,delay1,mag1.')
# 所以: X=doppler1, Y=delay1, Z=mag1转置
# 但meshgrid需要维度匹配
X, Y = np.meshgrid(doppler1, delay1)  # X形状: (len(delay1), len(doppler1))
Z = mag1.T  # Z形状: (len(delay1), len(doppler1))

print(f"X形状: {X.shape}, Y形状: {Y.shape}, Z形状: {Z.shape}")

surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
fig1.colorbar(surf, ax=ax1, shrink=0.6)
ax1.set_xlabel('Doppler (Hz)')
ax1.set_ylabel('Delay (sec)')
ax1.set_zlabel('Level')
ax1.set_title(f'WAF of {type_name[type_index-1]} Signal')
fig1.patch.set_facecolor('white')

# 绘图2 - 等高线图
fig2 = plt.figure(2, figsize=(10, 10))
# MATLAB: contour(delay1,doppler1,mag1)
plt.contour(Y, X, mag1.T)
plt.grid(True)
plt.xlabel('Delay (Sec)')
plt.ylabel('Doppler (Hz)')
plt.title('Contour of AF')

# 绘图3 - 零延迟和零多普勒切面
fig3 = plt.figure(3, figsize=(12, 8))

# 零延迟切面
plt.subplot(211)
zero_delay_idx = len(indext) // 2
plt.plot(doppler1, mag1[:, zero_delay_idx], 'b', linewidth=1.5)
plt.xlabel('Doppler (Hz)')
plt.ylabel('Amp')
plt.title('Zero Delay')
plt.grid(True)

# 零多普勒切面
plt.subplot(212)
zero_doppler_idx = len(indexf) // 2
plt.plot(delay1, mag1[zero_doppler_idx, :], 'b', linewidth=1.5)
plt.xlabel('Delay (sec)')
plt.ylabel('Amp')
plt.title('Zero Doppler')
plt.grid(True)

plt.tight_layout()
plt.show()

print("所有图形绘制完成")










