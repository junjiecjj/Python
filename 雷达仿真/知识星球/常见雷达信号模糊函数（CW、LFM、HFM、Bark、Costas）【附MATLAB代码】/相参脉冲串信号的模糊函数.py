#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 20:48:54 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzg3ODkwOTgyMw==&mid=2247488206&idx=1&sn=2ab04f20552319c7e30f782cf36d04e9&chksm=ce00efdd88a347949bd1d470099daba8b7dc53520ef405ec2fabb4ca178f2bd6ca26bf348f46&mpshare=1&scene=1&srcid=1023ZUmBe1NlGLk8ALg0u5vT&sharer_shareinfo=bd5b270cc7a54f5de83c2617b6202aeb&sharer_shareinfo_first=bd5b270cc7a54f5de83c2617b6202aeb&exportkey=n_ChQIAhIQLacZHWPVswbhtN7fpl4cdBKRAgIE97dBBAEAAAAAAKBgJLKGlGsAAAAOpnltbLcz9gKNyK89dVj0zuM5Ds3wEcTy9viQd%2BWixAMboMzhq%2F5oppKV8SWgCQLq4qJDD1NYrMV6U1ulrvhsu3aWJ6JDC59oYAo2DnBG%2B4rQCqDXYhKOTB7Kaj2er4VSQA7aXZfW5EdwJSAjQryQS%2FAGaI8dHZ8pIdB7AcepAw9WFOtjiuYcxxtR1W%2B7cHqmyGZ96e2sRrp9HHueJ4K3VgImH4xcRsntVwLaald15zEQbFjNXNwu18I2iEHDCUE%2BX%2BfmEMLvnNFIdjKe7NQkQKn%2BRMfG4VOIYJTT4lXjtzsvgfps6f5RxAnz74%2BvFBKl2hqjaM2OXU9R8A%3D%3D&acctmode=0&pass_ticket=EdBFqlk4MRirQdXpzcmKN2MSBvpAunId6YCsmTw4fTZNIytaY2xDyjLbS%2BkMRll3&wx_header=0#rd

仿真的结果，并不是按照公式实现
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# def uniform_pulse_train_analysis():
"""
均匀脉冲串信号分析
"""
# 1. 基本参数
Np = 6           # 脉冲个数
Tp = 1e-6        # 脉宽 1 us
T = 4e-6         # 周期 4 us
fs = 10e6        # 采样率 10 MHz
t_total = Np * T  # 总时长
t = np.arange(0, t_total, 1/fs)

# 2. 构造均匀脉冲串
s = np.zeros(len(t))
for n in range(Np):
    idx = (t >= n*T) & (t < n*T + Tp)
    s[idx] = 1
s = s.reshape(-1, 1)  # 列向量

# 3. 时域图
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t * 1e6, s, linewidth=1.2)
ax.set_xlabel('时间 (μs)')
ax.set_ylabel('幅度')
ax.set_title('时域图')
ax.grid(True)
ax.set_xlim([t[0]*1e6, t[-1]*1e6])
plt.show()
plt.close()

# 4. 频谱图 (FFT)
N = len(s)
f = np.arange(-N/2, N/2) * (fs / N)
S = np.fft.fftshift(np.fft.fft(s.flatten()))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(f / 1e6, np.abs(S) / np.max(np.abs(S)))
ax.set_xlabel('频率 (MHz)')
ax.set_ylabel('幅度')
ax.set_title('均匀脉冲串信号频谱图')
ax.grid(True)
ax.set_xlim([-20, 20])
plt.show()
plt.close()

# 5. 自定义模糊函数
# 参数
delay_max = 3 * T           # 最大延迟
doppler_max = 1e6           # 最大多普勒
Nd = 256                    # 延迟点数
Nf = 256                    # 多普勒点数
delay_axis = np.linspace(-delay_max, delay_max, Nd)
doppler_axis = np.linspace(-doppler_max, doppler_max, Nf)

# 预分配
af = np.zeros((Nf, Nd))

# 归一化
s_norm = s.flatten() / np.linalg.norm(s)

# 计算模糊函数
for k in range(Nf):
    fd = doppler_axis[k]
    # 构造复指数
    doppler_vec = np.exp(1j * 2 * np.pi * fd * t)
    for m in range(Nd):
        tau = delay_axis[m]
        delay_samples = int(round(tau * fs))
        if abs(delay_samples) < len(s):
            if delay_samples >= 0:
                s_tau = np.concatenate([np.zeros(delay_samples), s_norm[:-delay_samples] if delay_samples > 0 else s_norm])
            else:
                s_tau = np.concatenate([s_norm[-delay_samples:], np.zeros(-delay_samples)])
        else:
            s_tau = np.zeros_like(s_norm)

        af[k, m] = np.abs(np.dot(s_norm, s_tau * doppler_vec))

# 6. 模糊图 (2D)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(af / np.max(af), extent=[delay_axis[0]*1e6, delay_axis[-1]*1e6, doppler_axis[0]/1e3, doppler_axis[-1]/1e3], aspect='auto', origin='lower', cmap='jet')
plt.colorbar(im, ax=ax)
ax.set_xlabel('τ(μs)')
ax.set_ylabel('fd (kHz)')
ax.set_title('模糊图')
plt.show()
plt.close()

# 7. 模糊度图 (3D)
fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(delay_axis*1e6, doppler_axis/1e3)
cbar = ax.plot_surface(X, Y, af/np.max(af), rstride=2, cstride=2, cmap=plt.get_cmap('jet'))
# plt.colorbar(cbar)
ax.set_xlabel('τ(μs)')
ax.set_ylabel('fd (kHz)')
ax.set_zlabel('幅度')
ax.set_title('模糊度')
ax.grid(False)
ax.set_proj_type('ortho')
plt.show()
plt.close()

# 8. 速度模糊图 (零延迟切面)
idx_zero_delay = np.argmin(np.abs(delay_axis))
doppler_cut = af[:, idx_zero_delay]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(doppler_axis/1e3, doppler_cut/np.max(doppler_cut))
ax.set_xlabel('fd (kHz)')
ax.set_ylabel('幅度')
ax.set_title('速度模糊图')
ax.grid(True)
plt.show()
plt.close()

# 9. 距离模糊图 (零多普勒切面)
idx_zero_doppler = np.argmin(np.abs(doppler_axis))
delay_cut = af[idx_zero_doppler, :]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(delay_axis*1e6, delay_cut/np.max(delay_cut))
ax.set_xlabel('τ(μs)')
ax.set_ylabel('幅度')
ax.set_title('距离模糊图')
ax.grid(True)
plt.show()
plt.close()












