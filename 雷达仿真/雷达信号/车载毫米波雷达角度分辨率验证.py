#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 10:09:51 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzE5ODQ2NTg0NA==&mid=2247483685&idx=1&sn=6b19366e906756c110cfb7b554feb175&chksm=97416233e816c928797168cde46da8779d39e8d4d17898ba5101fa65a8eaf3322e86b4d20cba&mpshare=1&scene=1&srcid=090563AEpuqAba3Chr8xbyAb&sharer_shareinfo=a75664803b3441fa29888090931cc604&sharer_shareinfo_first=a75664803b3441fa29888090931cc604&exportkey=n_ChQIAhIQXmLvt5Dp4WszRPrcM8DfuhKfAgIE97dBBAEAAAAAAJDPETCMXQgAAAAOpnltbLcz9gKNyK89dVj0Inr605OLi1ccPXTYaZ%2BPDKCmR7BsdPhR9hK79tVTk%2FRRhh4cxD70csGWOC%2FAVZPb1W7WZEwsSh3gtGBQgcsLmHtReDndxkqeL4GaX4%2Ba9Hj1sP%2BD%2Fjfu7XozBCV54sTowbUKasG9obRdzW7X7Em9k02lkaNhqdkoRt2f7WCbiqcKOD7F0zkzyoQrCBfLyl3qUyMsMbrGfgw%2FhOnpcRqR1Mz185tRF%2BtJnqK3Gg8ZxKvUTXVT9FG%2B%2FMd%2FOWWs1uhONhyZlq75T%2Fvmc1Ts0XtsUydyHEDLTtOY5aNSIAJsQlzVJ8yDYLN9SnSMXQSXoQ3PY5FTnvI7LwnD&acctmode=0&pass_ticket=fwFLhukBPX2SBQcybSkWvSCzJm6e1SsZlEThxspgicv5ZiXtM5GnZ4KkNeS%2Bj6gW&wx_header=0#rd


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D

# 关闭所有图形窗口
plt.close('all')

# 1. 阵列参数配置
# 阵列几何配置（所有天线单元的x坐标相同，形成y-z平面阵列）
sub3_pos = np.array([
    [0, 1.5, 1.2],   # 天线1 [x, y, z] (λ)
    [0, 4.25, 1.2],
    [0, 4.75, 1.2],
    [0, 7.5, 1.2],
    [0, 10.5, 1.2],
    [0, 13.25, 1.2],
    [0, 13.75, 1.2],
    [0, 16.5, 1.2]
])

N = sub3_pos.shape[0]      # 阵元总数
fc = 76.5e9               # 工作频率76.5GHz
c = 3e8                   # 光速(m/s)
lambda_val = c / fc       # 波长(m)
k = 2 * np.pi / lambda_val  # 波数

# 将归一化位置转换为实际坐标（单位：米）
x_actual = sub3_pos[:, 0] * lambda_val
y_actual = sub3_pos[:, 1] * lambda_val
z_actual = sub3_pos[:, 2] * lambda_val

# 2. 信号参数设置
# 目标角度定义 [方位角φ, 俯仰角θ]（度）
target_angles = np.array([
    [4, 30],     # 目标1: φ=5°, θ=30°
    [0, 30]      # 目标2: φ=0°, θ=30°
])
n_targets = target_angles.shape[0]  # 目标数量

snr_db = 20              # 信噪比(dB)
noise_flag = True

# 3. 扫描角度范围设置
# 方位角扫描范围（φ从-10°到10°，步长0.1°）
phi_deg = np.arange(-10, 10.1, 0.1)
# 俯仰角扫描范围（θ从25°到35°，步长0.1°）
theta_deg = np.arange(25, 35.1, 0.1)

# 生成扫描角度网格
PHI, THETA = np.meshgrid(phi_deg, theta_deg)
phi_rad = np.deg2rad(PHI)
theta_rad = np.deg2rad(THETA)

# 4. 生成目标导向矢量
target_steering_vectors = np.zeros((N, n_targets), dtype=complex)

for t in range(n_targets):
    # 获取当前目标的方位角和俯仰角（转换为弧度）
    phi = np.deg2rad(target_angles[t, 0])
    theta = np.deg2rad(target_angles[t, 1])

    # 计算波达方向单位向量
    u = np.array([
        np.cos(phi) * np.cos(theta),
        np.sin(phi) * np.cos(theta),
        np.sin(theta)
    ])

    # 计算各阵元相对于参考点的相位延迟
    phase_shift = k * (x_actual * u[0] + y_actual * u[1] + z_actual * u[2])

    # 生成导向矢量
    target_steering_vectors[:, t] = np.exp(1j * phase_shift)

# 5. 生成接收信号
signal_amplitude = 1  # 信号幅度
signals = signal_amplitude * np.sum(target_steering_vectors, axis=1)

# 添加高斯白噪声
if noise_flag:
    # 计算噪声功率
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
    # 将噪声添加到信号中
    signals = signals + noise

# 6. 波束形成扫描计算
beam_power = np.zeros(PHI.shape)

# 遍历所有扫描角度
for i in range(PHI.size):
    phi = phi_rad.flat[i]
    theta = theta_rad.flat[i]

    u_scan = np.array([
        np.cos(phi) * np.cos(theta),
        np.sin(phi) * np.cos(theta),
        np.sin(theta)
    ])

    # 计算各阵元的相位延迟
    phase_shift = k * (x_actual * u_scan[0] + y_actual * u_scan[1] + z_actual * u_scan[2])

    # 形成波束权重矢量
    w = np.exp(1j * phase_shift)

    # 计算波束形成输出功率
    beam_power.flat[i] = np.abs(w.conj().T @ signals) ** 2

# 功率归一化（将最大功率设为0dB）
beam_power_normalized = beam_power / np.max(beam_power)
beam_power_dB = 10 * np.log10(beam_power_normalized)

# 7. 三维方向图可视化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(PHI, THETA, beam_power_dB, cmap='jet', edgecolor='none', alpha=0.8)
ax.set_xlabel('方位角φ (°)')
ax.set_ylabel('俯仰角θ (°)')
ax.set_zlabel('归一化功率 (dB)')
ax.set_title('y-z平面阵列三维波束形成方向图')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)

# 标记目标位置
for t in range(n_targets):
    ax.scatter(target_angles[t, 0], target_angles[t, 1], np.max(beam_power_dB),
              c='red', marker='o', s=100, label='目标位置' if t == 0 else "")

ax.view_init(elev=30, azim=45)
plt.legend()
plt.tight_layout()
plt.show()

# 8. 方位角切面分析
theta_fix = 30
theta_idx = np.argmin(np.abs(theta_deg - theta_fix))
slice_power = beam_power_dB[theta_idx, :]

# 峰值检测（寻找高于最大值-3dB的峰值）
peaks, peak_locs = find_peaks(slice_power, height=np.max(slice_power)-3)

# 计算实际分辨率（如果检测到至少两个峰值）
if len(peak_locs['peak_heights']) >= 2:
    peak_phi = phi_deg[peak_locs['peak_heights'].argsort()[-2:]]
    actual_resolution = np.abs(peak_phi[0] - peak_phi[1])
    print(f'实际方位角分辨率: {actual_resolution:.2f}°')
else:
    print('警告：未检测到足够多的峰值。')

# 显示理论分辨率（示例值）
resolution_theoretical = 2.92  # 理论分辨率（基于阵列孔径）
print(f'理论方位角分辨率: {resolution_theoretical:.2f}°')

# 9. 绘制方位角切面图
plt.figure(figsize=(10, 6))
plt.plot(phi_deg, slice_power, 'b-', linewidth=1.5)
plt.xlabel('方位角φ (°)')
plt.ylabel('归一化功率 (dB)')
plt.title(f'俯仰角θ={theta_fix}°时的方位角功率谱')
plt.grid(True)

# 标记目标位置
target_phi = target_angles[:, 0]
plt.plot(target_phi, np.interp(target_phi, phi_deg, slice_power), 'r*', markersize=10, label='目标方位角')

# 标记-3dB点
plt.axhline(y=-3, color='k', linestyle='--', label='-3dB参考线')
plt.text(np.mean(phi_deg), -2.8, '-3dB参考线', horizontalalignment='center')

plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
