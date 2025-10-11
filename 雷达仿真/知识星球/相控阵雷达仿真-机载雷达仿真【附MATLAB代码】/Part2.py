#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 21:50:56 2025

@author: jack

下面是对的，和matlab完全一样
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.constants import c, pi
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# %% 参数设定
eps = 1e-6

# 雷达参数
c = 3e8
N_R = 16                    # 阵元数
P_t = 2e3                   # 单阵元功率(W)
H_c = 5e3                   # 载机高度(m)
v_c = 150                   # 载机速度(m/s)
fc = 1e9                    # 载频(Hz)
lambda_ = c / fc            # 波长(m)
B = 1e6                     # 带宽(Hz)
T_p = 100e-6                # 脉宽(s)
PRF = 1e3                   # 脉冲重复频率(Hz)
CPI = 256                   # 积累脉冲数(CPI内脉冲数)
d = lambda_ / 2             # 阵元间距(m)
Ls_dB = 10                  # 接收机损耗(dB)
Ls = 10**(Ls_dB / 10)       # 转化为线性值
F_dB = 5                    # 噪声系数(dB)
F = 10**(F_dB / 10)         # 转化为线性值

# 目标参数
R_t = 90e3                  # 目标距离(m)
RCS_t = 5                   # 目标RCS(m²)
v_t = 60                    # 目标径向速度(m/s)

# 杂波参数
sigma0 = 0.01               # 杂波后向散射系数
N_bin = 101                 # 杂波块个数
T0 = 290                    # 标准温度(K)
k_B = 1.38e-23              # 玻尔兹曼常数

# 仿真参数
k_sam = 20                  # 样本个数（杂波距离环个数）
azimuth_target = 0          # 目标方位（°）
azimuth_array = 90          # 阵列与载机飞行方向夹角（°）

# %% 杂波生成函数
def clutter_gen(H_c, R_t, v_c, azimuth_target, azimuth_array, N_bin, CPI_c, N_R, d, lambda_, PRF, B, k_sam, sigma0, P_t, Ls):
    # 常数设置
    c_val = c
    H = H_c
    v = v_c
    L = CPI_c
    delta_R = c_val / (2 * B)  # 距离环间隔

    # 计算待测距离环和参考距离环的俯仰角
    R = R_t
    R_all = R + delta_R * np.arange(-k_sam//2, k_sam//2 + 1)
    phi_all = np.arcsin(H / R_all)

    # 杂波块数目和方向角设置
    azimuth_c = np.linspace(-pi/2, pi/2, N_bin)
    theta_rel = azimuth_c - np.deg2rad(azimuth_array)
    d_theta = pi / (N_bin - 1)
    azimuth_special = np.concatenate([
            np.linspace(-pi/2, 0, (N_bin - 1) // 2, endpoint=1),
            [0],
            np.linspace(0, pi/2, (N_bin - 1) // 2, endpoint=1)
            ])

    # 各距离环杂波块的空时频率
    f_s = (d/lambda_) * np.outer(np.cos(phi_all), np.sin(azimuth_special))
    f_d = (2*v/lambda_) * np.outer(np.cos(phi_all), np.sin(azimuth_special)) / PRF

    Amplitude_all = np.zeros((k_sam+1, N_bin))
    x_all = np.zeros((N_R * L, k_sam+1), dtype=complex)

    for ring_num in range(len(R_all)):
        R_ring = R_all[ring_num]
        phi = phi_all[ring_num]
        R_ground = R_ring * np.cos(phi)

        # 计算各杂波块CNR和幅度
        area_patch = delta_R * R_ground * d_theta
        RCS_patch = sigma0 * area_patch

        # 雷达方程计算幅度
        Pr = (P_t * N_R**2 * lambda_**2 * RCS_patch) / ((4*pi)**3 * R_ring**4 * Ls)
        Amplitude_all[ring_num, :] = np.sqrt(Pr)

        # 空时导向矢量
        a_s = np.exp(1j * 2 * pi * np.arange(N_R)[:, np.newaxis] * f_s[ring_num, :])
        a_t = np.exp(1j * 2 * pi * np.arange(L)[:, np.newaxis] * f_d[ring_num, :])

        # 回波数据
        for clutterpatch_num in range(N_bin):
            x_all[:, ring_num] += (Amplitude_all[ring_num, clutterpatch_num] * np.kron(a_t[:, clutterpatch_num], a_s[:, clutterpatch_num]))

    return x_all, f_s, f_d, azimuth_c

# %% 阵列接收信号生成函数
def rx_array_airborneradar(St, array_phase, N_R, num_pulses, fd, PRF, td, N_PRT, N_d, N_st, P_t, RCS_t, R_t, lambda_, Ls):
    # 计算目标接收功率
    Pr = (P_t * lambda_**2 * RCS_t) / ((4*pi)**3 * R_t**4 * Ls)
    A_t = np.sqrt(Pr)

    # 初始化多通道接收信号
    rx_array_signal = np.zeros((num_pulses * N_PRT, N_R), dtype=complex) * A_t

    for n in range(num_pulses):
        # 生成单脉冲回波
        doppler_phase = np.exp(1j * 2 * pi * fd * (n / PRF + td))

        # 生成阵列接收信号
        for k in range(N_R):
            R_phase = array_phase[k] * doppler_phase
            start_idx = n * N_PRT + N_d
            end_idx = start_idx + N_st

            if end_idx > num_pulses * N_PRT:
                end_idx = num_pulses * N_PRT
                valid_len = end_idx - start_idx
                rx_array_signal[start_idx:start_idx+valid_len, k] += St[:valid_len] * R_phase
            else:
                rx_array_signal[start_idx:end_idx, k] += St * R_phase

    return rx_array_signal

# %% 二维CFAR检测函数
def cfar_2d(input_data, guard_win, train_win, P_fa, range_bins, speed_bins, R_true, v_true):
    num_range, num_doppler = input_data.shape
    detection_map = np.zeros((num_range, num_doppler))

    # 计算阈值因子
    num_ref_cells = ((2*train_win[0]+2*guard_win[0]+1) * (2*train_win[1]+2*guard_win[1]+1) - (2*guard_win[0]+1) * (2*guard_win[1]+1))
    alpha = num_ref_cells * (P_fa**(-1/num_ref_cells) - 1)

    # 滑动窗口检测
    for range_idx in range(train_win[0]+guard_win[0], num_range-train_win[0]-guard_win[0]):
        for doppler_idx in range(train_win[1]+guard_win[1], num_doppler-train_win[1]-guard_win[1]):
            # 定义检测区域
            range_win = slice(range_idx-train_win[0]-guard_win[0], range_idx+train_win[0]+guard_win[0]+1)
            doppler_win = slice(doppler_idx-train_win[1]-guard_win[1], doppler_idx+train_win[1]+guard_win[1]+1)

            # 提取参考单元
            ref_cells = input_data[range_win, doppler_win].copy()

            # 去除保护单元
            ref_cells[train_win[0]:-train_win[0], train_win[1]:-train_win[1]] = np.nan

            # 计算噪声基底
            noise_level = np.nanmean(ref_cells)
            threshold = alpha * noise_level

            # 检测判决
            if input_data[range_idx, doppler_idx] > threshold:
                detection_map[range_idx, doppler_idx] = 1

    # 匹配真实目标
    true_range_idx = np.argmin(np.abs(range_bins - R_true/1e3))
    true_speed_idx = np.argmin(np.abs(speed_bins - v_true))

    # 判断真实目标是否被检测到
    is_detected = False
    if detection_map[true_range_idx, true_speed_idx] == 1:
        detected_range = range_bins[true_range_idx]
        detected_speed = speed_bins[true_speed_idx]
        is_detected = True

    if not is_detected:
        detected_ranges, detected_speeds = np.where(detection_map == 1)
        min_dist = np.inf
        detected_range = np.nan
        detected_speed = np.nan

        for k in range(len(detected_ranges)):
            current_dist = np.sqrt((range_bins[detected_ranges[k]] - R_true/1e3)**2 + (speed_bins[detected_speeds[k]] - v_true)**2)
            if current_dist < min_dist:
                min_dist = current_dist
                detected_range = range_bins[detected_ranges[k]]
                detected_speed = speed_bins[detected_speeds[k]]

    # 错误处理：未检测到目标时给出警告
    if np.isnan(detected_range):
        print('警告: 未检测到真实目标，请调整CFAR参数!')

    # 计算误差
    if not np.isnan(detected_range):
        range_error = abs(detected_range - R_true/1e3) / (R_true/1e3) * 100
        speed_error = abs(detected_speed - v_true) / abs(v_true) * 100
    else:
        range_error = np.nan
        speed_error = np.nan

    # 输出结果
    target_info = {
        'TrueRange': R_true/1e3,
        'TrueSpeed': v_true,
        'DetectedRange': detected_range,
        'DetectedSpeed': detected_speed,
        'RangeError': range_error,
        'SpeedError': speed_error
    }

    return detection_map, target_info

# %% T5 - 杂波空时谱分析
print("正在进行T5 - 杂波空时谱分析...")

# 获取杂波数据
x_clutter, f_s, f_d, azimuth_c = clutter_gen(H_c, R_t, v_c, azimuth_target, azimuth_array, N_bin, CPI, N_R, d, lambda_, PRF, B, k_sam, sigma0, P_t, Ls)

# 空时二维FFT
N_fft_s = 512
N_fft_d = 512

# 选择第一个距离环的数据
clutter_data = x_clutter[:, 0].reshape(N_R, CPI, order = 'F')
st_spectrum = np.fft.fftshift(np.fft.fft2(clutter_data, [N_fft_s, N_fft_d]), )
st_spectrum_db = 10 * np.log10(np.abs(st_spectrum.T) + eps)

# 生成频率轴
fs_axis = np.linspace(-0.5, 0.5, N_fft_s)       # 归一化空间频率
fd_axis = np.linspace(-0.5, 0.5, N_fft_d)       # 归一化多普勒频率
FS, FD = np.meshgrid(fs_axis, fd_axis)


# 绘制三维空时谱

# fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
# surf = ax.plot_wireframe(FS, FD, st_spectrum_db, color = [0.6,0.6,0.6], rstride = 10, cstride=10, linewidth = 0.25)
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
# # ax.scatter(FS, FD, st_spectrum_db, c = 'blue', s = 1,  )
# ax.set_proj_type('ortho')
# # 另外一种设定正交投影的方式
# ax.set_xlabel('归一化空间频率 (d/λ)')
# ax.set_ylabel('归一化多普勒频率F_d (Hz)')
# ax.set_zlabel('幅度 (dB)')
# ax.view_init(azim=-135, elev=30)
# ax.grid(False)

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, figsize = (8,8))
ax.set_proj_type('ortho')
norm_plt = plt.Normalize(st_spectrum_db.min(), st_spectrum_db.max())
colors = cm.RdYlBu_r(norm_plt(st_spectrum_db))
# colors = cm.Blues_r(norm_plt(ff))

surf = ax.plot_surface(FS, FD, st_spectrum_db, cmap=cm.jet, rstride = 5, cstride = 5, linewidth=0, antialiased=False, shade = False) # 删除阴影
surf.set_facecolor((0,0,0,0)) # 网格面填充为空, 利用 set_facecolor((0, 0, 0, 0)) 将曲面的表面颜色设置为透明,这样仅仅显示曲线。
ax.set_xlabel('归一化空间频率 (d/λ)')
ax.set_ylabel('归一化多普勒频率F_d (Hz)')
ax.set_zlabel('幅度 (dB)')
ax.view_init(azim = -135, elev = 30)
ax.grid(False)



# 理论杂波脊线叠加
slope = (2 * v_c) / (d * PRF)  # 理论斜率
fs_theo = np.linspace(-0.5, 0.5, 100)
fd_theo = slope * fs_theo
max_db = np.max(st_spectrum_db)
ax.plot(fs_theo, fd_theo, max_db * np.ones_like(fs_theo), 'r--', linewidth=2, label='理论脊')
ax.legend()

plt.tight_layout()
plt.show()

# %% T6 - 目标检测处理
print("正在进行T6 - 目标检测处理...")

# 信号波形生成
fs = 2 * B
Ts = 1 / fs
td = 2 * R_t / c  # 目标时延 (s)
N_d = int(td * fs)  # 目标时延采样点数
fd = 2 * v_t / lambda_  # 目标多普勒频率
K = B / T_p  # 调频斜率
t_chirp = np.linspace(0, T_p - Ts, int(T_p * fs))
St = np.exp(1j * pi * K * t_chirp**2)  # LFM信号
N_st = len(St)  # 单个脉冲采样点数
N_PRT = int(1 / PRF * fs)  # 单个PRT采样点数
num_pulses = 256  # 脉冲数

# 阵列接收信号生成
array_phase = np.exp(-1j * 2 * pi * d / lambda_ * np.sin(np.deg2rad(azimuth_target)) * np.arange(N_R))

# 生成目标回波
rx_target = rx_array_airborneradar(St, array_phase, N_R, num_pulses, fd, PRF, td, N_PRT, N_d, N_st, P_t, RCS_t, R_t, lambda_, Ls)

# 生成杂波
clutter, _, _, _ = clutter_gen(H_c, R_t, v_c, azimuth_target, azimuth_array, N_bin, num_pulses * N_PRT, N_R, d, lambda_, PRF, B, k_sam, sigma0, P_t, Ls)
rx_clutter = clutter[:, 0].reshape(num_pulses * N_PRT, N_R, order = 'F')

# 生成噪声
noise_power = F * k_B * T0 * B
noise = np.sqrt(noise_power/2) * (np.random.randn(num_pulses * N_PRT, N_R) + 1j * np.random.randn(num_pulses * N_PRT, N_R))

# 合成接收信号
rx_array_signal = rx_target + rx_clutter + noise

# DBF处理
weights_hamming = np.hamming(N_R)  # 汉明窗
steering_vector = array_phase * weights_hamming
w = steering_vector / np.dot(array_phase.conj(), steering_vector)  # 归一化权值
rx_beamformed = np.dot(rx_array_signal, w.conj())  # 波束形成

# 分帧处理
rx_pulses = rx_beamformed.reshape(N_PRT, num_pulses, order = 'F')

# 脉冲压缩
Pr = (P_t * lambda_**2 * RCS_t) / ((4*pi)**3 * R_t**4 * Ls)
Ar = np.sqrt(Pr)
h_mf = (Ar * St[::-1]).conj()  # 匹配滤波器
dbf_mf_output = np.zeros((N_PRT + N_st - 1, num_pulses), dtype=complex)

for i in range(num_pulses):
    dbf_mf_output[:, i] = np.convolve(rx_pulses[:, i], h_mf, mode='full')

# 距离轴校正
range_axis = (np.arange(dbf_mf_output.shape[0]) - (N_st - 1)) * c / (2 * fs)
valid_idx = range_axis >= 0
range_axis = range_axis[valid_idx] / 1e3  # 转换为km
dbf_mf_output = dbf_mf_output[valid_idx, :]

# MTD处理（多普勒FFT）
mtd_output = np.fft.fftshift(np.fft.fft(dbf_mf_output, axis=1), axes=1)
mtd_output_abs = np.abs(mtd_output)
mtd_output_db = 20 * np.log10(mtd_output_abs / np.max(mtd_output_abs) + eps)

# 速度轴计算
doppler_axis = (np.arange(num_pulses) - num_pulses//2) * PRF / num_pulses
speed_axis = doppler_axis * lambda_ / 2

# 三维可视化
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(range_axis, speed_axis)
surf = ax.plot_surface(X, Y, mtd_output_db.T, cmap='jet', linewidth=0, antialiased=False)
ax.set_xlabel('距离 (km)')
ax.set_ylabel('速度 (m/s)')
ax.set_zlabel('幅度 (dB)')
ax.set_title('距离-速度-幅度三维图')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# 标记目标
max_idx = np.argmax(mtd_output_db)
row, col = np.unravel_index(max_idx, mtd_output_db.shape)
detected_range = range_axis[row]
detected_speed = speed_axis[col]
ax.scatter(detected_range, detected_speed, mtd_output_db[row, col], c='r', marker='o', s=100, label='检测目标')
ax.text(detected_range, detected_speed, mtd_output_db[row, col] + 3, f'({detected_range:.1f} km, {detected_speed:.1f} m/s)', fontsize=12, color='r', ha='center')
ax.legend()

plt.tight_layout()
plt.show()

# %% CFAR检测（二维CA-CFAR）
print("正在进行CFAR检测...")

# CFAR参数设置
guard = [5, 5]     # 距离/速度保护单元
train = [10, 10]   # 距离/速度参考单元
P_fa = 1e-6        # 虚警概率

# 执行CFAR检测
detection_map, target_info = cfar_2d(mtd_output_db, guard, train, P_fa,
                                    range_axis, speed_axis, R_t, v_t)

# 打印检测结果
print('===== 目标检测结果 =====')
print(f'真实目标位置: {target_info["TrueRange"]:.2f} km, {target_info["TrueSpeed"]:.2f} m/s')
print(f'检测目标位置: {target_info["DetectedRange"]:.2f} km, {target_info["DetectedSpeed"]:.2f} m/s')
print(f'距离相对误差: {target_info["RangeError"]:.2f}%')
print(f'速度相对误差: {target_info["SpeedError"]:.2f}%')

# 绘制检测结果
plt.figure(figsize=(10, 6))
plt.imshow(detection_map.T, extent=[range_axis[0], range_axis[-1], speed_axis[0], speed_axis[-1]],
           aspect='auto', origin='lower', cmap='jet')
plt.xlabel('距离 (km)')
plt.ylabel('速度 (m/s)')
plt.title('CFAR检测结果')
plt.colorbar()

# 标记真实目标位置
plt.plot(target_info['TrueRange'], target_info['TrueSpeed'], 'go', markersize=10, linewidth=2, label='真实目标')
plt.text(target_info['TrueRange'], target_info['TrueSpeed'] + 3, f'({target_info["TrueRange"]:.2f} km, {target_info["TrueSpeed"]:.2f} m/s)', fontsize=12, color='g', ha='center')

# 标记检测目标位置
if not np.isnan(target_info['DetectedRange']):
    plt.plot(target_info['DetectedRange'], target_info['DetectedSpeed'], 'rp',  markersize=12, linewidth=2, label='检测目标')
    plt.text(target_info['DetectedRange'], target_info['DetectedSpeed'] - 3, f'({target_info["DetectedRange"]:.2f} km, {target_info["DetectedSpeed"]:.2f} m/s)', fontsize=12, color='r', ha='center')
    plt.legend()

plt.tight_layout()
plt.show()

# %% T7 - SCNR分析
print("正在进行T7 - SCNR分析...")

# 参数设置
v_max = PRF * lambda_ / 4  # 归一化多普勒频率±0.5对应的最大速度
v_values = np.linspace(-v_max, v_max, 31)  # 生成速度范围
SNCR_dB = np.zeros(len(v_values))  # 存储各速度对应的SCNR

# 背景功率计算 (无目标时)
rx_background = rx_clutter + noise
rx_beamformed_bg = np.dot(rx_background, w.conj())
rx_pulses_bg = rx_beamformed_bg.reshape(N_PRT, num_pulses, order = 'F')

# 脉冲压缩和MTD处理
dbf_mf_output_bg = np.zeros((N_PRT + N_st - 1, num_pulses), dtype=complex)
for i in range(num_pulses):
    dbf_mf_output_bg[:, i] = np.convolve(rx_pulses_bg[:, i], h_mf, mode='full')

# 距离轴校正
range_axis_bg = (np.arange(dbf_mf_output_bg.shape[0]) - (N_st - 1)) * c / (2 * fs)
valid_idx_bg = range_axis_bg >= 0
range_axis_bg = range_axis_bg[valid_idx_bg] / 1e3  # 转换为km
dbf_mf_output_bg = dbf_mf_output_bg[valid_idx_bg, :]

# MTD处理（多普勒FFT）
mtd_output_bg = np.fft.fftshift(np.fft.fft(dbf_mf_output_bg, axis=1), axes=1)
mtd_output_bg_abs = np.abs(mtd_output_bg)
mtd_power_bg = np.mean(mtd_output_bg_abs**2)  # 背景平均功率

# 遍历目标速度计算SCNR
for k in range(len(v_values)):
    v_k = v_values[k]
    fd_k = 2 * v_k / lambda_  # 当前速度对应的多普勒频率

    # 生成目标信号
    rx_target_k = rx_array_airborneradar(St, array_phase, N_R, num_pulses, fd_k, PRF, td, N_PRT, N_d, N_st, P_t, RCS_t, R_t, lambda_, Ls)

    # 合成接收信号
    rx_array_signal_k = rx_target_k + rx_background

    # 处理链
    rx_beamformed_k = np.dot(rx_array_signal_k, w.conj())
    rx_pulses_k = rx_beamformed_k.reshape(N_PRT, num_pulses, order = 'F')

    # 脉冲压缩
    dbf_mf_output_k = np.zeros((N_PRT + N_st - 1, num_pulses), dtype=complex)
    for i in range(num_pulses):
        dbf_mf_output_k[:, i] = np.convolve(rx_pulses_k[:, i], h_mf, mode='full')
    dbf_mf_output_k = dbf_mf_output_k[valid_idx_bg, :]

    # MTD处理
    mtd_output_k = np.fft.fftshift(np.fft.fft(dbf_mf_output_k, axis=1), axes=1)
    mtd_output_k_abs = np.abs(mtd_output_k)
    mtd_power_k = mtd_output_k_abs**2

    # 定位目标单元
    range_idx = np.argmin(np.abs(range_axis_bg * 1e3 - R_t))
    doppler_idx = np.argmin(np.abs(speed_axis - v_k))

    # 计算SCNR
    signal_power = mtd_power_k[range_idx, doppler_idx]
    SNCR_dB[k] = 10 * np.log10(signal_power / mtd_power_bg)

# 绘制SCNR曲线
plt.figure(figsize=(10, 6))
plt.plot(v_values, SNCR_dB, linewidth=1.5)
plt.xlabel('目标速度 (m/s)')
plt.ylabel('SCNR (dB)')
plt.title('目标SCNR随速度变化曲线')
plt.grid(True)

# 标记杂波脊位置
v_clutter_ridge = v_c * np.sin(np.deg2rad(azimuth_target))  # 正侧视杂波速度
plt.axvline(x=v_clutter_ridge, color='r', linestyle='--', linewidth=1.2, label='杂波脊位置')
plt.legend()

plt.tight_layout()
plt.show()

print("所有仿真完成!")
