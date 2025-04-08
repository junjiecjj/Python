#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 17:49:57 2025

@author: jack

和差波束测角和多波束测角原理与仿真
"""

import numpy as np
import matplotlib.pyplot as plt

def planar_sum_diff_monopulse():
    # 参数设置
    c = 3e8
    fc = 10e9
    wavelength = c / fc
    d = wavelength / 2
    Nx, Ny = 8, 8  # 平面阵行列数
    target_az, target_el = 10, 5  # 目标方位角和俯仰角（度）
    snr = 20  # 信噪比

    # 生成平面阵位置
    xx, yy = np.meshgrid(np.arange(Nx)*d, np.arange(Ny)*d)
    positions = np.vstack((xx.ravel(), yy.ravel())).T

    # 和差波束权重
    sum_weights = np.ones(Nx*Ny) / np.sqrt(Nx*Ny)

    # 方位差波束（x方向）
    diff_x_weights = np.tile(np.linspace(-1, 1, Nx), Ny).ravel()
    diff_x_weights /= np.linalg.norm(diff_x_weights)

    # 俯仰差波束（y方向）
    diff_y_weights = np.repeat(np.linspace(-1, 1, Ny), Nx)
    diff_y_weights /= np.linalg.norm(diff_y_weights)

    # 阵列响应
    theta_az, theta_el = np.deg2rad(target_az), np.deg2rad(target_el)
    phase = 2*np.pi*(positions[:,0]*np.sin(theta_az) + positions[:,1]*np.sin(theta_el))/wavelength
    array_response = np.exp(1j * phase)

    # 加噪声
    noise = np.random.randn(Nx*Ny) + 1j*np.random.randn(Nx*Ny)
    noise *= 10**(-snr/20) / np.sqrt(2)
    received_signal = array_response + noise

    # 和差处理
    sum_signal = np.dot(sum_weights, received_signal)
    diff_x_signal = np.dot(diff_x_weights, received_signal)
    diff_y_signal = np.dot(diff_y_weights, received_signal)

    # 角度估计
    monopulse_ratio_x = np.real(diff_x_signal / sum_signal)
    monopulse_ratio_y = np.real(diff_y_signal / sum_signal)

    est_az = np.rad2deg(np.arcsin(monopulse_ratio_x * wavelength / (np.pi*d)))
    est_el = np.rad2deg(np.arcsin(monopulse_ratio_y * wavelength / (np.pi*d)))

    print(f"真实角度: 方位角={target_az}°, 俯仰角={target_el}°")
    print(f"估计角度: 方位角={est_az:.2f}°, 俯仰角={est_el:.2f}°")

planar_sum_diff_monopulse()


def planar_multiple_beam():
    # 参数设置
    c = 3e8
    fc = 10e9
    wavelength = c / fc
    d = wavelength / 2
    Nx, Ny = 8, 8
    targets = [(10, 5), (-5, 8)]  # 多个目标的(az,el)
    snr = 15

    # 生成平面阵
    xx, yy = np.meshgrid(np.arange(Nx)*d, np.arange(Ny)*d)
    positions = np.vstack((xx.ravel(), yy.ravel())).T

    # 生成接收信号
    received_signal = np.zeros(Nx*Ny, dtype=np.complex_)
    for az, el in targets:
        theta_az, theta_el = np.deg2rad(az), np.deg2rad(el)
        phase = 2*np.pi*(positions[:,0]*np.sin(theta_az) + positions[:,1]*np.sin(theta_el))/wavelength
        signal = np.exp(1j * phase)
        noise = (np.random.randn(Nx*Ny) + 1j*np.random.randn(Nx*Ny)) * 10**(-snr/20)/np.sqrt(2)
        received_signal += signal + noise

    # 多波束形成（方位和俯仰各31个波束）
    beam_az = np.linspace(-20, 20, 31)
    beam_el = np.linspace(-10, 10, 31)
    beam_power = np.zeros((len(beam_az), len(beam_el)))

    for i, az in enumerate(np.deg2rad(beam_az)):
        for j, el in enumerate(np.deg2rad(beam_el)):
            steering_vector = np.exp(-1j*2*np.pi*(positions[:,0]*np.sin(az) + positions[:,1]*np.sin(el))/wavelength)
            beam_power[i,j] = np.abs(np.dot(steering_vector, received_signal))**2

    # 检测峰值
    peaks = np.unravel_index(np.argmax(beam_power), beam_power.shape)
    est_az, est_el = beam_az[peaks[0]], beam_el[peaks[1]]

    # 可视化
    plt.figure()
    plt.imshow(10*np.log10(beam_power.T), extent=[beam_az[0], beam_az[-1], beam_el[0], beam_el[-1]],
               origin='lower', aspect='auto', cmap='jet')
    plt.colorbar(label='Power (dB)')
    for az, el in targets:
        plt.plot(az, el, 'rx', markersize=10, lw = 12, label='真实角度')
    plt.plot(est_az, est_el, 'go', fillstyle='none',lw = 12, markersize=10, label='估计角度')
    plt.xlabel('方位角 (度)')
    plt.ylabel('俯仰角 (度)')
    plt.title('平面阵多波束测角')
    plt.legend()
    plt.show()

planar_multiple_beam()

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from mpl_toolkits.mplot3d import Axes3D

def fmcw_radar_3d_tracking():
    # ==================== 雷达参数设置 ====================
    c = 3e8  # 光速 (m/s)
    fc = 77e9  # 载频 (Hz)
    bw = 500e6  # 带宽 (Hz)
    tm = 50e-6  # 扫频时间 (s)
    slope = bw / tm  # 调频斜率 (Hz/s)

    # ==================== 平面阵列参数 ====================
    num_x, num_y = 8, 8  # x和y方向阵元数
    d = 0.5 * (c / fc)  # 阵元间距
    array_pos = np.zeros((num_x * num_y, 3))  # 平躺的平面阵 (z=0)
    for i in range(num_x):
        for j in range(num_y):
            array_pos[i*num_y + j] = [i*d, j*d, 0]

    # ==================== 目标参数 ====================
    targets = [
        {"pos": [50, 20, 30], "vel": [5, 2, 0]},  # 目标1 [x,y,z] (m), [vx,vy,vz] (m/s)
        {"pos": [80, -15, 40], "vel": [-3, 1, 0]}  # 目标2
    ]

    # ==================== 信号参数 ====================
    fs = 2.5 * bw  # 采样率
    num_samples = 1024 # int(tm * fs)  # 每个chirp采样点数
    num_chirps = 128  # 每个帧的chirp数
    pri = tm  # 脉冲重复间隔

    # ==================== 信号生成 ====================
    t = np.linspace(0, tm, num_samples)
    rx_signal = np.zeros((num_x*num_y, num_chirps, num_samples), dtype=np.complex_)

    for target in targets:
        # 计算距离和速度
        range_dist = np.linalg.norm(target["pos"])
        radial_vel = np.dot(target["vel"], target["pos"]) / range_dist

        # 计算角度 (方位角az和俯仰角el)
        az = np.arctan2(target["pos"][1], target["pos"][0])  # 方位角 (rad)
        el = np.arctan2(target["pos"][2], np.linalg.norm(target["pos"][:2]))  # 俯仰角 (rad)

        # 时延和多普勒
        tau = 2 * range_dist / c
        fd = 2 * fc * radial_vel / c

        for chirp in range(num_chirps):
            # 差频信号
            phase = 2 * np.pi * (slope * tau * t + fd * chirp * pri)

            # 阵列响应
            for n in range(num_x*num_y):
                xn, yn, _ = array_pos[n]
                array_phase = 2 * np.pi * (xn*np.sin(az)*np.cos(el) + yn*np.sin(el)) / (c/fc)
                rx_signal[n, chirp, :] += np.exp(1j * (phase + array_phase))

    # 添加噪声
    noise_power = 0.1
    rx_signal += np.sqrt(noise_power/2) * (np.random.randn(*rx_signal.shape) +
                                         1j*np.random.randn(*rx_signal.shape))

    # ==================== 信号处理 ====================
    # 1. 距离FFT
    range_fft = np.fft.fft(rx_signal, axis=2)
    range_bins = np.fft.fftfreq(num_samples, 1/fs) * c / (2 * slope)
    range_bins = range_bins[:num_samples//2]

    # 2. 多普勒FFT
    doppler_fft = np.fft.fftshift(np.fft.fft(range_fft, axis=1), axes=1)
    velocity_bins = np.fft.fftshift(np.fft.fftfreq(num_chirps, pri)) * c / (2 * fc)

    # 3. 峰值检测 (距离-多普勒域)
    power_rd = np.sum(np.abs(doppler_fft), axis=0)[:, :num_samples//2]
    peaks_rd = find_peaks(power_rd.ravel(), height=np.max(power_rd)*0.2, distance=10)[0]
    peak_indices = np.unravel_index(peaks_rd, power_rd.shape)

    # ==================== 角度估计 ====================
    detected_targets = []

    for i in range(len(peaks_rd)):
        r_idx, d_idx = peak_indices[1][i], peak_indices[0][i]
        range_est = range_bins[r_idx]
        vel_est = velocity_bins[d_idx]

        # 提取当前距离-多普勒单元的信号
        cell_signal = doppler_fft[:, d_idx, r_idx]

        # ===== 方法1: 和差波束测角 =====
        # 和波束
        sum_beam = np.mean(cell_signal)

        # 方位差波束 (x方向)
        diff_x_weights = np.tile(np.linspace(-1, 1, num_x), num_y).flatten()
        diff_x_weights /= np.linalg.norm(diff_x_weights)
        diff_x_beam = np.dot(diff_x_weights, cell_signal)
        az_est_sd = np.arcsin(np.real(diff_x_beam / sum_beam) * (c/fc) / (np.pi*d))

        # 俯仰差波束 (y方向)
        diff_y_weights = np.repeat(np.linspace(-1, 1, num_y), num_x)
        diff_y_weights /= np.linalg.norm(diff_y_weights)
        diff_y_beam = np.dot(diff_y_weights, cell_signal)
        el_est_sd = np.arcsin(np.real(diff_y_beam / sum_beam) * (c/fc) / (np.pi*d))

        # ===== 方法2: 多波束测角 =====
        az_grid = np.linspace(-np.pi/2, np.pi/2, 61)
        el_grid = np.linspace(-np.pi/4, np.pi/4, 31)
        beam_power = np.zeros((len(az_grid), len(el_grid)))

        for ai, az in enumerate(az_grid):
            for ei, el in enumerate(el_grid):
                steering_vec = np.exp(-1j*2*np.pi*(array_pos[:,0]*np.sin(az)*np.cos(el) +
                                              array_pos[:,1]*np.sin(el)) / (c/fc))
                beam_power[ai, ei] = np.abs(np.dot(steering_vec, cell_signal))**2

        # 寻找峰值
        peak_idx = np.unravel_index(np.argmax(beam_power), beam_power.shape)
        az_est_mb, el_est_mb = az_grid[peak_idx[0]], el_grid[peak_idx[1]]

        detected_targets.append({
            "range": range_est,
            "velocity": vel_est,
            "angle_sd": (np.rad2deg(az_est_sd), np.rad2deg(el_est_sd)),
            "angle_mb": (np.rad2deg(az_est_mb), np.rad2deg(el_est_mb))
        })

    # ==================== 结果可视化 ====================
    print("\n=== 检测结果 ===")
    for i, target in enumerate(detected_targets, 1):
        print(f"目标{i}:")
        print(f"  距离: {target['range']:.1f}m")
        print(f"  径向速度: {target['velocity']:.1f}m/s")
        print(f"  和差波束估计角度 (az,el): ({target['angle_sd'][0]:.1f}°, {target['angle_sd'][1]:.1f}°)")
        print(f"  多波束估计角度 (az,el): ({target['angle_mb'][0]:.1f}°, {target['angle_mb'][1]:.1f}°)")

    # 3D轨迹可视化
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制阵列位置
    ax.scatter(array_pos[:,0], array_pos[:,1], array_pos[:,2], c='r', label='阵列')

    # 绘制真实目标位置
    for target in targets:
        ax.scatter(*target["pos"], c='b', s=100, label='真实位置')
        ax.quiver(*target["pos"], *target["vel"], length=10, color='b')

    # 绘制估计位置
    for target in detected_targets:
        # 转换为笛卡尔坐标
        x_sd = target["range"] * np.cos(np.deg2rad(target["angle_sd"][0])) * np.cos(np.deg2rad(target["angle_sd"][1]))
        y_sd = target["range"] * np.sin(np.deg2rad(target["angle_sd"][0])) * np.cos(np.deg2rad(target["angle_sd"][1]))
        z_sd = target["range"] * np.sin(np.deg2rad(target["angle_sd"][1]))

        x_mb = target["range"] * np.cos(np.deg2rad(target["angle_mb"][0])) * np.cos(np.deg2rad(target["angle_mb"][1]))
        y_mb = target["range"] * np.sin(np.deg2rad(target["angle_mb"][0])) * np.cos(np.deg2rad(target["angle_mb"][1]))
        z_mb = target["range"] * np.sin(np.deg2rad(target["angle_mb"][1]))

        ax.scatter(x_sd, y_sd, z_sd, c='g', marker='^', s=100, label='和差波束估计')
        ax.scatter(x_mb, y_mb, z_mb, c='m', marker='s', s=100, label='多波束估计')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D目标跟踪结果')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    fmcw_radar_3d_tracking()















