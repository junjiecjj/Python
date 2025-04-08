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
