#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 22:32:53 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt

# 全局变量
c = 3e8  # 光速 (m/s)

def db(x):
    """计算分贝值"""
    return 20 * np.log10(np.abs(x) + 1e-12)

def initialize_parameters():
    """初始化雷达参数"""
    radar_params = {}
    radar_params['PRT'] = 1000e-6                       # 脉冲重复时间 (s)
    radar_params['Tao'] = 40e-6                         # 脉冲持续时间/脉宽 (s)
    radar_params['B'] = 5e6                             # 信号带宽 (Hz)
    radar_params['Kr'] = radar_params['B'] / radar_params['Tao']  # 线性调频率
    radar_params['Fs'] = 2 * radar_params['B']          # 距离采样频率
    radar_params['Ts'] = 1 / radar_params['Fs']         # 距离采样时间间隔 (s)
    radar_params['f0'] = 3e9                            # 雷达工作频率 (Hz)
    radar_params['lamda'] = c / radar_params['f0']      # 雷达工作波长 (m)
    radar_params['dx'] = 0.05                           # 横向单元间距
    radar_params['dy'] = 0.05                           # 纵向单元间距
    radar_params['Nx'] = 40                             # 横向单元数
    radar_params['Ny'] = 40                             # 纵向单元数
    radar_params['Nx_main'] = np.arange(1, radar_params['Nx'] + 1)
    radar_params['Ny_main'] = np.arange(1, radar_params['Ny'] + 1)
    radar_params['Nx_slc'] = np.arange(11, 19)          # 11到18
    radar_params['Ny_slc'] = np.arange(11, 19)          # 11到18
    radar_params['azi'] = 0
    radar_params['ele'] = 0
    radar_params['num_slc'] = 1
    radar_params['azi_slc'] = np.array([4])
    radar_params['ele_slc'] = np.array([4])

    return radar_params

def initialize_control_params(radar_params):
    """初始化控制参数"""
    ctrl_params = {}
    ctrl_params['Bomen_st'] = max([60e3, radar_params['Tao'] * 1e6 * 150])           # 波门起始距离
    ctrl_params['Bomen_end'] = min([150e3, radar_params['PRT'] * 1e6 * 150])         # 波门结束距离
    ctrl_params['N'] = int(np.ceil(2 * (ctrl_params['Bomen_end'] - ctrl_params['Bomen_st']) / c / radar_params['Ts']))

    return ctrl_params

def calculate_array_response(positions_x, positions_y, azi, ele, dx, dy, lamda):
    """计算阵列响应"""
    B = 0
    for row in range(len(positions_x)):
        for col in range(len(positions_y)):
            fai = (positions_x[row] - 1) * dx * np.sin(np.deg2rad(azi)) + \
                  (positions_y[col] - 1) * dy * np.sin(np.deg2rad(ele))
            B = B + np.exp(-1j * 2 * np.pi * fai / lamda)
    return B

def generate_echo_signal(radar_params, ctrl_params, target_params, false_target_params):
    """生成回波信号"""
    # 生成噪声
    noise = np.random.randn(ctrl_params['N']) + 1j * np.random.randn(ctrl_params['N'])
    echo = noise.copy()

    # 计算目标回波位置
    Nst = int(np.floor(2 * (target_params['R0'] - ctrl_params['Bomen_st']) / c / radar_params['Ts']))
    Ned = int(np.floor((2 * (target_params['R0'] - ctrl_params['Bomen_st']) / c + radar_params['Tao']) / radar_params['Ts']))
    Ned = min(Ned, ctrl_params['N'] - 1)

    # 生成目标信号
    ts = np.arange(Nst, Ned + 1) * radar_params['Ts']
    phrase = 0.5 * radar_params['Kr'] * ts**2

    target_response = calculate_array_response(
        radar_params['Nx_main'], radar_params['Ny_main'],
        target_params['Azi'], target_params['Ele'],
        radar_params['dx'], radar_params['dy'], radar_params['lamda']
    )

    echo[Nst:Ned+1] = echo[Nst:Ned+1] + target_response * np.exp(1j * 2 * np.pi * phrase)

    # 生成假目标信号（副瓣干扰）
    Nst2 = int(np.floor(2 * (false_target_params['R0'] - ctrl_params['Bomen_st']) / c / radar_params['Ts']))
    Ned2 = int(np.floor((2 * (false_target_params['R0'] - ctrl_params['Bomen_st']) / c + radar_params['Tao']) / radar_params['Ts']))
    Ned2 = min(Ned2, ctrl_params['N'] - 1)

    ts2 = np.arange(Nst2, Ned2 + 1) * radar_params['Ts']
    phrase2 = 0.5 * radar_params['Kr'] * ts2**2

    false_target_response = calculate_array_response(
        radar_params['Nx_main'], radar_params['Ny_main'],
        radar_params['azi_slc'][0], radar_params['ele_slc'][0],
        radar_params['dx'], radar_params['dy'], radar_params['lamda']
    )

    echo[Nst2:Ned2+1] = echo[Nst2:Ned2+1] + false_target_response * np.exp(1j * 2 * np.pi * phrase2)

    return echo, Nst, Ned, Nst2, Ned2

def generate_slc_signals(radar_params, ctrl_params, false_target_params):
    """生成对消通道信号"""
    num_slc = radar_params['num_slc']
    echo_slc = np.zeros((num_slc, ctrl_params['N']), dtype=complex)

    for i in range(num_slc):
        # 生成噪声
        noise = np.random.randn(ctrl_params['N']) + 1j * np.random.randn(ctrl_params['N'])
        echo_slc[i, :] = noise

        # 计算假目标位置
        Nst2 = int(np.floor(2 * (false_target_params['R0'] - ctrl_params['Bomen_st']) / c / radar_params['Ts']))
        Ned2 = int(np.floor((2 * (false_target_params['R0'] - ctrl_params['Bomen_st']) / c + radar_params['Tao']) / radar_params['Ts']))
        Ned2 = min(Ned2, ctrl_params['N'] - 1)

        ts2 = np.arange(Nst2, Ned2 + 1) * radar_params['Ts']
        phrase2 = 0.5 * radar_params['Kr'] * ts2**2

        # 计算对消通道阵列响应
        slc_response = calculate_array_response(
            radar_params['Nx_slc'], radar_params['Ny_slc'],
            radar_params['azi_slc'][i], radar_params['ele_slc'][i],
            radar_params['dx'], radar_params['dy'], radar_params['lamda']
        )

        echo_slc[i, Nst2:Ned2+1] = echo_slc[i, Nst2:Ned2+1] + slc_response * np.exp(1j * 2 * np.pi * phrase2)

    return echo_slc

def plot_signals(radar_params, ctrl_params, echo, echo_slc, Nst, Ned, Nst2):
    """绘制信号图形"""
    t = np.arange(ctrl_params['N']) * radar_params['Ts'] + ctrl_params['Bomen_st'] / 150 * 1e-6
    r = t * 1e6 * 150

    plt.figure(figsize=(10, 12))

    # 主通道信号
    plt.subplot(3, 1, 1)
    plt.plot(r / 1e3, db(echo), '.-')
    plt.grid(True)
    plt.xlabel('R/km')
    plt.ylabel('幅度/dB')
    plt.title('主通道信号')

    # 对消通道信号1
    plt.subplot(3, 1, 2)
    plt.plot(r / 1e3, db(echo_slc[0, :]), '.-')
    plt.grid(True)
    plt.xlabel('R/km')
    plt.ylabel('幅度/dB')
    plt.title('对消通道信号1')

    # 对消通道信号2
    if echo_slc.shape[0] > 1:
        plt.subplot(3, 1, 3)
        plt.plot(r / 1e3, db(echo_slc[1, :]), '.-')
        plt.grid(True)
        plt.xlabel('R/km')
        plt.ylabel('幅度/dB')
        plt.title('对消通道信号2')

    plt.tight_layout()
    plt.show()

    # 相位图
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(np.arange(Nst, Ned+1), np.angle(echo[Nst:Ned+1]), '.-')
    plt.grid(True)
    plt.title('主通道相位')

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(Nst, Ned+1), np.angle(echo_slc[0, Nst:Ned+1]), '.-')
    plt.grid(True)
    plt.title('对消通道相位')

    plt.tight_layout()
    plt.show()

    # 相位差
    angle_var = np.angle(echo[Nst:Ned+1]) - np.angle(echo_slc[0, Nst:Ned+1])
    angle_var = np.mod(angle_var + np.pi, 2 * np.pi) - np.pi  # 归一化到[-π, π]

    plt.figure()
    plt.plot(np.arange(len(angle_var)), angle_var, '.')
    plt.grid(True)
    plt.title('主通道与对消通道相位差')
    plt.show()

def slc_cancellation(radar_params, ctrl_params, echo, echo_slc):
    """副瓣对消处理"""
    num_slc = echo_slc.shape[0]

    # 选干扰样本
    R_starts = np.arange(0, len(echo), 128)
    Result = np.zeros((num_slc, len(R_starts)-1), dtype=complex)

    for j in range(num_slc):
        for i in range(len(R_starts)-1):
            Rx = slice(R_starts[i], R_starts[i+1])
            correlation = np.dot(echo[Rx], np.conj(echo_slc[j, Rx]))
            norm_echo = np.linalg.norm(echo[Rx])
            norm_echo_slc = np.linalg.norm(echo_slc[j, Rx])
            Result[j, i] = np.abs(correlation) / (norm_echo * norm_echo_slc)

    Result1 = np.max(Result, axis=0)
    index_slc = np.where(Result1 > 0.5)[0]

    # 绘制样本选择结果
    r = ctrl_params['Bomen_st'] + R_starts[:-1] * radar_params['Ts'] * 1e6 * 150
    r = r / 1e3

    plt.figure()
    plt.plot(r, Result.T, '.-')
    plt.grid(True)
    plt.xlabel('R/km')
    plt.ylim([-1, 2])
    plt.title('副瓣对消选样本')
    plt.show()

    # 计算权值
    W = 0
    for i in index_slc:
        Rx = slice(R_starts[i], R_starts[i+1])
        sample_echo = echo[Rx]
        sample_echo_slc = echo_slc[0, Rx]  # 使用第一个对消通道

        R_xy = np.mean(sample_echo * np.conj(sample_echo_slc))
        R_yy = np.mean(sample_echo_slc * np.conj(sample_echo_slc))

        W = W + R_xy / R_yy

    if len(index_slc) > 0:
        W = W / len(index_slc)

    # 副瓣对消
    result_slc = echo - W * echo_slc[0, :]

    # 绘制对消结果
    t = np.arange(ctrl_params['N']) * radar_params['Ts'] + ctrl_params['Bomen_st'] / 150 * 1e-6
    r = t * 1e6 * 150

    plt.figure()
    plt.plot(r / 1e3, db(result_slc), '.-')
    plt.grid(True)
    plt.ylim([-20, 80])
    plt.title('副瓣对消结果')
    plt.show()

    return result_slc

def main():
    """主函数"""
    # 初始化参数
    radar_params = initialize_parameters()
    ctrl_params = initialize_control_params(radar_params)

    # 目标参数
    target_params = {
        'R0': 100e3,
        'Azi': radar_params['azi'],
        'Ele': radar_params['ele']
    }

    false_target_params = {
        'R0': 110e3,
        'Azi': target_params['Azi'],
        'Ele': target_params['Ele']
    }

    # 生成回波信号
    echo, Nst, Ned, Nst2, Ned2 = generate_echo_signal(radar_params, ctrl_params, target_params, false_target_params)

    # 生成对消通道信号
    echo_slc = generate_slc_signals(radar_params, ctrl_params, false_target_params)

    # 绘制信号
    plot_signals(radar_params, ctrl_params, echo, echo_slc, Nst, Ned, Nst2)

    # 执行副瓣对消
    result_slc = slc_cancellation(radar_params, ctrl_params, echo, echo_slc)

    print("副瓣对消处理完成")

if __name__ == "__main__":
    main()
