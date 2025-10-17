#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 21:40:48 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import windows
from scipy.fft import fft, ifft, fftshift
from matplotlib.animation import FuncAnimation

# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)


def rectpuls(t, width):
    """生成矩形脉冲"""
    return np.where((t >= -width/2) & (t <= width/2), 1, 0)

def ca_cfar(RDM_dB, numGuard, numTrain, P_fa, SNR_OFFSET):
    """CA-CFAR检测"""
    numTrain2D = numTrain * numTrain - numGuard * numGuard
    RDM_mask = np.zeros_like(RDM_dB)

    rows, cols = RDM_mask.shape

    for r in range(numTrain + numGuard, rows - (numTrain + numGuard)):
        for d in range(numTrain + numGuard, cols - (numTrain + numGuard)):
            # 计算噪声功率
            Pn = (np.sum(RDM_dB[r-(numTrain+numGuard):r+(numTrain+numGuard)+1,
                               d-(numTrain+numGuard):d+(numTrain+numGuard)+1]) -
                  np.sum(RDM_dB[r-numGuard:r+numGuard+1, d-numGuard:d+numGuard+1])) / numTrain2D

            # 计算阈值
            a = numTrain2D * (P_fa**(-1/numTrain2D) - 1)
            threshold = a * Pn

            # 检测
            if RDM_dB[r, d] > threshold and RDM_dB[r, d] > SNR_OFFSET:
                RDM_mask[r, d] = 1

    # 找到检测点
    cfar_ranges, cfar_dopps = np.where(RDM_mask == 1)

    # 去除冗余检测点
    to_remove = []
    for i in range(1, len(cfar_ranges)):
        if (abs(cfar_ranges[i] - cfar_ranges[i-1]) <= 5 and
            abs(cfar_dopps[i] - cfar_dopps[i-1]) <= 5):
            to_remove.append(i)

    cfar_ranges = np.delete(cfar_ranges, to_remove)
    cfar_dopps = np.delete(cfar_dopps, to_remove)
    K = len(cfar_dopps)

    return RDM_mask, cfar_ranges, cfar_dopps, K

def lookup_angle_from_sumdiffratio(AB_ybili, sum_diff_ratio, theta, theta_min, theta_max):
    """根据和差比查找角度"""
    if len(AB_ybili) != len(theta):
        raise ValueError("AB_ybili 的长度必须与 theta 的长度一致")

    # 限制查找范围
    theta_range_idx = (theta >= theta_min) & (theta <= theta_max)
    theta_limited = theta[theta_range_idx]
    AB_ybili_limited = AB_ybili[theta_range_idx]

    # 找到最接近的和差比
    diff = np.abs(AB_ybili_limited - sum_diff_ratio)
    idx = np.argmin(diff)

    return theta_limited[idx]

def plot_results(Xk, Zk, Detect_Result):
    """绘制结果"""
    # 航迹解算
    r_all = Detect_Result[0, :]
    theta_all = Detect_Result[2, :] + 90
    xk_out = np.array([
        r_all * np.cos(np.deg2rad(theta_all)),
        r_all * np.sin(np.deg2rad(theta_all))
    ])

    # 绘图1: 雷达位置和航迹
    plt.figure(4)
    # 绘制雷达位置（简化表示）
    plt.plot(0, 0, 'g^', markersize=10, label='雷达位置')
    plt.plot(Xk[0, :], Xk[1, :], 'b--', linewidth=1.1, label='真实航迹')
    plt.plot(xk_out[0, :], xk_out[1, :], 'rx', linewidth=1.1, label='点迹估计结果')
    plt.legend(loc='best')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('航迹检测结果')
    plt.grid(True)
    plt.show()

    # 绘图2: 航迹放大图
    plt.figure(5)
    plt.plot(Xk[0, :], Xk[1, :], 'b--', linewidth=1.1, label='真实航迹')
    plt.plot(xk_out[0, :], xk_out[1, :], 'rx', linewidth=1.1, label='点迹估计结果')
    plt.legend(loc='best')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('航迹放大图')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_trajectory_animation(Detect_Result):
    """绘制航迹动态显示"""
    distance = Detect_Result[0, :]
    angle_xy = Detect_Result[2, :]
    angle_polar = -angle_xy + 90

    # 极坐标动画
    fig1, ax1 = plt.subplots(subplot_kw={'projection': 'polar'})

    def update_polar(frame):
        ax1.clear()
        ax1.plot(angle_polar[frame] * np.pi / 180, distance[frame], 'bo')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_thetalim(0, np.pi)
        ax1.set_title('航迹动态显示 - 极坐标')
        return ax1

    # 将动画赋值给变量，避免被垃圾回收
    ani1 = FuncAnimation(fig1, update_polar, frames=len(distance), interval=50, repeat=True)

    # 直角坐标动画
    X = distance * np.sin(np.deg2rad(angle_xy))
    Y = distance * np.cos(np.deg2rad(angle_xy))

    fig2, ax2 = plt.subplots()

    def update_xy(frame):
        ax2.clear()
        ax2.plot(X[frame], Y[frame], 'bo')
        ax2.grid(True)
        ax2.set_xlabel('水平距离 (m)')
        ax2.set_ylabel('垂直距离 (m)')
        ax2.set_xlim([-4500, 4500])
        ax2.set_ylim([0, 70000])
        ax2.set_title('航迹动态显示 - 直角坐标')
        return ax2

    # 将动画赋值给变量，避免被垃圾回收
    ani2 = FuncAnimation(fig2, update_xy, frames=len(distance), interval=50, repeat=True)

    # 显示动画
    plt.show()

    # 返回动画对象，确保它们不会被垃圾回收
    return ani1, ani2


"""运行主仿真"""
# 基础参数
c = 3.0e8  # 光速(m/s)
Fc = 35e9  # 雷达射频
Br = 10e6  # 发射信号带宽
fs = 20e6  # 采样频率
PRF = 2e3  # 脉冲重复频率
PRT = 1 / PRF               # 脉冲重复周期
lamda = c / Fc              # 雷达工作波长
N_pulse = 128               # 回波脉冲数
N_sample = round(fs * PRT)  # 每个脉冲周期的采样点数
Tr = 3e-6  # 发射信号时宽

# 时间序列和距离序列
t1 = np.arange(0, N_sample) / fs
Range = c * t1 / 2  # 距离序列
RangeMax = c * t1[-1] / 2  # 最大不模糊距离
Vmax = lamda * PRF / 2  # 最大可检测速度
Velocity = np.linspace(-Vmax/2, Vmax/2 - Vmax/N_pulse, N_pulse)
searching_doa = np.arange(-15, 15+0.01, 0.01)  # 角度搜索区间

# 阵列参数
M = 16  # 阵元数量
d = lamda / 2  # 阵元间隔
d_LinearArray = np.arange(M).reshape(-1, 1) * d
SNR = 10
SNR_linear = 10**(SNR / 10)

# 初始化目标
# 场景参数
V = 50
T = 1  # 采样间隔，没有采用PRT是因为：T=PRT时，nT=300001,跑一次代码太久
nT = len(np.arange(-4e3, 4e3 + V * T, V * T))  # 采样帧数
# 目标状态变化
Xk = np.zeros((4, nT))
Xk[0, :] = np.arange(-4e3, 4e3 + V * T, V * T)    # x (-4km -> 4km)
Xk[1, :] = 64e3 * np.ones(nT)                     # y (64 km)
Xk[2, :] = V * np.ones(nT)                        # vx
Xk[3, :] = np.zeros(nT)                           # vy

Zk = np.zeros((3, nT))
for i in range(nT):
    r = np.sqrt(Xk[0, i] ** 2 + Xk[1, i] ** 2)  # 径向距离
    v = -(Xk[0, i] * Xk[2, i] + Xk[1, i] * Xk[3, i]) / r  # 径向速度
    phi = -(np.arctan2(Xk[1, i], Xk[0, i]) * 180 / np.pi - 90)  # 角度
    Zk[:, i] = [r, v, phi]  # i时刻目标极坐标状态

#%% 波束鉴角曲线的生成
# w_1, w_2, AB_ybili, theta, theta_min, theta_max = generate_beams(lamda, d_LinearArray)
"""生成波束和鉴角曲线"""
theta = np.arange(-90, 90, 0.01)
theta1 = -3  # 波束A指向的方向
theta2 = 3   # 波束B指向的方向
theta_min = -3.8
theta_max = 3.8

# 导向矢量
look_a = np.exp(1j * 2 * np.pi * d_LinearArray * np.sin(np.deg2rad(theta)) / lamda)

# 波束加权权向量
w_1 = np.exp(1j * 2 * np.pi * d_LinearArray * np.sin(np.deg2rad(theta1)) / lamda)
w_2 = np.exp(1j * 2 * np.pi * d_LinearArray * np.sin(np.deg2rad(theta2)) / lamda)

# 波束方向图 - 修复维度问题
yA = np.abs(w_1.conj().T @ look_a).flatten()  # 展平为1D数组
yB = np.abs(w_2.conj().T @ look_a).flatten()  # 展平为1D数组

# 和差波束
ABSum = yA + yB
ABDiff = yA - yB
AB_ybili = ABDiff / ABSum

# 绘制波束
plt.figure(1)
plt.plot(theta, yA / np.max(yA), linewidth=1, label='波束A')
plt.plot(theta, yB / np.max(yB), linewidth=1, label='波束B')
plt.xlabel('方位角/°')
plt.ylabel('归一化方向图')
plt.legend()
plt.title('波束A、B示意图')
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制和差波束
plt.figure(2)
plt.plot(theta, ABSum, linewidth=1, label='和波束')
plt.plot(theta, ABDiff, linewidth=1, label='差波束')
plt.xlabel('方位角/°')
plt.ylabel('功率增益')
plt.legend()
plt.title('和差波束示意图')
plt.grid(True)
plt.tight_layout()
plt.show()

# 绘制鉴角曲线
plt.figure(3)
plt.plot(theta, AB_ybili)
plt.xlim([theta_min, theta_max])
plt.xlabel('方位角/°')
plt.ylabel('差和比')
plt.title('鉴角曲线')
plt.grid(True)
plt.tight_layout()
plt.show()


#%% 主程序
# 初始化结果数组
Detect_Result = np.zeros((3, nT))

# 匹配滤波系数生成
sr = rectpuls(t1 - Tr/2, Tr) * np.exp(1j * np.pi * (Br/Tr) * (t1 - Tr/2)**2)
win = windows.hamming(N_sample)
h_w = np.conj(sr[::-1]) * win
h_w_freq = fft(h_w)

# 主循环
for t in range(min(10, nT)):  # 为了测试，只运行前10个时间点
    print(f"处理时间点 {t+1}/{nT}")
    data = Zk[:, t]
    a_tar_LinearArray = np.exp(1j * 2 * np.pi * d_LinearArray * np.sin(np.deg2rad(data[2])) / lamda)

    # 初始化信号矩阵
    signal_i = np.ones((N_pulse, N_sample), dtype=complex)
    y1_out = np.ones((N_pulse, N_sample), dtype=complex)
    y2_out = np.ones((N_pulse, N_sample), dtype=complex)

    # 噪声
    clutter = (np.sqrt(2)/2 * np.random.randn(M, N_sample) +
              np.sqrt(2)/2 * 1j * np.random.randn(M, N_sample))

    for i_n in range(N_pulse):
        ta = i_n * PRT
        tao = 2 * (data[0] - data[1] * (ta + t1)) / c

        # 生成LFM信号
        signal_i[i_n, :] = (SNR_linear * rectpuls(t1 - tao - Tr/2, Tr) * np.exp(1j * 2 * np.pi * Fc * (t1 - tao - Tr/2) + 1j * np.pi * (Br/Tr) * (t1 - tao - Tr/2)**2))

        # 阵列信号
        signal_LFM = a_tar_LinearArray @ signal_i[i_n:i_n+1, :] + clutter

        # 波束形成
        y1 = w_1.conj().T @ signal_LFM
        y2 = w_2.conj().T @ signal_LFM

        # 脉冲压缩
        y1_out[i_n, :] = ifft(fft(y1.flatten()) * h_w_freq)
        y2_out[i_n, :] = ifft(fft(y2.flatten()) * h_w_freq)

    # MTD
    win2 = np.tile(windows.hamming(N_pulse), (N_sample, 1)).T
    FFT_y1out = fftshift(fft(y1_out * win2, axis=0), axes=0)
    FFT_y2out = fftshift(fft(y2_out * win2, axis=0), axes=0)

    # CA-CFAR
    RDM_dB_y1 = 10 * np.log10(np.abs(FFT_y1out) / np.max(np.abs(FFT_y1out)))
    RDM_dB_y2 = 10 * np.log10(np.abs(FFT_y2out) / np.max(np.abs(FFT_y2out)))

    numGuard = 2
    numTrain = numGuard * 2
    P_fa = 1e-5
    SNR_OFFSET = -5

    RDM_mask_A, cfar_ranges_A, cfar_dopps_A, K_A = ca_cfar(RDM_dB_y1, numGuard, numTrain, P_fa, SNR_OFFSET)
    RDM_mask_B, cfar_ranges_B, cfar_dopps_B, K_B = ca_cfar(RDM_dB_y2, numGuard, numTrain, P_fa, SNR_OFFSET)

    if len(cfar_ranges_A) > 0 and len(cfar_dopps_A) > 0:
        # 获取目标距离和速度
        TrgtR = Range[cfar_dopps_A[0]]
        TrgtV = Velocity[cfar_ranges_A[0]]

        # 获取对应目标在波束A和B中的强度
        intensity_A = np.abs(FFT_y1out[cfar_ranges_A[0], cfar_dopps_A[0]])
        intensity_B = np.abs(FFT_y2out[cfar_ranges_B[0], cfar_dopps_B[0]])

        # 计算和差比
        sum_val = intensity_A + intensity_B
        diff_val = intensity_A - intensity_B
        sum_diff_ratio = diff_val / sum_val

        # 根据和差比估计角度
        TrgtAngle = lookup_angle_from_sumdiffratio(AB_ybili, sum_diff_ratio, theta, theta_min, theta_max)

        # 存储结果
        Detect_Result[:, t] = [TrgtR, TrgtV, TrgtAngle]
    else:
        # 如果没有检测到目标，使用真实值（在实际应用中可能需要其他处理）
        Detect_Result[:, t] = [data[0], data[1], data[2]]

# 计算误差
valid_indices = np.where(Detect_Result[0, :] != 0)[0]  # 只计算有效检测点
if len(valid_indices) > 0:
    RMSE_R_ave = np.mean(np.abs(Detect_Result[0, valid_indices] - Zk[0, valid_indices]))
    RMSE_V_ave = np.mean(np.abs(Detect_Result[1, valid_indices] - Zk[1, valid_indices]))
    RMSE_phi_ave = np.mean(np.abs(Detect_Result[2, valid_indices] - Zk[2, valid_indices]))

    print(f'平均测距误差{RMSE_R_ave:.2f} m')
    print(f'平均测速误差{RMSE_V_ave:.3f} m/s')
    print(f'平均测角误差{RMSE_phi_ave:.4f} °')
else:
    print("没有有效检测到目标")


plot_results(Xk, Zk, Detect_Result)

# 显示动态航迹，并将动画对象赋值给变量
ani1, ani2 = plot_trajectory_animation(Detect_Result)
# 如果需要保存动画，可以取消下面的注释
ani1.save('trajectory_animation1.gif', writer='pillow', fps=10)
ani2.save('trajectory_animation2.gif', writer='pillow', fps=10)




