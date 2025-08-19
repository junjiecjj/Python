#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 23:46:10 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from commpy.modulation import QAMModem
# from commpy.utilities import bitarray2dec
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

def ISFFT(X):
    """
    Inverse Symplectic Finite Fourier Transform
    Parameters:
        X : 2D numpy array (m x n)
    Returns:
        X_out : 2D numpy array after ISFFT
    """
    M, N = X.shape
    # ISFFT: DFT along rows (delay domain) and IDFT along columns (Doppler domain)
    X_out = np.fft.ifft(np.fft.fft(X, n=M, axis=0), n=N, axis=1) * np.sqrt(N / M)
    return X_out

def SFFT(X):
    """
    Symplectic Finite Fourier Transform
    Parameters:
        X : 2D numpy array (m x n)
    Returns:
        X_out : 2D numpy array after SFFT
    """
    m, n = X.shape
    # SFFT: IDFT along rows (delay domain) and DFT along columns (Doppler domain)
    X_out = np.fft.fft(np.fft.ifft(X, n=M, axis=0), n=N, axis=1) * np.sqrt(m / n)
    return X_out

# 基本参数
M = 1024   # 时延域数据点数
N = 120    # 多普勒域数据点数

c = 3e8
fc = 30e9
deltaf = 240e3  # 子载波间隔 240kHz
T = 1/deltaf    # 脉冲成形滤波器周期
lambda_ = c/fc

QAM_mod = 4
bitPerSymbol = int(np.log2(QAM_mod))

eng_sqrt = 1 if QAM_mod == 2 else np.sqrt((QAM_mod-1)/6*(2**2))  # QAM符号平均功率
SNR_dB = np.arange(-10, 25, 5)
NFrame = 100

sigmaSquare = eng_sqrt**2 * np.exp(-SNR_dB*np.log(10)/10)

Nsymbolperframe = M*N  # 每OTFS帧的QAM符号数
Nbitsperframe = Nsymbolperframe * bitPerSymbol  # 每OTFS帧的比特数

# 目标信息
targetRange = 35  # 目标距离35m
targetSpeed = 30  # 目标速度30m/s

delay = 2*targetRange/c
kp = round(delay*M*deltaf)

doppler = 2*targetSpeed/lambda_
lp = doppler*N*T

# 初始化误差数组
errorRange_TF = np.zeros_like(SNR_dB, dtype=float)
errorVelo_TF = np.zeros_like(SNR_dB, dtype=float)
errorRange_MF = np.zeros_like(SNR_dB, dtype=float)
errorVelo_MF = np.zeros_like(SNR_dB, dtype=float)

# 生成相位旋转向量
dd = np.arange(0, M*N)
d = np.exp(1j * 2*np.pi/(M*N) * dd)

# 创建QAM调制器
qam = QAMModem(QAM_mod)
map_table, demap_table = qam.getMappTable()
for ii, snr in enumerate(SNR_dB):
    print(f'SNR = {snr} dB')
    for jj in range(NFrame):
        uu = np.random.randint(0, 2, size = Nbitsperframe).astype(np.int8)
        x = qam.modulate(uu)
        data = np.array([demap_table[sym] for sym in x]).reshape(M, N)
        X = x.reshape(M, N)  # 原始QAM符号映射到时延-多普勒域

        # ISFFT
        X_TF = ISFFT(X)

        # Heisenberg变换
        X_ht = np.fft.ifft(X_TF, axis=0)
        s = X_ht.reshape(-1)

        # 通过时延多普勒信道
        r = np.zeros_like(s, dtype=complex)
        # temp = s * (d**lp)
        # r += np.exp(1j*2*np.pi*np.random.rand()) * np.concatenate([temp[-kp:], temp[:-kp]])
        # 修正为（确保正确循环移位）
        temp = s * (d**lp)
        r += np.exp(1j*2*np.pi*np.random.rand()) * np.roll(temp, kp)
        # 添加噪声
        noise = (np.random.randn(*r.shape) + 1j*np.random.randn(*r.shape)) * np.sqrt(sigmaSquare[ii]/2)
        r += noise

        # Wigner变换
        Y = r.reshape(M, N)
        y_TF = np.fft.fft(Y, axis=0)

        # 基于TF域的感知
        H_tf = y_TF * np.conj(X_TF)
        rdm_tf = np.fft.fft(np.fft.ifft(H_tf, axis=1).T, n=N*10, axis=1).T * np.sqrt(M/N)
        MM = np.max(np.abs(rdm_tf))
        I1, I2 = np.where(np.abs(rdm_tf) == MM)
        rangeEst = (I1[0])/(M*deltaf)*c/2
        veloEst = (I2[0])/(N*10*T)*lambda_/2

        errorRange_TF[ii] += (rangeEst - targetRange)**2/NFrame
        errorVelo_TF[ii] += (veloEst - targetSpeed)**2/NFrame

        # SFFT
        y = SFFT(y_TF)

        # 基于DD域匹配滤波的感知
        y_vec = y.reshape(-1)
        h_vec = np.fft.ifft(np.conj(np.fft.fft(x)) * np.fft.fft(y_vec), n=M*N)
        H_est = h_vec.reshape(M, N)

        MM = np.max(np.abs(H_est))
        I1, I2 = np.where(np.abs(H_est) == MM)
        rangeEst = (I1[0])/(M*deltaf)*c/2
        veloEst = (I2[0])/(N*T)*lambda_/2

        errorRange_MF[ii] += (rangeEst - targetRange)**2/NFrame
        errorVelo_MF[ii] += (veloEst - targetSpeed)**2/NFrame

# 计算RMSE
errorRange_TF = np.sqrt(errorRange_TF)
errorVelo_TF = np.sqrt(errorVelo_TF)
errorRange_MF = np.sqrt(errorRange_MF)
errorVelo_MF = np.sqrt(errorVelo_MF)

# 绘制结果
plt.figure(1)
plt.semilogy(SNR_dB, errorRange_TF, label='基于时频域的方法')
plt.semilogy(SNR_dB, errorRange_MF, label='基于匹配滤波的方法')
plt.xlabel('SNR(dB)')
plt.ylabel('距离RMSE(m)')
plt.legend()
plt.grid(True)
plt.savefig('range_rmse_comparison.png')
plt.show()




















