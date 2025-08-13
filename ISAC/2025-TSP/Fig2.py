#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:18:11 2025

@author: Junjie Chen,
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import commpy
from Modulations import modulator

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
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

#%%
def generateJk(L, N, k):
    if k < 0:
        k = L*N+k
    if k == 0:
        Jk = np.eye(L*N)
    elif k > 0:
        tmp1 = np.zeros((k, L*N-k))
        tmp2 = np.eye(k)
        tmp3 = np.eye(L*N-k)
        tmp4 = np.zeros((L*N - k, k))
        Jk = np.block([[tmp1, tmp2], [tmp3, tmp4]])
    return Jk

# 产生傅里叶矩阵
def FFTmatrix(row, col):
     mat = np.zeros((row, col), dtype = complex)
     for i in range(row):
          for j in range(col):
               mat[i,j] = 1.0*np.exp(-1j*2.0*np.pi*i*j/row) / (np.sqrt(row)*1.0)
     return mat

# 生成 循环矩阵
def CirculantMatric(gen, row):
     if type(gen) == list:
         col = len(gen)
     elif type(gen) == np.ndarray:
         col = gen.size
     row = col
     mat = np.zeros((row, col), np.complex128)
     mat[0, :] = gen
     for i in range(1, row):
         mat[i, :] = np.roll(gen, i)
     return mat

def srrcFunction(beta, L, span):
    # Function for generating rectangular pulse for the given inputs
    # L - oversampling factor (number of samples per symbol)
    # span - filter span in symbol durations
    # Returns the output pulse p(t) that spans the discrete-time base -span:1/L:span. Also returns the filter delay.

    Tsym = 1
    t = np.arange(-span/2, span/2 + 0.5/L, 1/L)
    A = np.sin(np.pi*t*(1-beta)/Tsym) + 4*beta*t/Tsym * np.cos(np.pi*t*(1+beta)/Tsym)
    B = np.pi*t/Tsym * (1-(4*beta*t/Tsym)**2)
    p = 1/np.sqrt(Tsym) * A/B
    p[np.argwhere(np.isnan(p))] = 1
    p[np.argwhere(np.isinf(p))] = beta/(np.sqrt(2*Tsym)) * ((1+2/np.pi)*np.sin(np.pi/(4*beta)) + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
    filtDelay = (len(p)-1)/2
    p = p / np.sqrt(np.sum(np.power(p, 2)))
    return p, t, filtDelay

#%%
M = 16   # 16QAM
N = 128  # the number of symbols
L = 8    # oversampling ratio
alpha = 0.35

M_array = [4, 16, 64, 256, 1024]
for M in M_array:
    MOD_TYPE = "qam"
    modem, Es, bps = modulator(MOD_TYPE, M)
    Constellation = modem.constellation/np.sqrt(Es)
    kurtosis = np.mean(np.abs(Constellation)**4)
    print(f"{M}-{MOD_TYPE.upper()}, kurtosis = {kurtosis}")

MOD_TYPE = "qam"
modem, Es, bps = modulator(MOD_TYPE, M)
Constellation = modem.constellation/np.sqrt(Es)
AvgEnergy = np.mean(np.abs(Constellation)**2)




#%%

import numpy as np
from scipy.signal import firwin, fftconvolve
import matplotlib.pyplot as plt

def generate_rc_pulse(alpha, L, span):
    """生成升余弦脉冲"""
    t = np.arange(-span*L//2, span*L//2) / L
    pulse = np.sinc(t) * np.cos(np.pi*alpha*t) / (1 - (2*alpha*t)**2)
    pulse[np.abs(t) == 1/(2*alpha)] = np.pi/4 * np.sinc(1/(2*alpha))  # 处理奇异点
    return pulse / np.linalg.norm(pulse)  # 能量归一化

def build_circulant_matrix(pulse, N, L):
    """构建循环矩阵P"""
    LN = L * N
    P = np.zeros((LN, LN))
    pulse_padded = np.pad(pulse, (0, LN - len(pulse)))
    for k in range(LN):
        P[:, k] = np.roll(pulse_padded, k)
    return P

def ofdm_modulation(symbols, N, L):
    """OFDM调制（包含IFFT和上采样）"""
    time_domain = np.fft.ifft(symbols) * np.sqrt(N)
    upsampled = np.zeros(N*L, dtype=complex)
    upsampled[::L] = time_domain  # 插入L-1个零
    return upsampled

def calculate_acf(signal):
    """计算周期自相关函数（公式15）"""
    N = len(signal)
    acf = np.zeros(N, dtype=complex)
    for k in range(N):
        acf[k] = np.dot(signal.conj(), np.roll(signal, k))
    return acf

# 参数设置（与论文Fig.2一致）
N = 128       # 符号数
L = 10        # 过采样率
alpha = 0.35  # 滚降因子
span = 6      # 滤波器跨度（根据旁瓣要求调整）

# 1. 生成升余弦脉冲
rc_pulse = generate_rc_pulse(alpha, L, span)

# 2. 构建循环矩阵P（公式11）
P = build_circulant_matrix(rc_pulse, N, L)

# 3. 生成随机QAM符号（16-QAM）
qam_symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j,
                               1+3j, 1-3j, -1+3j, -1-3j,
                               3+1j, 3-1j, -3+1j, -3-1j,
                               3+3j, 3-3j, -3+3j, -3-3j], N)
qam_symbols = qam_symbols / np.sqrt(10)  # 16-QAM能量归一化

# 4. OFDM调制
ofdm_signal = ofdm_modulation(qam_symbols, N, L)

# 5. 脉冲成形（公式10）
pulse_shaped_signal = P @ ofdm_signal

# 6. 计算ACF（公式15）
acf = calculate_acf(pulse_shaped_signal)

# 7. 绘制结果（对应论文Fig.2）
plt.figure(figsize=(10, 6))
plt.plot(np.arange(-N*L//2, N*L//2), 10*np.log10(np.abs(np.fft.fftshift(acf))**2),
         label='Simulated ACF')
plt.xlabel('Lag (samples)')
plt.ylabel('Squared ACF (dB)')
plt.title('Auto-correlation Function of Pulse-shaped OFDM Signal')
plt.grid(True)
plt.legend()
plt.show()


#%%

import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

def rrc_filter(alpha, L, span):
    """改进的根升余弦滤波器生成"""
    t = np.linspace(-span//2, span//2, span*L+1)
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-6:  # t=0
            h[i] = (1 - alpha + 4*alpha/np.pi)
        elif abs(abs(ti) - 1/(4*alpha)) < 1e-6:  # t=±1/(4α)
            h[i] = (alpha/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*alpha)) +
                    (1-2/np.pi)*np.cos(np.pi/(4*alpha)))
        else:
            num = np.sin(np.pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(np.pi*ti*(1+alpha))
            den = np.pi*ti*(1 - (4*alpha*ti)**2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h**2))  # 严格能量归一化

def accurate_acf(signal, M=1):
    """精确周期ACF计算"""
    N = len(signal)
    acf = np.zeros(N, dtype=complex)
    for _ in range(M):
        # 周期自相关计算
        acf += np.array([np.sum(signal * np.roll(signal, k)) for k in range(N)])
    return acf / M

# 参数设置
N = 128       # 符号数
L = 10        # 过采样率
alpha = 0.35  # 滚降因子
span = 12     # 滤波器跨度
M = 100       # 蒙特卡洛次数

# 1. 生成RRC滤波器
rrc_pulse = rrc_filter(alpha, L, span)

# 2. 生成随机16-QAM符号
constellation = np.array([1+1j,1-1j,-1+1j,-1-1j, 1+3j,1-3j,-1+3j,-1-3j,
                        3+1j,3-1j,-3+1j,-3-1j, 3+3j,3-3j,-3+3j,-3-3j]) / np.sqrt(10)
symbols = np.random.choice(constellation, N)

# 3. OFDM调制 + 脉冲成形
time_signal = np.fft.ifft(symbols) * np.sqrt(N)
upsampled = np.zeros(N*L, dtype=complex)
upsampled[::L] = time_signal
pulse_shaped = fftconvolve(upsampled, rrc_pulse, mode='same')

# 4. 计算ACF - 现在返回与输入信号相同长度的ACF
acf = accurate_acf(pulse_shaped, M)

# 5. 绘图设置
plt.figure(figsize=(12,6))
lags = np.arange(-len(pulse_shaped)//2, len(pulse_shaped)//2)  # 修正lags范围

# 添加小量避免log(0)
acf_power = np.abs(acf)**2 + 1e-12
acf_db = 10*np.log10(acf_power)

plt.plot(lags, np.fft.fftshift(acf_db), 'b-', linewidth=1.5)
plt.xlabel('Lag (Samples)')
plt.ylabel('Squared ACF (dB)')
plt.title('Auto-correlation Function of Pulse-shaped OFDM Signal')
plt.grid(True)
plt.xlim([-50, 50])  # 限制显示范围
plt.show()





#%%







#%%







#%%







#%%







#%%







#%%







#%%







#%%
























































