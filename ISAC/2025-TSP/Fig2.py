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
def FFTmatrix(L, ):
     mat = np.zeros((L, L), dtype = complex)
     for i in range(L):
          for j in range(L):
               mat[i,j] = 1.0*np.exp(-1j*2.0*np.pi*i*j/L) / (np.sqrt(L)*1.0)
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

def srrcFunction(beta, L, span, Tsym = 1):
    # Function for generating rectangular pulse for the given inputs
    # L - oversampling factor (number of samples per symbol)
    # span - filter span in symbol durations
    # Returns the output pulse p(t) that spans the discrete-time base -span:1/L:span. Also returns the filter delay.
    t = np.arange(-span*Tsym/2, span*Tsym/2 + 0.5/L, Tsym/L)
    A = np.sin(np.pi*t*(1-beta)/Tsym) + 4*beta*t/Tsym * np.cos(np.pi*t*(1+beta)/Tsym)
    B = np.pi*t/Tsym * (1-(4*beta*t/Tsym)**2)
    p = 1/np.sqrt(Tsym) * A/B
    p[np.argwhere(np.isnan(p))] = 1
    p[np.argwhere(np.isinf(p))] = beta/(np.sqrt(2*Tsym)) * ((1+2/np.pi)*np.sin(np.pi/(4*beta)) + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
    filtDelay = (len(p)-1)/2
    p = p / np.sqrt(np.sum(np.power(p, 2))) # both Add and Delete this line is OK.
    return p, t, filtDelay

#%% Table. I
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


#%% Fig.2
# 参数设置
Tsym = 1
pi = np.pi
N = 128       # 符号数
L = 10        # 过采样率
alpha = 0.3  # 滚降因子
span = 6      # 滤波器跨度（根据旁瓣要求调整）

# p, t, filtDelay = srrcFunction(alpha, L, span, Tsym = Tsym)
# p = np.pad(p, (0, L*N - p.size))

t, p = commpy.filters.rrcosfilter(L*N , alpha, Tsym, L/Tsym)
p = p / np.sqrt(np.sum(np.power(p, 2)))

norm2p = np.linalg.norm(p)
FLN = FFTmatrix(L*N )
FN = FFTmatrix(N )

###>>>>> OFDM, Eq.(27)(34)
M = 100
kappa = 1.32
U = FN.conj().T
V = np.eye(N)  # U.conj().T @ FN.conj().T
tilde_V = V * V.conj()
g = (N * (FLN@p) * (FLN.conj() @ p.conj()))
# g = np.fft.fftshift(g)

TheoAveACF_Iceberg = np.zeros(L*N)
TheoAveACF_OFDM_M1 = np.zeros(L*N)
TheoAveACF_OFDM_M100 = np.zeros(L*N)
for k in range(L*N):
    fk = FLN[:,k]
    fk_tilde = fk[:N]
    gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)
    r1 = L * N * np.abs(fk_tilde.conj().T @ gk)**2
    r2 = np.linalg.norm(gk)**2
    r3 = (kappa - 2) * L * N * np.linalg.norm(tilde_V @ (gk * fk_tilde.conj()))**2


    TheoAveACF_OFDM_M1[k] = r1 + (r2 + r3)/1
    TheoAveACF_OFDM_M100[k] = r1 + (r2 + r3)/100
    TheoAveACF_Iceberg[k] = r1

TheoAveACF_OFDM_M1 = TheoAveACF_OFDM_M1/TheoAveACF_OFDM_M1.max() + 1e-10
# TheoAveACF_OFDM_M1 = np.fft.fftshift(TheoAveACF_OFDM_M1)

TheoAveACF_OFDM_M100 = TheoAveACF_OFDM_M100/TheoAveACF_OFDM_M100.max() + 1e-10
# TheoAveACF_OFDM_M100 = np.fft.fftshift(TheoAveACF_OFDM_M100)

TheoAveACF_Iceberg = TheoAveACF_Iceberg/TheoAveACF_Iceberg.max() + 1e-10
# TheoAveACF_Iceberg = np.fft.fftshift(TheoAveACF_Iceberg)

# x = np.arange(-N//2, N//2, 1/((L)))
x = np.arange(-N*L//2, N*L//2,)
fig, axs = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)
axs.plot(x, 10 * np.log10(TheoAveACF_Iceberg), color='k', linestyle='--', label='Squared ACF of the Pulse ("Iceberg")',)
axs.plot(x, 10 * np.log10(TheoAveACF_OFDM_M1 ), color='b', linestyle='-', label='Average Squared ACF, Theoretical',)
axs.plot(x, 10 * np.log10(TheoAveACF_OFDM_M100), color='r', linestyle='-', label='100 Coherent Integration, Theoretical',)

legend1 = axs.legend(loc='best', borderaxespad=0,  edgecolor='black', fontsize = 18)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')                  # 设置图例legend背景透明

axs.set_xlabel(r'Delay Index', )
axs.set_ylabel(r'Ambiguity Level (dB)', )
# axs.set_xlim([-30, 30])
out_fig = plt.gcf()
plt.show()
plt.close()

#%% OFDM, Eq.(36)

TheoAveACF_Iceberg = np.zeros(L*N)
TheoAveACF_OFDM_M1 = np.zeros(L*N)
TheoAveACF_OFDM_M100 = np.zeros(L*N)

for k in range(L*N):

    gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)
    fk = np.exp(-1j * 2*pi * k * np.arange(N)/(L*N))

    r1 = np.abs(gk @ fk.conj())**2
    TheoAveACF_Iceberg[k] = r1 #+ r2

    M = 1
    r2 = (kappa - 1) / M * (N - 2 *(1-np.cos(2*pi*k/L))*(g[:N] * (1- g[:N])).sum())
    TheoAveACF_OFDM_M1[k] = r1 + r2

    M = 100
    r2 = (kappa - 1) / M * (N - 2 *(1-np.cos(2*pi*k/L))*(g[:N] * (1- g[:N])).sum())
    TheoAveACF_OFDM_M100[k] = r1 + r2

TheoAveACF_Iceberg = TheoAveACF_Iceberg/TheoAveACF_Iceberg.max() + 1e-10
TheoAveACF_Iceberg = np.fft.fftshift(TheoAveACF_Iceberg)

TheoAveACF_OFDM_M1 = TheoAveACF_OFDM_M1/TheoAveACF_OFDM_M1.max() + 1e-10
TheoAveACF_OFDM_M1 = np.fft.fftshift(TheoAveACF_OFDM_M1)

TheoAveACF_OFDM_M100 = TheoAveACF_OFDM_M100/TheoAveACF_OFDM_M100.max() + 1e-10
TheoAveACF_OFDM_M100 = np.fft.fftshift(TheoAveACF_OFDM_M100)


# x = np.arange(-N//2, N//2, 1/((L)))
x = np.arange(-N*L//2, N*L//2,)
fig, axs = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)
axs.plot(x, 10 * np.log10(TheoAveACF_Iceberg), color='k', linestyle='--', label='Squared ACF of the Pulse ("Iceberg")',)
axs.plot(x, 10 * np.log10(TheoAveACF_OFDM_M1 ), color='b', linestyle='-', label='Average Squared ACF, Theoretical',)
axs.plot(x, 10 * np.log10(TheoAveACF_OFDM_M100), color='r', linestyle='-', label='100 Coherent Integration, Theoretical',)

legend1 = axs.legend(loc='best', borderaxespad=0,  edgecolor='black', fontsize = 18)
# frame1 = legend1.get_frame()
# frame1.set_alpha(1)
# frame1.set_facecolor('none')                  # 设置图例legend背景透明

axs.set_xlabel(r'Delay Index', )
axs.set_ylabel(r'Ambiguity Level (dB)', )
axs.set_xlim([-300, 300])
out_fig = plt.gcf()
plt.show()
plt.close()

#%% Average Squared ACF, Numerical / 100 Coherent Integration, Numerical, Eq.(26)

# M = 100
kappa = 1.32
U = FN.conj().T
V = np.eye(N)  # U.conj().T @ FN.conj().T
tilde_V = V * V.conj()
g = (N * (FLN@p) * (FLN.conj() @ p.conj()))
# g = np.fft.fftshift(g)

MOD_TYPE = "qam"
Order = 16
modem, Es, bps = modulator(MOD_TYPE, Order)
Constellation = modem.constellation/np.sqrt(Es)
AvgEnergy = np.mean(np.abs(Constellation)**2)

Iter = 1000

###>>>>>>>>>>>>>>>>> M = 1
M = 1
SimAveACF_OFDM_M1 = np.zeros((Iter, L*N))

for k in range(L*N):
    fk = FLN[:,k]
    fk_tilde = fk[:N]
    gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)
    for it in range(Iter):
        d = np.random.randint(Order, size = N)
        s = Constellation[d]
        VHs = np.abs(V.conj().T @ s)**2
        SimAveACF_OFDM_M1[it, k] = np.abs((gk * VHs * fk_tilde.conj()).sum())**2

Sim_M1_avg = SimAveACF_OFDM_M1.mean(axis = 0)
Sim_M1_avg = Sim_M1_avg/Sim_M1_avg.max() + 1e-10
Sim_M1_avg = np.fft.fftshift(Sim_M1_avg)

Sim_M1_max = SimAveACF_OFDM_M1.max(axis = 0)
Sim_M1_max = Sim_M1_max/Sim_M1_max.max() + 1e-10
Sim_M1_max = np.fft.fftshift(Sim_M1_max)

Sim_M1_min = SimAveACF_OFDM_M1.min(axis = 0)
Sim_M1_min = Sim_M1_min/Sim_M1_min.max() + 1e-10
Sim_M1_min = np.fft.fftshift(Sim_M1_min)

###>>>>>>>>>>>>>>>>> M = 100
M = 100
SimAveACF_OFDM_M100 = np.zeros((M, Iter, L*N), dtype = complex)
for k in range(L*N):
    fk = FLN[:,k]
    fk_tilde = fk[:N]
    gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)
    for m in range(M):
        for it in range(Iter):
            d = np.random.randint(Order, size = N)
            s = Constellation[d]
            VHs = np.abs(V.conj().T @ s)**2

            ##这里注意，一定不要取平方，因为这里需要先对M均值，在对Iter取均值，才是与Eq(33)对得上
            SimAveACF_OFDM_M100[m, it, k] = (gk * VHs * fk_tilde.conj()).sum()
## Eq(33)
RkBar = SimAveACF_OFDM_M100.mean(axis = 0)
RkBar2 = np.abs(RkBar)**2

Sim_M100_avg = RkBar2.mean(axis = 0)
Sim_M100_avg = Sim_M100_avg/Sim_M100_avg.max() + 1e-10
Sim_M100_avg = np.fft.fftshift(Sim_M100_avg)

Sim_M100_max = RkBar2.max(axis = 0)
Sim_M100_max = Sim_M100_max/Sim_M100_max.max() + 1e-10
Sim_M100_max = np.fft.fftshift(Sim_M100_max)

Sim_M100_min = RkBar2.min(axis = 0)
Sim_M100_min = Sim_M100_min/Sim_M100_min.max() + 1e-10
Sim_M100_min = np.fft.fftshift(Sim_M100_min)


colors = plt.cm.jet(np.linspace(0, 1, 5))
# x = np.arange(-N//2, N//2, 1/((L)))
x = np.arange(-N*L//2, N*L//2,)
fig, axs = plt.subplots(1, 1, figsize=(15, 10), constrained_layout=True)
axs.plot(x, 10 * np.log10(TheoAveACF_Iceberg), color='k', linestyle='-', lw = 1, label='Squared ACF of the Pulse ("Iceberg")',)


axs.plot(x, 10 * np.log10(TheoAveACF_OFDM_M1 ), color='#13A8F9', linestyle='-',  label='Average Squared ACF, Theoretical',)
axs.plot(x, 10 * np.log10(Sim_M1_avg ), color=colors[1], linestyle='-', lw = 1, marker = 'o', markevery = 20, ms = 12, markerfacecolor = 'none', label='Average Squared ACF, Simular',)
axs.fill_between(x, 10 * np.log10(Sim_M1_min), 10 * np.log10(Sim_M1_max), color='#13A8F9', alpha = 0.2)

axs.plot(x, 10 * np.log10(TheoAveACF_OFDM_M100), color='#F0760A', linestyle='-', label='100 Coherent Integration, Theoretical',)
axs.plot(x, 10 * np.log10(Sim_M100_avg ), color='#F97213', linestyle='-', lw = 1, marker = 'o', markevery = 20, ms = 12, markerfacecolor = 'none', label='100 Coherent Integration, Simular',)
axs.fill_between(x, 10 * np.log10(Sim_M100_min), 10 * np.log10(Sim_M100_max), color='#F0760A', alpha = 0.2)

legend1 = axs.legend(loc='best', borderaxespad=0,  edgecolor='black', fontsize = 18)
axs.set_xlabel(r'Delay Index', )
axs.set_ylabel(r'Ambiguity Level (dB)', )
axs.set_xlim([-300, 300])

plt.show()
plt.close()


#%%


#%%

#%%

#%%

#%%

#%%

#%%


#%%

#%%



#%%

# import numpy as np
# from scipy.signal import firwin, fftconvolve
# import matplotlib.pyplot as plt

# def generate_rc_pulse(alpha, L, span):
#     """生成升余弦脉冲"""
#     t = np.arange(-span*L//2, span*L//2) / L
#     pulse = np.sinc(t) * np.cos(np.pi*alpha*t) / (1 - (2*alpha*t)**2)
#     pulse[np.abs(t) == 1/(2*alpha)] = np.pi/4 * np.sinc(1/(2*alpha))  # 处理奇异点
#     return pulse / np.linalg.norm(pulse)  # 能量归一化

# def build_circulant_matrix(pulse, N, L):
#     """构建循环矩阵P"""
#     LN = L * N
#     P = np.zeros((LN, LN))
#     pulse_padded = np.pad(pulse, (0, LN - len(pulse)))
#     for k in range(LN):
#         P[:, k] = np.roll(pulse_padded, k)
#     return P

# def ofdm_modulation(symbols, N, L):
#     """OFDM调制（包含IFFT和上采样）"""
#     time_domain = np.fft.ifft(symbols) * np.sqrt(N)
#     upsampled = np.zeros(N*L, dtype=complex)
#     upsampled[::L] = time_domain  # 插入L-1个零
#     return upsampled

# def calculate_acf(signal):
#     """计算周期自相关函数（公式15）"""
#     N = len(signal)
#     acf = np.zeros(N, dtype=complex)
#     for k in range(N):
#         acf[k] = np.dot(signal.conj(), np.roll(signal, k))
#     return acf

# # 参数设置（与论文Fig.2一致）
# N = 128       # 符号数
# L = 10        # 过采样率
# alpha = 0.35  # 滚降因子
# span = 6      # 滤波器跨度（根据旁瓣要求调整）

# # 1. 生成升余弦脉冲
# rc_pulse = generate_rc_pulse(alpha, L, span)

# # 2. 构建循环矩阵P（公式11）
# P = build_circulant_matrix(rc_pulse, N, L)

# # 3. 生成随机QAM符号（16-QAM）
# qam_symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j,
#                                1+3j, 1-3j, -1+3j, -1-3j,
#                                3+1j, 3-1j, -3+1j, -3-1j,
#                                3+3j, 3-3j, -3+3j, -3-3j], N)
# qam_symbols = qam_symbols / np.sqrt(10)  # 16-QAM能量归一化

# # 4. OFDM调制
# ofdm_signal = ofdm_modulation(qam_symbols, N, L)

# # 5. 脉冲成形（公式10）
# pulse_shaped_signal = P @ ofdm_signal

# # 6. 计算ACF（公式15）
# acf = calculate_acf(pulse_shaped_signal)

# # 7. 绘制结果（对应论文Fig.2）
# plt.figure(figsize=(10, 6))
# plt.plot(np.arange(-N*L//2, N*L//2), 10*np.log10(np.abs(np.fft.fftshift(acf))**2),
#          label='Simulated ACF')
# plt.xlabel('Lag (samples)')
# plt.ylabel('Squared ACF (dB)')
# plt.title('Auto-correlation Function of Pulse-shaped OFDM Signal')
# plt.grid(True)
# plt.legend()
# plt.show()























































