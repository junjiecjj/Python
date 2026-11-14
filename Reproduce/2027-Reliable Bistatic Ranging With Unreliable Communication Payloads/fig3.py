#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 20:21:46 2026

@author: jack
"""

"""Fig.3：Reliable Bistatic Ranging With Unreliable Communication Payloads。
同目录放用户的Modulations.py；依赖numpy/scipy/matplotlib/scikit-commpy。
参数和主流程在顶层；只封装后验矩和加权LS两个数学功能。
已确认：沿用Fig.2信道；纵轴为sqrt(mean_k |mean_MC(g_hat-g)|^2 / N)。
信道复系数与标量偏差定义不是Fig.3原文公开参数，不能据此宣称纵轴数值精确一致。
"""
import os
import numpy as np
from scipy.linalg import solve_toeplitz
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from Modulations import modulator

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markersize"] = 6
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams["lines.linestyle"] = "-"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
np.random.seed(42)

# ==================== 参数：直接修改 ====================
N = 3276  # [原文Table I] 子载波数。
L = 288  # [原文Table I] CIR支撑长度。
M_values = np.array([3, 6, 13, 27, 41, 55, 69])  # [原图矢量曲线读取]。
M_max = int(np.max(M_values))
SNR_values = [3, 8]  # [原文Fig.3]。
EM_checkpoints = [5, 50]  # [原文Fig.3] 保存同一条EM迭代路径的第5、50轮。
MonteCarlo = 128  # [补充] 独立试验次数；调试可先设4或8，再检查统计误差并增加次数。
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig3_results")
os.makedirs(output_dir, exist_ok=True)

MOD_TYPE = "qam"
QAM_mod = 16  # [原文Table I] 16-QAM，bps=4；不是4-QAM。
modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
Constellation = np.asarray(modem.constellation, dtype=complex)/np.sqrt(Es)  # 星座只从用户模块读取。
pilot_modem, pilot_Es, pilot_bps = modulator("psk", 4)  # [原文Table I] QPSK导频。
real_levels = np.unique(Constellation.real)
imag_levels = np.unique(Constellation.imag)
if len(real_levels)*len(imag_levels) != len(Constellation):
    raise ValueError("下方后验矩函数采用均匀方形QAM的精确分解。")

# [已确认的补充设置] 沿用Fig.2信道，所有MC试验中保持g不变。
target_indices = np.array([15, 34, 35, 145, 254, 282])
target_bins = target_indices - 1
target_power_dB = np.array([-0.277667, 0.0, -31.093638, -38.628508, -37.993070, -19.071398])
CIR_PEAK_GAIN = 1.0870961165435937
target_phase = np.zeros(len(target_indices))
if np.any(target_bins < 0) or np.any(target_bins >= L):
    raise ValueError("目标时延超出L；修改L时需同步检查target_indices。")
g = np.zeros(L, dtype=complex)
g[target_bins] = np.sqrt(N)*CIR_PEAK_GAIN*10**(target_power_dB/20)*np.exp(1j*target_phase)  # [原文(1)]。
h = np.fft.fft(g, n=N, norm='ortho')  # [原文(2)(3)] h=F_L g。

# ==================== 必要数学功能：加权LS与后验矩 ====================
def weighted_ls(p, y_p, Y, symbol_mean, symbol_second, sigma2, L):
    w = (1 + np.sum(symbol_second, axis=1))/sigma2  # [原文(24)] 导频和数据噪声方差相同。
    r = (p.conj()*y_p + np.sum(symbol_mean.conj()*Y, axis=1))/sigma2  # [原文(25)]。
    first_column = np.fft.ifft(w)[:L]  # [由原文(5)/(27)推得] F_L^H Diag(w) F_L的首列，此处采用1/N IFFT。
    first_column[0] = first_column[0].real
    rhs = np.fft.ifft(r, norm='ortho')[:L]  # F_L^H r。
    return solve_toeplitz((first_column, first_column.conj()), rhs, check_finite=False)  # 精确求解，不显式求逆。

def posterior_moments(Y, h_hat, sigma2, real_levels, imag_levels):
    # [原文(22)] 均匀方形QAM后验可精确分解成实部和虚部两个PAM后验。
    # levels全部从modem.constellation获得；这里没有自行生成星座。
    sufficient = Y*h_hat[:, None].conj()
    h_power = np.abs(h_hat[:, None, None])**2
    component_moments = []
    for component, levels in [(sufficient.real, real_levels), (sufficient.imag, imag_levels)]:
        log_weights = (2*component[..., None]*levels - h_power*levels**2)/sigma2
        log_weights -= np.max(log_weights, axis=-1, keepdims=True)  # 移除公共常数，避免指数溢出。
        probability = np.exp(log_weights)
        probability /= np.sum(probability, axis=-1, keepdims=True)
        mean = np.sum(probability*levels, axis=-1)
        second = np.sum(probability*levels**2, axis=-1)
        component_moments.append((mean, second))
    symbol_mean = component_moments[0][0] + 1j*component_moments[1][0]  # [原文(23)] E[x|y]。
    symbol_second = component_moments[0][1] + component_moments[1][1]  # [原文(23)] E[|x|^2|y]，不是|E[x|y]|^2。
    return symbol_mean, symbol_second

# ==================== 主流程：共享观测，分别计算HD/EM5/EM50 ====================
# 维度依次为：SNR、方法、M采样点、时延单元；方法顺序HD、EM5、EM50。
error_sum = np.zeros((len(SNR_values), 3, len(M_values), L), dtype=complex)
error_power_sum = np.zeros((len(SNR_values), 3, len(M_values)))
for mc in range(MonteCarlo):
    pilot_bits = np.random.randint(0, 2, N*pilot_bps)
    p = pilot_modem.modulate(pilot_bits)/np.sqrt(pilot_Es)
    data_bits = np.random.randint(0, 2, N*M_max*bps)
    X = (modem.modulate(data_bits)/np.sqrt(Es)).reshape((N, M_max), order='F')  # 数据调制调用用户模块。
    pilot_noise_unit = (np.random.randn(N)+1j*np.random.randn(N))/np.sqrt(2)
    data_noise_unit = (np.random.randn(N, M_max)+1j*np.random.randn(N, M_max))/np.sqrt(2)
    for snr_index, SNRdB in enumerate(SNR_values):
        sigma2 = 10**(-SNRdB/10)  # [原文Sec.IV] 导频与数据采用相同SNR。
        y_p = p*h + np.sqrt(sigma2)*pilot_noise_unit  # [原文(2)]。
        Y_all = X*h[:, None] + np.sqrt(sigma2)*data_noise_unit  # [原文(3)]。
        g_pilot = np.fft.ifft(p.conj()*y_p, norm='ortho')[:L]  # [原文(11)/(28)]。
        h_pilot = np.fft.fft(g_pilot, n=N, norm='ortho')
        h_safe = np.where(np.abs(h_pilot)>1e-14, h_pilot, 1e-14)
        for m_index, M in enumerate(M_values):
            Y = Y_all[:, :M]  # 不同M使用同一帧的前M列；各方法共享同一观测。
            equalized = Y/h_safe[:, None]
            hard_bits = modem.demodulate((equalized*np.sqrt(Es)).ravel(order='F'), 'hard')
            X_hard = (modem.modulate(hard_bits)/np.sqrt(Es)).reshape((N, M), order='F')
            g_hard = weighted_ls(p, y_p, Y, X_hard, np.abs(X_hard)**2, sigma2, L)  # [原文(5)] 必须保留16-QAM幅度权重。
            g_hat = g_pilot.copy()  # [原文(28)] 每个M从同一个导频初值开始。
            EM_saved = {}
            for iteration in range(1, max(EM_checkpoints)+1):
                h_hat = np.fft.fft(g_hat, n=N, norm='ortho')  # [原文(20)]。
                symbol_mean, symbol_second = posterior_moments(Y, h_hat, sigma2, real_levels, imag_levels)  # [原文(22)(23)] E-step。
                g_hat = weighted_ls(p, y_p, Y, symbol_mean, symbol_second, sigma2, L)  # [原文(24)--(27)] M-step。
                if iteration in EM_checkpoints:
                    EM_saved[iteration] = g_hat.copy()  # 第5轮保存后继续迭代，不重新初始化。
            errors = np.stack([g_hard-g, EM_saved[5]-g, EM_saved[50]-g])
            error_sum[snr_index, :, m_index] += errors  # [原文(6)] 先累计复误差。
            error_power_sum[snr_index, :, m_index] += np.mean(np.abs(errors)**2, axis=-1)/N
        print(f"Fig.3: MC={mc+1}/{MonteCarlo}, SNR={SNRdB} dB", flush=True)

# ==================== 偏差统计：不是MSE ====================
bias_vector = error_sum/MonteCarlo  # [原文(6)] E[g_hat]-g的MC估计。
Bias_squared = np.mean(np.abs(bias_vector)**2, axis=-1)/N  # [已确认的标量定义] ||bias_vector||^2/(N*L)。
Bias = np.sqrt(Bias_squared)
MC_mean_SE = np.sqrt(np.maximum(error_power_sum/MonteCarlo-Bias_squared, 0)/max(MonteCarlo-1, 1))  # MC均值的RMS标准误差。
print('Bias at the largest M, rows=SNR and columns=HD/EM5/EM50:')
print(Bias[:, :, -1])
print('Corresponding Monte Carlo mean standard errors:')
print(MC_mean_SE[:, :, -1])
if MonteCarlo < 2:
    print('MonteCarlo=1无法估计MC均值标准误差；请增加试验次数。')
elif np.any(Bias[:, 1:, :] < 3*MC_mean_SE[:, 1:, :]):
    print('部分EM偏差与MC统计误差同量级；请增加MonteCarlo后再判断小偏差及迭代次数排序。')

# ==================== 绘图：六条曲线，不额外平移或缩放 ====================
fig, ax = plt.subplots(num=3, figsize=(8, 6), layout='constrained')
curve_order = [(0, 0, '--', '*'), (1, 0, '--', '*'), (0, 1, '-', 'o'), (0, 2, '-', 'o'), (1, 1, '-', '>'), (1, 2, '-', '>')]
colors = ['#00A1F1', '#F65314', '#EDB120', '#8A2BE2', '#77AC30', '#4DBEEE']
for color, (s, method, ls, marker) in zip(colors, curve_order):
    label = f'Hard-Decision-Based, SNR={SNR_values[s]} dB' if method == 0 else f'EM-Based, SNR={SNR_values[s]} dB, Iter={EM_checkpoints[method-1]}'
    ax.semilogy(M_values, np.maximum(Bias[s, method], np.finfo(float).tiny), color=color, ls=ls, marker=marker, markerfacecolor='none', label=label)
ax.set(xlabel='M', ylabel='Normalized RMS CIR bias', title='Finite M and finite iteration bias')
ax.set_xlim(M_values[0], M_values[-1])
ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
for spine in ax.spines.values():
    spine.set_linewidth(1)
ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, prop=FontProperties(family='Times New Roman', size=13))
fig.savefig(os.path.join(output_dir, 'Fig3.pdf'), facecolor='white', bbox_inches='tight')
plt.show()
plt.close(fig)




