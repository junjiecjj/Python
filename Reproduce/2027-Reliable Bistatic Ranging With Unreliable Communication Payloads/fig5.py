#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:29:04 2026

@author: jack
"""

"""Fig.5：Reliable Bistatic Ranging With Unreliable Communication Payloads。
同目录放用户的Modulations.py；依赖numpy/scipy/matplotlib/scikit-commpy。
参数、主流程、统计与绘图均在顶层；仅封装加权LS和后验矩。
已确认：16-QAM、M=30/300、50轮EM，沿用Fig.2--4信道，先平均PDP再取dB。
信道系数、MC次数、EM轮数及共同归一化参考属于补充设置，不能宣称原图数值完全一致。
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
M_values = np.array([30, 300])  # [原文Fig.5]。
M_max = int(np.max(M_values))
theory_asymptotic_reduction_dB = 10*np.log10(M_values[1]/M_values[0])  # [原文(37)] 1/M渐近标度对应的理论参考。
SNRdB = 3  # [原文Fig.5] 导频与数据SNR相同。
sigma2 = 10**(-SNRdB/10)  # [原文Sec.IV] SNR=1/sigma^2。
EM_iterations = 50  # [已确认的补充设置] Fig.5未注明轮数，两个M统一使用50轮。
EM_tail_start = max(0, EM_iterations-10)  # 记录最后10轮的迭代位移；不作为提前停止条件。
MonteCarlo = 64  # [补充] 独立试验次数；调试可先改为4或8，完整运行较慢。
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig5_results")
os.makedirs(output_dir, exist_ok=True)

MOD_TYPE = "qam"
QAM_mod = 16  # [原文Table I] Fig.5使用16-QAM；Fig.2中的BPSK不沿用到这里。
modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
Constellation = np.asarray(modem.constellation, dtype=complex)/np.sqrt(Es)  # 星座只从用户模块读取。
pilot_modem, pilot_Es, pilot_bps = modulator("psk", 4)  # [原文Table I] QPSK导频。
real_levels = np.unique(Constellation.real)
imag_levels = np.unique(Constellation.imag)
if len(real_levels)*len(imag_levels) != len(Constellation):
    raise ValueError("下方后验矩函数采用均匀方形QAM的精确分解。")
if len(M_values) != 2 or M_values[0] >= M_values[1]:
    raise ValueError("M_values需为两组递增的正整数。")
if MonteCarlo < 1 or EM_iterations < 1 or np.any(M_values < 1) or not 1 <= L <= N:
    raise ValueError("要求MonteCarlo>=1、EM_iterations>=1、M>=1、1<=L<=N。")

# [已确认的补充设置] 沿用Fig.2--4信道，所有MC试验中保持g不变。
target_indices = np.array([15, 34, 35, 145, 254, 282])  # 作图采用1起始索引。
target_bins = target_indices - 1  # Python数组采用0起始索引。
target_power_dB = np.array([-0.277667, 0.0, -31.093638, -38.628508, -37.993070, -19.071398])
CIR_PEAK_GAIN = 1.0870961165435937
target_phase = np.zeros(len(target_indices))
if np.any(target_bins < 0) or np.any(target_bins >= L):
    raise ValueError("目标时延超出L；修改L时需同步检查target_indices。")
g = np.zeros(L, dtype=complex)
g[target_bins] = np.sqrt(N)*CIR_PEAK_GAIN*10**(target_power_dB/20)*np.exp(1j*target_phase)  # [原文(1)] 稀疏CIR；系数为补充设置。
h = np.fft.fft(g, n=N, norm='ortho')  # [原文(2)(3)] h=F_L g，采用单位化DFT。
range_axis = np.arange(1, L+1)
non_target_mask = np.ones(L, dtype=bool)
non_target_mask[target_bins] = False  # [已确认的补充设置] 仅在背景统计中排除目标单元；绘图保留全部单元。
P_ref = np.max(np.abs(g)**2)  # 所有方法共用真值最大抽头功率，不逐条曲线归一化。

# ==================== 必要数学功能：与Fig.3/4一致 ====================
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

# ==================== 主流程：共享导频，30列数据取自300列的前缀 ====================
method_labels = ['Pilot-Only'] + [f'M={M}, {name}' for M in M_values for name in ['Hard-Decision-Based', 'EM-Based', 'Perfect Payload Knowledge']]
power_sum = np.zeros((len(method_labels), L))
power_square_sum = np.zeros_like(power_sum)
EM_relative_change = np.zeros((MonteCarlo, len(M_values), EM_iterations))
EM_tail_change_power = np.zeros((MonteCarlo, len(M_values)))
for mc in range(MonteCarlo):
    pilot_bits = np.random.randint(0, 2, N*pilot_bps)
    p = pilot_modem.modulate(pilot_bits)/np.sqrt(pilot_Es)
    data_bits = np.random.randint(0, 2, N*M_max*bps)
    X_all = (modem.modulate(data_bits)/np.sqrt(Es)).reshape((N, M_max), order='F')  # 数据调制调用用户模块。
    pilot_noise = np.sqrt(sigma2/2)*(np.random.randn(N)+1j*np.random.randn(N))
    data_noise = np.sqrt(sigma2/2)*(np.random.randn(N, M_max)+1j*np.random.randn(N, M_max))
    y_p = p*h + pilot_noise  # [原文(2)]。
    Y_all = X_all*h[:, None] + data_noise  # [原文(3)] 每列对应一个数据OFDM符号。
    g_pilot = np.fft.ifft(p.conj()*y_p, norm='ortho')[:L]  # [原文(11)/(28)]。
    h_pilot = np.fft.fft(g_pilot, n=N, norm='ortho')
    h_safe = np.where(np.abs(h_pilot)>1e-14, h_pilot, 1e-14)
    equalized_all = Y_all/h_safe[:, None]
    hard_bits = modem.demodulate((equalized_all*np.sqrt(Es)).ravel(order='F'), 'hard')  # 硬解调调用用户模块。
    X_hard_all = (modem.modulate(hard_bits)/np.sqrt(Es)).reshape((N, M_max), order='F')  # 重调制调用用户模块。
    trial_estimates = [g_pilot]  # 同一条导频曲线只统计一次。
    for m_index, M in enumerate(M_values):
        X, Y, X_hard = X_all[:, :M], Y_all[:, :M], X_hard_all[:, :M]
        g_hard = weighted_ls(p, y_p, Y, X_hard, np.abs(X_hard)**2, sigma2, L)  # [原文(5)] 用硬判决替代真实符号，保留16-QAM幅度权重。
        g_perfect = weighted_ls(p, y_p, Y, X, np.abs(X)**2, sigma2, L)  # [原文(5)] 完美载荷加权LS基准；真实X只用于此基准。
        g_hat = g_pilot.copy()  # [原文(28)] 两个M分别从同一个导频估计开始。
        g_tail = g_pilot.copy()
        for iteration in range(1, EM_iterations+1):
            h_hat = np.fft.fft(g_hat, n=N, norm='ortho')  # [原文(20)]。
            symbol_mean, symbol_second = posterior_moments(Y, h_hat, sigma2, real_levels, imag_levels)  # [原文(22)(23)] E-step。
            g_next = weighted_ls(p, y_p, Y, symbol_mean, symbol_second, sigma2, L)  # [原文(24)--(27)] M-step。
            EM_relative_change[mc, m_index, iteration-1] = np.linalg.norm(g_next-g_hat)/max(np.linalg.norm(g_hat), np.finfo(float).tiny)
            g_hat = g_next
            if iteration == EM_tail_start:
                g_tail = g_hat.copy()
        EM_tail_change_power[mc, m_index] = np.mean(np.abs(g_hat[non_target_mask]-g_tail[non_target_mask])**2)  # 末尾迭代在非目标单元上的平均位移功率。
        trial_estimates.extend([g_hard, g_hat, g_perfect])
        print(f"Fig.5: Monte Carlo {mc+1}/{MonteCarlo}, M={M}", flush=True)
    trial_power = np.abs(np.stack(trial_estimates))**2  # [原文Sec.II-B、(10)] 每次试验的PDP。
    power_sum += trial_power  # 先累加功率，不先平均复CIR，也不先取dB。
    power_square_sum += trial_power**2

# ==================== 统计：完整PDP、实际背景下降量和末尾迭代变化 ====================
PDP_mean = power_sum/MonteCarlo  # [原文(10)的MC实现] E[|g_hat[k]|^2]。
PDP_dB = 10*np.log10(np.maximum(PDP_mean/P_ref, np.finfo(float).tiny))  # 所有曲线共用真值参考功率。
PDP_MC_SE = np.full_like(PDP_mean, np.nan)
if MonteCarlo > 1:
    PDP_MC_SE = np.sqrt(np.maximum(power_square_sum/MonteCarlo-PDP_mean**2, 0)/(MonteCarlo-1))  # 平均PDP在线性功率域的MC标准误差。
background_power = np.mean(PDP_mean[:, non_target_mask], axis=1)  # 只在这一辅助统计中排除真实目标单元。
reduction_dB = 10*np.log10(background_power[1:4]/background_power[4:7])  # 顺序：HD、EM、完美载荷；测量30到300的实际下降量。
EM_tail_fraction = np.mean(EM_tail_change_power, axis=0)/background_power[[2, 5]]  # 末尾位移功率相对于EM非目标背景功率。
print('Measured background reductions, HD/EM/Perfect (dB):', reduction_dB)
print('Mean relative CIR change at the final EM step:', np.mean(EM_relative_change[:, :, -1], axis=0))
print('Mean tail displacement / EM background power:', EM_tail_fraction)
print(f'Asymptotic reference from Eq.(37): {theory_asymptotic_reduction_dB:.3f} dB; plotted data use the measured estimates.')

# ==================== 绘图：七条曲线，保留全部目标单元 ====================
fig, ax = plt.subplots(num=5, figsize=(8, 6), layout='constrained')
colors = ['#00A1F1', '#F65314', '#EDB120', '#8A2BE2', '#77AC30', '#4DBEEE', '#A2142F']
line_styles = ['--', '-', '-', '-.', '-', '-', '-.']  # [原文Fig.5] 两条完美载荷曲线均为点划线，无marker。
for method, label in enumerate(method_labels):
    ax.plot(range_axis, PDP_dB[method], color=colors[method], ls=line_styles[method], label=label)
for k in target_indices:
    ax.axvline(k, color='black', ls='--', lw=1.2, alpha=0.4)  # 真值位置辅助线，不额外增加图例曲线。
background_dB = 10*np.log10(background_power/P_ref)
ax.annotate('', xy=(8, background_dB[5]), xytext=(8, background_dB[2]), arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
ax.text(20, (background_dB[2]+background_dB[5])/2, f'{reduction_dB[1]:.2f} dB (EM)', color='red', fontsize=12)  # 标注实际下降量，不将10dB写死。
ax.set(xlabel='Range index', ylabel='Response (dB)', title=f'SNR={SNRdB:g} dB, direct coherent integration')
ax.set_xlim(1, L)
ax.set_ylim(5*np.floor(np.min(PDP_dB)/5), max(1, np.max(PDP_dB)+0.5))
ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
for spine in ax.spines.values():
    spine.set_linewidth(1)
ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, prop=FontProperties(family='Times New Roman', size=13))
fig.savefig(os.path.join(output_dir, 'Fig5.pdf'), facecolor='white', bbox_inches='tight')
plt.show()
plt.close(fig)
