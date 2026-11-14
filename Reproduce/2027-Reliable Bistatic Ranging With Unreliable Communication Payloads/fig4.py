#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 22:29:06 2026

@author: jack
"""

"""Fig.4：Reliable Bistatic Ranging With Unreliable Communication Payloads。
同目录放用户的Modulations.py；依赖numpy/scipy/matplotlib/scikit-commpy。
参数、主流程、统计和绘图全部在顶层；只封装加权LS与后验矩。
已确认：沿用Fig.2/3信道，先平均PDP，再对非目标单元计算ECDF。
信道复系数、MC次数、PDP归一化与ECDF统计口径属于补充设置，不能宣称原图数值完全一致。
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
M = 13  # [原文Table I] 默认数据OFDM符号数；Fig.4未另行指定。
SNRdB = 3  # [原文Fig.4] 导频与数据SNR相同。
sigma2 = 10**(-SNRdB/10)  # [原文Sec.IV] SNR=1/sigma^2。
EM_checkpoints = [3, 10, 30]  # [原文Fig.4] 同一条EM轨迹的三个迭代次数。
MonteCarlo = 128  # [补充] 独立试验次数；调试可先改为4或8。
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig4_results")
os.makedirs(output_dir, exist_ok=True)

MOD_TYPE = "qam"
QAM_mod = 16  # [原文Table I] 16-QAM数据。
modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
Constellation = np.asarray(modem.constellation, dtype=complex)/np.sqrt(Es)  # 星座只从用户模块读取。
pilot_modem, pilot_Es, pilot_bps = modulator("psk", 4)  # [原文Table I] QPSK导频。
real_levels = np.unique(Constellation.real)
imag_levels = np.unique(Constellation.imag)
if len(real_levels)*len(imag_levels) != len(Constellation):
    raise ValueError("下方后验矩函数采用均匀方形QAM的精确分解。")
if MonteCarlo < 1 or M < 1 or not 1 <= L <= N:
    raise ValueError("要求MonteCarlo>=1、M>=1、1<=L<=N。")

# [已确认的补充设置] 沿用Fig.2/3信道，所有MC试验中保持g不变。
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
non_target_mask[target_bins] = False  # [已确认的补充设置] 只排除精确on-grid目标单元。
P_ref = np.max(np.abs(g)**2)  # 所有方法共用真值最大抽头功率，不逐条曲线归一化。

# ==================== 必要数学功能：与Fig.3一致 ====================
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

# ==================== 主流程：六种估计共享同一组观测 ====================
method_labels = ['Pilot-Only', 'Hard-Decision-Based', 'EM-Based, 3x Iterations', 'EM-Based, 10x Iterations', 'EM-Based, 30x Iterations', 'Perfect Payload Knowledge']
power_sum = np.zeros((len(method_labels), L))
power_square_sum = np.zeros_like(power_sum)
for mc in range(MonteCarlo):
    pilot_bits = np.random.randint(0, 2, N*pilot_bps)
    p = pilot_modem.modulate(pilot_bits)/np.sqrt(pilot_Es)
    data_bits = np.random.randint(0, 2, N*M*bps)
    X = (modem.modulate(data_bits)/np.sqrt(Es)).reshape((N, M), order='F')  # 数据调制调用用户模块。
    pilot_noise = np.sqrt(sigma2/2)*(np.random.randn(N)+1j*np.random.randn(N))
    data_noise = np.sqrt(sigma2/2)*(np.random.randn(N, M)+1j*np.random.randn(N, M))
    y_p = p*h + pilot_noise  # [原文(2)]。
    Y = X*h[:, None] + data_noise  # [原文(3)]，每列是一个OFDM数据符号的观测。
    g_pilot = np.fft.ifft(p.conj()*y_p, norm='ortho')[:L]  # [原文(11)/(28)] 导频估计。
    h_pilot = np.fft.fft(g_pilot, n=N, norm='ortho')
    h_safe = np.where(np.abs(h_pilot)>1e-14, h_pilot, 1e-14)
    equalized = Y/h_safe[:, None]
    hard_bits = modem.demodulate((equalized*np.sqrt(Es)).ravel(order='F'), 'hard')  # 硬解调调用用户模块。
    X_hard = (modem.modulate(hard_bits)/np.sqrt(Es)).reshape((N, M), order='F')  # 重调制调用用户模块。
    g_hard = weighted_ls(p, y_p, Y, X_hard, np.abs(X_hard)**2, sigma2, L)  # [原文(5)] 用硬判决替代真实符号，保留幅度权重。
    g_perfect = weighted_ls(p, y_p, Y, X, np.abs(X)**2, sigma2, L)  # [原文(5)] 完美载荷加权LS基准；真实X只用于此基准。
    g_hat = g_pilot.copy()  # [原文(28)] EM从导频估计初始化。
    EM_saved = {}
    for iteration in range(1, max(EM_checkpoints)+1):
        h_hat = np.fft.fft(g_hat, n=N, norm='ortho')  # [原文(20)]。
        symbol_mean, symbol_second = posterior_moments(Y, h_hat, sigma2, real_levels, imag_levels)  # [原文(22)(23)] E-step。
        g_hat = weighted_ls(p, y_p, Y, symbol_mean, symbol_second, sigma2, L)  # [原文(24)--(27)] M-step。
        if iteration in EM_checkpoints:
            EM_saved[iteration] = g_hat.copy()  # 保存第3/10/30轮；中途不重新初始化。
    g_estimates = np.stack([g_pilot, g_hard, *[EM_saved[t] for t in EM_checkpoints], g_perfect])
    trial_power = np.abs(g_estimates)**2  # [原文Sec.II-B、(10)] 每次试验的PDP。
    power_sum += trial_power  # 先累加功率，不先累加复CIR，也不先转换为dB。
    power_square_sum += trial_power**2
    if (mc+1) % max(1, MonteCarlo//8) == 0 or mc+1 == MonteCarlo:
        print(f"Fig.4: Monte Carlo {mc+1}/{MonteCarlo}", flush=True)

# ==================== 已确认的统计口径：平均PDP与跨非目标单元ECDF ====================
PDP_mean = power_sum/MonteCarlo  # [原文(10)的MC实现] E[|g_hat[k]|^2]。
PDP_dB = 10*np.log10(np.maximum(PDP_mean/P_ref, np.finfo(float).tiny))  # [补充归一化] 所有曲线共用P_ref。
PDP_MC_SE = np.full_like(PDP_mean, np.nan)
if MonteCarlo > 1:
    PDP_MC_SE = np.sqrt(np.maximum(power_square_sum/MonteCarlo-PDP_mean**2, 0)/(MonteCarlo-1))  # 平均PDP的MC标准误差，在线性功率域。
profile_dB = PDP_dB.copy()
profile_dB[:, ~non_target_mask] = np.nan  # Fig.4(a)目标位置断开，保留其余非目标单元。
ECDF_x = np.sort(PDP_dB[:, non_target_mask], axis=1)  # Fig.4(b)对平均PDP的非目标单元排序，不混合各次试验。
ECDF_y = np.arange(1, np.count_nonzero(non_target_mask)+1)/np.count_nonzero(non_target_mask)  # [补充ECDF定义] F(x)=count(S[k]<=x)/K0。
print('Mean non-target PDP (dB), method order:', method_labels)
print(10*np.log10(np.mean(PDP_mean[:, non_target_mask], axis=1)/P_ref))  # 仅打印辅助统计；不是ECDF横轴样本。

# ==================== 绘图：两个子图分别保存，线型按原图 ====================
colors = ['#00A1F1', '#F65314', '#EDB120', '#8A2BE2', '#77AC30', '#4DBEEE']
line_styles = ['-', '-', '--', '--', '--', '-']
fig_a, ax_a = plt.subplots(num='Fig.4(a)', figsize=(8, 6), layout='constrained')
for method in [1, 2, 3, 4]:
    ax_a.plot(range_axis, profile_dB[method], color=colors[method], ls=line_styles[method], label=method_labels[method])
ax_a.set(xlabel='Range index', ylabel='Response (dB)', title=f'SNR={SNRdB:g} dB, range sidelobe profiles')
ax_a.set_xlim(1, L)
fig_b, ax_b = plt.subplots(num='Fig.4(b)', figsize=(8, 6), layout='constrained')
for method in range(len(method_labels)):
    ecdf_x = np.r_[np.min(ECDF_x)-0.5, ECDF_x[method], np.max(ECDF_x)+0.5]  # 补上0/1两端，使当前视窗内的ECDF完整。
    ecdf_y = np.r_[0.0, ECDF_y, 1.0]
    ax_b.step(ecdf_x, ecdf_y, where='post', color=colors[method], ls=line_styles[method], label=method_labels[method])
ax_b.set(xlabel='Sidelobe level (dB)', ylabel='F(x)', title=f'SNR={SNRdB:g} dB, ECDF of sidelobes')
ax_b.set_xlim(float(np.min(ECDF_x))-0.5, float(np.max(ECDF_x))+0.5)  # 显示全部数据，不强行套用原图未公开信道对应的横轴范围。
ax_b.set_ylim(0, 1.02)
for fig, ax, suffix in [(fig_a, ax_a, 'a'), (fig_b, ax_b, 'b')]:
    ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, prop=FontProperties(family='Times New Roman', size=13))
    fig.savefig(os.path.join(output_dir, f'Fig4{suffix}.pdf'), facecolor='white', bbox_inches='tight')
plt.show()
plt.close(fig_a)
plt.close(fig_b)
