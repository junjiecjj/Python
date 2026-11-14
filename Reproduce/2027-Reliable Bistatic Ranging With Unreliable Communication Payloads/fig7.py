#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 20:25:48 2026

@author: jack
"""

"""Fig.7：Reliable Bistatic Ranging With Unreliable Communication Payloads。
同目录放用户原始Modulations.py与ca_cfar.py；只导出最终PNG/PDF图。
四种方法共享每次试验的发送符号与接收噪声；ROC按延迟单元统计，不筛选局部峰。
原文未给出完整信道复系数与ROC统计细节，补充设置已在参数旁注明。
"""
import os
import numpy as np
from scipy.linalg import solve_toeplitz
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from Modulations import modulator
from ca_cfar import cacfar

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
M = 13  # [原文Table I] 默认数据OFDM符号数。
SNRdB = 3  # [原文Fig.7] 导频与数据SNR相同。
sigma2 = 10**(-SNRdB/10)  # [原文Sec.IV] SNR=1/sigma^2。
MonteCarlo = 200  # [已确认的补充设置] 独立发送符号与噪声试验次数；信道固定。
EM_max_iter = 500  # [补充] 与当前Fig.6一致。
EM_tol = 1e-8  # [补充] 相对CIR变化停止条件，与当前Fig.6一致。
training_per_side = 16  # [由Fig.6阈值反推，Fig.7沿用] 每侧训练单元数。
guard_per_side = 4  # [由Fig.6阈值反推，Fig.7沿用] 每侧保护单元数。
PFA_DESIGN_VALUES = np.geomspace(1e-10, 0.1, 81)  # [补充] 仅用于扫描阈值，横轴另行统计。
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Fig7_results")
os.makedirs(output_dir, exist_ok=True)

MOD_TYPE = "qam"
QAM_mod = 16  # [原文Table I] 16-QAM数据。
modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
Es = float(Es)
Constellation = np.asarray(modem.constellation, dtype=complex)/np.sqrt(Es)  # 星座只从用户模块读取。
pilot_modem, pilot_Es, pilot_bps = modulator("psk", 4)  # [原文Table I] QPSK导频。
pilot_Es = float(pilot_Es)
real_levels = np.unique(Constellation.real)
imag_levels = np.unique(Constellation.imag)
if len(real_levels)*len(imag_levels) != len(Constellation):
    raise ValueError("后验矩函数采用均匀方形QAM的精确分解。")
if not 1 <= L <= N or M < 1 or MonteCarlo < 1 or EM_max_iter < 1 or EM_tol <= 0:
    raise ValueError("要求1<=L<=N、M>=1、MonteCarlo>=1、EM_max_iter>=1、EM_tol>0。")
if training_per_side < 1 or guard_per_side < 0 or 2*(training_per_side+guard_per_side) >= L:
    raise ValueError("CFAR训练/保护窗口不合法。")
if np.any((PFA_DESIGN_VALUES <= 0) | (PFA_DESIGN_VALUES >= 1)) or np.any(np.diff(PFA_DESIGN_VALUES) <= 0):
    raise ValueError("设计Pfa应在(0,1)内严格递增。")

# ==================== 信道：沿用当前Fig.6，不为ROC另调参数 ====================
target_indices = np.array([40, 103, 159, 164, 234, 252])  # [Fig.6图中读取并沿用] 1起始位置。
target_bins = target_indices-1
relative_power_dB = np.array([-36.0, -6.0, -28.0, -36.0, 0.0, -6.0])  # [补充] 非原文公开的真实抽头功率。
target_phase = np.zeros(len(target_indices))  # [补充] 各径相位暂设0。
if np.any(target_bins < 0) or np.any(target_bins >= L) or len(np.unique(target_bins)) != len(target_bins):
    raise ValueError("目标位置应在CIR支撑内且不能重复。")
alpha_raw = 10**(relative_power_dB/20)*np.exp(1j*target_phase)
g = np.zeros(L, dtype=complex)
g[target_bins] = np.sqrt(N)*alpha_raw/np.linalg.norm(alpha_raw)  # [原文(1)]；补充归一化mean(|h|^2)=1。
h = np.fft.fft(g, n=N, norm='ortho')  # [原文(2)(3)] h=F_L g。
non_target_mask = np.ones(L, dtype=bool)
non_target_mask[target_bins] = False
K = len(target_bins)
padding = training_per_side+guard_per_side

# ==================== 必要数学功能：与Fig.6的加权LS、后验矩一致 ====================
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

# ==================== Monte Carlo：每帧四种方法共享同一组观测 ====================
method_names = ['Pilot-Only', 'Hard-Decision-Based', 'EM-Based', 'Perfect Payload Knowledge']
detection_count = np.zeros((4, len(PFA_DESIGN_VALUES)), dtype=np.int64)
false_alarm_count = np.zeros_like(detection_count)
EM_iterations_used = np.zeros(MonteCarlo, dtype=int)
EM_converged = np.zeros(MonteCarlo, dtype=bool)
for mc in range(MonteCarlo):
    pilot_bits = np.random.randint(0, 2, N*pilot_bps)
    p = pilot_modem.modulate(pilot_bits)/np.sqrt(pilot_Es)
    data_bits = np.random.randint(0, 2, N*M*bps)
    X = (modem.modulate(data_bits)/np.sqrt(Es)).reshape((N, M), order='F')
    pilot_noise = np.sqrt(sigma2/2)*(np.random.randn(N)+1j*np.random.randn(N))
    data_noise = np.sqrt(sigma2/2)*(np.random.randn(N, M)+1j*np.random.randn(N, M))
    y_p = p*h+pilot_noise  # [原文(2)]。
    Y = X*h[:, None]+data_noise  # [原文(3)]。
    g_pilot = np.fft.ifft(p.conj()*y_p, norm='ortho')[:L]  # [原文(11)/(28)] 导频估计和EM初值。
    h_pilot = np.fft.fft(g_pilot, n=N, norm='ortho')
    h_safe = np.where(np.abs(h_pilot)>1e-14, h_pilot, 1e-14)
    equalized = Y/h_safe[:, None]
    hard_bits = modem.demodulate((equalized*np.sqrt(Es)).ravel(order='F'), 'hard')  # 硬解调调用用户模块。
    X_hard = (modem.modulate(hard_bits)/np.sqrt(Es)).reshape((N, M), order='F')  # 重调制调用用户模块。
    g_hard = weighted_ls(p, y_p, Y, X_hard, np.abs(X_hard)**2, sigma2, L)  # [原文(5)] 代入硬判决参考。
    g_EM = g_pilot.copy()
    for iteration in range(1, EM_max_iter+1):
        h_hat = np.fft.fft(g_EM, n=N, norm='ortho')  # [原文(20)]。
        symbol_mean, symbol_second = posterior_moments(Y, h_hat, sigma2, real_levels, imag_levels)  # [原文(22)(23)] E-step。
        g_next = weighted_ls(p, y_p, Y, symbol_mean, symbol_second, sigma2, L)  # [原文(24)--(27)] M-step。
        relative_change = np.linalg.norm(g_next-g_EM)/max(np.linalg.norm(g_EM), np.finfo(float).tiny)
        g_EM = g_next
        if relative_change < EM_tol:
            EM_converged[mc] = True
            break
    EM_iterations_used[mc] = iteration
    # [Fig.7正文：已知载荷匹配滤波] 由(2)(3)构造；16-QAM下不等于式(5)的加权LS。
    g_perfect = np.fft.ifft((p.conj()*y_p+np.sum(X.conj()*Y, axis=1))/(M+1), norm='ortho')[:L]
    for method, g_hat in enumerate([g_pilot, g_hard, g_EM, g_perfect]):
        power = np.abs(g_hat)**2  # [原文Sec.II-B] 单帧PDP；不先平均PDP再检测。
        padded_power = np.pad(power, (padding, padding), mode='wrap')  # [补充] 沿用Fig.6的循环边缘。
        for j, pfa_design in enumerate(PFA_DESIGN_VALUES):
            positions, threshold, start_cell, stop_cell = cacfar(padded_power, pfa_design, training_per_side, guard_per_side)
            detected = np.zeros(L, dtype=bool)
            detected[positions.astype(int)-padding-1] = True  # 原函数返回1起始位置；不筛局部峰。
            detection_count[method, j] += np.count_nonzero(detected[target_bins])
            false_alarm_count[method, j] += np.count_nonzero(detected[non_target_mask])
    print(f'Fig.7: {mc+1}/{MonteCarlo}, EM iterations={iteration}, relative change={relative_change:.3e}', flush=True)

# ==================== ROC：横轴为实测虚警概率，不是设计Pfa ====================
PD = detection_count/(MonteCarlo*K)  # [补充统计定义] 全部试验的目标命中数/(试验数×目标数)。
PFA = false_alarm_count/(MonteCarlo*(L-K))  # [补充统计定义] 非目标超阈值单元数/(试验数×非目标单元数)。
if np.any(np.diff(detection_count, axis=1)<0) or np.any(np.diff(false_alarm_count, axis=1)<0):
    raise RuntimeError("设计Pfa递增时检测数应非递减，请检查CFAR与统计。")
print(f'EM tolerance reached: {np.count_nonzero(EM_converged)}/{MonteCarlo}; iterations min/mean/max={EM_iterations_used.min()}/{EM_iterations_used.mean():.1f}/{EM_iterations_used.max()}', flush=True)
for method, name in enumerate(method_names):
    visible = np.flatnonzero(PFA[method] <= 0.01)
    if len(visible):
        j = visible[-1]  # 只报告现有扫描点，不插值或移动ROC。
        print(f'{name}: last sampled point with empirical PFA<=0.01: PFA={PFA[method,j]:.6f}, PD={PD[method,j]:.4f}', flush=True)

# ==================== 绘图：用户统一风格，只保存最终图片 ====================
fig, ax = plt.subplots(num=7, figsize=(8, 6), layout='constrained')
colors = ['#00A1F1', '#F65314', '#EDB120', '#8A2BE2']
markers = ['o', 's', '^', 'd']
for method, name in enumerate(method_names):
    ax.plot(PFA[method], PD[method], color=colors[method], marker=markers[method], markevery=6, markerfacecolor='none', label=name)
ax.set(xlabel=r'Empirical false-alarm probability, $P_{FA}$', ylabel=r'Average detection probability, $P_D$', title=f'ROC curve under CA-CFAR, SNR={SNRdB:g} dB')
ax.set_xlim(0, 0.01)
ax.set_ylim(0.4, 1.02)
ax.set_xticks(np.linspace(0, 0.01, 6))
ax.tick_params(direction='in', axis='both', top=True, right=True, width=1)
for spine in ax.spines.values():
    spine.set_linewidth(1)
ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, alpha=0.4)
ax.legend(loc='lower right', prop=FontProperties(family='Times New Roman', size=13))
fig.savefig(os.path.join(output_dir, 'Fig7.pdf'), facecolor='white')
fig.savefig(os.path.join(output_dir, 'Fig7.png'), facecolor='white', dpi=300)
plt.show()
plt.close(fig)
