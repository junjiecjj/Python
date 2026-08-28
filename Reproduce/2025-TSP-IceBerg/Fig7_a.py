#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 02:18:09 2026

@author: jack
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 7(a) reproduction: Ranging RMSE of weak target vs SNR
for SC and OFDM with RRC and optimized pulse (ISL-based).
"""

import numpy as np
import matplotlib.pyplot as plt
import commpy
import cvxpy as cp
from Modulations import modulator
from scipy.signal import resample

# ------------------ 全局设置 ------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 18
np.random.seed(42)

# ------------------ 辅助函数 ------------------
def FFTmatrix(L):
    mat = np.zeros((L, L), dtype=complex)
    ll = np.arange(L)
    for i in range(L):
        mat[i,:] = np.exp(-1j * 2.0 * np.pi * i * ll / L) / np.sqrt(L)
    return mat

def solve_iceberg_shaping_isl(N, L, alpha, K_sl):
    """
    Minimize ISL (Integrated Sidelobe Level) in delay region K_sl.
    min sum_{k in K_sl} |f_k^H g_k|^2
    subject to constraints (45)-(48).
    """
    N_alpha = int(alpha * N)
    N_non_rolloff = N - N_alpha
    N_zeros = N_non_rolloff // 2
    N_ones = N_non_rolloff // 2

    g = cp.Variable(N, nonneg=True)
    constraints = []
    constraints.append(g[0:N_zeros] == 1)
    constraints.append(g[N-N_ones:N] == 0)
    for n in range(N-1):
        constraints.append(g[n+1] - g[n] <= 0)
    constraints.append(cp.sum(g) == N/2)

    isl_terms = []
    for k in K_sl:
        f_k = np.exp(-1j * 2 * np.pi * k * np.arange(N) / (L * N))
        gk = g + (1 - g) * np.exp(-1j * 2 * np.pi * k / L)
        isl_terms.append(cp.abs(f_k.conj().T @ gk) ** 2)

    objective = cp.Minimize(cp.sum(cp.hstack(isl_terms)))
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"ISL optimization successful, objective value: {prob.value}")
        return g.value
    else:
        raise RuntimeError(f"Optimization failed: {prob.status}")

# ------------------ 物理参数 ------------------
B = 122.88e6          # 带宽 Hz
N_sub = 128           # 子载波个数 (与符号数相同)
df = B / N_sub        # 子载波间隔 960 kHz
T_sym = 1 / df        # 符号周期 (不含CP) 约 1.0417 us
T_cp = 500e-9         # CP 长度 500 ns
T_total = T_sym + T_cp   # 总符号时长 1.5417 us

# 仿真过采样率（足够高以近似连续时间）
L_sim = 64            # 每个符号周期的采样点数（含CP？仿真中我们使用总时长）
Ts = T_total / L_sim  # 采样间隔
c = 3e8

# 距离转换
dist_strong = 20.0    # m
dist_weak = 35.0      # m
delay_strong = 2 * dist_strong / c   # 秒
delay_weak = 2 * dist_weak / c
idx_strong = delay_strong / Ts
idx_weak = delay_weak / Ts

# 关注区域 [26.0993, 38.2979] m -> 采样索引范围
dist_min = 26.0993
dist_max = 38.2979
idx_min = 2 * dist_min / (c * Ts)
idx_max = 2 * dist_max / (c * Ts)

# 脉冲成形参数 (用于优化)
alpha = 0.35
N = 128               # 符号数
L_design = 10         # 设计用的过采样率（与论文一致）

# 优化区域（符号级采样）：对应 [5, 15] 个符号间隔，即 k = 5*L_design ~ 15*L_design
K_sl = np.arange(5*L_design, 15*L_design + 1)

# 生成 ISL 优化的脉冲平方频谱
gN_opt = solve_iceberg_shaping_isl(N, L_design, alpha, K_sl)
gN_opt_full = np.hstack((gN_opt, np.zeros((L_design-2)*N), 1 - gN_opt))
FLN_design = FFTmatrix(L_design * N)
P_opt = np.sqrt(np.maximum(np.real(gN_opt_full), 0) / N)
p_opt = FLN_design.conj().T @ P_opt
p_opt = p_opt / np.sqrt(np.sum(np.abs(p_opt)**2))  # 单位能量

# 将脉冲插值到高采样率 (L_sim)
p_opt_high = resample(p_opt, L_sim * N)  # 简单重采样，实际可做更精确插值

# 生成 RRC 脉冲（同样插值到高采样率）
t_rrc, p_rrc = commpy.filters.rrcosfilter(L_design * N, alpha, T_sym, L_design / T_sym)
p_rrc = p_rrc / np.sqrt(np.sum(np.power(p_rrc, 2)))
p_rrc_high = resample(p_rrc, L_sim * N)

# 16-PSK 星座
MOD_TYPE = "psk"
Order = 16
modem, Es, bps = modulator(MOD_TYPE, Order)
Constellation = modem.constellation / np.sqrt(Es)   # 单位功率

num_symbols = N   # 符号数
num_MC = 200      # 蒙特卡洛次数（可增大）
SNR_dB_range = np.arange(-10, 31, 5)   # dB

# 强目标幅度归一化，弱目标低 44 dB
ampl_strong = 1.0
ampl_weak = 10 ** (-44/20)

# 预生成随机符号 (所有MC共享，但每个SNR独立噪声)
symbols_all = np.random.choice(Constellation, size=(num_MC, num_symbols))

# 存储RMSE
rmse_results = {'SC_RRC': [], 'SC_Opt': [], 'OFDM_RRC': [], 'OFDM_Opt': []}

# 循环 SNR
for snr_dB in SNR_dB_range:
    snr_lin = 10 ** (snr_dB / 10)
    errors = {'SC_RRC': [], 'SC_Opt': [], 'OFDM_RRC': [], 'OFDM_Opt': []}

    for mc in range(num_MC):
        s = symbols_all[mc, :]
        # ---------- 构造发送信号 ----------
        # 1) SC: 符号直接脉冲成形（上采样 + 卷积）
        up_sc = np.zeros(num_symbols * L_sim, dtype=complex)
        up_sc[0::L_sim] = s
        tx_sc_rrc = np.convolve(up_sc, p_rrc_high, mode='same')
        tx_sc_opt = np.convolve(up_sc, p_opt_high, mode='same')

        # 2) OFDM: 先 IFFT，再脉冲成形
        X_freq = np.zeros(N_sub, dtype=complex)
        X_freq[:num_symbols] = s
        x_time = np.fft.ifft(X_freq) * np.sqrt(N_sub)   # 保持功率
        up_ofdm = np.zeros(num_symbols * L_sim, dtype=complex)
        up_ofdm[0::L_sim] = x_time
        tx_ofdm_rrc = np.convolve(up_ofdm, p_rrc_high, mode='same')
        tx_ofdm_opt = np.convolve(up_ofdm, p_opt_high, mode='same')

        # 统一信号长度
        signal_len = len(tx_sc_rrc)
        # 对四种信号做相同处理（使用循环移位模拟回波）
        def generate_echo(tx):
            echo = ampl_strong * np.roll(tx, int(round(idx_strong))) \
                   + ampl_weak * np.roll(tx, int(round(idx_weak)))
            return echo

        echo_sc_rrc = generate_echo(tx_sc_rrc)
        echo_sc_opt = generate_echo(tx_sc_opt)
        echo_ofdm_rrc = generate_echo(tx_ofdm_rrc)
        echo_ofdm_opt = generate_echo(tx_ofdm_opt)

        # 加噪声
        # 计算信号功率（基于回波）
        P_signal = np.mean(np.abs(echo_sc_rrc)**2)  # 假设所有信号功率相近
        noise_var = P_signal / snr_lin
        noise = np.sqrt(noise_var/2) * (np.random.randn(signal_len) + 1j*np.random.randn(signal_len))

        rx_sc_rrc = echo_sc_rrc + noise
        rx_sc_opt = echo_sc_opt + noise
        rx_ofdm_rrc = echo_ofdm_rrc + noise
        rx_ofdm_opt = echo_ofdm_opt + noise

        # 匹配滤波（循环互相关）
        def mf(tx, rx):
            # 零填充使长度相同（已同长）
            corr = np.fft.ifft(np.fft.fft(rx) * np.conj(np.fft.fft(tx)))
            return np.fft.fftshift(corr)

        corr_sc_rrc = mf(tx_sc_rrc, rx_sc_rrc)
        corr_sc_opt = mf(tx_sc_opt, rx_sc_opt)
        corr_ofdm_rrc = mf(tx_ofdm_rrc, rx_ofdm_rrc)
        corr_ofdm_opt = mf(tx_ofdm_opt, rx_ofdm_opt)

        # 取幅度
        profile_sc_rrc = np.abs(corr_sc_rrc)
        profile_sc_opt = np.abs(corr_sc_opt)
        profile_ofdm_rrc = np.abs(corr_ofdm_rrc)
        profile_ofdm_opt = np.abs(corr_ofdm_opt)

        # 在关注区域内找弱目标峰值
        half = signal_len // 2
        # 将索引映射到相对于中心的偏移
        def find_peak_in_region(profile, center_idx, radius):
            # 以center_idx为中心，半径radius内找最大值，返回相对于中心的偏移
            center = (center_idx + half) % signal_len
            start = center - radius
            end = center + radius
            # 处理循环边界
            if start < 0:
                idx = np.concatenate((np.arange(start+signal_len, signal_len), np.arange(0, end+signal_len)))
            elif end >= signal_len:
                idx = np.concatenate((np.arange(start, signal_len), np.arange(0, end-signal_len)))
            else:
                idx = np.arange(start, end+1)
            region_vals = profile[idx % signal_len]
            max_local = np.argmax(region_vals)
            return idx[max_local] - center   # 相对于中心偏移

        # 弱目标真实索引（相对于中心的偏移）
        # 由于中心在0，偏移应为 idx_weak - half? 但我们使用循环移位，实际中心为0。
        # 我们在找峰值时使用绝对索引，然后计算距离。
        # 定义搜索半径（覆盖关注区域）
        radius = int( (idx_max - idx_min) / 2 ) + 10
        # 估计弱目标索引（绝对索引）
        def get_est_idx(profile, true_idx_abs, radius):
            # 在profile中找峰值，返回绝对索引
            half = len(profile)//2
            center = (true_idx_abs + half) % len(profile)
            # 取区间
            start = center - radius
            end = center + radius
            if start < 0:
                idx_abs = np.concatenate((np.arange(start+len(profile), len(profile)), np.arange(0, end+len(profile))))
            elif end >= len(profile):
                idx_abs = np.concatenate((np.arange(start, len(profile)), np.arange(0, end-len(profile))))
            else:
                idx_abs = np.arange(start, end+1)
            vals = profile[idx_abs % len(profile)]
            max_local = np.argmax(vals)
            return idx_abs[max_local] % len(profile)  # 绝对索引

        est_idx_sc_rrc = get_est_idx(profile_sc_rrc, idx_weak, radius)
        est_idx_sc_opt = get_est_idx(profile_sc_opt, idx_weak, radius)
        est_idx_ofdm_rrc = get_est_idx(profile_ofdm_rrc, idx_weak, radius)
        est_idx_ofdm_opt = get_est_idx(profile_ofdm_opt, idx_weak, radius)

        # 转换为距离估计
        # 索引相对于中心的偏移 = idx - half
        est_dist = lambda idx: (idx - half) * c * Ts / 2
        error = lambda idx: est_dist(idx) - dist_weak

        errors['SC_RRC'].append(error(est_idx_sc_rrc)**2)
        errors['SC_Opt'].append(error(est_idx_sc_opt)**2)
        errors['OFDM_RRC'].append(error(est_idx_ofdm_rrc)**2)
        errors['OFDM_Opt'].append(error(est_idx_ofdm_opt)**2)

    # 计算RMSE
    rmse_results['SC_RRC'].append(np.sqrt(np.mean(errors['SC_RRC'])))
    rmse_results['SC_Opt'].append(np.sqrt(np.mean(errors['SC_Opt'])))
    rmse_results['OFDM_RRC'].append(np.sqrt(np.mean(errors['OFDM_RRC'])))
    rmse_results['OFDM_Opt'].append(np.sqrt(np.mean(errors['OFDM_Opt'])))

    print(f"SNR={snr_dB} dB done")

# ------------------ 绘图 ------------------
plt.figure(figsize=(10,8))
plt.semilogy(SNR_dB_range, rmse_results['SC_RRC'], 'b-o', label='SC RRC')
plt.semilogy(SNR_dB_range, rmse_results['SC_Opt'], 'b--s', label='SC Opt')
plt.semilogy(SNR_dB_range, rmse_results['OFDM_RRC'], 'r-o', label='OFDM RRC')
plt.semilogy(SNR_dB_range, rmse_results['OFDM_Opt'], 'r--s', label='OFDM Opt')
plt.xlabel('SNR (dB)')
plt.ylabel('RMSE (m)')
plt.grid(True)
plt.legend()
plt.savefig('Fig7a.pdf', dpi=300)
plt.savefig('Fig7a.png', dpi=300)
plt.show()
