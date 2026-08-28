#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 6(c) reproduction
"Uncovering the Iceberg in the Sea: ..."
对比优化脉冲与 RRC 脉冲的 10k 相干集成结果（理论 + 蒙特卡洛仿真）。
理论：实线；仿真：虚线（带标记）。
"""

import numpy as np
import matplotlib.pyplot as plt
import commpy
import cvxpy as cp
from Modulations import modulator   # 您的星座生成模块

# ------------------ 绘图设置（与您的代码一致） ------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['legend.fontsize'] = 16
np.random.seed(42)

# ------------------ 从您的代码中复制的函数 ------------------
def FFTmatrix(L):
    mat = np.zeros((L, L), dtype=complex)
    ll = np.arange(L)
    for i in range(L):
        mat[i,:] = np.exp(-1j * 2.0 * np.pi * i * ll / L) / np.sqrt(L)
    return mat

def solve_iceberg_shaping_psl(N, L, alpha, K_s1):
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
    psl_terms = []
    for k in K_s1:
        f_k = np.exp(-1j * 2 * np.pi * k * np.arange(N) / (L * N))
        gk = g + (1 - g) * np.exp(-1j * 2 * np.pi * k / L)
        psl_terms.append(cp.abs(f_k.conj().T @ gk) ** 2)
    objective = cp.Minimize(cp.max(cp.hstack(psl_terms)))
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return g.value
    else:
        raise RuntimeError(f"Optimization failed: {prob.status}")


# ------------------ 参数设置 ------------------
Tsym = 1
pi = np.pi
N = 128
L = 10
alpha = 0.35
kappa = 1.32          # 16-QAM kurtosis

FLN = FFTmatrix(L * N)
FN = FFTmatrix(N)

# ---------- 生成优化脉冲的平方频谱 g_opt ----------
K_s1 = np.arange(5 * L, 15 * L + 1)   # 优化区域
gN_opt = solve_iceberg_shaping_psl(N, L, alpha, K_s1)
gN_opt_full = np.hstack((gN_opt, np.zeros((L-2)*N), 1 - gN_opt))
P_opt = np.sqrt(np.maximum(np.real(gN_opt_full), 0) / N)
p_opt = FLN.conj().T @ P_opt
p_opt = p_opt / np.sqrt(np.sum(np.abs(p_opt)**2))
g_opt = np.real(N * (FLN @ p_opt) * (FLN.conj() @ p_opt.conj()))

# ---------- 生成 RRC 脉冲的平方频谱 g_rrc ----------
t, p_rrc = commpy.filters.rrcosfilter(L * N, alpha, Tsym, L / Tsym)
p_rrc = p_rrc / np.sqrt(np.sum(np.power(p_rrc, 2)))
g_rrc = np.real(N * (FLN @ p_rrc) * (FLN.conj() @ p_rrc.conj()))

# ---------- 理论平均平方 ACF (M=10000 相干集成) ----------
def compute_theory_acf(g, kappa, N, L):
    ACF = np.zeros(L * N)
    for k in range(L * N):
        gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)
        f_k = np.exp(-1j * 2 * pi * k * np.arange(N) / (L * N))
        r1 = np.abs(gk @ f_k.conj()) ** 2
        M = 100
        r2 = (kappa - 1) / M * (N - 2 * (1 - np.cos(2 * pi * k / L)) * np.sum(g[:N] * (1 - g[:N])))
        ACF[k] = r1 + r2
    # 归一化
    ACF = np.fft.fftshift(ACF / ACF.max() + 1e-14)
    return ACF

ACF_opt_theo = compute_theory_acf(g_opt, kappa, N, L)
ACF_rrc_theo = compute_theory_acf(g_rrc, kappa, N, L)

# ---------- 蒙特卡洛仿真 (M=10000 相干集成) ----------
# 使用 16-QAM 星座（来自您的 Modulations 模块）
MOD_TYPE = "qam"
Order = 16
modem, Es, bps = modulator(MOD_TYPE, Order)
Constellation = modem.constellation / np.sqrt(Es)   # 单位功率

V = np.eye(N)   # OFDM 时 V=I

# 仿真参数：总集成次数 = M_blocks * Iter，这里尽量接近 10000
Iter = 10       # 独立实验块数，总平均次数 = 100*100 = 10000
M_blocks = 100   # 每个实验块内的平均次数
Sim_opt = np.zeros((Iter, L * N))
Sim_rrc = np.zeros((Iter, L * N))

for k in range(L * N):
    fk = FLN[:, k]
    fk_tilde = fk[:N]
    # 优化脉冲
    gk_opt = g_opt[:N] + (1 - g_opt[:N]) * np.exp(-1j * 2 * pi * k / L)
    # RRC 脉冲
    gk_rrc = g_rrc[:N] + (1 - g_rrc[:N]) * np.exp(-1j * 2 * pi * k / L)

    for it in range(Iter):
        # ---- 优化脉冲的平均 ACF (复数平均) ----
        Rk_avg_opt = 0.0
        Rk_avg_rrc = 0.0
        for _ in range(M_blocks):
            d = np.random.randint(Order, size=N)
            s = Constellation[d]
            VHs = np.abs(V.conj().T @ s) ** 2   # 因为 V=I，即 |s|^2
            Rk_avg_opt += np.sum(gk_opt * VHs * fk_tilde.conj())
            Rk_avg_rrc += np.sum(gk_rrc * VHs * fk_tilde.conj())
        Rk_avg_opt /= M_blocks
        Rk_avg_rrc /= M_blocks
        Sim_opt[it, k] = np.abs(Rk_avg_opt) ** 2
        Sim_rrc[it, k] = np.abs(Rk_avg_rrc) ** 2

# 对 Iter 取平均，并归一化
Sim_opt_avg = Sim_opt.mean(axis=0)
Sim_opt_avg = Sim_opt_avg / Sim_opt_avg.max() + 1e-14
Sim_opt_avg = np.fft.fftshift(Sim_opt_avg)

Sim_rrc_avg = Sim_rrc.mean(axis=0)
Sim_rrc_avg = Sim_rrc_avg / Sim_rrc_avg.max() + 1e-14
Sim_rrc_avg = np.fft.fftshift(Sim_rrc_avg)

# ------------------ 绘图 (理论实线 + 仿真虚线) ------------------
x = np.arange(-N * L // 2, N * L // 2)

fig, ax = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)

# 优化脉冲：理论（红色实线），仿真（红色虚线）
ax.plot(x, 10 * np.log10(ACF_opt_theo), color='tab:red', linestyle='-', linewidth=2, label='Optimized Pulse (Theory)')
ax.plot(x, 10 * np.log10(Sim_opt_avg), color='tab:red', linestyle='--', linewidth=2, label='Optimized Pulse (MC)')

# RRC 脉冲：理论（蓝色实线），仿真（蓝色虚线）
ax.plot(x, 10 * np.log10(ACF_rrc_theo), color='tab:blue', linestyle='-', linewidth=2, label='RRC Pulse (Theory)')
ax.plot(x, 10 * np.log10(Sim_rrc_avg), color='tab:blue', linestyle='--', linewidth=2, label='RRC Pulse (MC)')

ax.set_xlabel("Delay Index")
ax.set_ylabel("Ambiguity Level (dB)")
ax.set_xlim([-300, 300])
ax.set_ylim([-80, 0])
ax.set_xticks(np.arange(-300, 301, 100))
ax.set_yticks(np.arange(-80, 1, 20))
ax.grid(True)
ax.legend(loc='lower center', edgecolor='black', ncol=2)  # 两列图例

# plt.savefig("./Figs/Fig_6c_py.pdf",)
# plt.savefig("./Figs/Fig_6c_py.png", dpi=300)
plt.show()
plt.close()







