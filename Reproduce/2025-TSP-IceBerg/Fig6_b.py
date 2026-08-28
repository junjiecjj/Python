#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 01:15:41 2026

@author: jack
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 6(b) reproduction
"Uncovering the Iceberg in the Sea: ..."
使用与现有代码完全一致的函数和模块。
"""

import numpy as np
import matplotlib.pyplot as plt
import commpy
from Modulations import modulator

# ------------------ 全局绘图设置（与您的代码一致） ------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = [10, 8]
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)

# ------------------ 您已有的函数定义（复制保持一致） ------------------
def FFTmatrix(L):
    mat = np.zeros((L, L), dtype=complex)
    ll = np.arange(L)
    for i in range(L):
        mat[i,:] = np.exp(-1j * 2.0 * np.pi * i * ll / L) / np.sqrt(L)
    return mat

# ------------------ 参数设置 ------------------
Tsym = 1
pi = np.pi
N = 128
L = 10
alpha = 0.35
kappa = 1.32          # 16-QAM kurtosis

FLN = FFTmatrix(L * N)
FN = FFTmatrix(N)

# ------------------ 生成 RRC 脉冲（使用 commpy） ------------------
t, p = commpy.filters.rrcosfilter(L * N, alpha, Tsym, L / Tsym)
p = p / np.sqrt(np.sum(np.power(p, 2)))          # 能量归一化
g = np.real(N * (FLN @ p) * (FLN.conj() @ p.conj()))   # Eq.(23)
g_rrc = np.fft.fftshift(g)       # 仅供画频谱用，此处我们使用未移位的 g 做计算

# ------------------ 生成 16-QAM 星座（使用您的 modulator） ------------------
MOD_TYPE = "qam"
Order = 16
modem, Es, bps = modulator(MOD_TYPE, Order)
Constellation = modem.constellation / np.sqrt(Es)   # 单位功率

# ------------------ 理论平均平方 ACF (OFDM, Eq. (36)) ------------------
Theo_Iceberg = np.zeros(L * N)
Theo_M1 = np.zeros(L * N)
Theo_M10000 = np.zeros(L * N)

for k in range(L * N):
    gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)
    f_k = np.exp(-1j * 2 * pi * k * np.arange(N) / (L * N))
    r1 = np.abs(gk @ f_k.conj()) ** 2
    Theo_Iceberg[k] = r1
    # M=1
    r2 = (kappa - 1) * (N - 2 * (1 - np.cos(2 * pi * k / L)) * np.sum(g[:N] * (1 - g[:N])))
    Theo_M1[k] = r1 + r2
    M = 100
    Theo_M10000[k] = r1 + r2 / M

# 归一化（与您 Fig.6a 风格一致，除以最大值）
Theo_Iceberg = np.fft.fftshift(Theo_Iceberg / Theo_Iceberg.max() + 1e-14)
Theo_M1 = np.fft.fftshift(Theo_M1 / Theo_M1.max() + 1e-14)
Theo_M10000 = np.fft.fftshift(Theo_M10000 / Theo_M10000.max() + 1e-14)

# ------------------ 数值仿真（平均平方 ACF） ------------------
Iter = 10      # 独立实验次数
V = np.eye(N)   # OFDM 时 V = I，详见论文推导
M = 100   # 每个实验块内做 10000 次平均，共 Iter 个块
Sim_M1 = np.zeros((Iter, L * N))
Sim_M10000 = np.zeros((Iter, L * N))

for k in range(L * N):
    fk = FLN[:, k]
    fk_tilde = fk[:N]
    gk = g[:N] + (1 - g[:N]) * np.exp(-1j * 2 * pi * k / L)

    for it in range(Iter):
        # ---- 无相干集成 (M=1) ----
        d = np.random.randint(Order, size=N)
        s = Constellation[d]
        VHs = np.abs(V.conj().T @ s) ** 2   # 因 V=I，即 |s|^2
        Rk = np.sum(gk * VHs * fk_tilde.conj())
        Sim_M1[it, k] = np.abs(Rk) ** 2

        # ---- 10000 次相干集成：平均复数 ACF ----
        Rk_avg = 0.0
        for _ in range(M):
            d = np.random.randint(Order, size=N)
            s = Constellation[d]
            VHs = np.abs(V.conj().T @ s) ** 2
            Rk_avg += np.sum(gk * VHs * fk_tilde.conj())
        Rk_avg /= M
        Sim_M10000[it, k] = np.abs(Rk_avg) ** 2

# 对 Iter 取平均
Sim_M1_avg = Sim_M1.mean(axis=0)
Sim_M1_avg = Sim_M1_avg / Sim_M1_avg.max() + 1e-14
Sim_M1_avg = np.fft.fftshift(Sim_M1_avg)

Sim_M10000_avg = Sim_M10000.mean(axis=0)
Sim_M10000_avg = Sim_M10000_avg / Sim_M10000_avg.max() + 1e-14
Sim_M10000_avg = np.fft.fftshift(Sim_M10000_avg)

# ------------------ 绘图 (Fig. 6b) ------------------
x = np.arange(-N * L // 2, N * L // 2)

fig, ax = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)

ax.plot(x, 10 * np.log10(Sim_M1_avg), color='tab:blue', linestyle='-', linewidth=1.2, label='No Integration, Numerical')
ax.plot(x, 10 * np.log10(Theo_M1), color='tab:blue', linestyle=':', linewidth=1.8, label='No Integration, Theoretical')
ax.plot(x, 10 * np.log10(Sim_M10000_avg), color='tab:orange', linestyle='-', linewidth=1.2, label='10k Coh Integration, Numerical')
ax.plot(x, 10 * np.log10(Theo_M10000), color='tab:orange', linestyle=':', linewidth=1.8, label='10k Coh Integration, Theoretical')
ax.plot(x, 10 * np.log10(Theo_Iceberg), color='black', linestyle='--', linewidth=1.2, label='"Iceberg" of RRC Pulse')

ax.set_xlabel("Delay Index")
ax.set_ylabel("Ambiguity Level (dB)")
ax.set_xlim([-300, 300])
ax.set_ylim([-120, 0])
ax.set_xticks(np.arange(-300, 301, 100))
ax.set_yticks(np.arange(-120, 1, 20))
ax.grid(True)
ax.legend(loc='lower center', edgecolor='black')

# plt.savefig("./Figs/Fig_6b_py.pdf",)
# plt.savefig("./Figs/Fig_6b_py.png", dpi=300)
plt.show()
plt.close()







