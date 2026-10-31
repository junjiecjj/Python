#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单目标、单帧：直接由B11计算16-PSK与16-QAM的Tr(J^{-1})。"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from Modulations import modulator

#%%>>>>>>>>>>>>>>>>>>>>>>>>  全局绘图设置
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markersize"] = 6
plt.rcParams["figure.facecolor"] = "white"

#%%>>>>>>>>>>>>>>>>>>>>>>>>  仿真参数
np.random.seed(42)
N = 32
M = 16
Ncp = N // 4
QAM_mod = 16
bps = int(np.log2(QAM_mod))
delta_f = 960e3
fc = 30e9
c = 3e8
Ts = (1 + Ncp / N) / delta_f
target_pos = 30.0
target_speed = 200.0
alpha = 1.0 + 0j
tau = 2 * target_pos / c
fd = 2 * fc * target_speed / c
SNR_dB = np.arange(50, 101, 5)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  构造B11中的G0
# 按列向量化：第k+mN行对应子载波k、OFDM符号m。
k = np.tile(np.arange(N), M)
m = np.repeat(np.arange(M), N)
A = np.exp(-2j * np.pi * k * delta_f * tau + 2j * np.pi * m * Ts * fd)
G0 = np.column_stack((-2j * np.pi * delta_f * k * A * alpha,
                      2j * np.pi * Ts * m * A * alpha, A, 1j * A))

#%%>>>>>>>>>>>>>>>>>>>>>>>>  蒙特卡洛B11计算与绘图
Iter = 1000
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

for MOD_TYPE, color, style in [("psk", "#F65314", "o-"),
                                ("qam", "#00A1F1", "s--")]:
    modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
    map_table, demap_table = modem.getMappTable()
    CRB = 0.0

    for trial in range(Iter):
        bits = np.random.randint(0, 2, N * M * bps)
        X = np.asarray(modem.modulate(bits)).reshape(N, M, order="F") / np.sqrt(Es)
        power = np.abs(X.flatten(order="F"))**2
        J = 2 * np.real(G0.conj().T @ (power[:, None] * G0))
        CRB += np.trace(np.linalg.inv(J))

    CRB = CRB / Iter * 10**(-SNR_dB / 10)
    ax.semilogy(SNR_dB, CRB, style, color=color, label=f"{QAM_mod}-{MOD_TYPE.upper()}")

ax.set_xlabel("SNR (dB)")
ax.set_ylabel(r"$\mathbb{E}_{X}\{\mathrm{Tr}(\mathbf{J}^{-1})\}$")
ax.grid(linestyle=(0, (5, 10)), linewidth=0.5)
ax.tick_params(direction="in", top=True, right=True, labelsize=16)
ax.legend(edgecolor="black", framealpha=1)
plt.show()
plt.close(fig)
