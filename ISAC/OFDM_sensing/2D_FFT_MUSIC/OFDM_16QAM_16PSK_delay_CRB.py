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

#%%>>>>>>>>>>>>>>>>>>>>>>>>  每种星座仅生成一帧X，直接计算B11
bits = np.random.randint(0, 2, N * M * bps)
curves = {}
for MOD_TYPE in ["psk", "qam"]:
    modem, Es, bps = modulator(MOD_TYPE, QAM_mod)
    map_table, demap_table = modem.getMappTable()
    X = np.asarray(modem.modulate(bits)).reshape(N, M, order="F") / np.sqrt(Es)
    power = np.abs(X.flatten(order="F"))**2
    CRB = np.zeros(len(SNR_dB))
    for i, snr in enumerate(SNR_dB):
        sigma_W2 = 10**(-snr / 10)
        J = 2 / sigma_W2 * np.real(G0.conj().T @ (power[:, None] * G0))
        J_inv = np.linalg.inv(J)
        CRB[i] = np.trace(J_inv)
    curves[MOD_TYPE] = CRB

#%%>>>>>>>>>>>>>>>>>>>>>>>>  画图
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
ax.semilogy(SNR_dB, curves["psk"], "o-", color="#F65314", label="16-PSK")
ax.semilogy(SNR_dB, curves["qam"], "s--", color="#00A1F1", label="16-QAM")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel(r"$\mathrm{Tr}(\mathbf{J}^{-1})$")
ax.grid(linestyle=(0, (5, 10)), linewidth=0.5)
ax.tick_params(direction="in", top=True, right=True, labelsize=16)
ax.legend(edgecolor="black", framealpha=1)
output = Path(__file__).with_suffix(".pdf")
fig.savefig(output)
plt.show()
plt.close(fig)
