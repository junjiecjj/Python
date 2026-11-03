#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按E9直接构造完整FIM并求逆，比较距离与速度CRB。"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from DigiCommPy.modem import PSKModem, QAMModem

#%%>>>>>>>>>>>>>>>>>>>>>>>>  全局绘图设置
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

#%%>>>>>>>>>>>>>>>>>>>>>>>>  物理参数
N = 128                              # 子载波数
M = 256                              # OFDM符号数
Ncp = N // 4                         # CP采样点数
Q = 64                              # 目标数

N = 16                              # 子载波数
M = 4                              # OFDM符号数
Ncp = N // 4                         # CP采样点数
Q = 6                              # 目标数


QAM_mod = 16
TARGET_SEED = 42                     # 已验证的固定随机目标示例
Iter = 10000                         # QAM随机数据帧数；PSK条件CRB与数据相位无关
delta_f = 960e3                      # Hz
fc = 30e9                           # Hz
c = 3e8                             # m/s
dt = 1 / (N * delta_f)               # 快时间采样间隔，s
Tcp = Ncp * dt                      # CP时长，s
Tsym = (N + Ncp) * dt                # 含CP的符号时长，s
SNR_dB = np.arange(0, 21, 2)         # 归一化平均符号能量1 / sigma_W^2
modem_dict = {"psk": PSKModem, "qam": QAMModem}

#%%>>>>>>>>>>>>>>>>>>>>>>>>  目标物理参数与cell/bin换算
rng_targets = np.random.default_rng(TARGET_SEED)
range_max = c * Tcp / 2
# 距离由N(range_max/2, (range_max/2)^2)在(0, range_max)内截断生成。
target_pos = rng_targets.normal(range_max / 2, range_max / 2, Q)
while np.any((target_pos <= 0) | (target_pos >= range_max)):
    invalid = (target_pos <= 0) | (target_pos >= range_max)
    target_pos[invalid] = rng_targets.normal(range_max / 2, range_max / 2, np.count_nonzero(invalid))
# 速度由N(0, 100^2)在(-200, 200) m/s内截断生成。
target_speed = rng_targets.normal(0.0, 100.0, Q)
while np.any(np.abs(target_speed) >= 200):
    invalid = np.abs(target_speed) >= 200
    target_speed[invalid] = rng_targets.normal(0.0, 100.0, np.count_nonzero(invalid))
alpha = np.ones(Q, dtype=complex)                     # 已吸收载波时延相位
tau = 2 * target_pos / c                              # s
fd = 2 * fc * target_speed / c                        # Hz

range_cell = c / (2 * N * delta_f)                    # m/bin
doppler_cell = 1 / (M * Tsym)                         # Hz/bin
velocity_cell = c / (2 * fc * M * Tsym)                # (m/s)/bin
tau_bin = tau / dt                                   # 等于target_pos / range_cell
nu_bin = fd * M * Tsym                               # 等于target_speed / velocity_cell

#%%>>>>>>>>>>>>>>>>>>>>>>>>  CP覆盖与无模糊区间检查
if np.any((tau_bin < 0) | (tau_bin > Ncp)):
    raise ValueError(f"目标时延超出CP：要求0 <= tau_bin <= {Ncp}，对应距离0至{c * Tcp / 2:.6f} m。")
if np.any((nu_bin < -M / 2) | (nu_bin >= M / 2)):
    raise ValueError(f"目标多普勒超出无模糊区间：要求{-M / 2:g} <= nu_bin < {M / 2:g}。")
print(f"CP覆盖距离：{c * Tcp / 2:.6f} m；目标最大距离：{target_pos.max():.6f} m。")
print(f"距离cell：{range_cell:.6f} m；速度cell：{velocity_cell:.6f} m/s。")
print(f"多普勒频率cell：{doppler_cell:.6f} Hz。")

#%%>>>>>>>>>>>>>>>>>>>>>>>>  仅由cell/bin构造多目标导向矩阵
# X为(M,N)，按行展开；第mN+k行对应第m个符号、第k个子载波。
k = np.tile(np.arange(N), M)
m = np.repeat(np.arange(M), N)
A = np.exp(-2j * np.pi * k[:, None] * tau_bin[None, :] / N
           + 2j * np.pi * m[:, None] * nu_bin[None, :] / M)

#%%>>>>>>>>>>>>>>>>>>>>>>>>  逐帧计算条件CRB，再对CRB取平均
samples = {}
delay_curves = {}
doppler_curves = {}
for MOD_TYPE in ["psk", "qam"]:
    modem = modem_dict[MOD_TYPE](QAM_mod)
    Es = np.mean(np.abs(modem.modulate(np.arange(QAM_mod)))**2)
    rng_symbols = np.random.default_rng(42)
    # 归一化PSK的|X[m,k]|^2恒为1，其条件FIM对所有帧相同。
    nFrame = 1 if MOD_TYPE == "psk" else Iter
    samples[MOD_TYPE] = np.empty((nFrame, 2))
    for trial in range(nFrame):
        d = rng_symbols.integers(0, QAM_mod, M * N)
        X = modem.modulate(d).reshape(M, N) / np.sqrt(Es)
        Bmat = X.flatten()[:, None] * A
        D_tau = -2j * np.pi / N * k[:, None] * Bmat * alpha[None, :]
        D_nu = 2j * np.pi / M * m[:, None] * Bmat * alpha[None, :]
        # 参数顺序：[tau_bin, nu_bin, alpha_R, alpha_I]，每组含Q个参数。
        G = np.column_stack((D_tau, D_nu, Bmat, 1j * Bmat))
        # E9：sigma_W^2=1，直接构造完整FIM并求逆。
        J = 2 * np.real(G.conj().T @ G)
        C = np.linalg.pinv(J)
        # C对应归一化参数；乘以各自cell尺度的平方，得到物理量CRB。
        samples[MOD_TYPE][trial, 0] = range_cell**2 * np.trace(C[:Q, :Q])
        samples[MOD_TYPE][trial, 1] = velocity_cell**2 * np.trace(C[Q:2 * Q, Q:2 * Q])
        if (trial + 1) % 500 == 0:
            print(f"{MOD_TYPE.upper()}: {trial + 1}/{nFrame} frames", flush=True)
    name = f"{QAM_mod}-{MOD_TYPE.upper()}"
    delay_curves[name] = samples[MOD_TYPE][:, 0].mean() * 10**(-SNR_dB / 10)
    doppler_curves[name] = samples[MOD_TYPE][:, 1].mean() * 10**(-SNR_dB / 10)
    print(f"{name}, SNR=0 dB: 平均距离CRB之和={samples[MOD_TYPE][:, 0].mean():.12e} m^2")
    print(f"{name}, SNR=0 dB: 平均速度CRB之和={samples[MOD_TYPE][:, 1].mean():.12e} (m/s)^2")

# 相对差异使用“平均CRB之比”；PSK基准不含蒙特卡洛采样误差。
relative_gap = 100 * (samples["qam"].mean(axis=0) / samples["psk"][0] - 1)
relative_se = 100 * samples["qam"].std(axis=0, ddof=1) / np.sqrt(Iter) / samples["psk"][0]
for i, label in enumerate(["距离", "速度"]):
    print(f"{label}: QAM相对PSK的平均差异={relative_gap[i]:.6f}%，MC标准误差={relative_se[i]:.6f}个百分点")
    print(f"  近似95%置信区间=[{relative_gap[i] - 1.96 * relative_se[i]:.6f}, {relative_gap[i] + 1.96 * relative_se[i]:.6f}]%")

#%%>>>>>>>>>>>>>>>>>>>>>>>>  画图
#%%>>>>>>>>>>>>>>>>>>>>>>>>  画图并保存PDF
colors = ["#F65314", "#00A1F1"]
styles = ["o-", "s--"]
font1 = FontProperties(family="Times New Roman", style="normal", size=16)
for filename, curves, ylabel in [("Range_CRB.pdf", delay_curves, r"Range CRB (m$^2$)"), ("Velocity_CRB.pdf", doppler_curves, r"Velocity CRB ((m/s)$^2$)")]:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for i, (name, CRB) in enumerate(curves.items()):
        ax.plot(SNR_dB, CRB, styles[i], color=colors[i], label=name)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(SNR_dB[0], SNR_dB[-1])
    ax.grid(linestyle=(0, (5, 10)), linewidth=0.5, which="both")
    ax.legend(edgecolor="black", framealpha=1, prop=font1)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.tick_params(direction="in", top=True, right=True, labelsize=16, width=1)
    fig.savefig(filename, format="pdf", bbox_inches="tight")
    print(f"Saved {filename}")
    plt.show()
    plt.close(fig)



