#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 17:00:34 2025

@author: jack



"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift
import os
import warnings

# 设置随机种子
np.random.seed(1)

# ================= 信号与频谱参数 =================
fs = 200e6  # 采样率 [Hz]
PRI = 100e-6  # 脉冲重复间隔 [s]
PRF = 1 / PRI
tau = 5e-6  # 脉宽 [s]
Npulse = 128  # 脉冲数
fc = 0  # 基带演示（设非零可搬移到载频）

# 非相参扰动
use_random_phase = True
use_PRI_jitter = True
use_amp_jitter = True
sigma_phi = np.pi  # 相位抖动
sigma_PRI = 0.02 * PRI  # 2% PRI 抖动
sigma_amp = 0.05  # 5% 幅度抖动

# 频谱
pad_factor = 2
zoom_bins = 2  # "±N×PRF 细节"中的 N
BAND_VIEW_HZ = 2e5  # 周期图仅显示 ±BAND_VIEW_HZ

# ================= 绘图/导出控制 =================
OUTDIR = "out_show_and_save"
RES_DPI = 220  # 导出分辨率
Y_LIM_BAND = [-120, 5]  # 周期图 y 轴
Y_LIM_WELCH = [-120, 5]  # Welch y 轴
MAXPTS_MAIN = 80000  # 周期图主曲线抽点上限（保极值）
MAXPTS_REF = 30000  # 参考曲线抽点上限
SAVE_PDF = False  # 同名 PDF（矢量），如需请置 true
SAVE_FIG = False  # 同名 FIG（MATLAB 可编辑），如需请置 true

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

# ================= 生成信号 =================
T_total = Npulse * PRI
Ns = int(round(T_total * fs))
t = np.arange(Ns) / fs
x_coh = np.zeros(Ns, dtype=complex)
x_ncoh = np.zeros(Ns, dtype=complex)
L_p = max(1, int(round(tau * fs)))

for n in range(Npulse):
    t0_nom = n * PRI
    t0 = t0_nom + (use_PRI_jitter * np.random.randn() * sigma_PRI)
    k0 = int(round(t0 * fs))
    k1 = k0 + L_p - 1

    if k1 < 0 or k0 >= Ns:
        continue

    k0 = max(k0, 0)
    k1 = min(k1, Ns - 1)

    if k0 > k1:
        continue

    seg = slice(k0, k1 + 1)

    A = 1 + (use_amp_jitter * sigma_amp * np.random.randn())
    phi_coh = 0

    if use_random_phase:
        if np.isfinite(sigma_phi) and sigma_phi > 0:
            phi_ncoh = max(-np.pi, min(np.pi, np.random.randn() * sigma_phi))
        else:
            phi_ncoh = 2 * np.pi * np.random.rand()
    else:
        phi_ncoh = 0

    tt = t[seg]
    x_coh[seg] = x_coh[seg] + A * np.exp(1j * (2 * np.pi * fc * tt + phi_coh))
    x_ncoh[seg] = x_ncoh[seg] + A * np.exp(1j * (2 * np.pi * fc * tt + phi_ncoh))

# ================= 周期图 & 参考曲线 =================
Nfft = 2**int(np.ceil(np.log2(Ns * pad_factor)))
Xc = fftshift(fft(x_coh, Nfft))
Xn = fftshift(fft(x_ncoh, Nfft))
f = np.arange(-Nfft//2, Nfft//2) * (fs / Nfft)  # Hz

Pc_dB = 10 * np.log10((np.abs(Xc) / Ns)**2 + np.finfo(float).eps)
Pn_dB = 10 * np.log10((np.abs(Xn) / Ns)**2 + np.finfo(float).eps)
ref0 = np.max(Pc_dB)
Pc_dB = Pc_dB - ref0
Pn_dB = Pn_dB - ref0

# 单脉冲包络 + Dirichlet（视觉参考）
env_dB = 20 * np.log10(np.abs(np.sinc(f * tau)) + np.finfo(float).eps)
env_dB = env_dB - np.max(env_dB)
T = PRI
dirich = Npulse * (np.abs(np.sinc(Npulse * f * T)) / (np.abs(np.sinc(f * T)) + np.finfo(float).eps))
dirich = dirich / np.max(dirich)
dir_dB = 20 * np.log10(dirich + np.finfo(float).eps)
theo_coh_env_dB = env_dB + dir_dB

# ================= Welch PSD =================
win = signal.windows.hann(4096)
nover = int(0.5 * len(win))
nfft_w = 8192
fw, Pw_c = signal.welch(x_coh, fs, window=win, noverlap=nover, nfft=nfft_w, return_onesided=False, scaling='spectrum')
_, Pw_n = signal.welch(x_ncoh, fs, window=win, noverlap=nover, nfft=nfft_w, return_onesided=False, scaling='spectrum')

# 转换为中心频率
fw = fftshift(fw)
Pw_c = fftshift(Pw_c)
Pw_n = fftshift(Pw_n)

Pw_c_dB = 10 * np.log10(Pw_c + np.finfo(float).eps)
Pw_n_dB = 10 * np.log10(Pw_n + np.finfo(float).eps)
Pw_c_dB = Pw_c_dB - np.max(Pw_c_dB)
Pw_n_dB = Pw_n_dB - np.max(Pw_c_dB)

# ================= 工具函数 =================
def clip2nan(y, yMin):
    """将低于阈值的值设为NaN"""
    y = y.copy()
    y[y < yMin] = np.nan
    return y

def decimate_for_plot(x, y, maxPts, keepExtrema=False):
    """
    抽点以减少绘图伪影与数据量；keepExtrema=true 每组保 min/max
    """
    n = len(x)
    if n <= maxPts:
        return x, y

    g = int(np.ceil(n / maxPts))

    if keepExtrema:
        nb = int(n // g)
        x2 = np.zeros(2 * nb)
        y2 = np.zeros(2 * nb)
        k = 0

        for i in range(nb):
            start_idx = i * g
            end_idx = min((i + 1) * g, n)
            idx = slice(start_idx, end_idx)

            y_seg = y[idx]
            x_seg = x[idx]

            ymin_idx = np.argmin(y_seg)
            ymax_idx = np.argmax(y_seg)

            y2[k] = y_seg[ymin_idx]
            x2[k] = x_seg[ymin_idx]
            k += 1

            y2[k] = y_seg[ymax_idx]
            x2[k] = x_seg[ymax_idx]
            k += 1

        # 排序
        sort_idx = np.argsort(x2[:k])
        return x2[:k][sort_idx], y2[:k][sort_idx]
    else:
        return x[::g], y[::g]

def save_figure(fig, base, dpi, save_pdf, save_fig):
    """
    同时导出 PNG（位图）/可选 PDF（矢量）/可选 FIG（MATLAB）
    """
    fig.savefig(base + ".png", dpi=dpi, bbox_inches='tight')

    if save_pdf:
        fig.savefig(base + ".pdf", bbox_inches='tight')

    if save_fig:
        # Python中没有直接的FIG格式保存，可以保存为pickle或其他格式
        pass

# ================= 1) 时域（前三个 PRI） =================
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.plot(t * 1e3, np.real(x_coh), linewidth=1)
ax.plot(t * 1e3, np.real(x_ncoh), linewidth=1)
ax.grid(True)
ax.set_xlim([0, 3 * PRI * 1e3])
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Amplitude')
ax.set_title('时域（放大到前三个 PRI）')
ax.legend(['相参', '非相参'], loc='best')
# save_figure(fig, os.path.join(OUTDIR, '01_time_domain'), RES_DPI, SAVE_PDF, SAVE_FIG)
plt.show()

# ================= 2) 周期图（仅 ±BAND_VIEW_HZ） ==========
idx_band = np.abs(f) <= BAND_VIEW_HZ
f_band = f[idx_band]
Pc_band = Pc_dB[idx_band]
Pn_band = Pn_dB[idx_band]
env_band = env_dB[idx_band]
ref_band = theo_coh_env_dB[idx_band]

fxc, yc = decimate_for_plot(f_band, Pc_band, MAXPTS_MAIN, True)
fxn, yn = decimate_for_plot(f_band, Pn_band, MAXPTS_MAIN, True)
fxe, ye = decimate_for_plot(f_band, env_band, MAXPTS_REF, False)
fxr, yr = decimate_for_plot(f_band, ref_band, MAXPTS_REF, False)

ye = clip2nan(ye, Y_LIM_BAND[0])
yr = clip2nan(yr, Y_LIM_BAND[0])

fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(fxc, yc, linewidth=0.9)
ax.plot(fxn, yn, linewidth=0.9)
ax.plot(fxe, ye, '--', linewidth=1)
ax.plot(fxr, yr, ':', linewidth=1)
ax.grid(True)
ax.set_xlim([-BAND_VIEW_HZ, BAND_VIEW_HZ])
ax.set_ylim(Y_LIM_BAND)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Relative PSD [dB]')
ax.set_title(f'全带频谱（周期图，窄带视图：±{BAND_VIEW_HZ} Hz）')
ax.legend(['相参(周期图)', '非相参(周期图)', '单脉冲sinc包络(对齐)', '相参Dirichlet参考'], loc='best')
# save_figure(fig, os.path.join(OUTDIR, '02_periodogram_pmBAND'), RES_DPI, SAVE_PDF, SAVE_FIG)
plt.show()

# ================= 3) 细节：± zoom_bins × PRF（kHz） ======
f_win = zoom_bins * PRF
idx_zoom = np.abs(f) <= f_win

fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(f[idx_zoom] / 1e3, Pc_dB[idx_zoom], linewidth=1)
ax.plot(f[idx_zoom] / 1e3, Pn_dB[idx_zoom], linewidth=1)

yl = ax.get_ylim()
stem_k = np.arange(-zoom_bins, zoom_bins + 1) * PRF

for kk in stem_k:
    ax.plot([kk / 1e3, kk / 1e3], yl, 'k:')

ax.grid(True)
ax.set_xlabel('Frequency [kHz]')
ax.set_ylabel('Relative PSD [dB]')
ax.set_title(f'细节：±{zoom_bins}×PRF = ±{f_win/1e3:.1f} kHz')
ax.legend(['相参(线谱清晰)', '非相参(线谱被抹平)'], loc='best')
# save_figure(fig, os.path.join(OUTDIR, '03_zoom_pmNPRF'), RES_DPI, SAVE_PDF, SAVE_FIG)
plt.show()

# ================= 4) Welch 平滑 PSD（kHz） ===============
fx_env_w, env_w_plot = decimate_for_plot(f / 1e3, env_dB, MAXPTS_REF, False)
env_w_plot = clip2nan(env_w_plot, Y_LIM_WELCH[0])

fig, ax = plt.subplots(figsize=(11, 3.6))
ax.plot(fw / 1e3, Pw_c_dB, linewidth=1)
ax.plot(fw / 1e3, Pw_n_dB, linewidth=1)
ax.plot(fx_env_w, env_w_plot, '--', linewidth=1)
ax.grid(True)
ax.set_xlim([-fs/1e3*0.2, fs/1e3*0.2])
ax.set_ylim(Y_LIM_WELCH)
ax.set_xlabel('Frequency [kHz]')
ax.set_ylabel('Relative PSD [dB]')
ax.set_title('Welch 平滑 PSD')
ax.legend(['相参 Welch', '非相参 Welch', '单脉冲sinc包络(对齐)'], loc='best')
# save_figure(fig, os.path.join(OUTDIR, '04_welch'), RES_DPI, SAVE_PDF, SAVE_FIG)
plt.show()

print(f'已在屏幕显示并导出 PNG 到：{OUTDIR}')
