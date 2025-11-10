#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 17:47:00 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.signal.windows import hann
import matplotlib.cm as cm

# ----------------------- Angle Estimation ---------------------
def music_aoa(x_m, d, lamda, Nfft_a, K):
    """MUSIC algorithm for AoA estimation"""
    # 确保x_m是二维数组 (Rx x 1)
    if x_m.ndim == 1:
        x_m = x_m.reshape(-1, 1)

    M = x_m.shape[0]
    Rxx = (x_m @ x_m.conj().T) / M

    # 添加正则化项确保矩阵可逆
    Rxx = Rxx + (1e-3 * np.trace(Rxx) / M) * np.eye(M)

    # 计算特征值和特征向量
    D, V = eigh((Rxx + Rxx.conj().T) / 2)
    idx = np.argsort(np.real(D))
    K = min(K, M - 1)
    En = V[:, idx[:M-K]]

    mu_axis = np.linspace(-1, 1, Nfft_a)
    P = np.zeros(Nfft_a)
    m = np.arange(M).reshape(-1, 1)

    for i in range(Nfft_a):
        a = np.exp(1j * 2 * np.pi * (m * d / lamda) * mu_axis[i])
        denom = np.real(a.conj().T @ (En @ En.conj().T) @ a)
        if denom <= 0:
            denom = np.finfo(float).eps
        P[i] = 1 / denom

    ia = np.argmax(P)
    return np.degrees(np.arcsin(mu_axis[ia]))


def nms_peaks(mag, det_map, nms_sz):
    """Simple Non-Maximum Suppression on det_map using local neighborhood"""
    H, W = mag.shape
    hh = nms_sz[0] // 2
    ww = nms_sz[1] // 2
    idx = np.where(det_map)
    keep = np.ones(len(idx[0]), dtype=bool)

    for k in range(len(idx[0])):
        if not keep[k]:
            continue

        r, c = idx[0][k], idx[1][k]
        r1 = max(0, r - hh)
        r2 = min(H, r + hh + 1)
        c1 = max(0, c - ww)
        c2 = min(W, c + ww + 1)

        patch = mag[r1:r2, c1:c2]
        if mag[r, c] < np.max(patch) - np.finfo(float).eps:
            keep[k] = False
            continue

        # Find neighbors in detection map
        nbr_idx = np.where(det_map[r1:r2, c1:c2])
        nbr_r = nbr_idx[0] + r1
        nbr_c = nbr_idx[1] + c1

        for t in range(len(nbr_r)):
            if nbr_r[t] == r and nbr_c[t] == c:
                continue

            # Find index of this neighbor
            j = np.where((idx[0] == nbr_r[t]) & (idx[1] == nbr_c[t]))[0]
            if len(j) > 0:
                keep[j[0]] = False

    sel_r = idx[0][keep]
    sel_c = idx[1][keep]
    peaks = np.column_stack((sel_r, sel_c))

    return peaks
# def fmcw_radar_demo():
    # =============================================================
    # FMCW Radar: Range / Velocity / Angle All-in-One Demo
    # - Angle–Range uses per-range strongest Doppler slice
    # - Legends + Truth Annotations on RD, RA, 3D
    # Single-file, one-click runnable
    # =============================================================

# ----------------------- User Parameters -----------------------
# Radar & waveform
c = 3e8                  # speed of light
fc = 77e9                # carrier frequency (Hz)
lamda = c / fc
BW = 1e9                 # sweep bandwidth (Hz)
Tc = 40e-6               # chirp duration (s)
S = BW / Tc              # chirp slope (Hz/s)
fs = 20e6                # ADC sampling rate (Hz)
Ns = int(round(fs * Tc)) # samples per chirp
Nchirp = 128             # number of chirps (slow-time)
Tx = 1                   # 1-Tx (TDM off)
Rx = 8                   # number of Rx (ULA)
d = lamda / 2            # element spacing

# Scene (multi-target): [R(m), v(m/s), azimuth(deg), RCS/amp]
targets = np.array([
    [40, -10, -10, 1.0],  # T1
    [55,   8,  20, 0.8],  # T2
    [85,   0,   5, 0.6]   # T3
])

SNR_dB = 20             # SNR of dechirped baseband

# Processing params
Nfft_r = 2**int(np.ceil(np.log2(Ns * 2)))     # range FFT size (zero-pad)
Nfft_d = 2**int(np.ceil(np.log2(Nchirp * 2))) # doppler FFT size
Nfft_a = 256            # angle FFT size (beamforming)
use_MUSIC = True        # also compute MUSIC AoA (optional)
MUSIC_K = targets.shape[0]  # number of sources assumed for MUSIC
guard_r = 2
train_r = 8              # 2D-CFAR params (range)
guard_d = 2
train_d = 8              # 2D-CFAR params (doppler)
Pfa = 1e-3               # CFAR false alarm

# Derived limits
R_max = fs * c / (2 * S)    # unambiguous range
V_max = lamda / (4 * Tc)    # +/- V_max unambiguous
print(f'Unambiguous Range ~ {R_max:.1f} m, Unambiguous Velocity ~ +/- {V_max:.1f} m/s')

# ----------------------- Time & Grids -------------------------
t_fast = np.arange(Ns) / fs                 # fast-time within a chirp
n_slow = np.arange(Nchirp).reshape(-1, 1)   # chirp index (slow-time)
f_fast = np.arange(Nfft_r) * (fs / Nfft_r)  # beat freq axis (0..fs)
R_axis = c * f_fast / (2 * S)               # beat freq -> range

# Slow-time (Doppler) axis: PRF = 1/Tc, centered
f_doppler = np.fft.fftshift(np.fft.fftfreq(Nfft_d, Tc))
V_axis = (lamda / 2) * f_doppler

# Angle axis for Rx-ULA (broadside 0 deg; left negative)
mu_axis = np.linspace(-1, 1, Nfft_a)
ang_axis = np.degrees(np.arcsin(mu_axis))

# ----------------------- Signal Simulation --------------------
# Data cube: Rx x Nchirp x Ns (dechirped baseband)
X = np.zeros((Rx, Nchirp, Ns), dtype=complex)

for k in range(targets.shape[0]):
    R0 = targets[k, 0]
    vk = targets[k, 1]
    thet = np.radians(targets[k, 2])
    amp = targets[k, 3]

    Rn = R0 + vk * n_slow * Tc              # Nchirp x 1
    tau = 2 * Rn / c                        # Nchirp x 1
    fd = 2 * vk / lamda                     # Doppler (Hz)

    m_idx = np.arange(Rx).reshape(-1, 1)
    phi_arr_m = 2 * np.pi * (m_idx * d * np.sin(thet) / lamda)  # Rx x 1

    S_tau = S * tau                         # Nchirp x 1

    # Create meshgrids
    S_tau_mat, t_fast_mat = np.meshgrid(S_tau.flatten(), t_fast, indexing='ij')
    n_slow_mat, _ = np.meshgrid(n_slow.flatten(), t_fast, indexing='ij')

    phase_nt = 2 * np.pi * ((S_tau_mat + fd) * t_fast_mat + fd * n_slow_mat * Tc)  # Nchirp x Ns

    for m in range(Rx):
        X[m, :, :] += amp * np.exp(1j * (phase_nt + phi_arr_m[m, 0]))

# Add AWGN
sig_pow = np.mean(np.abs(X)**2)
noise_pw = sig_pow / (10**(SNR_dB / 10))
noise = np.sqrt(noise_pw / 2) * (np.random.randn(*X.shape) + 1j * np.random.randn(*X.shape))
X += noise

# ----------------------- Range FFT ----------------------------
win_r = hann(Ns)
X_r = np.zeros((Rx, Nchirp, Nfft_r), dtype=complex)

for m in range(Rx):
    for n in range(Nchirp):
        x = X[m, n, :] * win_r
        X_r[m, n, :] = np.fft.fft(x, Nfft_r)

valid_r = R_axis <= R_max
R_hat = R_axis[valid_r]
Nr = len(R_hat)

# ----------------------- Doppler FFT (fixed) ------------------
win_d = hann(Nchirp)
Nd = Nfft_d
X_rd = np.zeros((Rx, Nd, Nr), dtype=complex)  # Rx x Nd x Nr

for m in range(Rx):
    Xr = X_r[m, :, :]                         # Nchirp x Nfft_r
    Xr = Xr[:, valid_r]                       # Nchirp x Nr
    Xr = Xr * win_d.reshape(-1, 1)
    Xrd = np.fft.fftshift(np.fft.fft(Xr, Nd, axis=0), axes=0)  # Nd x Nr
    X_rd[m, :, :] = Xrd

# Non-coherent sum over Rx for detection
RD = np.sum(np.abs(X_rd)**2, axis=0)          # Nd x Nr

# ----------------------- 2D-CFAR ------------------------------
def ca_cfar_alpha(Nref, Pfa):
    """Return CA-CFAR scaling alpha for given reference cells and Pfa"""
    return Nref * (Pfa**(-1/Nref) - 1)

mag = RD.copy()
Nref = (2 * train_r * 2 * train_d)
alpha = ca_cfar_alpha(Nref, Pfa)

det_map = np.zeros((Nd, Nr), dtype = bool)

for ii in range(train_d + guard_d, Nd - (train_d + guard_d)):
    d_idx = list(range(ii - (train_d + guard_d), ii - guard_d)) + list(range(ii + guard_d, ii + train_d + guard_d))
    for ir in range(train_r + guard_r, Nr - (train_r + guard_r)):
        r_idx = list(range(ir - (train_r + guard_r), ir - guard_r)) + list(range(ir + guard_r, ir + train_r + guard_r))
        noise_est = np.mean(mag[np.ix_(d_idx, r_idx)])
        if mag[ii, ir] > alpha * (noise_est + np.finfo(float).eps):
            det_map[ii, ir] = True
peak_idx = nms_peaks(mag, det_map, (3, 3))   # [ii, ir]

est_list = []  # [R, V, AoA_FFT, Mag, AoA_MUSIC]
for p in range(peak_idx.shape[0]):
    ii = peak_idx[p, 0]   # doppler bin
    ir = peak_idx[p, 1]   # range bin

    fd_hat = f_doppler[ii]
    V_est = (lamda / 2) * fd_hat
    R_est = R_hat[ir]

    x_m = X_rd[:, ii, ir]                     # Rx x 1 snapshot

    ang_spectrum = np.abs(np.fft.fftshift(np.fft.fft(x_m, Nfft_a)))**2
    ia = np.argmax(ang_spectrum)
    ang_fft = ang_axis[ia]

    ang_music = np.nan
    if use_MUSIC and Rx >= (MUSIC_K + 1):
        ang_music = music_aoa(x_m, d, lamda, Nfft_a, MUSIC_K)

    est_list.append([R_est, V_est, ang_fft, np.linalg.norm(x_m), ang_music])

est_list = np.array(est_list)

# ----------------------- Truth (for annotations) ---------------
R_true = targets[:, 0]
v_true = targets[:, 1]
th_true = targets[:, 2]
fd_true = 2 * v_true / lamda           # Hz

# ----------------------- Visualization ------------------------
# 1) Range profile
RP = np.sum(np.sum(np.abs(X_r[:, :, valid_r])**2, axis=0), axis=0)  # 1 x Nr
plt.figure(figsize=(10, 6))
plt.plot(R_hat, 10 * np.log10(RP / np.max(RP) + 1e-12), linewidth=1.3)
plt.grid(True)
plt.xlabel('Range (m)')
plt.ylabel('Normalized Power (dB)')
plt.title('Range Profile')
plt.show()

# 2) Range-Doppler map + detections + TRUTH + LEGEND
plt.figure(figsize=(12, 8))
plt.imshow(10 * np.log10(mag / np.max(mag) + 1e-12), extent=[R_hat[0], R_hat[-1], f_doppler[0], f_doppler[-1]], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar()
plt.xlabel('Range (m)')
plt.ylabel('Doppler (Hz)')
plt.title('Range-Doppler Map (sum over Rx)')

hDet = None
if peak_idx.size > 0:
    hDet = plt.scatter(R_hat[peak_idx[:, 1]], f_doppler[peak_idx[:, 0]], 40, 'white', label='Detections')
hTruth = plt.plot(R_true, fd_true, 'p', markersize=12, markerfacecolor='yellow', markeredgecolor='black', linewidth=1.5, label='Truth')[0]
for i in range(len(R_true)):
    plt.text(R_true[i] + 0.5, fd_true[i], f'T{i+1}', color='black', fontweight='bold', verticalalignment='center')
if hDet is None:
    plt.legend([hTruth], ['Truth'], loc='best')
else:
    plt.legend([hDet, hTruth], ['Detections', 'Truth'], loc='best')
plt.show()

# 3) Angle spectrum for top-1 detection (FFT beamforming)
if peak_idx.size > 0:
    ii = peak_idx[0, 0]
    ir = peak_idx[0, 1]
    x_m = X_rd[:, ii, ir]
    ang_spec = np.abs(np.fft.fftshift(np.fft.fft(x_m, Nfft_a)))**2
    ang_spec = ang_spec / (np.max(ang_spec) + 1e-12)

    plt.figure(figsize=(10, 6))
    plt.plot(ang_axis, 10 * np.log10(ang_spec + 1e-12), linewidth=1.3)
    plt.grid(True)
    plt.xlabel('Azimuth (deg)')
    plt.ylabel('dB')
    plt.title('Angle Spectrum (Top-1, FFT Beamforming)')
    plt.show()

# 4) Angle–Range 2D Map using "per-range strongest Doppler slice" + TRUTH + LEGEND
id_max_per_r = np.argmax(RD, axis=0)   # 1 x Nr (each range bin's strongest Doppler bin index)
RA = np.zeros((Nfft_a, Nr))

for ir in range(Nr):
    x_m = X_rd[:, id_max_per_r[ir], ir]    # Rx x 1
    ang_spec = np.abs(np.fft.fftshift(np.fft.fft(x_m, Nfft_a)))**2   # Nfft_a x 1
    RA[:, ir] = ang_spec

RA = RA / (np.max(RA) + 1e-12)

plt.figure(figsize=(12, 8))
plt.imshow(10 * np.log10(RA + 1e-12), extent=[R_hat[0], R_hat[-1], ang_axis[0], ang_axis[-1]], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar()
plt.xlabel('Range (m)')
plt.ylabel('Azimuth (deg)')
plt.title('Angle–Range Map (per-range strongest Doppler slice)')

# truth overlay
df = np.abs(f_doppler[1] - f_doppler[0])    # Doppler frequency resolution
tol = 1.5 * df                              # tolerance: ~1.5 bins
truth_mask = np.zeros(len(R_true), dtype=bool)

for i in range(len(R_true)):
    ir0 = np.argmin(np.abs(R_hat - R_true[i]))  # nearest range-bin to target range
    fd_slice = f_doppler[id_max_per_r[ir0]]     # doppler slice used for this range
    truth_mask[i] = abs(fd_true[i] - fd_slice) <= tol

hTruthRA = plt.plot(R_true[truth_mask], th_true[truth_mask], 'p', markersize=11, markeredgecolor='black', markerfacecolor='yellow', linewidth=1.3, label='Truth near slice')[0]

for i in range(len(R_true)):
    if truth_mask[i]:
        plt.text(R_true[i] + 0.5, th_true[i], f'T{i+1}', color = 'black', fontweight = 'bold', verticalalignment='center')

plt.legend([hTruthRA], ['Truth near slice'], loc = 'best')
plt.show()

# 5) 3D Scatter of Detections in (Range, Velocity, Angle) + TRUTH + LEGEND
if est_list.size > 0:
    ang_use = est_list[:, 4].copy()  # prefer MUSIC if available
    nan_idx = np.isnan(ang_use)
    ang_use[nan_idx] = est_list[nan_idx, 2]  # fallback to FFT AoA

    mag_lin = est_list[:, 3]
    mag_db = 20 * np.log10(mag_lin / (np.max(mag_lin) + 1e-12))
    sz = 30 + 70 * (mag_lin / (np.max(mag_lin) + 1e-12))  # marker size by magnitude

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(est_list[:, 0], est_list[:, 1], ang_use, s=sz, c=mag_db, cmap='viridis', label='Detections')

    ax.grid(True)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_zlabel('Azimuth (deg)')
    ax.set_title('3D Detection Cloud: Range–Velocity–Angle')

    cb = plt.colorbar(sc)
    cb.set_label('Magnitude (dB, relative)')

    ax.view_init(elev=25, azim=45)

    # Plot truth targets
    ax.plot(R_true, v_true, th_true, 'p', markersize=12, markerfacecolor='yellow', markeredgecolor='black', linewidth=1.5, label='Truth')

    for i in range(len(R_true)):
        ax.text(R_true[i], v_true[i], th_true[i] + 1.0, f'T{i+1}', color='black', fontweight='bold')

    ax.legend(['Detections', 'Truth'], loc='best')
    plt.show()

# ----------------------- Print Detections ---------------------
if est_list.size == 0:
    print('No detections. Try lowering threshold (increase Pfa) or raising SNR.')
else:
    # Sort by magnitude (desc)
    ord_idx = np.argsort(est_list[:, 3])[::-1]
    est_sorted = est_list[ord_idx, :]

    print('\nDetections (after 2D-CFAR + AoA):')
    if use_MUSIC:
        print(f'   {"Range(m)":>9}  {"Vel(m/s)":>10}  {"AoA_FFT":>10}  {"Mag":>10}  {"AoA_MUSIC":>12}  {"Mag(dB)":>10}')
        for i in range(est_sorted.shape[0]):
            mag_db = 20 * np.log10(est_sorted[i, 3] / (np.max(est_sorted[:, 3]) + 1e-12))
            print(f'   {est_sorted[i, 0]:9.2f}  {est_sorted[i, 1]:10.2f}  {est_sorted[i, 2]:10.1f}  {est_sorted[i, 3]:10.2f}  {est_sorted[i, 4]:12.1f}  {mag_db:10.1f}')
    else:
        print(f'   {"Range(m)":>9}  {"Vel(m/s)":>10}  {"AoA_FFT":>10}  {"Mag":>10}  {"Mag(dB)":>10}')
        for i in range(est_sorted.shape[0]):
            mag_db = 20 * np.log10(est_sorted[i, 3] / (np.max(est_sorted[:, 3]) + 1e-12))
            print(f'   {est_sorted[i, 0]:9.2f}  {est_sorted[i, 1]:10.2f}  {est_sorted[i, 2]:10.1f}  {est_sorted[i, 3]:10.2f}  {mag_db:10.1f}')


# # Run the demo
# if __name__ == "__main__":
#     fmcw_radar_demo()
