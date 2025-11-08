#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 17:41:49 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import dft
from scipy.signal import resample
import warnings

from DigiCommPy.modem import PSKModem, QAMModem, PAMModem, FSKModem
from DigiCommPy.channels import awgn
from DigiCommPy.errorRates import ser_awgn

def AF_fullAnalysis(s, params, OS_del, OS_dop):
    """
    =========================================================================
    Author    : Dr. (Eric) Hyeon Seok Rou
    Version   : v1.0
    Date      : Oct 13, 2025
    =========================================================================
    """

    # Compute AF cuts
    tau_norm, AFdB_zerodop, AF_zerodop = AF_zerodopplercut(s, OS_del, 4)
    nu_norm, AFdB_zerodel, AF_zerodel = AF_zerodelaycut(s, OS_dop, 1)

    lenDopGuard = len(nu_norm) // 20
    nu_norm = nu_norm[lenDopGuard:-lenDopGuard]
    AFdB_zerodel = AFdB_zerodel[lenDopGuard:-lenDopGuard]
    AF_zerodel = AF_zerodel[lenDopGuard:-lenDopGuard]

    # Delay cut metrics
    metrics_delay = compute_delay_metrics(tau_norm, AF_zerodop, params)
    # Doppler cut metrics
    metrics_dopp = compute_doppler_metrics(nu_norm, AF_zerodel, params)

    return tau_norm, nu_norm, AFdB_zerodop, AFdB_zerodel, metrics_delay, metrics_dopp

def AF_zerodopplercut(s, OS_delay, Lh):
    """
        Computes the zero-Doppler cut A(τ,0) with optional fractional-delay Aperiodic (zero-padded overlap) model, built from first principles.
        INPUTS
          s          : length-N complex/real vector (row or column)
          OS_delay   : delay oversampling factor (default 1 => integer delays only)
          Lh         : half-length of windowed-sinc kernel
        OUTPUTS
          tau_norm   : normalized delay grid (tau/N), dimensionless, centered at 0
          cut_dB     : |A(tau,0)| in dB, normalized to 0 dB peak
          cut        : complex A(tau,0)
    """
    # ----- defaults -----
    if OS_delay < 1:
        OS_delay = 1
    if Lh < 1:
        Lh = 1
    # ----- validate & prep -----
    s = np.array(s, dtype=complex).flatten()
    N = len(s)
    # energy Es = sum |s[n]|^2
    Es = np.sum(np.abs(s)**2)
    # ----- fractional delay grid -----
    tau_int = np.arange(-(N-1), N)                    # integer endpoints
    tau_frac = np.arange(tau_int[0], tau_int[-1] + 1/OS_delay, 1/OS_delay)
    M = len(tau_frac)
    cut = np.zeros(M, dtype=complex)
    # ----- smooth compact window, |x|<=Lh -----
    def winCompact(x):
        """ w(x) = (1 - (|x|/Lh)^2)^2 for |x|<=Lh; 0 otherwise. Ensures w(0)=1. """
        ax = np.abs(x)
        w = np.zeros_like(x)
        mask = ax <= Lh
        u = ax[mask] / Lh
        w[mask] = (1 - u*u)**2
        return w
    # ----- main computation: A(tau,0) with fractional reconstruction of s[n-τ] -----
    for idx in range(M):
        tau_m = tau_frac[idx]     # possibly fractional delay
        # Sum over n where interpolation footprint overlaps data region.
        acc = 0.0 + 0.0j
        for n in range(N):
            # Reconstruct s[n - tau_m] by finite windowed-sinc from data samples s[m]
            xcenter = n - tau_m
            # Limit m to kernel support around xcenter
            mMin = max(0, int(np.ceil(xcenter - Lh)))
            mMax = min(N-1, int(np.floor(xcenter + Lh)))
            srec = 0.0 + 0.0j
            for m in range(mMin, mMax + 1):
                dx = xcenter - m      # distance from grid sample m to fractional point
                # sinc(x) with sinc(0)=1
                if dx == 0:
                    sincv = 1.0
                else:
                    sincv = np.sin(np.pi * dx) / (np.pi * dx)
                wv = winCompact(dx)
                srec += s[m] * (sincv * wv)
            # s[n] * conj( s_rec )
            acc += s[n] * np.conj(srec)
        cut[idx] = acc
    # ----- normalization so A(0)=1 when Es>0-----
    if Es > 0:
        cut = cut / Es
    # ----- normalized delay axis -----
    tau_norm = tau_frac / N   # dimensionless, centered
    # ----- dB output 0 dB at peak -----
    mag = np.abs(cut)
    peak = np.max(mag) if np.max(mag) > 0 else 1.0
    mag_normalized = mag / peak

    dB_floor = -300  # instead of -inf for practical purposes
    cut_dB = 20 * np.log10(mag_normalized + 1e-10)
    cut_dB[cut_dB < dB_floor] = dB_floor

    return tau_norm, cut_dB, cut

def AF_zerodelaycut(s, OS, doppler_span=5):
    """
    Zero-delay cut A(0,nu) evaluated densely and exactly

    INPUTS
      s            : length-N complex/real vector (row or column)
      OS           : Doppler oversampling factor (e.g., 32, 64, 128)
      doppler_span : plot span in normalized units; evaluate nu in [-span, +span]

    OUTPUTS
      nu_dense  : 1×Nd vector of normalized Doppler points (includes 0 exactly)
      AdB_dense : 1×Nd zero-delay cut in dB, normalized (peak = 0 dB)
    """

    if OS < 1:
        OS = 1

    s = np.array(s, dtype=complex).flatten()
    N = len(s)

    # Energy of s: Es = sum |s[n]|^2
    Es = np.sum(np.abs(s)**2)
    if Es == 0:
        Nd = max(2, OS * N)
        nu_os = np.linspace(-doppler_span, doppler_span, Nd)
        AdB_os = -300 * np.ones(len(nu_os))
        A_os = np.zeros(len(nu_os), dtype=complex)
        return nu_os, AdB_os, A_os

    # Number of dense Doppler samples
    Nd = max(2, OS * N)

    # Build symmetric Doppler grid that GUARANTEES nu = 0 is on-grid
    if Nd % 2 == 1:
        nu_os = np.linspace(-doppler_span, doppler_span, Nd)
    else:
        Nd_eff = Nd + 1
        nu_full = np.linspace(-doppler_span, doppler_span, Nd_eff)
        nu_os = nu_full[:-1]  # drop last endpoint (+span) so 0 is included

    # A(0,nu) = sum_{n=0}^{N-1} |s[n]|^2 * exp(-j 2 pi nu n)
    A0 = np.zeros(len(nu_os), dtype=complex)

    for k, nu_k in enumerate(nu_os):
        acc = 0.0 + 0.0j
        for n in range(N):
            # |s[n]|^2
            p = np.abs(s[n])**2

            # exp(-j 2 pi nu_k (n))
            phase = -2 * np.pi * nu_k * n
            c = np.cos(phase)
            d = np.sin(phase)

            # accumulate p * (c + j d)
            acc += p * (c + 1j * d)

        A0[k] = acc

    A_os = A0
    A0abs = np.abs(A0)
    A0abs = A0abs / (np.max(A0abs) + 1e-10)
    AdB_os = 20 * np.log10(A0abs + 1e-10)

    return nu_os, AdB_os, A_os

def compute_delay_metrics(x_norm, A_cut, params):
    """
        x_norm   : tau_norm = tau / N  (dimensionless, centered)
        A_cut    : complex A(tau,0)    (no dB floor)
        params   : dict with fields:
                   'N'   = #samples
                   'Fs'  = sampling frequency [Hz]
                   'B'   = (optional) occupied bandwidth [Hz] (for reference limits)
                   'Tsym'= (optional) symbol time [s]         (for reporting only)
                   'Tobs'= (optional) observation time [s]; default N/Fs
    """

    N = params['N']
    Fs = params['Fs']
    Ts = 1 / Fs
    Tobs = params.get('Tobs', N / Fs)
    B = params.get('B', None)
    Tsym = params.get('Tsym', None)

    # ---- magnitude / power ----
    mag = np.abs(A_cut)
    if np.all(mag == 0):
        raise ValueError('Delay cut is all zeros.')

    mag = mag / np.max(mag)
    pwr = mag**2

    # ---- find mainlobe bounds ----
    ipk = np.argmax(mag)
    iL, iR = first_minima_around_peak(mag, ipk)  # 找出主瓣边界
    if iL is None or iR is None or iL >= iR:
        iL, iR = halfpower_bounds(mag, ipk)

    # ---- 3 dB crossings (linear interp) ----
    xL = halfpower_crossing(x_norm, mag, ipk, -1)
    xR = halfpower_crossing(x_norm, mag, ipk, +1)
    width_norm = xR - xL        # in tau/N  (dimensionless)
    width_samp = width_norm * N # samples
    width_sec = width_samp * Ts

    # ---- PSLR / ISLR ----
    max_sl = max([
        peak_outside(mag, 0, iL-1) or 0,
        peak_outside(mag, iR+1, len(mag)-1) or 0
    ])
    PSLR_dB = 20 * np.log10(max_sl + 1e-10)

    P_main = np.sum(pwr[iL:iR+1])
    P_side = np.sum(pwr) - P_main
    ISLR_dB = 10 * np.log10(max(P_side, 1e-10) / max(P_main, 1e-10))

    # ---- package ----
    M = {}
    M['width_norm'] = width_norm          # tau/N
    M['width_samples'] = width_samp       # samples
    M['width_seconds'] = width_sec        # seconds
    M['PSLR_dB'] = PSLR_dB
    M['ISLR_dB'] = ISLR_dB
    M['bounds_idx'] = [iL, iR]
    M['crossings_norm'] = [xL, xR]
    M['scales'] = {
        'N': N, 'Fs': Fs, 'Ts': Ts, 'Tobs': Tobs,
        'Tsym': Tsym, 'B': B
        }

    return M

def compute_doppler_metrics(nu_norm, A_cut, params):
    """
    nu_norm : normalized Doppler (cycles/sample), typically [-0.5,0.5]
    A_cut   : complex A(0,nu)
    params  : dict like above (N, Fs, optional B, Tsym, Tobs)
    """

    N = params['N']
    Fs = params['Fs']
    Ts = 1 / Fs
    Tobs = params.get('Tobs', N / Fs)
    B = params.get('B', None)
    Tsym = params.get('Tsym', None)

    mag = np.abs(A_cut)
    mag = mag / np.max(mag)
    pwr = mag**2

    ipk = np.argmax(mag)
    iL, iR = first_minima_around_peak(mag, ipk)
    if iL is None or iR is None or iL >= iR:
        iL, iR = halfpower_bounds(mag, ipk)

    nL = halfpower_crossing(nu_norm, mag, ipk, -1)
    nR = halfpower_crossing(nu_norm, mag, ipk, +1)
    width_nu = nR - nL          # cycles/sample (normalized Doppler)
    width_Hz = width_nu * Fs    # Hz (since f_D = nu * Fs)

    max_sl = max([
                peak_outside(mag, 0, iL-1) or 0,
                peak_outside(mag, iR+1, len(mag)-1) or 0
                ])
    PSLR_dB = 20 * np.log10(max_sl + 1e-10)

    P_main = np.sum(pwr[iL:iR+1])
    P_side = np.sum(pwr) - P_main
    ISLR_dB = 10 * np.log10(max(P_side, 1e-10) / max(P_main, 1e-10))

    M = {}
    M['width_norm'] = width_nu           # cycles/sample
    M['width_Hz'] = width_Hz             # Hz
    M['width_perT'] = width_Hz * Tobs    # dimensionless ~ (width * Tobs); mainlobe ~ O(1)
    M['PSLR_dB'] = PSLR_dB
    M['ISLR_dB'] = ISLR_dB
    M['bounds_idx'] = [iL, iR]
    M['crossings'] = [nL, nR]
    M['scales'] = {
                'N': N, 'Fs': Fs, 'Ts': Ts, 'Tobs': Tobs,
                'Tsym': Tsym, 'B': B
            }

    return M

def halfpower_bounds(mag, ipk):
    # 功能：
    #     基于-3dB点确定主瓣的左右边界
    # 算法：
    #     计算半功率点值：hp = 1/sqrt(2) ≈ 0.707（对应-3dB）
    #     在峰值左侧寻找最后一个低于半功率点的位置
    #     在峰值右侧寻找第一个低于半功率点的位置
    hp = 1 / np.sqrt(2)
    left_part = mag[:ipk+1]
    right_part = mag[ipk:]

    iL = np.where(left_part < hp)[0]
    iL = iL[-1] if len(iL) > 0 else 0

    tmp = np.where(right_part < hp)[0]
    if len(tmp) == 0:
        iR = len(mag) - 1
    else:
        iR = ipk + tmp[0]

    return iL, iR

def halfpower_crossing(x, mag, ipk, dirsign):
    # 功能：
    #     通过线性插值精确计算-3dB点的位置
    # 算法：
    #     从峰值位置向指定方向搜索
    #     找到幅度从高于半功率点变为低于半功率点的区间
    #     使用线性插值计算精确的交叉位置
    hp = 1 / np.sqrt(2)

    if dirsign < 0:
        idx = np.arange(ipk, -1, -1)
        for k in range(len(idx) - 1):
            i1 = idx[k]
            i0 = idx[k + 1]
            if (mag[i0] >= hp and mag[i1] < hp) or (mag[i0] <= hp and mag[i1] > hp):
                t = (hp - mag[i0]) / (mag[i1] - mag[i0] + 1e-10)
                xC = x[i0] + t * (x[i1] - x[i0])
                return xC
        return x[0]
    else:
        idx = np.arange(ipk, len(x))
        for k in range(len(idx) - 1):
            i0 = idx[k]
            i1 = idx[k + 1]
            if (mag[i0] >= hp and mag[i1] < hp) or (mag[i0] <= hp and mag[i1] > hp):
                t = (hp - mag[i0]) / (mag[i1] - mag[i0] + 1e-10)
                xC = x[i0] + t * (x[i1] - x[i0])
                return xC
        return x[-1]

def peak_outside(mag, a, b):
    # 功能：
    #     在指定区域外寻找最大的旁瓣峰值
    # 算法：
    #     在给定的索引范围[a, b]内搜索
    #     识别所有局部极大值点（比左右邻居都大的点）
    #     返回其中最大的幅度值
    if a > b:
        return None

    region = mag[a:b+1]
    if len(region) == 0:
        return None

    # Find local maxima in the region
    local_maxima = []
    for i in range(1, len(region) - 1):
        if region[i] >= region[i-1] and region[i] >= region[i+1]:
            local_maxima.append(region[i])

    if len(local_maxima) == 0:
        return None

    return max(local_maxima)

def first_minima_around_peak(mag, ipk):
    """
    它从峰值点开始向左和向右搜索，找到第一个满足条件的局部最小值
    ----------
    mag : 1D  array.
    ipk : int, index of mag.max().

    """
    iL = None
    iR = None

    # Search left
    prev = mag[ipk]
    for i in range(ipk-1, 0, -1):
        if mag[i] < mag[i-1] and mag[i] <= prev:
            if mag[i] <= mag[i+1]:
                iL = i
                break
        prev = mag[i]

    # Search right
    prev = mag[ipk]
    for i in range(ipk+1, len(mag)-1):
        if mag[i] < mag[i+1] and mag[i] <= prev:
            if mag[i] <= mag[i-1]:
                iR = i
                break
        prev = mag[i]

    return iL, iR

# System Parametrisation
N = 144                         # number of discrete time samples
OS_dop = 4                      # smoothing factor in doppler domain
OS_del = 4                      # smoothing factor in delay domain
allones = False                 # If "True", transmit all one symbols, if "False", transmit random QAM symbols

params = {}
params['Ts'] = 1e-6               # arbitrary sampling period (For physical translation)
params['N'] = N                   # number of time samples used in the AF (length(s))
params['Fs'] = 1 / params['Ts']   # sampling rate [Hz]

mod_type = 'QAM'                  # Set 'PSK' or 'QAM' or 'PAM' or 'FSK'
M = 16                            # Modulation order for 16-QAM
coherence = 'coherent'            # 'coherent'/'noncoherent'-only for FSK
modem_dict = {'psk': PSKModem, 'qam':QAMModem, 'pam':PAMModem, 'fsk':FSKModem}
modem = modem_dict[mod_type.lower()](M)

# Generate Signals
if allones:
    symbols = np.ones(N)
else:
    symbols = np.random.randint(0, M, N)
    symbols = modem.modulate(symbols)

# ---- OFDM
IDFT_matrix = dft(N, scale='sqrtn').conj().T  # dftmtx(N)'/sqrt(N)
s_OFDM = IDFT_matrix @ symbols

# ---- OTFS
L_OTFS = int(np.sqrt(N))
FH_OTFS = np.kron(dft(L_OTFS, scale='sqrtn').conj().T, np.eye(L_OTFS))
s_OTFS = FH_OTFS @ symbols

# ---- AFDM
n = np.arange(N)
ellmax = 6
fmax = 4
AFDM_guard = 1
AFDM_resources = (2*(fmax + AFDM_guard)*ellmax) + (2*(fmax + AFDM_guard)) + ellmax

if AFDM_resources > N:
    warnings.warn(f"AFDM orthogonality not satisfied (resources={AFDM_resources} > N={N})")

c1 = (2*(fmax + AFDM_guard) + 1) / (2*N)
c2 = 1 / (2*N)
chirp_c1 = np.exp(-2j * np.pi * c1 * (n**2))
chirp_c2 = np.exp(-2j * np.pi * c2 * (n**2))
s_AFDM = np.diag(chirp_c1).conj().T @ IDFT_matrix @ np.diag(chirp_c2).conj().T @ symbols

# ---- CP-AFDM
c2_perm = np.random.permutation(N)
chirpc2_perm = chirp_c2[c2_perm]
s_CPAFDM = np.diag(chirp_c1).conj().T @ IDFT_matrix @ np.diag(chirpc2_perm).conj().T @ symbols

# Obtain AFs and extract metrics using the helper functions
tau_norm, nu_norm, OFDM_AFdB_zerodop, OFDM_AFdB_zerodel, OFDM_metrics_delay, OFDM_metrics_dopp = AF_fullAnalysis(s_OFDM, params, OS_del, OS_dop)
_, _, AFDM_AFdB_zerodop, AFDM_AFdB_zerodel, AFDM_metrics_delay, AFDM_metrics_dopp = AF_fullAnalysis(s_AFDM, params, OS_del, OS_dop)
_, _, CPAFDM_AFdB_zerodop, CPAFDM_AFdB_zerodel, CPAFDM_metrics_delay, CPAFDM_metrics_dopp = AF_fullAnalysis(s_CPAFDM, params, OS_del, OS_dop)
_, _, OTFS_AFdB_zerodop, OTFS_AFdB_zerodel, OTFS_metrics_delay, OTFS_metrics_dopp = AF_fullAnalysis(s_OTFS, params, OS_del, OS_dop)

# ---------- Common Figure Settings ----------
fig_width = 6
fig_height = fig_width / 1.618
lw = 1.4

clr_OFDM = [0, 0, 0]        # black
clr_OTFS = [0, 0.35, 0.7]   # dark blue
clr_AFDM = [0, 0.5, 0.1]    # dark green
clr_CPAFDM = [0.6, 0, 0]    # dark red

plt.rcParams.update({
    'text.usetex': False,
    'font.size': 10
})

# ---------- Plot: Delay AF (Zero-Doppler cut) ----------

fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout = True)
axs.plot(tau_norm, OFDM_AFdB_zerodop, linewidth = lw, color = clr_OFDM, ls = '-', label='OFDM')
axs.plot(tau_norm, OTFS_AFdB_zerodop, linewidth = lw, color = clr_OTFS, ls = '-',label='OTFS')
axs.plot(tau_norm, AFDM_AFdB_zerodop, linewidth = lw, color = clr_AFDM, ls = '-', label='AFDM')
axs.plot(tau_norm, CPAFDM_AFdB_zerodop, linewidth = lw, color = clr_CPAFDM, ls = '-', label='CP-AFDM')
legend1 = axs.legend(loc='best', borderaxespad = 0,  edgecolor = 'black', fontsize = 18)
axs.set_xlabel(r'Normalized Delay', )
axs.set_ylabel(r'Magnitude [dB]', )
axs.set_xlim([-1, 1])
axs.set_ylim([-60, 10])
out_fig = plt.gcf()
# out_fig.savefig('Fig6_d.png', )
# out_fig.savefig('Fig6_d.pdf', )
plt.show()
plt.close()

# ---------- Plot: Doppler AF (Zero-delay cut) ----------

fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
axs.plot(nu_norm, OFDM_AFdB_zerodel, linewidth = lw, color = clr_OFDM, label='OFDM')
axs.plot(nu_norm, OTFS_AFdB_zerodel, linewidth = lw, color = clr_OTFS, label='OTFS')
axs.plot(nu_norm, AFDM_AFdB_zerodel, linewidth = lw, color = clr_AFDM, label='AFDM')
axs.plot(nu_norm, CPAFDM_AFdB_zerodel, linewidth = lw, color = clr_CPAFDM, label='CP-AFDM')
legend1 = axs.legend(loc='best', borderaxespad = 0,  edgecolor = 'black', fontsize = 18)
axs.set_xlabel(r'Normalized Doppler', )
axs.set_ylabel(r'Magnitude [dB]', )
axs.set_xlim([-0.5, 0.5])
axs.set_ylim([-60, 10])
out_fig = plt.gcf()
# out_fig.savefig('Fig6_d.png', )
# out_fig.savefig('Fig6_d.pdf', )
plt.show()
plt.close()


















































































































































































































































