#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 17:44:18 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi


# ======================================================================
# (I) parameters' definition
# ======================================================================
c = 3e8                            # speed of light
pi_val = 3.1415926                 # pi
j00 = 1j                           # square root of -1

res_a = 2                          # required azimuth resolution
res_r = 2                          # required range resolution
k_a = 1.2                          # azimuth factor
k_r = 1.2                          # range factor

Ra = 4000.0                        # radar working distance
va = 70.0                          # radar/platform forward velocity
Tp = 1e-6                          # transmitted pulse width
fc = 3e9                           # carrier frequency
FsFactor = 1.0
theta = 90 * pi_val / 180          # squint angle

# ======================================================================
# Derived parameters
lamda = c / fc                     # wavelength
Br = k_r * c / (2 * res_r)         # required transmitted bandwidth
Fs = Br * FsFactor                 # A/D sampling rate
bin_r = c / (2 * Fs)               # range bin
Kr = Br / Tp                       # range chirp rate

La = Ra * k_a * lamda / (2 * res_a)  # required synthetic aperture length
Ta = La / va                       # required synthetic aperture time
fdc = 2 * va * np.cos(theta) / lamda  # doppler centriod
fdr = -2 * (va * np.sin(theta))**2 / (lamda * Ra)  # doppler rate
Bd = abs(fdr) * Ta                 # doppler bandwidth
prf = round(Bd * 2)                # PRF

print(f"PRF: {prf} Hz")
print(f"Wavelength: {lamda:.4f} m")
print(f"Bandwidth: {Br/1e6:.2f} MHz")
print(f"Synthetic Aperture Length: {La:.2f} m")
print(f"Synthetic Aperture Time: {Ta:.4f} s")

# ======================================================================
# (II) echo return modelling (point target)
# ======================================================================

na = int(np.floor(Ta * prf / 2))   # azimuth sampling number
ta = np.arange(-na, na + 1)        # azimuth time indices
ta = ta / prf                      # slow time along azimuth
xa = va * ta - Ra * np.cos(theta)  # azimuth location along flight track
Na = 2 * int(na)                   # total azimuth samples

# Define target positions
# x0 = [0, 0, 0, 0, 0]            # define multi points if you want
# R0 = [-20, -10, 0, 10, 20]      # x0: azimuth location (positive towards forward velocity)
                                  # R0: slant range location (positive towards far range)

x0 = [0, 0]                       # only two points for demonstration
R0 = [0, 10]
Npt_num = len(x0)

# Calculate every point target's slant range history
ra = np.zeros((Npt_num, len(xa)))
for i in range(Npt_num):
    ra[i, :] = np.sqrt((Ra * np.sin(theta) + R0[i])**2 + (xa + x0[i])**2)

rmax = np.max(ra)                  # max. slant range
rmin = np.min(ra)                  # min. slant range
rmc = int((rmax - rmin) / bin_r)   # range migration, number

rg = np.zeros_like(ra)             # initialize
rg = ((ra - rmin) / bin_r + 1).astype(int)  # range gate index caused by range migration
rgmax = np.max(rg)
rgmin = np.min(rg)

nr = int(round(Tp * Fs))           # samples of a pulse
tr = np.arange(1, nr + 2)          # fast time indices (1-indexed like MATLAB)
tr = tr / Fs - Tp / 2              # fast time within a pulse duration
Nr = nr + rgmax                    # total range samples

print(f"Azimuth samples: {Na}")
print(f"Range samples per pulse: {nr}")
print(f"Total range samples: {Nr}")
print(f"Range migration cells: {rmc}")

# ======================================================================
# (III) Generate echo signal
# ======================================================================

sig = np.zeros((Na, Nr), dtype=complex)

for i in range(Na):
    for k in range(Npt_num):
        start_idx = rg[k, i] - 1  # Convert to 0-based indexing
        end_idx = start_idx + nr
        if end_idx <= Nr:
            range_phase = -j00 * 4 * pi_val / lamda * ra[k, i]
            chirp_phase = -j00 * pi_val * Kr * tr[:nr]**2
            sig[i, start_idx:end_idx] += np.exp(range_phase) * np.exp(chirp_phase)

sig_real = np.real(sig)

# ======================================================================
# (IV) Visualization
# ======================================================================

plt.figure(figsize=(12, 8))

# Create contour plot similar to MATLAB
X, Y = np.meshgrid(np.arange(Nr), np.arange(Na))
plt.contour(X, Y, sig_real, levels=20, cmap='gray')
plt.xlabel('Range Bin')
plt.ylabel('Azimuth Sample')
plt.title('Point Target Echo Signal (Real Part)')
plt.colorbar(label  = 'Amplitude')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Additional visualization: show as image
plt.figure(figsize=(12, 8))
plt.imshow(sig_real, cmap='gray', aspect='auto', extent=[0, Nr, 0, Na])
plt.xlabel('Range Bin')
plt.ylabel('Azimuth Sample')
plt.title('Point Target Echo Signal (Real Part) - Image View')
plt.colorbar(label='Amplitude')
plt.tight_layout()
plt.show()

return sig, sig_real, ra, rg

