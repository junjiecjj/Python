#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 20:37:08 2025

@author: jack

https://github.com/rjsummers/ambiguity_functions/blob/master/ambiguity_functions.py


"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns


def ambgfun(x, fs):
    """Calculate the ambiguity function of a radar waveform.

    This function calculates the monostatic narrowband ambiguity function of
    the radar waveform x.

    Parameters
    ----------
    x : array_like
        A vector containing a radar waveform.
    fs : float
        The sampling frequency of the radar waveform in Hz.

    Returns
    -------
    ambig : 2d ndarray
        A matrix containing the ambiguity function of the radar waveform
        evaluated at the delays and Doppler shifts contained in the
        corresponding return values. The first axis of ambig indexes the
        Doppler shift and the second axis indexes the delay.
    delay : 1d ndarray
        A vector containing the delays the ambiguity function was evaluated
        at, in seconds.
    doppler : 1d ndarray
        A vector containing the doppler shifts the ambiguity function was
        evaluated at, in Hz.
    """
    ts = 1 / fs   # Sampling interval (seconds)

    # Delay-related quantities
    nx = np.size(x)
    delay = np.arange(1-nx, nx) * ts
    n = np.size(delay)
    nfft = int(2**(np.ceil(np.log2(n))))

    # Doppler-related quantities
    m = nfft
    dshifts = np.arange(-m/2, m/2, dtype=int)
    doppler = dshifts * (fs / nfft)

    # Calculate the FFT of the radar waveform
    X0 = np.fft.fft(x, n=nfft)

    # Generate a matrix of Doppler-shifted copies of the radar waveform.
    # This is done by circularly shifting the FFT of the waveform by each
    # possible shift value.
    X = np.zeros((m, nfft), dtype=complex)
    for i in range(1, m):
        X[i, :] = np.roll(X0, dshifts[i])

    # Generate the matched filter for the radar waveform.
    H = np.fft.fft(np.conj(np.flip(x)), n=nfft)

    # Apply the matched filter to the doppler shifted waveforms to obtain
    # the ambiguity function.
    ambig = np.fft.ifft(X * H[None, :], axis=1)
    ambig = ambig[:, 0:n]
    return ambig, delay, doppler


sns.set()
sns.set_context("paper")

# ##################
# # Pulse waveform #
# ##################

fs = 1000000
ts = 1/fs
N = 100

t = np.arange(0, N,  ) * ts
x = np.ones( N)
ambig, delay, doppler = ambgfun(x, fs)

plt.figure(figsize=(10, 6))
plt.contour(delay , doppler , np.abs(ambig), levels = 10)
plt.xlabel('Delay (μs)')
plt.ylabel('Doppler Shift (kHz)')
plt.title('Ambiguity Function of Rectangular Pulse')
plt.colorbar(label='Normalized Magnitude')
plt.grid(True, alpha=0.3)
plt.show()


##################
# Pulse waveform #
##################

fs = 6.4e6
ts = 1/fs
tx = 1.0000e-05
prf = 10000

t = np.arange(0, tx, step=ts)

x = np.ones(np.shape(t))
ambig, delay, doppler = ambgfun(x, fs)

fig = plt.figure()
im = plt.pcolormesh(delay / 1e-6, doppler / 1e3, np.abs(ambig))
fig.colorbar(im)
plt.xlabel("Delay (μs)")
plt.ylabel("Doppler (kHz)")
# plt.savefig("images/pulse_ambigfun.png")

plt.figure()
plt.contour(delay / 1e-6, doppler / 1e3, np.abs(ambig))
plt.xlabel("Delay (μs)")
plt.ylabel("Doppler (kHz)")
# plt.savefig("images/pulse_ambigfun_contour.png")

fig = plt.figure()
ax = plt.axes(projection='3d')
DEL, DOP = np.meshgrid(delay, doppler)
ax.plot_surface(DEL, DOP, np.abs(ambig), cmap=cm.inferno, linewidth=0.1)
# plt.savefig("images/pulse_ambigfun_surf.png")

#########################
# Linear chirp waveform #
#########################

fs = 8e6
bw = 1e6
ts = 1/fs
tx = 15/bw
prf = 10e3

t = np.arange(0, tx, step=ts)

chirpyness = bw / tx
theta = - np.pi * bw * t + np.pi * chirpyness * t**2
x = np.exp(1j * theta)
ambig, delay, doppler = ambgfun(x, fs)

fig = plt.figure()
im = plt.pcolormesh(delay / 1e-6, doppler / 1e3, np.abs(ambig))
fig.colorbar(im)
plt.xlabel("Delay (μs)")
plt.ylabel("Doppler (kHz)")
# plt.savefig("images/lfm_ambigfun.png")

plt.figure()
plt.contour(delay / 1e-6, doppler / 1e3, np.abs(ambig))
plt.xlabel("Delay (μs)")
plt.ylabel("Doppler (kHz)")
# plt.savefig("images/lfm_ambigfun_contour.png")

fig = plt.figure()
ax = plt.axes(projection='3d')
DEL, DOP = np.meshgrid(delay, doppler)
ax.plot_surface(DEL, DOP, np.abs(ambig), cmap=cm.inferno, linewidth=0.1)
# plt.savefig("images/lfm_ambigfun_surf.png")

#########################
# Barker-coded waveform #
#########################

# code = [1, 1, 1, -1, 1]
code = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
x = np.zeros((117,))
for i in range(np.size(x)):
    x[i] = code[i // 9]
fs = 8e6
ambig, delay, doppler = ambgfun(x, fs)

plt.figure()
plt.plot(x)
# plt.savefig("images/barker_waveform.png")

fig = plt.figure()
im = plt.pcolormesh(delay / 1e-6, doppler / 1e3, np.abs(ambig))
fig.colorbar(im)
plt.xlabel("Delay (μs)")
plt.ylabel("Doppler (kHz)")
# plt.savefig("images/barker_ambigfun.png")

plt.figure()
plt.contour(delay / 1e-6, doppler / 1e3, np.abs(ambig))
plt.xlabel("Delay (μs)")
plt.ylabel("Doppler (kHz)")
# plt.savefig("images/barker_ambigfun_contour.png")

fig = plt.figure()
ax = plt.axes(projection='3d')
DEL, DOP = np.meshgrid(delay, doppler)
ax.plot_surface(DEL, DOP, np.abs(ambig), cmap=cm.inferno, linewidth=0.1)
# plt.savefig("images/barker_ambigfun_surf.png")
