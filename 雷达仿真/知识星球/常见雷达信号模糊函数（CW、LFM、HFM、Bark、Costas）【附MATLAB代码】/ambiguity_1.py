#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 22:03:29 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, Union, Optional

def ambgfun(x: np.ndarray, Fs: float, PRF: float,
           y: Optional[np.ndarray] = None,
           Cut: str = "2D",
           CutValue: Union[float, np.ndarray] = 0) -> Tuple[np.ndarray, ...]:
    """
    Python implementation of MATLAB's ambgfun - Ambiguity and crossambiguity function

    Parameters:
    -----------
    x : np.ndarray
        Input pulse waveform (complex-valued vector)
    Fs : float
        Sample frequency (Hz)
    PRF : float
        Pulse repetition frequency (Hz)
    y : np.ndarray, optional
        Second input pulse waveform for crossambiguity
    Cut : str, default="2D"
        Direction of cut: "2D", "Delay", or "Doppler"
    CutValue : float or np.ndarray, default=0
        Time delays or Doppler shifts for cuts

    Returns:
    --------
    afmag : np.ndarray
        Normalized ambiguity function magnitudes
    delay : np.ndarray, optional
        Time delays vector
    doppler : np.ndarray, optional
        Doppler frequencies vector
    """

    # Input validation
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if x.ndim > 1:
        x = x.flatten()

    Nx = len(x)

    # Calculate signal energy for normalization
    if y is None:
        # Auto-ambiguity function
        Ex = np.sum(np.abs(x) ** 2)
    else:
        # Cross-ambiguity function
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if y.ndim > 1:
            y = y.flatten()
        Ny = len(y)
        Ex = np.sum(np.abs(x) ** 2)
        Ey = np.sum(np.abs(y) ** 2)
        normalization = 1.0 / np.sqrt(Ex * Ey)

    # Determine output sizes based on MATLAB documentation
    if y is None:
        # Auto-ambiguity: N = 2*Nx - 1 delays
        N_delay = 2 * Nx - 1
    else:
        # Cross-ambiguity: N = Nx + Ny - 1 delays
        N_delay = Nx + len(y) - 1

    # Number of Doppler frequencies: M = 2^ceil(log2(N))
    M_doppler = 2 ** int(np.ceil(np.log2(N_delay)))

    # Create delay vector
    if y is None:
        # Auto-ambiguity delay vector
        delay = (np.arange(N_delay) - (Nx - 1)) / Fs
    else:
        # Cross-ambiguity delay vector
        if N_delay % 2 == 0:  # Even number of delays
            delay = (np.arange(N_delay) - (N_delay // 2 - 1)) / Fs
        else:  # Odd number of delays
            Nf = N_delay // 2
            delay = (np.arange(-Nf, Nf + 1)) / Fs

    # Create Doppler frequency vector
    doppler = np.linspace(-Fs/2, Fs/2 - Fs/M_doppler, int(M_doppler))

    # Initialize ambiguity function matrix
    afmag = np.zeros((len(doppler), len(delay)), dtype=complex)

    # Compute ambiguity function
    for i, fd in enumerate(doppler):
        for j, tau in enumerate(delay):
            # Convert delay to samples
            tau_samples = int(round(tau * Fs))

            if y is None:
                # Auto-ambiguity function
                if 0 <= tau_samples < Nx:
                    # Positive delay
                    u1 = x[tau_samples:]
                    u2 = x[:Nx - tau_samples] * np.exp(1j * 2 * np.pi * fd *  np.arange(Nx - tau_samples) / Fs)
                    afmag[i, j] = np.dot(u1, np.conj(u2))
                elif tau_samples < 0 and tau_samples > -Nx:
                    # Negative delay
                    tau_samples_abs = abs(tau_samples)
                    u1 = x[:Nx - tau_samples_abs]
                    u2 = x[tau_samples_abs:] * np.exp(1j * 2 * np.pi * fd *  np.arange(Nx - tau_samples_abs) / Fs)
                    afmag[i, j] = np.dot(u1, np.conj(u2))
            else:
                # Cross-ambiguity function
                # This is a simplified implementation
                if 0 <= tau_samples < min(Nx, len(y)):
                    u1 = x[tau_samples:]
                    u2 = y[:len(y) - tau_samples] * np.exp(1j * 2 * np.pi * fd *  np.arange(len(y) - tau_samples) / Fs)
                    afmag[i, j] = np.dot(u1, np.conj(u2))

    # Normalize
    if y is None:
        afmag = np.abs(afmag) / Ex
    else:
        afmag = np.abs(afmag) * normalization

    # Handle different cut types
    if Cut == "Doppler":
        if isinstance(CutValue, (int, float)):
            # Find closest Doppler index
            doppler_idx = np.argmin(np.abs(doppler - CutValue))
            afmag_cut = afmag[doppler_idx, :]
            return afmag_cut, delay
        else:
            # Multiple cut values
            afmag_cut = np.zeros((len(CutValue), len(delay)))
            for k, cv in enumerate(CutValue):
                doppler_idx = np.argmin(np.abs(doppler - cv))
                afmag_cut[k, :] = afmag[doppler_idx, :]
            return afmag_cut, delay

    elif Cut == "Delay":
        if isinstance(CutValue, (int, float)):
            # Find closest delay index
            delay_idx = np.argmin(np.abs(delay - CutValue))
            afmag_cut = afmag[:, delay_idx]
            return afmag_cut, doppler
        else:
            # Multiple cut values
            afmag_cut = np.zeros((len(doppler), len(CutValue)))
            for k, cv in enumerate(CutValue):
                delay_idx = np.argmin(np.abs(delay - cv))
                afmag_cut[:, k] = afmag[:, delay_idx]
            return afmag_cut, doppler

    else:  # "2D" cut
        return afmag, delay, doppler


def ambgfun(x, fs, prf = 1000):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    Nx = len(x)
    Ex = np.sum(np.abs(x) ** 2)

    # Auto-ambiguity: N = 2*Nx - 1 delays
    N_delay = 2 * Nx - 1

    # Number of Doppler frequencies: M = 2^ceil(log2(N))
    M_doppler = 2 ** int(np.ceil(np.log2(N_delay)))

    # Auto-ambiguity delay vector
    delay = np.arange(1-Nx, Nx) / fs

    # Create Doppler frequency vector
    doppler = np.linspace(-fs/2, fs/2 - fs/M_doppler, int(M_doppler))

    # Initialize ambiguity function matrix
    afmag = np.zeros((len(doppler), len(delay)), dtype=complex)

    # Compute ambiguity function
    for i, fd in enumerate(doppler):
        for j, tau in enumerate(delay):
            # Convert delay to samples
            tau_samples = int(round(tau * fs))

            # Auto-ambiguity function
            if 0 <= tau_samples < Nx:
                # Positive delay
                u1 = x[tau_samples:] * np.exp(1j * 2 * np.pi * fd *  np.arange(Nx - tau_samples) / fs)
                u2 = x[:Nx - tau_samples]
                afmag[i, j] = np.dot(u1, np.conj(u2))
            elif tau_samples < 0 and tau_samples > -Nx:
                # Negative delay
                tau_samples_abs = abs(tau_samples)
                u1 = x[:Nx - tau_samples_abs]
                u2 = x[tau_samples_abs:] * np.exp(1j * 2 * np.pi * fd *  np.arange(Nx - tau_samples_abs) / fs)
                afmag[i, j] = np.dot(u1, np.conj(u2))
    return




##################
# Pulse waveform #
##################


# Example usage similar to MATLAB documentation

"""Example usage similar to MATLAB documentation examples"""

# Create a rectangular pulse
pulse_width = 1e-6  # 1 microsecond
Fs = 1e6  # 100 MHz sampling rate
PRF = 10e3  # 10 kHz PRF

t = np.arange(0, 99.5/Fs, 1/Fs)
x = np.ones(len(t), dtype=complex)  # Rectangular pulse

# Compute ambiguity function
afmag, delay, doppler = ambgfun(x, Fs, PRF)

# Plot
plt.figure(figsize=(10, 6))
plt.contour(delay * 1e6, doppler * 1e-3, afmag, 20)
plt.xlabel('Delay (μs)')
plt.ylabel('Doppler Shift (kHz)')
plt.title('Ambiguity Function of Rectangular Pulse')
plt.colorbar(label='Normalized Magnitude')
plt.grid(True, alpha=0.3)
plt.show()


fs = 1000000
ts = 1/fs
N = 100
PRF = 10e3

t = np.arange(0, N,  ) * ts
x = np.ones( N)
ambig, delay, doppler = ambgfun(x, fs, PRF)

plt.figure(figsize=(10, 6))
plt.contour(delay , doppler , ambig, levels = 20)
plt.xlabel('Delay (μs)')
plt.ylabel('Doppler Shift (kHz)')
plt.title('Ambiguity Function of Rectangular Pulse')
plt.colorbar(label='Normalized Magnitude')
plt.grid(True, alpha=0.3)
plt.show()


