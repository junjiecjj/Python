#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 03:37:06 2025

@author: jack

https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows


"""
#%%
import numpy as np
from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

window = signal.windows.hamming(51)
# n = np.arange(51)
# window = 0.54 - 0.46 * np.cos(2*np.pi*n / (51-1))


plt.figure()
plt.plot(window)
plt.title("Hamming window")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
plt.show()

plt.figure()
A = fft(window, 2048) / (len(window)/2.0)
freq = np.linspace(-0.5, 0.5, len(A))
response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
plt.plot(freq, response)
plt.axis([-0.5, 0.5, -120, 0])
plt.title("Frequency response of the Hamming window")
plt.ylabel("Normalized magnitude [dB]")
plt.xlabel("Normalized frequency [cycles per sample]")
plt.show()

#%%
import numpy as np
from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

# window = signal.windows.hann(51)
n = np.arange(51)
window = 0.54 - 0.5 * np.cos(2*np.pi*n / (51-1))

plt.figure()
plt.plot(window)
plt.title("Hann window")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
plt.show()

plt.figure()
A = fft(window, 2048) / (len(window)/2.0)
freq = np.linspace(-0.5, 0.5, len(A))
response = np.abs(fftshift(A / abs(A).max()))
response = 20 * np.log10(np.maximum(response, 1e-10))
plt.figure()
plt.plot(freq, response)
plt.axis([-0.5, 0.5, -120, 0])
plt.title("Frequency response of the Hann window")
plt.ylabel("Normalized magnitude [dB]")
plt.xlabel("Normalized frequency [cycles per sample]")
plt.show()

#%% general_hamming
import numpy as np
from scipy.signal.windows import general_hamming
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt


fig1, spatial_plot = plt.subplots()
spatial_plot.set_title("Generalized Hamming Windows")
spatial_plot.set_ylabel("Amplitude")
spatial_plot.set_xlabel("Sample")



fig2, freq_plot = plt.subplots()
freq_plot.set_title("Frequency Responses")
freq_plot.set_ylabel("Normalized magnitude [dB]")
freq_plot.set_xlabel("Normalized frequency [cycles per sample]")

for alpha in [0.75, 0.7, 0.52]:
    window = general_hamming(41, alpha)
    spatial_plot.plot(window, label="{:.2f}".format(alpha))
    A = fft(window, 2048) / (len(window)/2.0)
    freq = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    freq_plot.plot(freq, response, label="{:.2f}".format(alpha))
freq_plot.legend(loc="upper right")
spatial_plot.legend(loc="upper right")





























