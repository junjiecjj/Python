#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:24:24 2024

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt

def signalCreate(_fs, _N, _f0):
    fs = _fs # 采样率
    N = _N # 数据点数
    f0 = _f0
    n = np.linspace(0, N-1, N)
    t = n / fs
    yt = np.exp(1j*2*np.pi*f0*t)
    f = n * fs / N - fs/2
    yf = np.fft.fftshift(np.fft.fft(yt))

    return t, yt, f, yf

t, yt, f, yf = signalCreate(5, 128, 5)
plt.subplot(5, 2, 1)
plt.plot(t, yt)
plt.subplot(5, 2, 2)
plt.plot(f, np.abs(yf))

t, yt, f, yf = signalCreate(10, 128, 5)
plt.subplot(5, 2, 3)
plt.plot(t, yt)
plt.subplot(5, 2, 4)
plt.plot(f, np.abs(yf))

t, yt, f, yf = signalCreate(20, 128, 5)
plt.subplot(5, 2, 5)
plt.plot(t, yt)
plt.subplot(5, 2, 6)
plt.plot(f, np.abs(yf))

t, yt, f, yf = signalCreate(40, 128, 5)
plt.subplot(5, 2, 7)
plt.plot(t, yt)
plt.subplot(5, 2, 8)
plt.plot(f, np.abs(yf))

t, yt, f, yf = signalCreate(100, 128, 5)
plt.subplot(5, 2, 9)
plt.plot(t, yt)
plt.subplot(5, 2, 10)
plt.plot(f, np.abs(yf))

plt.show()
