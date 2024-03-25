#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 19:09:29 2022

@author: jack

https://blog.csdn.net/weijifen000/article/details/79598258
"""

# 离散时间傅里叶变换的 python 实现
import numpy as np
import math
import pylab as pl
import scipy.signal as signal
import matplotlib.pyplot as plt

sampling_rate=1000
t1=np.arange(0, 10.0, 1.0/sampling_rate)
x1 =np.sin(15*np.pi*t1)

# 傅里叶变换
def fft1(xx):
#     t=np.arange(0, s)
    t=np.linspace(0, 1.0, len(xx))
    f = np.arange(len(xx)/2+1, dtype=complex)
    for index in range(len(f)):
        f[index]=complex(np.sum(np.cos(2*np.pi*index*t)*xx), -np.sum(np.sin(2*np.pi*index*t)*xx))
    return f

# len(x1)



xf=fft1(x1)/len(x1)
freqs = np.linspace(0, sampling_rate/2, int(len(x1)/2+1 ))
plt.figure(figsize=(16,4))
plt.plot(freqs, 2*np.abs(xf),'r--')

plt.xlabel("Frequency(Hz)")
plt.ylabel("Amplitude($m$)")
plt.title("Amplitude-Frequency curve")
plt.show()


plt.figure(figsize=(16,4))
plt.plot(freqs, 2*np.abs(xf),'b--')

plt.xlabel("Frequency(Hz)")
plt.ylabel("Amplitude($m$)")
plt.title("Amplitude-Frequency curve")
plt.xlim(0,20)
plt.show()

#===================================================================================

# 傅里叶变换
def fft1(xx):
#     t=np.arange(0, s)
    t=np.linspace(0, 1.0, len(xx))
    f = np.arange(len(xx) , dtype=complex)
    for index in range(len(f)):
        f[index]=complex(np.sum(np.cos(2*np.pi*index*t)*xx), -np.sum(np.sin(2*np.pi*index*t)*xx))
    return f

# len(x1)



xf=fft1(x1)/len(x1)
freqs = np.linspace(0, sampling_rate/2, int(len(x1) ))
plt.figure(figsize=(16,4))
plt.plot(freqs, 2*np.abs(xf),'r--')

plt.xlabel("Frequency(Hz)")
plt.ylabel("Amplitude($m$)")
plt.title("Amplitude-Frequency curve")
plt.show()


plt.figure(figsize=(16,4))
plt.plot(freqs, 2*np.abs(xf),'b--')

plt.xlabel("Frequency(Hz)")
plt.ylabel("Amplitude($m$)")
plt.title("Amplitude-Frequency curve")
plt.xlim(0,20)
plt.show()
