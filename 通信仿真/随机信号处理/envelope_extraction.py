#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:57:34 2025

@author: jack
"""

import math
import numpy as np
# from numpy import array, sign, zeros
from scipy.interpolate import interp1d
import scipy.signal

import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp

def general_equation(first_x,first_y,second_x,second_y):
    # 斜截式 y = kx + b
    A = second_y-first_y
    B = first_x-second_x
    C = second_x * first_y - first_x * second_y
    k = -1 * A / B
    b = -1 * C / B
    return k, b


#输入信号序列即可(list)
def envelope_extraction(signal):
    s = signal.astype(float )
    q_u = np.zeros(s.shape)
    q_l =  np.zeros(s.shape)

    #在插值值前加上第一个值。这将强制模型对上包络和下包络模型使用相同的起点。
    #Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.
    u_x = [0,] #上包络的x序列
    u_y = [s[0],] #上包络的y序列

    l_x = [0,] #下包络的x序列
    l_y = [s[0],] #下包络的y序列

    # 检测波峰和波谷，并分别标记它们在u_x,u_y,l_x,l_中的位置。
    #Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

    for k in range(1,len(s)-1):
        if (np.sign(s[k]-s[k-1])==1) and (np.sign(s[k]-s[k+1])==1):
            u_x.append(k)
            u_y.append(s[k])

        if (np.sign(s[k]-s[k-1])==-1) and ((np.sign(s[k]-s[k+1]))==-1):
            l_x.append(k)
            l_y.append(s[k])

    u_x.append(len(s)-1) #上包络与原始数据切点x
    u_y.append(s[-1]) #对应的值

    l_x.append(len(s)-1) #下包络与原始数据切点x
    l_y.append(s[-1]) #对应的值

    #u_x,l_y是不连续的，以下代码把包络转为和输入数据相同大小的数组[便于后续处理，如滤波]
    upper_envelope_y = np.zeros(len(signal))
    lower_envelope_y = np.zeros(len(signal))

    upper_envelope_y[0] = u_y[0]#边界值处理
    upper_envelope_y[-1] = u_y[-1]
    lower_envelope_y[0] =  l_y[0]#边界值处理
    lower_envelope_y[-1] =  l_y[-1]

    #上包络
    last_idx,next_idx = 0, 0
    k, b = general_equation(u_x[0], u_y[0], u_x[1], u_y[1]) #初始的k,b
    for e in range(1, len(upper_envelope_y)-1):

        if e not in u_x:
            v = k * e + b
            upper_envelope_y[e] = v
        else:
            idx = u_x.index(e)
            upper_envelope_y[e] = u_y[idx]
            last_idx = u_x.index(e)
            next_idx = u_x.index(e) + 1
            #求连续两个点之间的直线方程
            k, b = general_equation(u_x[last_idx], u_y[last_idx], u_x[next_idx], u_y[next_idx])

    #下包络
    last_idx,next_idx = 0, 0
    k, b = general_equation(l_x[0], l_y[0], l_x[1], l_y[1]) #初始的k,b
    for e in range(1, len(lower_envelope_y)-1):

        if e not in l_x:
            v = k * e + b
            lower_envelope_y[e] = v
        else:
            idx = l_x.index(e)
            lower_envelope_y[e] = l_y[idx]
            last_idx = l_x.index(e)
            next_idx = l_x.index(e) + 1
            #求连续两个切点之间的直线方程
            k, b = general_equation(l_x[last_idx], l_y[last_idx], l_x[next_idx], l_y[next_idx])

    #也可以使用三次样条进行拟合
    #u_p = interp1d(u_x,u_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
    #l_p = interp1d(l_x,l_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
    #for k in range(0,len(s)):
     #   q_u[k] = u_p(k)
     #   q_l[k] = l_p(k)

    return upper_envelope_y, lower_envelope_y



def get_envelope(x,):
    """use the Hilbert transform to determine the amplitude envelope.
    Parameters:
    x : ndarray
        Real sequence to compute  amplitude envelope.
    N : {None, int}, optional, Number of Fourier components. Default: x.shape[axis]
        Length of the hilbert.

    Returns:
    amplitude_envelope: ndarray
        The amplitude envelope.

    """

    analytic_signal = hilbert(x,)
    amplitude_envelope = np.abs(analytic_signal)
    # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)

    return amplitude_envelope

## 1
x = np.linspace(0, 20, 201)
y = np.sin(x)
amplitude_envelope = get_envelope(y)
plt.figure(figsize = (8, 6))
plt.plot(x, y,label='signal')
plt.plot(x,amplitude_envelope,label='envelope')
plt.ylabel('Amplitude')
plt.xlabel('Location (x)')
plt.legend()
plt.show()

## 2
duration = 1.0
fs = 400.0
samples = int(fs*duration)
t = np.arange(samples) / fs

signal = scipy.signal.chirp(t, 20.0, t[-1], 100.0)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )

# analytic_signal = scipy.signal.hilbert(signal)
amplitude_envelope = get_envelope(signal)

up_envelope, lw_envelope = envelope_extraction(signal)

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
ax0.plot(t, signal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()

ax1.plot(t, signal, label='signal')
ax1.plot(t, up_envelope, label='up envelope')
ax1.plot(t, lw_envelope, label='low envelope')
ax1.set_xlabel("time in seconds")
ax1.legend()

# ax2.plot(t[1:], instantaneous_frequency)
# ax2.set_xlabel("time in seconds")
# ax2.set_ylim(0.0, 120.0)
fig.tight_layout()



#%%

# 在Python中，使用科学计算库如NumPy和SciPy来进行包络谱分析。下面是一个使用Python进行包络谱分析的示例代码：

# import numpy as np
# from scipy.signal import hilbert
# import scipy

# def envelope_spectrum_analysis(signal, sample_rate):
#     # 计算信号的包络
#     analytic_signal = hilbert(signal)
#     amplitude_envelope = np.abs(analytic_signal)

#     # 应用汉宁窗函数
#     window =  scipy.signal.windows.hann(len(signal))
#     windowed_envelope = amplitude_envelope * window

#     # 计算包络谱
#     spectrum = np.fft.fft(windowed_envelope) / signal.size
#     spectrum = np.abs(spectrum[:len(signal)//2])
#     frequency = np.fft.fftfreq(len(signal), 1/sample_rate)
#     frequency = frequency[:len(signal)//2]

#     return frequency, spectrum

# # 示例用法
# # 生成示例信号
# fs = 1000
# T = 1.0
# t = np.arange(0, T, 1/fs)
# f1 = 10.0
# f2 = 100.0
# x = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t)

# # 进行包络谱分析
# frequency, spectrum = envelope_spectrum_analysis(x, fs)

# # 打印结果
# for freq, spec in zip(frequency, spectrum):
#     print("Frequency: {:.2f} Hz, Spectrum: {:.2f}".format(freq, spec))
# # 在上面的示例代码中，我们定义了一个envelope_spectrum_analysis函数，用于进行包络谱分析。函数接受信号和采样率作为输入，并返回频率和包络谱结果。
# # 我们生成了一个示例信号，然后调用envelope_spectrum_analysis函数进行包络谱分析。最后，我们打印出每个频率点对应的包络谱结果。
# # 请注意，示例代码中使用了希尔伯特变换来提取信号的包络，并应用了汉宁窗函数来减小频谱泄漏效应。这只是其中一种方法，你可以根据具体需求选择其他的包络提取和窗函数方法

# plt.plot(frequency, spectrum)



# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.signal.windows import gaussian
# # from scipy.signal import envelope
# n, n_out = 500, 40  # number of signal samples and envelope samples
# T = 2 / n  # sampling interval for 2 s duration
# t = np.arange(n) * T  # time stamps
# a_x = gaussian(len(t), 0.4/T)  # instantaneous amplitude
# phi_x = 30*np.pi*t + 35*np.cos(2*np.pi*0.25*t)  # instantaneous phase
# x_carrier = a_x * np.cos(phi_x)
# x_drift = 0.3 * gaussian(len(t), 0.4/T)  # drift
# x = x_carrier + x_drift
# bp_in = (int(4 * (n*T)), None)  # 4 Hz highpass input filter
# x_env, x_res = scipy.signal.envelope(x, bp_in, n_out=n_out)
# t_out = np.arange(n_out) * (n / n_out) * T
# fg0, ax0 = plt.subplots(1, 1, tight_layout=True)
# ax0.set_title(r"$4\,$Hz Highpass Envelope of Drifting Signal")
# ax0.set(xlabel="Time in seconds", xlim=(0, n*T), ylabel="Amplitude")
# ax0.plot(t, x, 'C0-', alpha=0.5, label="Signal")
# ax0.plot(t, x_drift, 'C2--', alpha=0.25, label="Drift")
# ax0.plot(t_out, x_res+x_env, 'C1.-', alpha=0.5, label="Envelope")
# ax0.plot(t_out, x_res-x_env, 'C1.-', alpha=0.5, label=None)
# ax0.grid(True)
# ax0.legend()
# plt.show()



































































