#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 15:29:05 2025

@author: jack

https://zhuanlan.zhihu.com/p/692354746

"""

import numpy as np
import matplotlib.pyplot as plt
# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 14        # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 22   # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 22   # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6] # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300      # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2     # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6 # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'  # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'        # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'          # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 22

#%%  图一:(未脉冲调制波形生成)
def generate_pulse_doppler_signal(carrier_freq, pulse_width, pri, amplitude, fs):
    """生成脉冲多普勒信号和相应的回波信号"""
    num_samples = int(pri * fs)
    num_samples_pulse = int(pulse_width * fs)

    # 生成载波信号
    t = np.linspace(0, pri, num_samples, endpoint = False)
    pulse = np.zeros(num_samples)
    pulse[:num_samples_pulse] = amplitude * np.cos(2 * np.pi * carrier_freq * t[:num_samples_pulse])
    return t, pulse

def generate_echo_signals(pulse, amplitude, delay, fs):
    """生成具有特定延时的回波信号"""
    num_samples_delay = int(delay * fs)
    echo = np.roll(pulse * amplitude, num_samples_delay)
    return echo

def plot_signals(t, pulse, echo1, echo2):
    """绘制发射信号和回波信号"""
    plt.figure(figsize = (10, 5))
    plt.plot(t * 1e6, pulse, label = 'Transmitted Pulse')
    plt.plot(t * 1e6, echo1, '--', label = 'Echo Signal 1 (Amplitude = 0.5, Delay = 4μs)')
    plt.plot(t * 1e6, echo2, ':', label = 'Echo Signal 2 (Amplitude = 0.3, Delay = 5μs)')
    plt.title('Pulse Doppler Radar Signals')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    return

# 设置参数
carrier_freq = 20e6  # 载波频率20MHz
pulse_width = 1e-6  # 脉冲宽度1μs
pri = 6e-6  # 脉冲重复间隔6μs，以包括较长的延时
amplitude = 1  # 发射信号幅度
fs = 10 * carrier_freq  # 采样率

# 生成信号
t, pulse = generate_pulse_doppler_signal(carrier_freq, pulse_width, pri, amplitude, fs)

# 生成回波信号
echo1 = generate_echo_signals(pulse, 0.5, 4e-6, fs)  # 回波1，幅度0.5，延时4μs
echo2 = generate_echo_signals(pulse, 0.3, 5e-6, fs)  # 回波2，幅度0.3，延时5μs

# 绘制结果
plot_signals(t, pulse, echo1, echo2)

#%%  图二代码:线性调频脉冲压缩的图像
import numpy as np
import matplotlib.pyplot as plt
import scipy

def generate_lfm_pulse(carrier_start_freq, carrier_end_freq, pulse_width, pri, amplitude, fs):
    """生成线性调频脉冲信号和相应的时间向量"""
    num_samples = int(pri * fs)
    num_samples_pulse = int(pulse_width * fs)

    # 生成时间向量
    t = np.linspace(0, pri, num_samples, endpoint=False)

    # 生成LFM信号
    pulse = np.zeros(num_samples)
    pulse[:num_samples_pulse] = amplitude * scipy.signal.chirp(t[:num_samples_pulse], f0 = carrier_start_freq, f1 = carrier_end_freq, t1 = pulse_width, method = 'linear')

    return t, pulse

def generate_echo_signals(pulse, amplitude, delay, fs):
    """生成具有特定延时的回波信号"""
    num_samples_delay = int(delay * fs)
    echo = np.roll(pulse * amplitude, num_samples_delay)
    return echo

def plot_signals(t, pulse, echo1, echo2):
    """绘制发射信号和回波信号"""
    plt.figure(figsize=(12, 6))
    plt.plot(t * 1e6, pulse, label='Transmitted LFM Pulse')
    plt.plot(t * 1e6, echo1, '--', label='Echo Signal 1 (Amplitude = 0.5, Delay = 4μs)')
    plt.plot(t * 1e6, echo2, ':', label='Echo Signal 2 (Amplitude = 0.3, Delay = 5μs)')
    plt.title('LFM Pulse Doppler Radar Signals')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

# 设置参数
carrier_start_freq = 10e6  # 起始频率19MHz
carrier_end_freq = 20e6  # 末尾频率20MHz
pulse_width = 1e-6  # 脉冲宽度1μs
pri = 6e-6  # 脉冲重复间隔6μs
amplitude = 1  # 发射信号幅度
fs = 10 * carrier_end_freq  # 采样率

# 生成信号
t, pulse = generate_lfm_pulse(carrier_start_freq, carrier_end_freq, pulse_width, pri, amplitude, fs)

# 生成回波信号
echo1 = generate_echo_signals(pulse, 0.5, 4e-6, fs)  # 回波1，幅度0.5，延时4μs
echo2 = generate_echo_signals(pulse, 0.3, 5e-6, fs)  # 回波2，幅度0.3，延时5μs

# 绘制结果
plot_signals(t, pulse, echo1, echo2)


#%%  图三：经过匹配滤波后的图像
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, correlate

def generate_lfm_pulse(carrier_start_freq, carrier_end_freq, pulse_width, pri, amplitude, fs):
    """生成线性调频脉冲信号和相应的时间向量"""
    num_samples = int(pri * fs)
    num_samples_pulse = int(pulse_width * fs)

    t = np.linspace(0, pri, num_samples, endpoint = False)
    pulse = np.zeros(num_samples)
    # 仅在脉冲宽度内生成LFM信号
    pulse[:num_samples_pulse] = amplitude * scipy.signal.chirp(t[:num_samples_pulse], f0 = carrier_start_freq, f1 = carrier_end_freq, t1 = pulse_width, method = 'linear')
    return t, pulse
def generate_echo_signals(pulse, amplitude, delay, fs):
    """ 生成具有特定延时的回波信号 """
    num_samples_delay = int(delay * fs)
    echo = np.roll(pulse * amplitude, num_samples_delay)
    return echo
def perform_match_filtering(echo, pulse):
    """ 对回波信号进行匹配滤波处理 """
    matched_signal = scipy.signal.correlate(echo, pulse, mode = 'full')
    return matched_signal
def plot_matched_signals(t, pulse, matched_signal1, matched_signal2, fs):
    """ 绘制匹配滤波后的信号 """
    dt = t[1] - t[0]
    offset = (len(matched_signal1) - 1) / 2
    matched_time = (np.arange(len(matched_signal1)) - offset) * dt

    plt.figure(figsize = (12, 6))
    plt.plot(matched_time * 1e6, matched_signal1, label = 'Matched Signal 1', )
    plt.plot(matched_time * 1e6, matched_signal2, label = 'Matched Signal 2')
    plt.title('Matched Filter Output for Echo Signals')
    plt.xlabel('Time (μs)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    return

# 参数设置
carrier_start_freq = 10e6  # 起始频率19MHz
carrier_end_freq = 20e6  # 末尾频率20MHz
pulse_width = 1e-6  # 脉冲宽度1μs
pri = 6e-6  # 脉冲重复间隔6μs
amplitude = 1  # 发射信号幅度
fs = 10 * carrier_end_freq  # 采样率

# 生成信号和回波
t, pulse = generate_lfm_pulse(carrier_start_freq, carrier_end_freq, pulse_width, pri, amplitude, fs)
echo1 = generate_echo_signals(pulse, 0.5, 4e-6, fs)  # 回波1，幅度0.5，延时4μs
echo2 = generate_echo_signals(pulse, 0.3, 5e-6, fs)  # 回波2，幅度0.3，延时5μs

# 匹配滤波处理
matched_signal1 = perform_match_filtering(echo1, pulse)
matched_signal2 = perform_match_filtering(echo2, pulse)

# 绘制结果
plot_matched_signals(t, pulse, matched_signal1, matched_signal2, fs)


###>>>>>>>>>  图四：非调制直接进行匹配滤波

# 参数设置
carrier_start_freq = 19e6   # 起始频率 19MHz
carrier_end_freq = 20e6     # 末尾频率 20MHz
pulse_width = 1e-6          # 脉冲宽度 1μs
pri = 6e-6                  # 脉冲重复间隔 6μs
amplitude = 1               # 发射信号幅度
fs = 10 * carrier_end_freq  # 采样率

# 生成信号和回波
t, pulse = generate_lfm_pulse(carrier_start_freq, carrier_end_freq, pulse_width, pri, amplitude, fs)
echo1 = generate_echo_signals(pulse, 0.5, 4e-6, fs)  # 回波1，幅度0.5，延时4μs
echo2 = generate_echo_signals(pulse, 0.3, 5e-6, fs)  # 回波2，幅度0.3，延时5μs

# 匹配滤波处理
matched_signal1 = perform_match_filtering(echo1, pulse)
matched_signal2 = perform_match_filtering(echo2, pulse)

# 绘制结果
plot_matched_signals(t, pulse, matched_signal1, matched_signal2, fs)


#  第一段代码的线性调频脉冲的频率范围是从10MHz到20MHz，而第二段代码是从19MHz到20MHz。因此，第一段代码的带宽为10MHz，第二段代码的带宽为1MHz。
# 带宽的不同会导致匹配滤波后的脉冲压缩结果不同。带宽越大，脉冲压缩后的主瓣越窄，分辨率越高。






















































































































