#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:03:56 2022

@author: jack

https://blog.csdn.net/innovationy/article/details/121572508?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-121572508-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-121572508-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=6

https://blog.csdn.net/jiangwenqixd/article/details/109521694?spm=1001.2101.3001.6650.5&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-109521694-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-5-109521694-blog-131543206.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=10

https://blog.csdn.net/RICEresearchNOTE/article/details/140855697

https://blog.51cto.com/u_16213651/8904378

https://blog.csdn.net/qq_44648285/article/details/143471871

https://zhuanlan.zhihu.com/p/692354746

https://blog.csdn.net/qq_43485394/article/details/122655901

https://blog.51cto.com/u_16213651/8904378






"""


import numpy as np
import matplotlib.pyplot as plt

# 参数设置
A = 1.0  # 信号幅度
fs = 1000  # 采样频率
T = 1  # 信号持续时间
t = np.linspace(0, T, fs * T, endpoint=False)

# 生成长脉冲信号
pulse_duration = 0.2  # 脉冲持续时间
pulse = np.zeros_like(t)
pulse[int(0.5 * fs):int((0.5 + pulse_duration) * fs)] = A

# 添加杂波
noise = np.random.normal(0, 0.1, pulse.shape)
received_signal = pulse + noise

# 匹配滤波器设计
matched_filter = np.flip(pulse)

# 进行匹配滤波
compressed_signal = np.convolve(received_signal, matched_filter, mode='same')

# 绘图
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, received_signal)
plt.title('Received Signal with Noise')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, compressed_signal)
plt.title('Compressed Signal After Matched Filtering')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()
