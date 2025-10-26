#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 20:19:11 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzg3ODkwOTgyMw==&mid=2247488206&idx=4&sn=cd3de726d00192cfc04db9dd16912dd9&chksm=cf0dd2def87a5bc879e5bf92131913fd7c45258c4c1493806f7e844e7eb5c1250764eda14872&cur_album_id=3692626176607780876&scene=189#wechat_redirect

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 理论计算的模糊函数
"""
线性调频脉冲信号模糊函数
"""
B = 4e6                # 信号带宽
Tp = 2e-6              # 脉冲宽度
Grid = 100             # 坐标轴点数
u = B / Tp             # 调频斜率

# 创建时间轴和频率轴 - 严格按MATLAB代码
t = np.arange(-Tp, Tp + Tp/Grid, Tp/Grid)
f = np.arange(-B, B + B/Grid, B/Grid)

# 创建网格
tau, fd = np.meshgrid(t, f)

# 计算线性调频脉冲模糊函数 - 严格按MATLAB代码
var1 = Tp - np.abs(tau)
var2 = np.pi * (fd - u * tau) * var1
amf = np.abs(np.sinc(var2 ) * var1 / Tp)  # 注意：numpy.sinc定义为sin(pi*x)/(pi*x)
amf = amf / np.max(amf)

# 单脉冲模糊函数 - 严格按MATLAB代码
tau1_sp = (Tp - np.abs(tau)) / Tp
mul1 = np.pi * fd * (Tp - np.abs(tau)) + np.finfo(float).eps
amf_sp = np.abs((np.sin(mul1) / mul1) * tau1_sp)

# 计算归一化距离模糊和速度模糊
var3 = np.pi * u * tau * var1
tau1 = np.abs(np.sinc(var3 ) * var1)  # 注意：numpy.sinc定义为sin(pi*x)/(pi*x)
tau1 = tau1 / np.max(tau1)                   # 归一化距离模糊

mul = Tp * np.abs(np.sinc(np.pi * fd * Tp))          # 注意：numpy.sinc定义为sin(pi*x)/(pi*x)
mul = mul / np.max(mul)                      # 归一化速度模糊

# 1. 3D曲面图 - 使用您提供的风格
fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')
cbar = ax.plot_surface(tau*1e6, fd*1e-6, amf, rstride=2, cstride=2, cmap=plt.get_cmap('jet'))
# plt.colorbar(cbar)
ax.set_xlabel('时间/us')
ax.set_ylabel('fd/MHz')
ax.set_zlabel('幅值')
ax.set_title('线性调频脉冲信号的模糊函数')
ax.grid(False)
ax.set_proj_type('ortho')
plt.show()
plt.close()

# 2. 2D等高线图 - 严格按MATLAB代码，使用您提供的风格
fig, ax = plt.subplots(figsize=(8, 6))
ax.contour(tau*1e6, fd*1e-6, amf, 1, colors='blue')
ax.contour(tau*1e6, fd*1e-6, amf_sp, 1, colors='red', linestyles='--')
ax.legend(['LFM', '脉冲'])
ax.grid(True)
ax.set_xlim([-2, 2])
ax.set_ylim([-4, 4])
ax.set_xlabel('时间/us')
ax.set_ylabel('fd/MHz')
ax.set_title('线性调频脉冲信号的模糊度图')
plt.show()
plt.close()

# 3. 距离模糊函数 - 严格按MATLAB代码
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t*1e6, tau1[Grid, :])
ax.grid(True)
ax.set_xlim([-2, 2])
ax.set_ylim([0, 1])
ax.set_xlabel('时间/us')
ax.set_ylabel('|x(t,0)|')
ax.set_title('距离模糊函数')
plt.show()
plt.close()

# 4. 速度模糊函数 - 严格按MATLAB代码
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fd*1e-6, mul[:, Grid])
ax.grid(True)
ax.set_xlim([-4, 4])
ax.set_ylim([0, 1])
ax.set_xlabel('fd/MHz')
ax.set_ylabel('|x(0,fd)|')
ax.set_title('速度模糊函数')
plt.show()
plt.close()


#%%  https://mp.weixin.qq.com/s?__biz=Mzk4ODgxMDc4Ng==&mid=2247484519&idx=1&sn=290bc630027a4c9be3e79f41f51a8de2&chksm=c4e11f200b209cfc4ea8cc70999f342a9f8ccc54cfe57cbe277a3442bf1add3f443aa0dfee81&mpshare=1&scene=1&srcid=1023XuUAQkrzulkj7Wdsz3U5&sharer_shareinfo=626afc0b7dac700e2834617953b29c4f&sharer_shareinfo_first=626afc0b7dac700e2834617953b29c4f&exportkey=n_ChQIAhIQ96a7wSivybAUbK3AyyDeuhKfAgIE97dBBAEAAAAAAN0hMjFNvTYAAAAOpnltbLcz9gKNyK89dVj0ZY4C4X%2Fm357iTL28PgFxsjKZX8JdO%2BLeLMRFURDUXqHmUf0HDLgNO3v3Dn8ezF5y608cA3dtJ7E1AHMkO6EnSlcCtvcrNpVpo56sCwKb9uV1uwkLUeXhEgiDyjbj2HlCP3hsAAqbDBAD0DZOMfvRrxuv9U%2BPUQXp64omhZzDu%2FRrBiXCjbBxMy4Y%2FP3bsUaPnfvE6kKW1W3BDwwcocGwy0S4CTizf5f1cLWqkyMJIh6kmj5wtNocTXQGnNYnSm0DPD%2FVWlw%2FrDHJf8smaAK%2BMw2RxWwnhPkGHwpwgeozS9jSon4LFyRc44dMR%2BQr7YZJ1%2FIJyHsyAGIO&acctmode=0&pass_ticket=edvjKQih7T%2BwXQB4guWi7mpdauITL%2FGoCnpLKKl605ud1X3yxRndq%2Ffp6s6JMKx1&wx_header=0#rd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# def lfm_ambiguity_function():
"""
线性调频信号模糊函数计算
"""
fs = 20e6        # 采样率
t = 10e-6        # 脉宽
tt = np.arange(0, t, 1/fs)  # 时间刻度
b = 10e6         # 带宽
Nz = round(b * t)
u = b / (2 * t)  # 调频斜率

# 线性调频信号
y = np.exp(1j * 2 * np.pi * u * tt**2)
len_y = len(y)

# 构造扩展信号
x = np.concatenate([np.zeros(2*len_y), y, np.zeros(2*len_y)])
x1 = np.concatenate([np.zeros(len_y), y, np.zeros(len_y)])
N1 = len(x1)

val1 = np.sum(np.abs(y)**2)

# 延迟和多普勒轴
tau1 = np.arange(-t*fs, t*fs + 1, 1)
fd1 = np.arange(-b*t, b*t + 1, 1)

# 预分配模糊函数矩阵
MF = np.zeros((len(fd1), len(tau1)))

k = 0
for tau in tau1:
    k += 1
    l = 0
    for fd in fd1:
        l += 1
        val = np.zeros(N1, dtype=complex)
        for m in range(N1):
            # 时间因子
            idx = round(-tau1[0]) + m + round(tau)
            if 0 <= idx < len(x):
                val2 = np.conj(x[idx])
            else:
                val2 = 0
            # 频率因子
            val3 = np.exp(1j * 2 * np.pi * fd * m / len_y)
            val[m] = x1[m] * val2 * val3

        MF[l-1, k-1] = np.abs(np.sum(1/val1 * val))

# 绘制3D模糊函数图
fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')

# 创建网格
Tau, Fd = np.meshgrid(tau1/fs*1e6, fd1/t/1e6)

# 绘制表面图
cbar = ax.plot_surface(Tau, Fd, MF, rstride=2, cstride=2, cmap=plt.get_cmap('jet'), linewidth=0, antialiased=False)
# plt.colorbar(cbar)

ax.set_xlabel('时延(us)')
ax.set_ylabel('多普勒频移(MHz)')
ax.set_zlabel('模糊函数值')
ax.set_title('线性调频模糊函数', fontsize=14)
ax.grid(False)
ax.set_proj_type('ortho')

plt.show()
plt.close()



