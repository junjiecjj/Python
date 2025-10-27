#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 20:09:38 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzg3ODkwOTgyMw==&mid=2247488206&idx=5&sn=0119a292f8249f240540a31e27e04238&chksm=ce36c6a7d265c53a4fa00b267aec7ff4904b9235fac333500dd14d9bcb051c1296a7d67d01ef&mpshare=1&scene=1&srcid=10235LJ7xtUvcYta6dfGcyyZ&sharer_shareinfo=dc43a15691342bbd827244ec52c24f4b&sharer_shareinfo_first=dc43a15691342bbd827244ec52c24f4b&exportkey=n_ChQIAhIQJLpqCvQ1oI1Oy5m6VMY7hxKfAgIE97dBBAEAAAAAAObHDhcTIiEAAAAOpnltbLcz9gKNyK89dVj0UdTprnoFQrxMuTz2UZN4mJ91j1Km6Uep5aI4AUVTEbostjeojcslujhCRlCDLno9%2BUK1vbyGdIxgHIuRa6ZyITLgyzI3Nfe4tcJS3iFEytwY8kxqz0EwKh8muZNEVgbYVNAIhCiaZ9QGxgFRvZJuA6ye1JgYliLJYsDfNhVHhi4MTE63COdfju5H%2FrVWnDiodVmbApULRcMLaYWaaYpDbLOUloWS7E1Al7WmzbjINA1IaHvLGO5yxZhVV9zJNdqlTLAvo0jU25Dx6GNK23RHcaMyZXzFBcMMn%2FEevgnjY4JHNg0z75rZDQSHKSBa1B6rOzTo8tV43Oa0&acctmode=0&pass_ticket=qDRrone76A7okNoWn8lSHH3qz4ZionicK3DW7G5GNGbHOuukkUq2AaAcjCY2JXGp&wx_header=0#rd

理论计算的模糊函数
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""
单载频矩形脉冲模糊函数
"""
Tp = 1e-6  # 脉冲宽度
Grid = 64  # 坐标轴点数

# 创建时间轴和频率轴
t = np.linspace(-Tp, Tp, 2*Grid + 1)
f = np.linspace(-10/Tp, 10/Tp, 2*Grid + 1)

# 创建网格
tau, fd = np.meshgrid(t, f)

# 计算模糊函数
tau1 = (Tp - np.abs(tau)) / Tp
mul1 = np.pi * fd * (Tp - np.abs(tau)) + np.finfo(float).eps
amf = np.abs((np.sin(mul1) / mul1) * tau1)

# 1. 3D曲面图 - 使用您提供的风格
fig = plt.figure(figsize=(8, 8), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')
cbar = ax.plot_surface(tau*1e6, fd*1e-6, amf, rstride=2, cstride=2, cmap=plt.get_cmap('jet'))
# plt.colorbar(cbar)
ax.set_xlabel('时间/us')
ax.set_ylabel('fd/MHz')
ax.set_zlabel('幅值')
ax.set_title('矩形脉冲信号的模糊函数')
ax.grid(False)
ax.set_proj_type('ortho')
plt.show()
plt.close()

# 2. 2D等高线图 - 严格按MATLAB代码，使用您提供的风格
fig, ax = plt.subplots(figsize=(8, 6))
ax.contour(tau*1e6, fd*1e-6, amf, 1, colors='blue')
ax.grid(True)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_xlabel('时间/us')
ax.set_ylabel('fd/MHz')
ax.set_title('矩形脉冲信号的模糊度图')
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
ff = np.abs(np.sin(mul1) / mul1)
ffd = ff[:, Grid]
ax.plot(fd*1e-6, ffd)
ax.grid(True)
ax.set_xlim([-10, 10])
ax.set_ylim([0, 1])
ax.set_xlabel('fd/MHz')
ax.set_ylabel('|x(0,fd)|')
ax.set_title('速度模糊函数')
plt.show()
plt.close()
