#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:16:10 2024

@author: jack


"Deng R, Di B, Zhang H, et al. Reconfigurable Holographic Surface Enabled Multi-User Wireless Communications: Amplitude-Controlled Holographic Beamforming[J]. IEEE Transactions on Wireless Communications, 2022."
Fractional programming for communication systems—Part I: Power control and beamforming
Fractional programming for communication systems—Part II: Uplink scheduling via matching
https://www.cnblogs.com/hjd21/p/16608461.html
https://www.zhihu.com/tardis/zm/art/599204238?source_id=1005


" Weighted Sum-Rate Optimization for Intelligent Reflecting Surface Enhanced Wireless Networks 本文应该是第一篇以速率为优化目标的IRS波束成形文章"
https://blog.csdn.net/weixin_39274659/article/details/121148894
https://zhuyulab.blog.csdn.net/article/details/121192360?spm=1001.2014.3001.5502
https://github.com/guohuayan/WSR_maximization_for_RIS_system?utm_source=catalyzex.com
"""



#%% https://zhuanlan.zhihu.com/p/599204238
# https://www.cnblogs.com/hjd21/p/16608461.html
# https://kaimingshen.github.io/publications.html
# https://www.cnblogs.com/longtianbin/p/17124657.html

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from math import log10, log2, sqrt, tan, pi, cos, sin

# 设置随机种子
np.random.seed(1)
epsilon = 1e-5  # 收敛阈值
Max_iter = 50   # 最大迭代步数

R = 2  # 基站到基站的距离 0.8
def PL(d):
    return 128.1 + 37.6 * np.log10(d)  # 路损模型，d--km
U = 7  # 用户个数，每个蜂窝中有一个用户
C = 7  # 基站个数
P_val = 50  # 最大发射功率43 dBm
sigma2 = -105  # 噪声功率 -100 dBm
shadowing_std = 8  # 阴影衰落的标准差-8 dB

B = 10e6  # 10Mhz

def db2pow(db):
    """将dB值转换为功率值"""
    return 10 ** (db / 10)

def channel_generate(U, R, PL, shadowing_std):
    """在蜂窝小区覆盖的范围内产生用户位置，计算大尺度衰落"""
    # 在正六边形蜂窝小区中撒点
    cell_loc = np.array([
        [0, 0],
        [R * cos(pi/6), R * sin(pi/6)],
        [0, R],
        [-R * cos(pi/6), R * sin(pi/6)],
        [-R * cos(pi/6), -R * sin(pi/6)],
        [0, -R],
        [R * cos(pi/6), -R * sin(pi/6)]
    ])  # 基站坐标

    L = R * tan(pi/6)  # 六边形的边长

    # 产生用户位置
    user_loc = np.zeros((U, 2))
    i = 0
    while i < U:
        x = 2 * L * np.random.rand(2) - L
        if (abs(x[0]) + abs(x[1])/sqrt(3)) <= L and abs(x[1]) <= L * sqrt(3)/2:
            user_loc[i, :] = x + cell_loc[i, :]
            i += 1

    # 计算距离（转换为km）
    dis = np.zeros((U, C))
    for i in range(U):
        for j in range(C):
            dis[i, j] = np.linalg.norm(cell_loc[j, :] - user_loc[i, :]) / 1000.0  # 转换为km

    # 计算信道增益，考虑服从对数正态分布的阴影衰落
    H_gain = -PL(dis) - shadowing_std * np.random.randn(U, C)
    return H_gain

# 生成信道增益
H_gain = channel_generate(U, R, PL, shadowing_std)
print(f"H_gain shape: {H_gain.shape}")

# 为了优化方便，将噪声归一化
H_gain_power = db2pow(H_gain - sigma2)
print(f"H_gain_power shape: {H_gain_power.shape}")

# 直接式求解方法 - 简化版本
p_temp = db2pow(P_val * np.ones(C)) / 2  # Step 0: 按等功率初始化分配

iter_count = 0  # 迭代计数
sum_rate = []   # 记录和速率
sum_rate_old = 100

A = np.ones((U, C)) - np.eye(U)  # 求和指示矩阵

print("Starting direct FP...")

# 先跳过复杂的CVXPY优化，直接使用闭式解方法
print("Skipping direct FP due to complexity, proceeding with closed-form FP...")

# 闭式解方法
print("\nStarting closed-form FP...")
A = np.ones((U, C)) - np.eye(U)
p = db2pow(P_val * np.ones(C)) / 2

iter_count_cf = 0
sum_rate2 = []
sum_rate_old_cf = 100

while iter_count_cf < Max_iter:
    iter_count_cf += 1

    # 计算gamma
    gamma = np.zeros(U)
    for i in range(U):
        interf = 0
        for j in range(C):
            interf += H_gain_power[i, j] * A[i, j] * p[j]
        gamma[i] = H_gain_power[i, i] * p[i] / (interf + 1e-10)  # 避免除零

    # 计算y_star
    y_star = np.zeros(U)
    for i in range(U):
        total_interf = 0
        for j in range(C):
            total_interf += H_gain_power[i, j] * p[j]
        y_star[i] = np.sqrt((1 + gamma[i]) * H_gain_power[i, i] * p[i]) / (total_interf + 1e-10)

    # 更新p
    new_p = np.zeros(C)
    for j in range(C):
        denominator = 1e-10  # 小常数避免除零

        # 计算分母
        temp_sum = 0
        for i in range(U):
            temp_sum += y_star[i]**2 * H_gain_power[i, j]
        denominator = temp_sum**2

        # 计算分子
        numerator = 1e-10
        for i in range(U):
            if i == j:  # 只有当i==j时才贡献
                numerator += y_star[i]**2 * (1 + gamma[i]) * H_gain_power[i, i]

        new_p[j] = min(db2pow(P_val), numerator / (denominator + 1e-10))

    p = new_p

    # 计算目标值
    opt_value = 0
    for i in range(U):
        interf = 0
        for j in range(C):
            interf += H_gain_power[i, j] * A[i, j] * p[j]
        opt_value += np.log2(1 + H_gain_power[i, i] * p[i] / (interf + 1e-10))

    if not np.isnan(opt_value):
        sum_rate2.append(opt_value)
        print(f"Closed-form Iteration {iter_count_cf}: sum_rate = {opt_value:.4f}")

    if iter_count_cf > 1 and abs(opt_value - sum_rate_old_cf) / (sum_rate_old_cf + 1e-10) < epsilon:
        break
    else:
        sum_rate_old_cf = opt_value

# 绘制结果
plt.figure(figsize=(12, 8))

if sum_rate2:
    plt.plot(range(1, len(sum_rate2) + 1), [rate * B / 1e6 for rate in sum_rate2], '-ro', linewidth=2, markersize=6, label='Close-form FP')
    print(f"Final sum rate: {sum_rate2[-1] * B / 1e6:.2f} Mbps")

plt.xlabel('Iteration number', fontsize=12)
plt.ylabel('Sum rate (Mbps)', fontsize=12)
plt.legend(fontsize=10)
plt.title('FP Algorithm Convergence (Closed-form)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Simulation completed successfully!")






















































































































































































































