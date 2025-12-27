#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 17:49:55 2025

@author: jack
"""

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
print(f"H_gain_power range: [{H_gain_power.min():.2e}, {H_gain_power.max():.2e}]")

# 直接式求解方法 - 重新实现
print("Starting direct FP...")
p_temp = db2pow(P_val * np.ones(C)) / 2  # Step 0: 按等功率初始化分配

iter_count = 0
sum_rate = []
sum_rate_old = 100

A = np.ones((U, C)) - np.eye(U)  # 求和指示矩阵

# 先进行几步手动计算来验证
print("Performing initial manual calculations...")
for i in range(3):  # 先做3次手动迭代
    # Step 1: 计算y_star
    interference = np.zeros(U)
    for u in range(U):
        for c in range(C):
            interference[u] += H_gain_power[u, c] * A[u, c] * p_temp[c]
        interference[u] += 1

    diag_H = np.diag(H_gain_power)
    y_star = np.sqrt(diag_H * p_temp) / interference

    print(f"Manual step {i+1}: y_star range = [{y_star.min():.2e}, {y_star.max():.2e}]")

    # 更新p_temp进行下一次迭代
    p_temp = p_temp * 0.9  # 简单衰减

# 重置p_temp
p_temp = db2pow(P_val * np.ones(C)) / 2

# 现在使用CVXPY进行直接FP
while iter_count < Max_iter:
    iter_count += 1

    # Step 1: 计算y_star
    interference = np.zeros(U)
    for u in range(U):
        for c in range(C):
            interference[u] += H_gain_power[u, c] * A[u, c] * p_temp[c]
        interference[u] += 1

    diag_H = np.diag(H_gain_power)
    y_star = np.sqrt(diag_H * p_temp) / interference

    # Step 2: CVXPY优化
    p = cp.Variable(C, nonneg=True)

    # 构建目标函数 - 使用更简单的表达式
    objective_terms = []
    for u in range(U):
        # 计算干扰项
        interf_u = 1  # 噪声项
        for c in range(C):
            interf_u += H_gain_power[u, c] * A[u, c] * p[c]

        # 计算信号项
        signal_u = 2 * y_star[u] * cp.sqrt(H_gain_power[u, u] * p[u])

        # 目标函数项
        obj_term = signal_u - y_star[u]**2 * interf_u
        objective_terms.append(cp.log(1 + obj_term))

    objective = cp.Maximize(cp.sum(objective_terms))
    constraints = [p <= db2pow(P_val)]

    # 尝试不同的求解器
    solvers_to_try = [cp.ECOS, cp.SCS, cp.CVXOPT]
    result = None

    for solver in solvers_to_try:
        try:
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=solver, verbose=False, max_iters=1000)
            if result is not None and not np.isnan(result) and p.value is not None:
                break
        except:
            continue

    if result is not None and not np.isnan(result) and p.value is not None:
        p_temp = p.value.copy()
        sum_rate.append(result)
        print(f"Direct FP Iteration {iter_count}: sum_rate = {result:.6f}")

        if iter_count > 1 and abs(result - sum_rate_old) / (abs(sum_rate_old) + 1e-10) < epsilon:
            break
        sum_rate_old = result
    else:
        print(f"Direct FP failed at iteration {iter_count}, using previous values")
        # 使用小的随机扰动继续
        p_temp = p_temp * (0.95 + 0.1 * np.random.rand(C))

# 闭式解方法
print("\nStarting closed-form FP...")
p_cf = db2pow(P_val * np.ones(C)) / 2
iter_count_cf = 0
sum_rate2 = []
sum_rate_old_cf = 100

while iter_count_cf < Max_iter:
    iter_count_cf += 1

    # 计算gamma
    gamma = np.zeros(U)
    for u in range(U):
        interf = 1
        for c in range(C):
            interf += H_gain_power[u, c] * A[u, c] * p_cf[c]
        gamma[u] = H_gain_power[u, u] * p_cf[u] / max(interf, 1e-10)

    # 计算y_star
    y_star_cf = np.zeros(U)
    for u in range(U):
        total_interf = 1
        for c in range(C):
            total_interf += H_gain_power[u, c] * p_cf[c]
        y_star_cf[u] = np.sqrt((1 + gamma[u]) * H_gain_power[u, u] * p_cf[u]) / max(total_interf, 1e-10)

    # 更新p
    new_p = np.zeros(C)
    for c in range(C):
        denominator = 0
        for u in range(U):
            denominator += y_star_cf[u]**2 * H_gain_power[u, c]
        denominator = max(denominator**2, 1e-10)

        numerator = 0
        for u in range(U):
            if u == c:
                numerator += y_star_cf[u]**2 * (1 + gamma[u]) * H_gain_power[u, u]

        new_p[c] = min(db2pow(P_val), numerator / denominator)

    p_cf = new_p

    # 计算目标值
    opt_value = 0
    for u in range(U):
        interf = 1
        for c in range(C):
            interf += H_gain_power[u, c] * A[u, c] * p_cf[c]
        opt_value += np.log2(1 + H_gain_power[u, u] * p_cf[u] / max(interf, 1e-10))

    sum_rate2.append(opt_value)
    print(f"Closed-form Iteration {iter_count_cf}: sum_rate = {opt_value:.6f}")

    if iter_count_cf > 1 and abs(opt_value - sum_rate_old_cf) / max(abs(sum_rate_old_cf), 1e-10) < epsilon:
        break
    sum_rate_old_cf = opt_value

# 绘制结果
plt.figure(figsize=(12, 8))

if sum_rate:
    plt.plot(range(1, len(sum_rate) + 1), [rate / np.log(2) * B / 1e6 for rate in sum_rate],
             '-b*', linewidth=2, markersize=8, label='Direct FP')
    print(f"Direct FP final rate: {sum_rate[-1] / np.log(2) * B / 1e6:.2f} Mbps")

if sum_rate2:
    plt.plot(range(1, len(sum_rate2) + 1), [rate * B / 1e6 for rate in sum_rate2],
             '-ro', linewidth=2, markersize=6, label='Close-form FP')
    print(f"Closed-form FP final rate: {sum_rate2[-1] * B / 1e6:.2f} Mbps")

plt.xlabel('Iteration number', fontsize=12)
plt.ylabel('Sum rate (Mbps)', fontsize=12)
plt.legend(fontsize=10)
plt.title('FP Algorithm Convergence', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Simulation completed!")
