#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:16:11 2025

@author: jack
"""



import numpy as np
import scipy
# import matplotlib.pyplot as plt

# from Channel import channelConfig
# from Channel import  Large_rayleigh_fast, Large_rician_fast


# np.random.seed(42)
# frame_len = 1024
# B      = 4e6                    # bandwidth, Hz
# n0     = -140                   # 噪声功率谱密度, dBm/Hz
# n0     = 10**(n0/10.0)/1000     # 噪声功率谱密度, Watts/Hz
# N0     = n0 * B                 # 噪声功率, Watts
# K      = 4                      # 用户num

# P_total = K
# # P_max   = 30                     # 用户发送功率, dBm
# # P_max   = 10**(P_max/10.0)/1000  # Watts
# P_max   = P_total / 3              # Watts

# ## 产生信道系数
# BS_locate, users_locate, beta_Au, PL_Au, d_Au = channelConfig(K, r = 100, rmin = 0.6)
# H1 = Large_rayleigh_fast(K, frame_len, beta_Au, PL_Au, noisevar = N0)
# # H2 = Large_rician_fast(K, frame_len, beta_Au, PL_Au, noisevar = N0)

# Hbar = np.mean(np.abs(H1)**2, axis = 1) # * np.sqrt(N0)/ np.sqrt(PL_Au.flatten())
# # Hbar = np.mean(np.abs(H2)**2, axis = 1) # * np.sqrt(N0)/ np.sqrt(PL_Au.flatten())
# # print(f"H1bar = \n{H1bar}, ")
# print(f"Hbar = {Hbar}, \n sqrt(PL_Au / N0) = {np.sqrt(PL_Au.flatten())/np.sqrt(N0)}")


# ======================== 优化目标函数 ========================
def objective_function(powers, num_users, h2, sigma2 = 1):
    total_capacity = 0.0
    # for h in channel_realizations:
    # 动态SIC顺序：按信道增益降序
    sorted_idx = np.argsort(h2)[::-1]
    sorted_h2 = h2[sorted_idx]
    sorted_p = powers[sorted_idx]

    # 计算每个用户的容量
    for k in range(num_users):
        # 仅考虑未被消除用户的干扰
        interference = np.sum(sorted_p[k+1:] * sorted_h2[k+1:])
        sinr = (sorted_p[k] * sorted_h2[k]) / (interference + sigma2)
        total_capacity += np.log2(1 + sinr)

    # 返回负平均容量用于最小化
    return -total_capacity

def getSINR(h2, optim_powers, noisevar = 1):
    K = optim_powers.size
    SINR = np.zeros(K)
    Capacity = np.zeros(K)
    # 动态SIC顺序：按信道增益降序
    sorted_idx = np.argsort(h2)[::-1]
    sorted_p = optim_powers[sorted_idx]
    sorted_h2 = h2[sorted_idx]

    # 计算每个用户的容量
    for k, i in enumerate(sorted_idx):
        # 仅考虑未被消除用户的干扰
        interference = np.sum(sorted_p[k+1:] * sorted_h2[k+1:])
        sinr = (sorted_p[k] * sorted_h2[k]) / (interference + noisevar)
        SINR[i] = sinr
        Capacity[i] = np.log2(1 + sinr)
    return SINR, Capacity

def NOMAcapacityOptim(H2bar, d_Au, P_total, P_max, noisevar = 1 ):
    K = H2bar.size
    # 约束条件
    sorted_idx = H2bar.argsort()[::-1]
    constraints = [
        {'type': 'ineq', 'fun': lambda p: P_total - np.sum(p)},    # 总功率约束
        {'type': 'ineq', 'fun': lambda p: P_max - p},              # 单用户功率约束
        {'type': 'ineq', 'fun': lambda p: p[sorted_idx[:-1]] * H2bar[sorted_idx[:-1]]/noisevar - p[sorted_idx[1:]] * H2bar[sorted_idx[1:]]/noisevar}
    ]

    ## 变量边界 (0 <= p_i <= P_max)
    bounds = [(P_total/(4*K), P_max) for _ in range(K)]

    # 初始猜测 (随机生成满足总功率约束)
    init = np.random.rand(K) * P_max
    init = init / np.sum(init) * P_total

    ## ======================== 执行优化 ========================
    result = scipy.optimize.minimize(
        objective_function,
        init,
        args = (K, H2bar.copy(), noisevar),
        method = 'SLSQP',
        bounds = bounds,
        constraints = constraints,
        options = {
            'maxiter': 1000,       # 增加最大迭代次数
            'ftol': 1e-8,          # 提高收敛精度
            'disp': False           # 显示优化过程信息
        }
    )

    optimized_powers = result.x
    total_capacity = -result.fun
    SINR, Capacity = getSINR(H2bar.copy(), result.x, noisevar = noisevar)

    ## ======================== 结果分析 ========================
    if result.success:
        print("优化成功:",)
        optimized_powers = np.round(result.x, 3)
        total_capacity = -result.fun

        print("\n优化结果:")
        print("  --------------------------")
        for k in sorted_idx:
            print(f"  用户{k:2d}功率: {optimized_powers[k]:.3f} W, d = {d_Au.flatten()[k]:.3f} m, sinr = {10 * np.log10(SINR[k]):.4f}(dB), C={Capacity[k]:.3f} bps/Hz, Pk*|hk|^2/N0 = {optimized_powers[k] * H2bar[k]/noisevar:.8f}")
        print("  --------------------------")
        print(f"  总功率消耗: {np.sum(optimized_powers):.3f} W (约束: {P_total:.3f} W)")
        print(f"  单用户最大功率: {np.max(optimized_powers):.3f} W (约束: {P_max:.3f} W)")
        # print(f"Pi*|hi|^2 > Pj*|hj|^2: {optimized_powers[idx] * H2bar[idx]**2}  ")
        print("  --------------------------")
        print(f"  系统总容量: {total_capacity:.3f} bps/Hz")
    else:
        print("优化失败:", result.message)

    return optimized_powers, total_capacity, SINR, Capacity


def JointCapacityOptim(PL_Au, P_total, ):
    alpha = P_total/np.sum(1/PL_Au)

    optimized_powers = alpha/PL_Au

    # H_compensate = H1 * np.sqrt(optimized_powers)

    # Hbar1 = np.mean(np.abs(H_compensate)**2, axis = 1)
    # SINR, Capacity = getSINR(H2bar.copy(), optimized_powers.flatten(), noisevar = noisevar)
    # total_capacity = Capacity.sum()

    return optimized_powers.flatten()

def EquaCapacityOptim(H2bar, d_Au, P_total, ):

    optimized_powers = np.ones(H2bar.size) * P_total/H2bar.size

    # H_compensate = H1 * np.sqrt(optimized_powers)

    # Hbar1 = np.mean(np.abs(H_compensate)**2, axis = 1)
    # SINR, Capacity = getSINR(H2bar.copy(), optimized_powers.flatten(), noisevar = noisevar)
    # total_capacity = Capacity.sum()

    return optimized_powers.flatten()  # , total_capacity, SINR, Capacity


# optimized_powers, total_capacity, SINR, Capacity = NOMAcapacityOptim(Hbar, d_Au, P_total, P_max, noisevar = 1, )
# print(f"{optimized_powers}, {total_capacity}, {SINR}, {Capacity}\n\n")

# optimized_powers1 = JointCapacityOptim(PL_Au, P_total, noisevar = 1, )
# print(f"{optimized_powers1}, \n\n")

# optimized_powers2, total_capacity2, SINR2, Capacity2 = EquaCapacityOptim(Hbar, d_Au, P_total, P_max, noisevar = 1, )
# print(f"{optimized_powers2}, {total_capacity2}, {SINR2}, {Capacity2}")




































