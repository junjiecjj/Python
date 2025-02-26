#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:48:21 2025

@author: JunJie Chen
"""




import numpy as np
import scipy
import matplotlib.pyplot as plt

from Channel import channelConfig
from Channel import  Large_rayleigh_fast, Large_rician_fast


np.random.seed(42)
frame_len = 1024
B      = 4e6                    # bandwidth, Hz
n0     = -150                   # 噪声功率谱密度, dBm/Hz
n0     = 10**(n0/10.0)/1000     # 噪声功率谱密度, Watts/Hz
N0     = n0 * B                 # 噪声功率, Watts
K      = 6                      # 用户num

P_total = K
# P_max   = 30                     # 用户发送功率, dBm
# P_max   = 10**(P_max/10.0)/1000  # Watts
P_max   = P_total / 3              # Watts

## 产生信道系数
BS_locate, users_locate, beta_Au, PL_Au, d_Au = channelConfig(K, r = 100, rmin = 0.6)
H1 = Large_rayleigh_fast(K, frame_len, PL_Au, noisevar = N0)
H2 = Large_rician_fast(K, frame_len, beta_Au, PL_Au, noisevar = N0)

Hbar = np.mean(np.abs(H1)**2, axis = 1) # * np.sqrt(N0)/ np.sqrt(PL_Au.flatten())
# Hbar = np.mean(np.abs(H2)**2, axis = 1) # * np.sqrt(N0)/ np.sqrt(PL_Au.flatten())
# print(f"H1bar = \n{H1bar}, ")
print(f"Hbar = {Hbar}, \n sqrt(PL_Au / N0) = {np.sqrt(PL_Au.flatten())/np.sqrt(N0)}")

# 约束条件
sorted_idx = Hbar.argsort()[::-1]
constraints = [
    {'type': 'ineq', 'fun': lambda p: P_total - np.sum(p)},    # 总功率约束
    {'type': 'ineq', 'fun': lambda p: P_max - p},              # 单用户功率约束
    {'type': 'ineq', 'fun': lambda p: p[sorted_idx[:-1]] * Hbar[sorted_idx[:-1]]/1 - p[sorted_idx[1:]] * Hbar[sorted_idx[1:]]/1}
]

# 变量边界 (0 <= p_i <= P_max)
bounds = [(P_total/(4*K), P_max) for _ in range(K)]

# 初始猜测 (随机生成满足总功率约束)
init = np.random.rand(K) * P_max
init = init / np.sum(init) * P_total

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

# ======================== 执行优化 ========================
result = scipy.optimize.minimize(
    objective_function,
    init,
    args = (K, Hbar.copy(), 1),
    method = 'SLSQP',
    bounds = bounds,
    constraints = constraints,
    options = {
        'maxiter': 1000,       # 增加最大迭代次数
        'ftol': 1e-6,          # 提高收敛精度
        'disp': True           # 显示优化过程信息
    }
)

def getSINR(h2, optim_powers, sigma2 = 1):
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
        sinr = (sorted_p[k] * sorted_h2[k]) / (interference + sigma2)
        SINR[i] = sinr
        Capacity[i] = np.log2(1 + sinr)
    return SINR, Capacity

SINR, Capacity = getSINR(Hbar.copy(), result.x, sigma2 = 1)
# ======================== 结果分析 ========================
if result.success:
    optimized_powers = np.round(result.x, 3)
    total_capacity = -result.fun

    print("\n优化结果:")
    print("--------------------------")
    for k in sorted_idx:
        print(f"用户{k:2d}功率: {optimized_powers[k]:.3f} W, d = {d_Au.flatten()[k]:.3f} m, sinr = {10 * np.log10(SINR[k]):.4f}(dB), C={Capacity[k]:.3f} bps/Hz, Pk*|hk|^2/N0 = {optimized_powers[k] * Hbar[k]/1:.8f}")
    print("--------------------------")
    print(f"总功率消耗: {np.sum(optimized_powers):.3f} W (约束: {P_total:.3f} W)")
    print(f"单用户最大功率: {np.max(optimized_powers):.3f} W (约束: {P_max:.3f} W)")
    # print(f"Pi*|hi|^2 > Pj*|hj|^2: {optimized_powers[idx] * H2bar[idx]**2}  ")
    print("--------------------------")
    print(f"系统总容量: {total_capacity:.3f} bps/Hz")
else:
    print("优化失败:", result.message)



#%%
# import numpy as np
# from scipy.optimize import minimize
# np.random.seed(42)  # 固定随机种子保证可重复性

# # ======================== 系统参数设置 ========================
# num_users = 3                # 用户数量
# P_total = 10                 # 总功率约束 (W)
# P_max = 5                    # 单用户最大功率 (W)
# cell_radius = 500            # 小区半径 (m)
# fc = 2e9                     # 载波频率 2GHz
# rice_K = 5                   # 莱斯信道 K 因子
# num_realizations = 1       # 信道实现的蒙特卡洛次数
# # ======================== 优化问题设置 ========================

# # ======================== 噪声功率计算 ========================
# BW = 10e6                     # 系统带宽 10MHz
# NF = 9                        # 接收机噪声系数 (dB)
# k = 1.38e-23                  # 玻尔兹曼常数
# T = 300                       # 温度 (K)

# # 热噪声功率计算
# noise_power = k * T * BW * 10**(NF/10)  # 线性值 (W)
# # noise_power = 0.00001

# # ======================== 信道模型 ========================
# def generate_channel(num_users, cell_radius, rice_K):
#     # 生成用户位置(距离基站50-500米)
#     user_distances = np.random.uniform(50, cell_radius, num_users)

#     # ---------- 大尺度衰落 ----------
#     # 3GPP UMa路径损耗模型 (单位: dB)
#     d_km = user_distances / 1000  # 转换为千米
#     fc_GHz = fc / 1e9            # 转换为GHz
#     path_loss = 28.0 + 22 * np.log10(d_km) + 20 * np.log10(fc_GHz)

#     # 阴影衰落 (对数正态分布，标准差8dB)
#     shadowing = np.random.normal(0, 8, num_users)
#     large_scale = 10**(-(path_loss + shadowing)/10)  # 转换为线性值

#     # ---------- 小尺度衰落 ----------
#     # 莱斯信道模型
#     los_component = np.sqrt(rice_K/(2*(rice_K+1))) * (np.random.randn(num_users) + 1j*np.random.randn(num_users))
#     nlos_component = np.sqrt(1/(2*(rice_K+1))) * (np.random.randn(num_users) + 1j*np.random.randn(num_users))
#     small_scale = los_component + nlos_component

#     # 总信道增益 (幅度)
#     h = np.sqrt(large_scale) * small_scale
#     return np.abs(h)
# # ======================== 优化目标函数 ========================
# def objective_function(powers, num_users, channel_realizations):
#     total_capacity = 0.0
#     for h in channel_realizations:
#         # 动态SIC顺序：按信道增益降序
#         sorted_idx = np.argsort(h)[::-1]
#         sorted_p = powers[sorted_idx]
#         sorted_h = h[sorted_idx]

#         # 计算每个用户的容量
#         for k in range(num_users):
#             # 仅考虑未被消除用户的干扰
#             interference = np.sum(sorted_p[k+1:] * sorted_h[k+1:]**2)
#             snr = (sorted_p[k] * sorted_h[k]**2) / (interference + noise_power)
#             total_capacity += np.log2(1 + snr)

#     # 返回负平均容量用于最小化
#     return -total_capacity / num_realizations

# # 生成信道实现
# channel_realizations = np.array([generate_channel(num_users, cell_radius, rice_K) for _ in range(num_realizations)])

# # 打印示例信道增益验证
# print("示例信道增益 (线性值):")
# print("用户1 | 用户2 | 用户3")
# print(f"{channel_realizations[0][0]:.4f} | {channel_realizations[0][1]:.4f} | {channel_realizations[0][2]:.4f}")

# # 约束条件
# constraints = [
#     {'type': 'ineq', 'fun': lambda p: P_total - np.sum(p)},   # 总功率约束
#     {'type': 'ineq', 'fun': lambda p: P_max - p}              # 单用户功率约束
# ]

# # 变量边界 (0 <= p_i <= P_max)
# bounds = [(1, P_max) for _ in range(num_users)]

# # 初始猜测 (随机生成满足总功率约束)
# np.random.seed(42)
# initial_guess = np.random.rand(num_users) * P_max
# initial_guess = initial_guess / np.sum(initial_guess) * P_total

# # ======================== 执行优化 ========================
# result = minimize(
#     objective_function,
#     initial_guess,
#     args=(num_users, channel_realizations),
#     method='SLSQP',
#     bounds=bounds,
#     constraints=constraints,
#     options={
#         'maxiter': 1000,       # 增加最大迭代次数
#         'ftol': 1e-6,          # 提高收敛精度
#         'disp': True           # 显示优化过程信息
#     }
# )

# # ======================== 结果分析 ========================
# if result.success:
#     optimized_powers = np.round(result.x, 3)
#     total_capacity = -result.fun

#     print("\n优化结果:")
#     print("--------------------------")
#     print(f"用户1功率: {optimized_powers[0]:.3f} W")
#     print(f"用户2功率: {optimized_powers[1]:.3f} W")
#     print(f"用户3功率: {optimized_powers[2]:.3f} W")
#     print("--------------------------")
#     print(f"总功率消耗: {np.sum(optimized_powers):.3f} W (约束: {P_total} W)")
#     print(f"单用户最大功率: {np.max(optimized_powers):.3f} W (约束: {P_max} W)")
#     print("--------------------------")
#     print(f"系统总容量: {total_capacity:.2f} bps/Hz")
# else:
#     print("优化失败:", result.message)

















































