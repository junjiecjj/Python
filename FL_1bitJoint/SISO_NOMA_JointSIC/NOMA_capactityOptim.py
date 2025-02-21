#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:48:21 2025

@author: JunJie Chen
"""








import numpy as np

from Channel import channelConfig
from Channel import AWGN_mac, BlockFading_mac, FastFading_mac, Large_mac


B      = 4e6                    # Hz
sigma2 = -60                    # dBm/Hz
sigma2 = 10**(sigma2/10.0)/1000 # Watts/Hz
N0     = sigma2 * B             # Watts

pmax = 30                     # dBm
pmax = 10**(pmax/10.0)/1000   # Watts
pmax = 0.1                    # Watts

K = 100
BS_locate, users_locate, beta_Au, PL_Au = channelConfig(K, r = 100)




#%%
import numpy as np
from scipy.optimize import minimize

# 定义参数
num_users = 3  # 用户数量
max_total_power = 10  # 总功率的最大值
max_individual_power = 5  # 单个用户功率的最大值
noise_power = 1.0  # 噪声功率

# 生成瑞利衰落信道增益（假设为随机变量）
np.random.seed(42)  # 设置随机种子以确保可重复性
channel_gains = np.random.rayleigh(scale = 1.0, size = num_users)  # 瑞利分布

# 定义目标函数：最大化总容量
def objective_function(powers):
    # 计算每个用户的信噪比(SNR), 考虑多用户干扰
    snr = np.zeros(num_users)
    for k in range(num_users):
        interference = np.sum([powers[j] * channel_gains[j]**2 for j in range(num_users) if j != k])
        snr[k] = (powers[k] * channel_gains[k]**2) / (interference + noise_power)
    # 计算每个用户的容量
    capacities = np.log2(1 + snr)
    # 目标是最小化负总容量（相当于最大化总容量）
    return -np.sum(capacities)

# 定义约束条件
def constraint_total_power(powers):
    # 总功率约束：总功率 <= max_total_power
    return max_total_power - np.sum(powers)

def constraint_individual_power(powers):
    # 单个用户功率约束：每个用户的功率 <= max_individual_power
    return max_individual_power - powers

# 初始猜测（均匀分配）
initial_guess = np.ones(num_users) * (max_total_power / num_users)

# 定义约束字典
constraints = (
    {'type': 'ineq', 'fun': constraint_total_power},      # 总功率约束
    {'type': 'ineq', 'fun': constraint_individual_power}  # 单个用户功率约束
)

# 定义边界（每个用户的功率至少为0）
bounds = [(0, max_individual_power) for _ in range(num_users)]

# 求解优化问题
result = minimize(objective_function, initial_guess, method = 'SLSQP', bounds = bounds, constraints = constraints)

# 输出结果
if result.success:
    optimized_powers = result.x
    print("Optimized power allocation:", optimized_powers)
    print("Total capacity:", -result.fun)  # 目标函数是最小化负容量，因此取负值
else:
    print("Optimization failed:", result.message)




#%%
import numpy as np
from scipy.optimize import minimize

# ------------------------- 参数设置 --------------------------
num_users = 3                # 用户数量
P_total = 10                 # 总功率约束 (W)
P_max = 5                    # 单用户最大功率 (W)
noise_power = 1e-9           # 噪声功率 (W)
cell_radius = 500            # 缩小小区半径以增大用户差异
fc = 2e9                     # 载波频率 2GHz
rice_K = 3                   # 降低莱斯K因子，增强散射分量
num_realizations = 50        # 减少蒙特卡洛次数以加速优化

# ------------------------- 信道模型 --------------------------
def generate_channel(num_users, cell_radius, rice_K):
    user_distances = np.random.uniform(100, cell_radius, num_users)  # 增大距离差异
    path_loss = 128.1 + 37.6 * np.log10(user_distances)
    shadowing = np.random.normal(0, 10, num_users)        # 增大阴影衰落标准差
    large_scale = 10**(-(path_loss + shadowing)/10)
    los = np.sqrt(rice_K / (rice_K + 1)) * np.ones(num_users)
    nlos = np.sqrt(1 / (2*(rice_K + 1))) * (np.random.randn(num_users) + 1j*np.random.randn(num_users))
    small_scale = los + nlos
    h = np.sqrt(large_scale) * small_scale
    return np.abs(h)

# ------------------------- 目标函数 --------------------------
def objective_function(powers, num_users, channel_realizations):
    total_capacity = 0
    weights = np.array([1.2, 1.0, 0.8])  # 引入用户权重
    for h in channel_realizations:
        sorted_indices = np.argsort(h)[::-1]
        sorted_powers = powers[sorted_indices]
        sorted_h = h[sorted_indices]
        for k in range(num_users):
            interference = np.sum(sorted_powers[k+1:] * sorted_h[k+1:]**2)
            snr = (sorted_powers[k] * sorted_h[k]**2) / (interference + noise_power)
            total_capacity += weights[k] * np.log2(1 + snr)  # 加权容量
    return -total_capacity / num_realizations

# ------------------------- 优化配置 --------------------------
np.random.seed(42)
channel_realizations = np.array([generate_channel(num_users, cell_radius, rice_K) for _ in range(num_realizations)])

# 随机初始猜测（满足总功率约束）
initial_guess = np.random.rand(num_users) * P_max
initial_guess = initial_guess / np.sum(initial_guess) * P_total

constraints = [
    {'type': 'ineq', 'fun': lambda p: P_total - np.sum(p)},
    {'type': 'ineq', 'fun': lambda p: P_max - p}
]
bounds = [(0, P_max) for _ in range(num_users)]

result = minimize(
    objective_function,
    initial_guess,
    args=(num_users, channel_realizations),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000, 'ftol': 1e-6}
)

# ------------------------- 结果验证 --------------------------
if result.success:
    optimized_powers = np.round(result.x, 2)
    print("优化后的功率分配:", optimized_powers)
    print("总功率:", np.sum(optimized_powers))
    print("单用户功率约束:", np.all(optimized_powers <= P_max))
    print("总容量 (bps/Hz):", -result.fun)
else:
    print("优化失败:", result.message)































































