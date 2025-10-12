#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 20:31:23 2025

@author: jack
"""

import cvxpy as cp
import numpy as np

import matplotlib.pyplot as plt
# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)

def water_filling(sigma2, lamba,  PT):
    """
    实现注水功率分配算法
    参数:
        lamba: Σc的特征值
        sigma2: 噪声功率
        PT: 总功率预算
    返回:
        power_allocation: 功率分配向量
        water_level: 注水水平
    """
    N = len(lamba)

    # 按特征值降序排列
    idx_sorted = np.argsort(lamba)[::-1]
    Lambda_sorted = lamba[idx_sorted]

    # 计算每个特征值对应的噪声归一化项
    noise_terms = sigma2 / Lambda_sorted

    # 初始化
    power_allocation = np.zeros(N)
    water_level = 0

    # 注水算法
    for k in range(1, N+1):
        # 计算当前可能的注水水平
        water_level_candidate = (PT + np.sum(noise_terms[:k])) / k

        # 检查是否所有分配的功率都为正
        if k == N or water_level_candidate <= noise_terms[k]:
            water_level = water_level_candidate
            break

    # 计算功率分配
    for i in range(N):
        if i < k:
            power_allocation[idx_sorted[i]] = max(0, water_level - noise_terms[i])
        else:
            power_allocation[idx_sorted[i]] = 0

    return power_allocation, water_level

def waterfilling_manual(sigma2, lamba, PT, tolerance=1e-10, max_iter=2000):
    """
    手动实现注水算法
    参数:
        sigma2: 各信道的噪声功率
        lamba: 特征值
        PT: 总可用功率
        tolerance: 收敛容差
        max_iter: 最大迭代次数
    返回:
        optimal_powers: 最优功率分配
        water_level: 水位线
    """

    # N = len(lamba)
    # 按特征值降序排列
    idx_sorted = np.argsort(lamba.copy())[::-1]
    Lambda_sorted = lamba.copy()[idx_sorted]
    sorted_noise = sigma2 / Lambda_sorted

    # 使用二分法寻找最优水位线
    low = 0
    high = PT + np.max(sorted_noise)

    for iter_count in range(max_iter):
        water_level = (low + high) / 2

        # 计算当前水位线下的功率分配
        powers = np.maximum(0, water_level - sorted_noise)
        total_used_power = np.sum(powers)

        # 检查功率约束
        if abs(total_used_power - PT) < tolerance:
            break
        elif total_used_power < PT:
            low = water_level
        else:
            high = water_level

    # 恢复原始顺序的功率分配
    optimal_powers = np.maximum(0, water_level - sigma2/lamba)

    return optimal_powers, water_level

def waterfilling_cvxpy(sigma2, lamba, total_power):
    """
    使用CVXPY实现注水算法

    参数:
    noise_powers: 各信道的噪声功率
    total_power: 总可用功率

    返回:
    optimal_powers: 最优功率分配
    problem_status: 问题状态
    optimal_value: 最优值
    """

    N = len(lamba)
    tmp = sigma2 / lamba
    # 定义优化变量
    p = cp.Variable(N, nonneg=True)

    # 定义目标函数：最大化总容量
    objective = cp.Maximize(cp.sum(cp.log(1 + p / tmp)))

    # 定义约束条件
    constraints = [
        p >= 0,  # 功率非负
        cp.sum(p) <= total_power  # 总功率约束
    ]

    # 构建优化问题
    problem = cp.Problem(objective, constraints)

    # 求解问题
    optimal_value = problem.solve()

    # 获取结果
    optimal_powers = p.value
    problem_status = problem.status

    # 计算水位线
    # 水位线 = 分配的功率 + 噪声功率 (对于激活的信道)
    active_channels = optimal_powers > 1e-6  # 避免数值误差
    if np.any(active_channels):
        # 水位线是激活信道的 (功率 + 噪声功率) 的平均值
        water_level = np.mean(optimal_powers[active_channels] + tmp[active_channels])
    else:
        # 如果没有激活的信道，水位线设为最小噪声功率
        water_level = np.min(tmp)

    return optimal_powers, water_level


def plot_waterfilling(noise_powers, optimal_powers, water_level = 1):
    """可视化注水结果"""
    N = len(noise_powers)

    plt.figure(figsize=(8, 6))

    # 绘制噪声功率和最优功率
    x = np.arange(1, N + 1)
    width = 0.35

    plt.bar(x  , noise_powers, width, label='噪声功率', alpha=0.7, color='red')
    plt.bar(x , optimal_powers, width, label='分配功率', alpha=0.7, color='blue', bottom = noise_powers)

    # 绘制水位线
    plt.axhline(y=water_level, color='green', linestyle='--', linewidth=2, label=f'水位线: {water_level:.3f}')

    plt.xlabel('信道索引')
    plt.ylabel('功率')
    plt.title('注水算法功率分配')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(x)
    plt.tight_layout()
    plt.show()

#### 生成测试数据

# num_channels = 8
# lamba = np.random.uniform(0.1, 2.0, num_channels)
# sigma2 = 1
# total_power = 4

# print("手动实现注水算法:")
# print(f"信道数量: {num_channels}")
# print(f"总功率: {total_power}")
# # print(f"噪声功率: {noise_powers}")

# ##>>>>>>>> 执行注水算法
# optimal_powers1, water_level1 = waterfilling_manual(sigma2, lamba, total_power)

# print(f"水位线: {water_level1:.4f}")
# print(f"最优功率分配: {optimal_powers1}")
# print(f"实际使用功率: {np.sum(optimal_powers1):.4f}")
# print(f"总容量: {np.sum(np.log(1 + optimal_powers1 * lamba/sigma2)):.4f}")

# # 可视化
# plot_waterfilling(sigma2/lamba, optimal_powers1, water_level1)

# ##>>>>>>>> 执行注水算法
# optimal_powers2, water_level2 = water_filling(sigma2, lamba, total_power)

# print(f"水位线: {water_level2:.4f}")
# print(f"最优功率分配: {optimal_powers2}")
# print(f"实际使用功率: {np.sum(optimal_powers2):.4f}")
# print(f"总容量: {np.sum(np.log(1 + optimal_powers2 * lamba/sigma2)):.4f}")

# # 可视化
# plot_waterfilling(sigma2/lamba, optimal_powers2, water_level2)


# ##>>>>>>>>  执行注水算法
# optimal_powers3, water_level3 = waterfilling_cvxpy(sigma2, lamba, total_power)

# print(f"水位线: {water_level3:.4f}")
# print(f"最优功率分配: {optimal_powers3}")
# print(f"实际使用功率: {np.sum(optimal_powers3):.4f}")
# print(f"总容量: {np.sum(np.log(1 + optimal_powers3 * lamba/sigma2)):.4f}")

# # 可视化
# plot_waterfilling(sigma2/lamba, optimal_powers3, water_level3)




