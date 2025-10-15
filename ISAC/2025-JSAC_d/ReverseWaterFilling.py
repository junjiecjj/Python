#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 15:20:00 2025

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


def reverse_waterfilling_binary(sigma_sq, D, tol=1e-10, max_iter=1000):
    """
    修正后的二分搜索方法
    """
    sigma_sq = np.array(sigma_sq)
    m = len(sigma_sq)

    # 检查边界情况
    total_variance = np.sum(sigma_sq)
    if D >= total_variance:
        # 所有信源都用最大失真
        return sigma_sq.copy(), 0.0, np.max(sigma_sq)
    if D <= 0:
        # 理论上不可行，实践中用极小值
        return np.full(m, 1e-10), 0.5 * np.sum(np.log(sigma_sq / 1e-10)), 0
    # 二分搜索范围：λ应该在 [0, max(σ_i²)] 之间
    low = 0
    high = np.max(sigma_sq)
    # 二分搜索找到最优λ
    for i in range(max_iter):
        lambda_val = (low + high) / 2

        # 计算当前λ对应的失真分配
        D_current = np.minimum(lambda_val, sigma_sq)
        total_distortion = np.sum(D_current)

        if abs(total_distortion - D) < tol:
            break
        if total_distortion < D:
            # 总失真不足，需要增大λ来增加失真
            low = lambda_val
        else:
            # 总失真过大，需要减小λ来减少失真
            high = lambda_val

        if high - low < tol:
            break
    # 最终的最优λ
    optimal_lambda = (low + high) / 2
    # 计算最优失真分配
    D_star = np.minimum(optimal_lambda, sigma_sq)
    # 计算率失真函数值
    R = 0.5 * np.sum(np.log(sigma_sq / D_star))
    return D_star, R, optimal_lambda


def reverse_waterfilling_cvxpy(sigma_sq, D):
    m = len(sigma_sq)
    sigma_sq = np.array(sigma_sq)

    D_i = cp.Variable(m)
    # 常数项
    constant = 0.5 * np.sum(np.log(sigma_sq))

    # 目标函数：最小化 -0.5 * sum(log(D_i))，这等价于最小化原目标函数（差一个常数）
    objective = cp.Minimize(0.5 * cp.sum(cp.log(sigma_sq) - cp.log(D_i)))
    # cp.Minimize(-0.5 * cp.sum(cp.log(D_i)))

    # 约束条件
    constraints = [
        D_i >= 1e-10,  # 避免log(0)，
        D_i <= sigma_sq,
        cp.sum(D_i) <= D
    ]

    # 求解
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError("优化问题未收敛")

    D_star = D_i.value
    # 计算原目标函数值
    R = 0.5 * np.sum(np.log(sigma_sq/D_star))

    # 计算最优lambda - 取所有未达到上限的失真值的最大值
    # 注意：在反注水法中，所有未达到上限的失真值应该等于lambda
    non_saturated = D_star < sigma_sq - 1e-8
    if np.any(non_saturated):
        optimal_lambda = np.max(D_star[non_saturated])
    else:
        # 所有信源都达到了上限，lambda应该大于等于最大方差
        optimal_lambda = np.max(sigma_sq)

    return D_star, R, optimal_lambda


def reverse_waterfilling_sorted(sigma_sq, D):
    """
    最终正确的排序方法实现
    """
    sigma_sq = np.array(sigma_sq)
    m = len(sigma_sq)
    # 检查边界情况
    total_variance = np.sum(sigma_sq)
    if D >= total_variance:
        return sigma_sq.copy(), 0.0, np.max(sigma_sq)
    if D <= 0:
        return np.full(m, 1e-10), 0.5 * np.sum(np.log(sigma_sq / 1e-10)), 0.0
    # 对信源方差排序（从大到小）
    sorted_indices = np.argsort(sigma_sq)
    sorted_sigma_sq = sigma_sq[sorted_indices]
    # 计算累积和
    cumsum = np.cumsum(sorted_sigma_sq)
    # 找到最优的k
    k = 0
    for i in range(m):
        # 计算如果前i个信源使用自身方差，剩下的使用相同λ
        if i == 0:
            sum_used = 0
        else:
            sum_used = cumsum[i-1]
        remaining = D - sum_used
        num_remaining = m - i
        if num_remaining <= 0:
            break
        lambda_candidate = remaining / num_remaining
        # 如果这是最后一个信源，或者λ小于等于下一个信源的方差
        if i == m - 1 or lambda_candidate <= sorted_sigma_sq[i]:
            k = i
            break
    # 计算最优λ
    if k == 0:
        optimal_lambda = D / m
    else:
        optimal_lambda = (D - np.sum(sorted_sigma_sq[:k])) / (m - k)
    # 计算最优失真分配
    D_star_sorted = np.minimum(optimal_lambda, sorted_sigma_sq)
    # 恢复原始顺序
    D_star = np.zeros_like(D_star_sorted)
    D_star[sorted_indices] = D_star_sorted
    # 计算率失真函数值
    R = 0.5 * np.sum(np.log(sigma_sq / D_star))

    return D_star, R, optimal_lambda

def plot_reversewaterfilling(noise_powers, optimal_powers, water_level = 1):
    """可视化注水结果"""
    N = len(noise_powers)

    plt.figure(figsize=(8, 6))

    # 绘制噪声功率和最优功率
    x = np.arange(1, N + 1)
    width = 0.35

    plt.bar(x , noise_powers, width, label='噪声功率', alpha=0.7, color='red')
    plt.bar(x , optimal_powers, width, label='分配功率', alpha=0.2, color='blue',)

    # 绘制水位线
    plt.axhline(y=water_level, color='green', linestyle='--', linewidth=2, label=f'水位线: {water_level:.3f}')

    plt.xlabel('信道索引')
    plt.ylabel('功率')
    plt.title('反注水算法功率分配')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(x)
    plt.tight_layout()
    plt.show()


def comprehensive_test():
    """综合测试三种方法"""
    test_cases = [
        # ([10, 8, 6, 4, 2], 20, "正常情况"),
        # ([10, 8, 6, 4, 2], 30, "大失真情况"),
        # ([10, 8, 6, 4, 2], 5, "小失真情况"),
        # ([5, 5, 5, 5], 10, "均匀方差"),
        ([100, 10, 10, 10], 50, "高方差差异"),
    ]

    print("=" * 60)
    print("反注水算法综合测试")
    print("=" * 60)

    for sigma_sq, D, description in test_cases:
        print(f"\n测试案例: {description}")
        print(f"信源方差: {sigma_sq}")
        print(f"总失真约束: {D}")
        print("-" * 40)

        # 方法1: CVXPY
        try:
            D1, R1, lambda1 = reverse_waterfilling_cvxpy(sigma_sq, D)
            print(f"CVXPY   - R(D): {R1:.6f}, {lambda1}, 总失真: {np.sum(D1):.6f}/{D1}")
        except Exception as e:
            print(f"CVXPY   - 失败: {e}")

        # 方法2: 二分搜索
        D2, R2, lambda2 = reverse_waterfilling_binary(sigma_sq, D)
        print(f"二分搜索 - R(D): {R2:.6f}, {lambda2}, 总失真: {np.sum(D2):.6f}/{D2}")

        # 方法3: 排序方法
        D3, R3, lambda3 = reverse_waterfilling_sorted(sigma_sq, D)
        print(f"排序方法 - R(D): {R3:.6f}, {lambda3}, 总失真: {np.sum(D3):.6f}/{D3}")

        plot_reversewaterfilling(sigma_sq, D1, lambda1)
        plot_reversewaterfilling(sigma_sq, D2, lambda2)
        plot_reversewaterfilling(sigma_sq, D3, lambda3)

        # 验证反注水特性
        print("失真分配验证:")
        print(f"  是否满足 D_i ≤ σ_i²: {np.all(D2 <= np.array(sigma_sq))}")
        print(f"  是否满足 ΣD_i ≈ D: {abs(np.sum(D2) - D) < 1e-6}")

### 运行综合测试
# comprehensive_test()

##  反向注水算法计算通信失真
def reverse_waterfill_D(R, sigma2, tol = 1e-10, max_iter = 1000):
    """
    反向注水算法计算通信失真
    Parameters:
    R: float, 可用编码率 I_c(p_c)
    sigma2: array, 源方差向量 [g_1, g_2, ..., g_N]

    Returns:
    Dc: float, 通信失真
    """
    # N = len(sigma2)
    sigma2 = np.array(sigma2)
        # 检查边界情况
    if R <= 0:
        # R=0时，所有信源用最大失真
        return np.sum(sigma2), np.max(sigma2)
    # 计算R的最大可能值（当D_i接近0时）
    R_max = np.sum(np.log(sigma2 / 1e-15))
    if R >= R_max:
        # R很大时，所有信源失真接近0
        return len(sigma2) * 1e-10, 0
    # 设置搜索范围
    low = 0
    high = np.max(sigma2)

    for _ in range(max_iter):
        xi = (low + high) / 2

        # 计算当前ξ对应的速率
        D_i = np.minimum(xi, sigma2)
        current_R = np.sum(np.log(sigma2 / np.maximum(D_i, 1e-15)))

        if abs(current_R - R) < tol:
            break
        elif current_R > R:
            # 当前速率太大，需要增大ξ来降低速率
            low = xi
        else:
            # 当前速率太小，需要减小ξ来增加速率
            high = xi

    # 计算最终的失真
    D_i = np.minimum(xi, sigma2)
    Dc = np.sum(D_i)

    return Dc, xi


