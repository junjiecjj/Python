#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 21:03:31 2025

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt


# 全局设置字体大小
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18                     # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18                # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18                # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18               # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18               # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False         # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]            # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300                 # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2                # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6               # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)


def blahut_arimoto_rate_distortion(distortion_matrix, max_distortion, epsilon=1e-6, max_iter=1000, beta=1.0):
    """
    使用Blahut-Arimoto算法计算率失真函数 R(D)

    参数:
    distortion_matrix: 2D numpy数组，失真矩阵 d(x, x̂)
                      行表示信源符号x，列表示重构符号x̂
    max_distortion: 最大允许失真D
    epsilon: 收敛阈值
    max_iter: 最大迭代次数
    beta: 初始的拉格朗日乘子（可选）

    返回:
    rate: 率失真函数值 R(D)
    distortion: 达到的实际失真
    q_opt: 最优的条件分布 q(x̂|x)
    history: 迭代历史
    """

    # 输入验证
    assert isinstance(distortion_matrix, np.ndarray)
    assert distortion_matrix.ndim == 2
    assert distortion_matrix.shape[0] > 0 and distortion_matrix.shape[1] > 0
    assert max_distortion >= 0

    n_source, n_reconstruction = distortion_matrix.shape

    # 初始化信源分布（通常假设为均匀分布）
    p_x = np.ones(n_source) / n_source

    # 初始化重构符号分布（均匀分布）
    q_x_hat = np.ones(n_reconstruction) / n_reconstruction

    # 初始化条件分布 q(x̂|x)（均匀分布）
    q_x_hat_given_x = np.ones((n_source, n_reconstruction)) / n_reconstruction

    history = {'rate': [], 'distortion': [], 'beta': []}

    # 使用二分法寻找合适的beta值，使得失真约束得到满足
    beta_low = 0.0
    beta_high = 1000.0
    beta_tolerance = 1e-6

    for beta_iter in range(50):  # 二分法迭代
        # 固定beta，运行BA算法
        current_rate, current_distortion, q_x_hat_given_x = run_ba_for_beta(
            p_x, distortion_matrix, beta, epsilon, max_iter
        )

        history['rate'].append(current_rate)
        history['distortion'].append(current_distortion)
        history['beta'].append(beta)

        # 检查失真约束
        if np.abs(current_distortion - max_distortion) < beta_tolerance:
            break
        elif current_distortion > max_distortion:
            # 失真太大，需要增加beta（更严格的约束）
            beta_low = beta
            beta = (beta + beta_high) / 2
        else:
            # 失真太小，可以降低beta
            beta_high = beta
            beta = (beta_low + beta) / 2

    return current_rate, current_distortion, q_x_hat_given_x, history

def run_ba_for_beta(p_x, distortion_matrix, beta, epsilon, max_iter):
    """对于给定的beta值运行BA算法"""
    n_source, n_reconstruction = distortion_matrix.shape

    # 初始化条件分布（均匀分布）
    q_x_hat_given_x = np.ones((n_source, n_reconstruction)) / n_reconstruction

    prev_rate = np.inf

    for iteration in range(max_iter):
        # 步骤1: 更新重构符号分布 q(x̂)
        # q(x̂) = Σ_x p(x) q(x̂|x)
        q_x_hat = p_x @ q_x_hat_given_x

        # 避免除零错误
        q_x_hat_safe = np.maximum(q_x_hat, 1e-12)

        # 步骤2: 更新条件分布 q(x̂|x)
        # q(x̂|x) = q(x̂) * exp(-beta * d(x, x̂)) / Z(x)
        # 其中 Z(x) = Σ_x̂ q(x̂) * exp(-beta * d(x, x̂))

        # 计算指数项
        exp_terms = np.exp(-beta * distortion_matrix)

        # 计算归一化常数 Z(x)
        Z_x = np.sum(q_x_hat * exp_terms, axis=1, keepdims=True)
        Z_x_safe = np.maximum(Z_x, 1e-12)

        # 更新条件分布
        q_x_hat_given_x = (q_x_hat * exp_terms) / Z_x_safe

        # 步骤3: 计算当前率和失真
        current_rate = calculate_rate(p_x, q_x_hat_given_x, q_x_hat)
        current_distortion = calculate_distortion(p_x, q_x_hat_given_x, distortion_matrix)

        # 检查收敛
        if np.abs(current_rate - prev_rate) < epsilon:
            break

        prev_rate = current_rate

    return current_rate, current_distortion, q_x_hat_given_x

def calculate_rate(p_x, q_x_hat_given_x, q_x_hat):
    """计算互信息 I(X;X̂) = ΣΣ p(x)q(x̂|x) log(q(x̂|x)/q(x̂))"""
    rate = 0.0
    for i in range(len(p_x)):
        for j in range(len(q_x_hat)):
            if q_x_hat_given_x[i, j] > 1e-12 and q_x_hat[j] > 1e-12:
                rate += p_x[i] * q_x_hat_given_x[i, j] * np.log(
                    q_x_hat_given_x[i, j] / q_x_hat[j]
                )
    return max(rate, 0.0)  # 率不能为负

def calculate_distortion(p_x, q_x_hat_given_x, distortion_matrix):
    """计算平均失真 E[d(X,X̂)] = ΣΣ p(x)q(x̂|x) d(x,x̂)"""
    distortion = 0.0
    for i in range(len(p_x)):
        for j in range(distortion_matrix.shape[1]):
            distortion += p_x[i] * q_x_hat_given_x[i, j] * distortion_matrix[i, j]
    return distortion

# 示例：二进制信源和汉明失真
def example_binary_source_hamming():
    """二进制信源 with 汉明失真示例"""
    print("=== 二进制信源 with 汉明失真 ===")

    # 信源分布（可以是非均匀的）
    p_x = np.array([0.6, 0.4])  # P(X=0)=0.6, P(X=1)=0.4

    # 汉明失真矩阵：d(x,x̂) = 0 if x=x̂, else 1
    distortion_matrix = np.array([
        [0, 1],  # x=0: d(0,0)=0, d(0,1)=1
        [1, 0]   # x=1: d(1,0)=1, d(1,1)=0
    ])

    # 测试不同的失真约束
    distortions_to_test = [0.0, 0.1, 0.2, 0.3, 0.4]

    results = []
    for D in distortions_to_test:
        rate, actual_dist, q_opt, history = blahut_arimoto_rate_distortion(
            distortion_matrix, D, epsilon=1e-8
        )

        results.append((D, rate, actual_dist))

        print(f"D_max={D:.2f}: R(D)={rate:.6f}, 实际失真={actual_dist:.6f}")

        # 显示最优条件分布
        if D == 0.2:  # 只显示一个示例的详细分布
            print("最优条件分布 q(x̂|x):")
            print("q(0|0) = {:.4f}, q(1|0) = {:.4f}".format(q_opt[0, 0], q_opt[0, 1]))
            print("q(0|1) = {:.4f}, q(1|1) = {:.4f}".format(q_opt[1, 0], q_opt[1, 1]))
            print()

    return results

# 示例：高斯信源的平方误差失真（离散化版本）
def example_discretized_gaussian():
    """离散化高斯信源 with 平方误差失真"""
    print("\n=== 离散化高斯信源 with 平方误差失真 ===")

    # 离散化信源和重构空间
    n_points = 10
    source_values = np.linspace(-2, 2, n_points)
    recon_values = np.linspace(-2, 2, n_points)

    # 高斯信源分布 N(0,1)
    p_x = np.exp(-0.5 * source_values**2)
    p_x = p_x / np.sum(p_x)  # 归一化

    # 平方误差失真矩阵
    distortion_matrix = np.zeros((n_points, n_points))
    for i, x in enumerate(source_values):
        for j, x_hat in enumerate(recon_values):
            distortion_matrix[i, j] = (x - x_hat)**2

    # 测试不同的失真约束
    D_values = [0.1, 0.5, 1.0, 2.0]

    for D in D_values:
        rate, actual_dist, q_opt, history = blahut_arimoto_rate_distortion(distortion_matrix, D, epsilon=1e-8 )

        print(f"D_max={D:.2f}: R(D)={rate:.6f}, 实际失真={actual_dist:.6f}")

    return source_values, recon_values, p_x, distortion_matrix

def plot_rate_distortion_curve(results):
    """绘制率失真曲线"""
    try:
        import matplotlib.pyplot as plt

        D_values = [r[0] for r in results]
        R_values = [r[1] for r in results]

        plt.figure(figsize=(10, 6))
        plt.plot(D_values, R_values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('失真 D')
        plt.ylabel('率 R(D) (bits)')
        plt.title('率失真函数曲线')
        plt.grid(True, alpha=0.3)
        plt.show()

    except ImportError:
        print("需要matplotlib库来绘制图形")

if __name__ == "__main__":
    # 运行二进制信源示例
    print("二进制信源率失真函数计算:")
    binary_results = example_binary_source_hamming()

    # 运行高斯信源示例
    print("\n离散化高斯信源率失真函数计算:")
    gaussian_params = example_discretized_gaussian()

    # 绘制率失真曲线
    plot_rate_distortion_curve(binary_results)

    # 验证理论值（对于二进制信源）
    print("\n理论验证:")
    print("对于二进制信源 with 汉明失真，理论率失真函数为:")
    print("R(D) = H(p) - H(D) for D ≤ min(p, 1-p)")
    print("其中 H(p) = -p log p - (1-p) log (1-p)")
