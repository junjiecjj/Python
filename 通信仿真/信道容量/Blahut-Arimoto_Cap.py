#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 20:31:15 2025

@author: jack
"""

import numpy as np

def blahut_arimoto_capacity(p_y_x, epsilon=1e-6, max_iter=1000):
    """
    使用Blahut-Arimoto算法计算离散信道的信道容量
    参数:
        p_y_x: 2D numpy数组，信道转移概率矩阵 p(y|x) 行表示输入x，列表示输出y
        epsilon: 收敛阈值
        max_iter: 最大迭代次数
    返回:
        capacity: 信道容量
        p_opt: 达到容量的最优输入分布
        history: 迭代历史(可选，用于调试)
    """
    # 输入验证
    assert isinstance(p_y_x, np.ndarray)
    assert p_y_x.ndim == 2
    assert np.all(p_y_x >= 0)
    assert np.allclose(p_y_x.sum(axis=1), 1.0), "每行的概率和必须为1"

    n_inputs, n_outputs = p_y_x.shape

    # 初始化：均匀输入分布
    p_x = np.ones(n_inputs) / n_inputs

    # 初始化历史记录
    history = {'p_x': [], 'capacity': []}
    prev_capacity = -np.inf

    for iteration in range(max_iter):
        # 步骤1: 计算输出分布 p(y) = Σ p(x)p(y|x)
        p_y = p_x @ p_y_x

        # 避免除零错误，给很小的概率值加上微小正数
        p_y_safe = np.maximum(p_y, 1e-12)

        ## 步骤2: 计算后验概率 q(x|y) = p(x)p(y|x) / p(y),使用广播机制进行高效计算
        q_x_given_y = (p_x[:, np.newaxis] * p_y_x) / p_y_safe

        # 步骤3: 计算指数项中的KL散度,对于每个输入x，计算 D_KL(p(y|x) || p(y))
        kl_terms = np.zeros(n_inputs)
        for x in range(n_inputs):
            # 计算 p(y|x) 和 p(y) 之间的KL散度
            p_y_given_x = p_y_x[x, :]
            mask = p_y_given_x > 0  # 只考虑非零概率
            kl_terms[x] = np.sum(p_y_given_x[mask] * np.log(p_y_given_x[mask] / p_y_safe[mask]))
        # 步骤4: 更新输入分布
        p_x_new = p_x * np.exp(kl_terms)
        p_x_new = p_x_new / np.sum(p_x_new)  # 归一化
        # 步骤5: 计算当前互信息（信道容量估计值）
        capacity = np.sum(kl_terms * p_x_new)

        # 记录历史
        history['p_x'].append(p_x.copy())
        history['capacity'].append(capacity)

        # 检查收敛
        if np.abs(capacity - prev_capacity) < epsilon:
            print(f"在 {iteration+1} 次迭代后收敛")
            break
        # 更新变量
        p_x = p_x_new
        prev_capacity = capacity
    else:
        print(f"达到最大迭代次数 {max_iter}，可能尚未完全收敛")

    return capacity, p_x, history

# 示例：计算二进制对称信道(BSC)的容量
def example_bsc_capacity():
    """二进制对称信道(BSC)示例"""
    print("=== 二进制对称信道(BSC)示例 ===")

    # 定义错误概率
    error_prob = 0.1

    # BSC信道矩阵
    # 行：输入0, 1
    # 列：输出0, 1
    bsc_matrix = np.array([
        [1 - error_prob, error_prob],      # 输入0
        [error_prob, 1 - error_prob]       # 输入1
    ])

    # 计算容量
    capacity, p_opt, history = blahut_arimoto_capacity(bsc_matrix)

    # 理论值：C = 1 - H(p)
    theoretical_capacity = 1 + error_prob * np.log2(error_prob) + (1 - error_prob) * np.log2(1 - error_prob)

    print(f"错误概率: {error_prob}")
    print(f"计算得到的容量: {capacity:.6f} (以nat为单位)")
    print(f"计算得到的容量: {capacity/np.log(2):.6f} (以bit为单位)")
    print(f"理论容量: {theoretical_capacity:.6f} (以bit为单位)")
    print(f"最优输入分布: p(0)={p_opt[0]:.6f}, p(1)={p_opt[1]:.6f}")

    return capacity, p_opt

# 示例：计算二进制擦除信道(BEC)的容量
def example_bec_capacity():
    """二进制擦除信道(BEC)示例"""
    print("\n=== 二进制擦除信道(BEC)示例 ===")

    # 定义擦除概率
    erase_prob = 0.3

    # BEC信道矩阵
    # 输入：0, 1
    # 输出：0, 1, E(擦除)
    bec_matrix = np.array([
        [1 - erase_prob, 0, erase_prob],      # 输入0
        [0, 1 - erase_prob, erase_prob]       # 输入1
    ])

    capacity, p_opt, history = blahut_arimoto_capacity(bec_matrix)

    # 理论值：C = 1 - p
    theoretical_capacity = 1 - erase_prob

    print(f"擦除概率: {erase_prob}")
    print(f"计算得到的容量: {capacity:.6f} (以nat为单位)")
    print(f"计算得到的容量: {capacity/np.log(2):.6f} (以bit为单位)")
    print(f"理论容量: {theoretical_capacity:.6f} (以bit为单位)")
    print(f"最优输入分布: p(0)={p_opt[0]:.6f}, p(1)={p_opt[1]:.6f}")

    return capacity, p_opt

# 可视化收敛过程
def plot_convergence(history, title):
    """绘制收敛曲线"""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(history['capacity'], 'b-', linewidth=2)
        plt.xlabel('迭代次数')
        plt.ylabel('信道容量估计值 (nat)')
        plt.title(f'{title} - 收敛过程')
        plt.grid(True, alpha=0.3)
        plt.show()

    except ImportError:
        print("需要matplotlib库来绘制图形")


#### In[76]:
e = 0.2
p1 = [1-e, e]
p2 = [e, 1-e]
p_y_x = np.asarray([p1, p2])
C,  p_x, history = blahut_arimoto_capacity(p_y_x)
print('Capacity: ', C/np.log(2))
print('The prior: ', p_x)

# The analytic solution of the capaciy
H_P_e = 1 + e * np.log2(e) + (1-e) * np.log2(1-e)
print('Anatliyic capacity: ', (H_P_e))

####  运行示例
cap_bsc, p_bsc = example_bsc_capacity()
cap_bec, p_bec = example_bec_capacity()

# 如果需要绘制收敛曲线，取消下面的注释
# _, _, history_bsc = blahut_arimoto_capacity(np.array([[0.9, 0.1], [0.1, 0.9]]))
# plot_convergence(history_bsc, "BSC信道")















