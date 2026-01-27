#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 21:19:44 2026

@author: jack
"""

import numpy as np

def objective_function(mu, nu, H_hat, d_hat, E_list, gamma, P_tot, epsilon=1e-8):
    """
    计算目标函数值 (31a)

    Parameters:
    -----------
    mu : numpy array (L,)
        变量 μ
    nu : numpy array (M,)
        变量 ν
    H_hat : numpy array (L, K)   # 注意: 这里是 L×K，不是 K×L
        矩阵 H_hat
    d_hat : numpy array (K,)
        向量 d_hat
    E_list : list of numpy arrays
        E_m 矩阵的列表，每个矩阵大小为 (K, K)
    gamma : numpy array (L,)
        向量 γ
    P_tot : float
        总功率
    epsilon : float
        数值稳定性的小常数

    Returns:
    --------
    f_val : float
        目标函数值
    """
    L, K = H_hat.shape  # H_hat 是 L×K 矩阵
    M = len(E_list)     # E_m 的数量

    # 检查维度一致性
    if mu.shape[0] != L:
        raise ValueError(f"mu 维度不匹配: mu.shape={mu.shape}, H_hat.shape={H_hat.shape}")

    if d_hat.shape[0] != K:
        raise ValueError(f"d_hat 维度不匹配: d_hat.shape={d_hat.shape}, K={K}")

    # 计算 H_hat^T μ - d_hat
    H_term = H_hat.T @ mu - d_hat

    # 计算 Σ_{m=1}^M ν_m E_m
    E_sum = np.zeros((K, K))
    for m in range(M):
        if nu[m] < 0:
            nu[m] = 0  # 确保非负
        E_sum += nu[m] * E_list[m]

    # 添加小常数以确保矩阵可逆
    E_sum += epsilon * np.eye(K)

    try:
        # 计算 (Σ ν_m E_m)^{-1} (H_hat^T μ - d_hat)
        # 使用 solve 代替直接求逆以提高数值稳定性
        inv_term = np.linalg.solve(E_sum, H_term)

        # 计算二次型部分: (H_term)^T (E_sum)^{-1} (H_term)
        quadratic = H_term @ inv_term

    except np.linalg.LinAlgError:
        # 如果矩阵奇异，返回一个大值作为惩罚
        return 1e10

    # 计算目标函数
    f_val = 0.25 * quadratic - mu @ gamma + (P_tot / M) * np.sum(nu)

    return f_val

def hooke_jeeves_for_problem31(H_hat, d_hat, E_list, gamma, P_tot,
                               mu_init=None, nu_init=None,
                               d_init=0.5, d_th=1e-6, N_max=1000):
    """
    使用 Hooke-Jeeves 方法求解问题 (31)

    Parameters:
    -----------
    H_hat : numpy array (L, K)   # 注意: 这里是 L×K
        矩阵 H_hat
    d_hat : numpy array (K,)
        向量 d_hat
    E_list : list of numpy arrays
        E_m 矩阵的列表
    gamma : numpy array (L,)
        向量 γ
    P_tot : float
        总功率
    mu_init : numpy array, optional
        μ 的初始值，如果为None则随机初始化
    nu_init : numpy array, optional
        ν 的初始值，如果为None则随机初始化
    d_init : float
        初始步长
    d_th : float
        最小步长阈值
    N_max : int
        最大迭代次数

    Returns:
    --------
    mu_opt : numpy array
        最优 μ
    nu_opt : numpy array
        最优 ν
    history : dict
        优化历史
    """
    # 确定维度
    L, K = H_hat.shape  # H_hat 是 L×K
    M = len(E_list)
    total_vars = L + M

    # 初始化变量
    if mu_init is None:
        mu_init = np.ones(L) * 0.1
    if nu_init is None:
        nu_init = np.ones(M) * 0.1

    # 确保初始值维度正确
    if len(mu_init) != L:
        raise ValueError(f"mu_init 维度不匹配: {len(mu_init)} != {L}")
    if len(nu_init) != M:
        raise ValueError(f"nu_init 维度不匹配: {len(nu_init)} != {M}")

    # 合并变量
    x_base = np.concatenate([mu_init, nu_init])
    x_current = x_base.copy()
    d = d_init
    iter_count = 0

    # 创建搜索方向（单位向量）
    directions = np.eye(total_vars)

    # 目标函数（处理合并变量）
    def f(x):
        mu = x[:L]
        nu = x[L:]
        return objective_function(mu, nu, H_hat, d_hat, E_list, gamma, P_tot)

    # 存储历史
    history = {
        'x': [x_base.copy()],
        'f': [f(x_base)],
        'd': [d],
        'iter': [0]
    }

    def apply_nonnegativity(x):
        """应用非负约束"""
        return np.maximum(x, 0)

    # 确保初始点满足非负约束
    x_base = apply_nonnegativity(x_base)
    x_current = x_base.copy()

    while iter_count < N_max and d >= d_th:
        improved = False

        # 探索移动
        for i in range(total_vars):
            # 正向搜索
            x_test = x_current + d * directions[i]
            x_test = apply_nonnegativity(x_test)

            f_test = f(x_test)
            f_current_val = f(x_current)

            if f_test < f_current_val:
                x_current = x_test.copy()
                improved = True
                break  # 重新开始探索

            # 负向搜索
            x_test = x_current - d * directions[i]
            x_test = apply_nonnegativity(x_test)

            f_test = f(x_test)

            if f_test < f_current_val:
                x_current = x_test.copy()
                improved = True
                break  # 重新开始探索

        # 检查是否改进
        f_current = f(x_current)
        f_base = f(x_base)

        if f_current >= f_base and not improved:
            # 未找到改进 - 减小步长
            if d > d_th:
                d = d / 2
                x_current = x_base.copy()
            else:
                break
        else:
            if f_current < f_base:
                # 模式移动
                if iter_count > 0:
                    pattern_dir = x_current - x_base
                    x_pattern = x_current + d * pattern_dir
                    x_pattern = apply_nonnegativity(x_pattern)

                    if f(x_pattern) < f(x_current):
                        x_current = x_pattern.copy()

                # 更新基点
                x_base = x_current.copy()

        iter_count += 1

        # 存储历史
        history['x'].append(x_base.copy())
        history['f'].append(f(x_base))
        history['d'].append(d)
        history['iter'].append(iter_count)

    # 分离结果
    mu_opt = x_base[:L]
    nu_opt = x_base[L:]

    return mu_opt, nu_opt, history

def generate_test_problem(K=5, L=3, M=4):
    """
    生成测试问题

    Parameters:
    -----------
    K : int
        d_hat 的维度，也是 E_m 矩阵的维度
    L : int
        μ 的维度，也是 H_hat 的行数
    M : int
        ν 的维度（也是 E_m 的数量）

    Returns:
    --------
    问题参数
    """
    # 生成随机但合理的问题参数
    np.random.seed(42)

    # H_hat: L × K 矩阵 (注意: 原公式中 H_hat^T μ，所以 H_hat 应该是 L×K)
    H_hat = np.random.randn(L, K)

    # d_hat: K 维向量
    d_hat = np.random.randn(K)

    # E_m: 生成 M 个 K×K 的正定矩阵
    E_list = []
    for m in range(M):
        # 生成随机正定矩阵
        A = np.random.randn(K, K)
        E = A @ A.T + 0.1 * np.eye(K)  # 确保正定
        E_list.append(E)

    # gamma: L 维向量
    gamma = np.random.randn(L)

    # P_tot: 总功率
    P_tot = 10.0

    return H_hat, d_hat, E_list, gamma, P_tot

# 示例用法
if __name__ == "__main__":
    # 生成测试问题
    K, L, M = 5, 3, 4
    H_hat, d_hat, E_list, gamma, P_tot = generate_test_problem(K, L, M)

    print(f"问题维度: K={K}, L={L}, M={M}")
    print(f"H_hat shape: {H_hat.shape} (应该是 L×K = {L}×{K})")
    print(f"d_hat shape: {d_hat.shape} (应该是 K = {K})")
    print(f"E_list length: {len(E_list)}, each shape: {E_list[0].shape}")
    print(f"gamma shape: {gamma.shape} (应该是 L = {L})")
    print(f"P_tot: {P_tot}")

    # 测试目标函数
    test_mu = np.ones(L) * 0.1
    test_nu = np.ones(M) * 0.1

    try:
        test_val = objective_function(test_mu, test_nu, H_hat, d_hat, E_list, gamma, P_tot)
        print(f"\n测试目标函数值: {test_val}")

        # 求解问题
        mu_opt, nu_opt, history = hooke_jeeves_for_problem31(
            H_hat=H_hat,
            d_hat=d_hat,
            E_list=E_list,
            gamma=gamma,
            P_tot=P_tot,
            d_init=0.5,
            d_th=1e-4,
            N_max=500
        )

        print(f"\n优化结果:")
        print(f"最优 μ: {mu_opt}")
        print(f"最优 ν: {nu_opt}")
        print(f"最优目标值: {history['f'][-1]:.6f}")
        print(f"迭代次数: {history['iter'][-1]}")

        # 检查非负约束
        print(f"\n约束满足情况:")
        print(f"μ ≥ 0 满足: {np.all(mu_opt >= -1e-10)} (最小值: {np.min(mu_opt):.6e})")
        print(f"ν ≥ 0 满足: {np.all(nu_opt >= -1e-10)} (最小值: {np.min(nu_opt):.6e})")

        # 计算梯度检查（近似）
        print(f"\n梯度检查（有限差分）:")
        h = 1e-5
        grad_mu = np.zeros(L)
        for i in range(L):
            mu_plus = mu_opt.copy()
            mu_plus[i] += h
            f_plus = objective_function(mu_plus, nu_opt, H_hat, d_hat, E_list, gamma, P_tot)

            mu_minus = mu_opt.copy()
            mu_minus[i] -= h
            f_minus = objective_function(mu_minus, nu_opt, H_hat, d_hat, E_list, gamma, P_tot)

            grad_mu[i] = (f_plus - f_minus) / (2*h)
        print(f"μ 的梯度近似: {grad_mu}")

        # 可视化收敛历史
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 4))

            # 目标函数值收敛图
            plt.subplot(1, 3, 1)
            plt.plot(history['iter'], history['f'])
            plt.xlabel('Iteration')
            plt.ylabel('Objective Value')
            plt.title('Convergence History')
            plt.grid(True)

            # 步长变化图
            plt.subplot(1, 3, 2)
            plt.plot(history['iter'], history['d'])
            plt.xlabel('Iteration')
            plt.ylabel('Step Size')
            plt.title('Step Size Adaptation')
            plt.grid(True)
            plt.yscale('log')

            # μ和ν的最终值
            plt.subplot(1, 3, 3)
            x_axis = np.arange(L + M)
            plt.bar(x_axis, np.concatenate([mu_opt, nu_opt]))
            plt.axvline(x=L-0.5, color='r', linestyle='--', label='μ/ν boundary')
            plt.xlabel('Variable Index')
            plt.ylabel('Optimal Value')
            plt.title('Optimal μ and ν')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available for plotting")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
