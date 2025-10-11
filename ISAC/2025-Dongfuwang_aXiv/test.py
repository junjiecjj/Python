#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 16:18:47 2025

@author: jack
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# 参数设置
Nt = 2
Mc = 2
Ms = 2
Iter = 100
max_iterations = 100
I_mc = np.eye(Mc)
I_nt = np.eye(Nt)
Pt = 1
K = 100
SNRcom = [-5, 0, 5]
SNRsen = np.arange(0, 22, 2)
Sigma_s = np.diag([0.4, 0.1])

# 预先计算常数矩阵
Sigma_s_inv = np.linalg.inv(Sigma_s)

def sensing_distortion(R_x, sigma_s):
    """计算感知失真 D_s(R_x)"""
    inner_matrix = (1/sigma_s**2) * R_x + Sigma_s_inv
    return Ms * np.trace(np.linalg.inv(inner_matrix))

def R_s_matrix(R_x, sigma_s2):
    """计算 R_s 矩阵"""
    inner_matrix = (1/sigma_s2) * R_x + Sigma_s_inv
    return Sigma_s - np.linalg.inv(inner_matrix)

def channel_capacity(R_x, H_c, sigma_c2):
    """计算信道容量 C(R_x)"""
    inner_matrix = (1/sigma_c2) * H_c @ R_x @ H_c.conj().T + I_mc
    return np.log(np.linalg.det(inner_matrix))

def solve_P3(R_0, H_c, sigma_s2, sigma_c2, max_iterations=100, tolerance=1e-6):
    """求解问题P3 - 使用SCA技术"""
    R_x_prev = R_0
    obj_values = []

    for iteration in range(max_iterations):
        # 计算常数矩阵 P
        inner_matrix_P = (1/sigma_s2) * R_x_prev + Sigma_s_inv
        P = np.linalg.inv(inner_matrix_P)

        # 定义优化变量
        R_x = cp.Variable((Nt, Nt), symmetric=True)
        D = cp.Variable((Nt, Nt), symmetric=True)

        # 计算线性化近似 ~R_s
        R_s_tilde = Sigma_s - P + (1/sigma_s2) * P @ (R_x - R_x_prev) @ P

        # 计算 f(R_x) 的线性化近似
        Sigma_minus_P = Sigma_s - P
        Sigma_minus_P_inv = np.linalg.inv(Sigma_minus_P)

        f_Rx = np.log(np.linalg.det(Sigma_minus_P)) + (1/sigma_s2) * cp.trace(Sigma_minus_P_inv @ P @ (R_x - R_x_prev) @ P)
        # 目标函数 - 使用矩阵分式方法避免直接求逆
        # 对于 Tr((A*R_x + B)^{-1})，我们使用辅助变量和矩阵不等式
        A = (1/sigma_s2) * I_nt
        B = Sigma_s_inv

        # 方法1：使用Schur补引理
        # 引入辅助变量 Z，约束 [A*R_x + B, I; I, Z] >= 0
        # 然后最小化 trace(Z)
        Z = cp.Variable((Nt, Nt), symmetric=True)
        schur_constraint = cp.bmat([
            [A @ R_x + B, I_nt],
            [I_nt, Z]
        ]) >> 0

        objective = Ms * cp.trace(Z) + cp.trace(D)

        # 约束条件
        inner_matrix_capacity = (1/sigma_c2) * H_c @ R_x @ H_c.conj().T + I_mc
        capacity_constraint = cp.log_det(inner_matrix_capacity) - Ms * (f_Rx - cp.log_det(D)) >= 0

        constraints = [
            schur_constraint,
            capacity_constraint,
            R_s_tilde - D >> 0,
            R_x >> 0,
            cp.trace(R_x) <= Pt
        ]

        # 求解问题
        problem = cp.Problem(cp.Minimize(objective), constraints)
        try:
            problem.solve(verbose=False,)
        except Exception as e:
            print(f"Solver error at iteration {iteration}: {e}")
            break

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"Iteration {iteration}: Problem status - {problem.status}")
            if iteration == 0:
                return R_x_prev, [sensing_distortion(R_x_prev, sigma_s2)]
            break

        R_x_opt = R_x.value

        if R_x_opt is None:
            print(f"Iteration {iteration}: No solution found")
            break

        obj_value = sensing_distortion(R_x_opt, sigma_s2) + np.trace(D.value)
        obj_values.append(obj_value)

        print(f"Iteration {iteration}: Objective = {obj_value:.6f}")

        if iteration > 0:
            relative_change = abs(obj_value - obj_values[-2]) / (abs(obj_values[-2]) + 1e-8)
            if relative_change < tolerance:
                print(f"Converged after {iteration+1} iterations")
                break

        R_x_prev = R_x_opt

    return R_x_opt, obj_values

def solve_P2(H_c, sigma_s, sigma_c, initial_R_0=None, max_iterations=100, tolerance=1e-6):
    """求解问题P2 - 通过迭代求解P3"""

    if initial_R_0 is None:
        initial_R_0 = (Pt / Nt) * I_nt

    print("Solving P2 using SCA technique...")
    print(f"Initial R_0 trace: {np.trace(initial_R_0):.6f}")
    R_x_opt, obj_values = solve_P3(initial_R_0, H_c, sigma_s, sigma_c, max_iterations, tolerance)

    return R_x_opt, obj_values

def evaluate_solution(R_x_opt, H_c, sigma_s, sigma_c):
    """评估解的各个指标"""
    D_s = sensing_distortion(R_x_opt, sigma_s)
    C = channel_capacity(R_x_opt, H_c, sigma_c)
    R_s = R_s_matrix(R_x_opt, sigma_s)

    print("\n=== Solution Evaluation ===")
    print(f"Sensing Distortion D_s: {D_s:.6f}")
    print(f"Channel Capacity C: {C:.6f}")
    print(f"Power Constraint: {np.trace(R_x_opt):.6f} <= {Pt}")
    print(f"R_x positive definite: {np.all(np.linalg.eigvals(R_x_opt) > 0)}")

    return D_s, C, R_s

np.random.seed(42)

# 生成随机信道矩阵 H_c
H_c = np.random.randn(Mc, Nt) + 1j * np.random.randn(Mc, Nt)

# 设置感知和通信噪声参数
sigma_s2 = 0.5
sigma_c2 = 0.5

print("System Parameters:")
print(f"Nt = {Nt}, Mc = {Mc}, Ms = {Ms}")
print(f"sigma_s2 = {sigma_s2}, sigma_c2 = {sigma_c2}, Pt = {Pt}")

# 求解问题P2
R_x_opt, obj_values = solve_P2(H_c, sigma_s2, sigma_c2, max_iterations=20, tolerance=1e-4)

if R_x_opt is not None:
    # 评估解
    D_s, C, R_s = evaluate_solution(R_x_opt, H_c, sigma_s2, sigma_c2)

    # 绘制收敛曲线
    if len(obj_values) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(obj_values, 'b-o', linewidth=2, markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('SCA Convergence')
        plt.grid(True, alpha=0.3)
        plt.show()
else:
    print("Failed to find solution")












