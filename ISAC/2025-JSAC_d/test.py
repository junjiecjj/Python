#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 09:12:58 2025

@author: jack
"""

# 测试修正后的方案
import numpy as np
import cvxpy as cp


def solve_mmse_waveform_design_trace_reformulation(sigma_s_sq, Sigma_s, P_T):
    """
    使用迹的线性代数变换求解
    """
    n = Sigma_s.shape[0]

    # 定义变量
    R_s = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((n, n), symmetric=True)  # 辅助变量

    # 约束条件
    Sigma_s_inv = np.linalg.inv(Sigma_s)
    M = (1/sigma_s_sq) * R_s + Sigma_s_inv

    constraints = [
        R_s >> 0,
        Y >> 0,
        cp.trace(R_s) <= P_T,
        cp.bmat([[M, np.eye(n)], [np.eye(n), Y]]) >> 0  # Schur补条件
    ]

    # 目标函数：最小化 trace(Y)
    objective = cp.trace(Y)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=cp.MOSEK)  # MOSEK处理SDP更好

    if problem.status in ['optimal', 'optimal_inaccurate']:
        return R_s.value, problem.value
    else:
        print(f"求解失败，状态: {problem.status}")
        return None, None
def solve_mmse_waveform_design_eigen(sigma_s_sq, Sigma_s, P_T):
    """
    使用特征值分解方法（当Σ_s有特殊结构时）
    """
    # 对先验协方差矩阵进行特征分解
    eigvals, U = np.linalg.eigh(Sigma_s)
    n = len(eigvals)

    # 在新的基上定义变量
    R_tilde = cp.Variable((n, n), diagonal=True)  # 假设在对角基上最优

    # 约束条件
    constraints = [
        R_tilde >> 0,
        cp.trace(R_tilde) <= P_T
    ]

    # 目标函数 - 在对角基上简化
    objective = 0
    for i in range(n):
        term = (1/sigma_s_sq) * R_tilde[i, i] + 1/eigvals[i]
        objective += cp.inv_pos(term)  # 使用inv_pos处理标量情况:cite[2]

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve()

    if problem.status in ['optimal', 'optimal_inaccurate']:
        # 转换回原始基
        R_s_opt = U @ R_tilde.value @ U.T
        return R_s_opt, problem.value
    else:
        print(f"求解失败，状态: {problem.status}")
        return None, None
def test_corrected_solution():
    # 参数设置
    np.random.seed(42)
    n = 3
    sigma_s_sq = 2.0
    P_T = 5.0

    # 生成正定协方差矩阵
    A = np.random.randn(n, n)
    Sigma_s = A.T @ A + n * np.eye(n)

    print("=== 测试修正后的MMSE波形设计求解 ===")

    # 测试方案一
    # print("\n1. 使用矩阵分式方法:")
    # R_opt1, obj_val1 = solve_mmse_waveform_design_matrix_frac(sigma_s_sq, Sigma_s, P_T)
    # if R_opt1 is not None:
    #     print(f"目标函数值: {obj_val1:.6f}")
    #     print(f"功率使用: {np.trace(R_opt1):.4f} / {P_T}")

    # 测试方案二（如果有MOSEK）
    try:
        print("\n2. 使用迹重构方法:")
        R_opt2, obj_val2 = solve_mmse_waveform_design_trace_reformulation(sigma_s_sq, Sigma_s, P_T)
        if R_opt2 is not None:
            print(f"目标函数值: {obj_val2:.6f}")
            print(f"功率使用: {np.trace(R_opt2):.4f} / {P_T}")
    except Exception as e:
        print(f"方案二求解失败: {e}")

    # # 验证结果
    # if R_opt1 is not None:
    #     print("\n=== 结果验证 ===")
    #     M_opt = (1/sigma_s_sq) * R_opt1 + np.linalg.inv(Sigma_s)
    #     verified_obj = np.trace(np.linalg.inv(M_opt))
    #     print(f"数值验证目标值: {verified_obj:.6f}")
    #     print(f"半正定性: {np.all(np.linalg.eigvals(R_opt1) >= -1e-6)}")

if __name__ == "__main__":
    test_corrected_solution()
