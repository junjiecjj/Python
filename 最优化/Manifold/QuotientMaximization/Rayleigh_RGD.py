#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 22:39:34 2026

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
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

def riemannian_gd_rayleigh(A, opts=None):
    """
    黎曼梯度下降求解瑞利商最小化问题

    求解：
        min x^H A x
        s.t. ||x||_2 = 1

    输入：
        A    : Hermitian 矩阵
        opts : 参数字典

    输出：
        x          : 估计得到的最小特征值对应的特征向量
        lambda_est : 估计得到的最小瑞利商
        info       : 迭代信息字典
    """

    if opts is None:
        opts = {}

    # 默认参数
    eta0 = opts.get("eta0", 1.0)
    rho = opts.get("rho", 0.5)
    c = opts.get("c", 1e-4)
    epsilon = opts.get("epsilon", 1e-10)
    Kmax = opts.get("Kmax", 1000)
    max_backtrack = opts.get("max_backtrack", 100)
    eta_min = opts.get("eta_min", 1e-16)

    # 保证 A 是 Hermitian 矩阵
    A = (A + A.conj().T) / 2

    # 初始化
    n = A.shape[0]
    if "x0" in opts:
        x = opts["x0"].astype(complex)
    else:
        x = np.random.randn(n) + 1j * np.random.randn(n)

    # 投影到单位球面
    x = x / np.linalg.norm(x)

    # 记录迭代历史
    obj_hist = np.zeros(Kmax)
    grad_hist = np.zeros(Kmax)
    eta_hist = np.zeros(Kmax)
    bt_hist = np.zeros(Kmax)

    exitflag = 2
    message = "Stopped: maximum number of iterations reached."

    for k in range(Kmax):

        # 当前瑞利商，由于 ||x|| = 1，所以分母为 1
        f = np.real(np.vdot(x, A @ x))

        # 黎曼梯度，省略常数因子 2
        g = A @ x - f * x
        g_norm = np.linalg.norm(g)

        # 保存迭代历史
        obj_hist[k] = f
        grad_hist[k] = g_norm

        # 收敛判断
        if g_norm <= epsilon:
            exitflag = 1
            message = "Converged: gradient norm is below tolerance."
            break

        # Armijo 回溯线搜索
        eta = eta0
        success = False
        bt_used = 0

        for bt in range(1, max_backtrack + 1):

            # 沿负黎曼梯度方向更新
            x_new = x - eta * g

            # 回缩到单位球面
            x_new = x_new / np.linalg.norm(x_new)

            # 候选点目标函数
            f_new = np.real(np.vdot(x_new, A @ x_new))

            # Armijo 充分下降条件
            if f_new <= f - c * eta * g_norm**2:
                success = True
                bt_used = bt
                break

            # 缩小步长
            eta = rho * eta

            # 数值保护
            if eta <= eta_min:
                bt_used = bt
                break

        # 若线搜索失败，通常表示已经接近数值精度极限
        if not success:
            exitflag = 0
            message = "Stopped: Armijo line search failed due to numerical precision."
            break

        # 接受更新
        x = x_new
        eta_hist[k] = eta
        bt_hist[k] = bt_used

    # 实际迭代次数
    iter_num = k + 1

    # 截断历史记录
    obj_hist = obj_hist[:iter_num]
    grad_hist = grad_hist[:iter_num]
    eta_hist = eta_hist[:iter_num]
    bt_hist = bt_hist[:iter_num]

    # 输出最小瑞利商
    lambda_est = np.real(np.vdot(x, A @ x))

    info = {
        "obj_hist": obj_hist,
        "grad_hist": grad_hist,
        "eta_hist": eta_hist,
        "bt_hist": bt_hist,
        "iter": iter_num,
        "exitflag": exitflag,
        "message": message,
    }

    return x, lambda_est, info


if __name__ == "__main__":

    np.random.seed(42)

    # 构造一个 3×3 Hermitian 矩阵 A
    A = np.array([
        [3, 1 + 1j, 0.5],
        [1 - 1j, 2, -0.3j],
        [0.5, 0.3j, 1]
    ], dtype=complex)

    A = (A + A.conj().T) / 2

    # 问题维度
    n = A.shape[0]

    # 设置算法参数
    opts = {}
    opts["eta0"] = 1.0
    opts["rho"] = 0.5
    opts["c"] = 1e-4
    opts["epsilon"] = 1e-10
    opts["Kmax"] = 1000
    opts["max_backtrack"] = 100
    opts["eta_min"] = 1e-16

    # 设置初始点
    opts["x0"] = np.random.randn(n) + 1j * np.random.randn(n)

    # 调用黎曼梯度下降算法
    x, lambda_est, info = riemannian_gd_rayleigh(A, opts)

    # 使用 eig 验证结果
    lambda_all, V = np.linalg.eigh(A)

    # np.linalg.eigh 会自动按照特征值从小到大排序
    lambda_true = lambda_all[0]
    x_true = V[:, 0]

    # 复特征向量存在全局相位不唯一，因此需要相位对齐
    phase_factor = np.vdot(x_true, x)
    x_aligned = x * np.exp(-1j * np.angle(phase_factor))

    # 验证误差
    eigval_err = abs(lambda_est - lambda_true)
    eigvec_err = np.linalg.norm(x_aligned - x_true)
    corr_val = abs(np.vdot(x_true, x))

    # 输出结果
    print("\nRiemannian GD result:")
    print(f"Status                              = {info['message']}")
    print(f"Estimated minimum Rayleigh quotient = {lambda_est:.12f}")
    print(f"True minimum eigenvalue             = {lambda_true:.12f}")
    print(f"Absolute eigenvalue error           = {eigval_err:.4e}")
    print(f"Number of iterations                = {info['iter']}")
    print(f"Final gradient norm                 = {info['grad_hist'][-1]:.4e}")
    print(f"Phase-aligned eigenvector error     = {eigvec_err:.4e}")
    print(f"Absolute inner product              = {corr_val:.12f}")

    print("\nEstimated eigenvector x:")
    print(x)

    print("\nTrue eigenvector from eig:")
    print(x_true)

    print("\nEstimated eigenvector after phase alignment:")
    print(x_aligned)

    # 绘制瑞利商下降曲线
    plt.figure()
    plt.plot(np.arange(1, info["iter"] + 1), info["obj_hist"], linewidth=1.5)
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Rayleigh quotient")
    plt.show()

    # 绘制黎曼梯度范数收敛曲线
    plt.figure()
    plt.semilogy(np.arange(1, info["iter"] + 1), info["grad_hist"], linewidth=1.5)
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Riemannian gradient norm")
    plt.show()

    # 绘制 Armijo 步长变化
    plt.figure()
    plt.semilogy(np.arange(1, info["iter"] + 1), info["eta_hist"], linewidth=1.5)
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Step size")
    plt.show()

    # 绘制每次迭代的回溯次数
    plt.figure()
    plt.stem(np.arange(1, info["iter"] + 1), info["bt_hist"])
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("Backtracking number")
    plt.show()
