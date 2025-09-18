#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 14:28:43 2025

@author: jack

https://www.cnblogs.com/longtianbin/p/17124657.html


"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

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


# 决策变量维度K
K = 3
r = np.array([20.8204, 24.8497, 22.5085])
p = np.array([0.97, 1, 0.99])
p_c = 1
c_1 = np.array([0.0451, 0.0408, 0.0312])
C_2 = np.array([[-0.0010, 0.0493, 0.0202],
                [0.0313, -0.0010, 0.0202],
                [0.0313, 0.0493, -0.0010]])
r_d = np.array([5.7464, 8.2832, 9.4225])
c_3 = 8

delta = 1E-6
IterMax = 20

# 5.2. Charnes-Cooper变换算法实现
def LFP_Charnes_Cooper(K, r, p, p_c, c_1, C_2, r_d, c_3):
    q = cp.Variable(K, nonneg=True)
    z = cp.Variable(nonneg=True)

    objective = cp.Maximize(r.T @ q)

    constraints = [
        q >= z * c_1,
        C_2 @ q >= 0,
        r_d.T @ q >= z * c_3,
        cp.sum(q) <= z,
        p.T @ q + z * p_c == 1
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    isFeasible = True
    if prob.status in ['infeasible', 'unbounded'] or np.isnan(prob.value) or np.isinf(prob.value):
        isFeasible = False

    obj_opt = prob.value
    tau_opt = q.value / z.value if z.value != 0 else np.full(K, np.nan)

    return isFeasible, obj_opt, tau_opt

# 5.3. Dinkelbach变换算法实现
def LFP_Dinkelbach(K, r, p, p_c, c_1, C_2, r_d, c_3, delta, IterMax):
    Q_log = []
    F_log = []
    isFeasible = True
    Q = 0

    for j in range(IterMax):
        tau = cp.Variable(K, nonneg=True)

        objective = cp.Maximize(r.T @ tau - Q * (p.T @ tau + p_c))

        constraints = [
            tau >= c_1,
            C_2 @ tau >= 0,
            r_d.T @ tau >= c_3,
            cp.sum(tau) <= 1
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.status in ['infeasible', 'unbounded'] or np.isnan(prob.value) or np.isinf(prob.value):
            isFeasible = False
            break

        Q_log.append(Q)
        F_log.append(prob.value)

        if prob.value <= delta:
            break
        else:
            Q = (r.T @ tau.value) / (p.T @ tau.value + p_c)

    tau_opt = tau.value
    obj_opt = (r.T @ tau_opt) / (p.T @ tau_opt + p_c) if isFeasible else np.nan

    return isFeasible, obj_opt, tau_opt, np.array(Q_log), np.array(F_log)

# 5.4. Quadratic变换算法实现
def LFP_Quadratic(K, r, p, p_c, c_1, C_2, r_d, c_3, delta, IterMax):
    # 计算初始可行解
    tau = cp.Variable(K, nonneg=True)
    constraints = [
        tau >= c_1,
        C_2 @ tau >= 0,
        r_d.T @ tau >= c_3,
        cp.sum(tau) <= 1
    ]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve()

    if prob.status in ['infeasible', 'unbounded']:
        return False, np.nan, np.full(K, np.nan), np.array([])

    tau_i = tau.value
    F_i = (r.T @ tau_i) / (p.T @ tau_i + p_c)
    F_log = [F_i]
    isFeasible = True

    for j in range(IterMax):
        y = np.sqrt(r.T @ tau_i) / (p.T @ tau_i + p_c)

        tau_var = cp.Variable(K, nonneg=True)
        objective = cp.Maximize(2 * y * cp.sqrt(r.T @ tau_var) - y**2 * (p.T @ tau_var + p_c))

        constraints = [
            tau_var >= c_1,
            C_2 @ tau_var >= 0,
            r_d.T @ tau_var >= c_3,
            cp.sum(tau_var) <= 1
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.status in ['infeasible', 'unbounded'] or np.isnan(prob.value) or np.isinf(prob.value):
            isFeasible = False
            break

        tau_i_1 = tau_var.value
        F_i_1 = (r.T @ tau_i_1) / (p.T @ tau_i_1 + p_c)
        F_log.append(F_i_1)

        if (F_i_1 - F_i) <= delta:
            break
        else:
            tau_i = tau_i_1
            F_i = F_i_1

    obj_opt = F_i_1
    tau_opt = tau_i_1

    return isFeasible, obj_opt, tau_opt, np.array(F_log)

# 主程序
c_3_vec = np.arange(7.6, 9.3, 0.2)
L = len(c_3_vec)
result_CC = np.zeros(L)
result_Dinkelbach = np.zeros(L)
result_Quadratic = np.zeros(L)

for j in range(L):
    isFeasible, obj_opt, tau_opt_CC = LFP_Charnes_Cooper(K, r, p, p_c, c_1, C_2, r_d, c_3_vec[j])
    result_CC[j] = obj_opt

    isFeasible, obj_opt, tau_opt_Dink, Q_log, F_log = LFP_Dinkelbach(K, r, p, p_c, c_1, C_2, r_d, c_3_vec[j], delta, IterMax)
    result_Dinkelbach[j] = obj_opt

    isFeasible, obj_opt, tau_opt_Quad, F_log = LFP_Quadratic(K, r, p, p_c, c_1, C_2, r_d, c_3_vec[j], delta, IterMax)
    result_Quadratic[j] = obj_opt

    print(f'# {j+1}')
    print(f'CC: {tau_opt_CC}')
    print(f'Dink: {tau_opt_Dink}')
    print(f'Quad: {tau_opt_Quad}')

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(c_3_vec, result_CC, '-o', label='Charnes-Cooper', markersize=8, linewidth=2)
plt.plot(c_3_vec, result_Dinkelbach, '-p', label='Dinkelbach', markersize=8, linewidth=2)
plt.plot(c_3_vec, result_Quadratic, '-+', label='Quadratic', markersize=8, linewidth=2)
plt.legend()
plt.ylabel('最优目标值')
plt.xlabel('c_3')
plt.title('不同变换方法求解的最优目标函数值与参数c_3的关系')
plt.grid(True)
plt.show()
