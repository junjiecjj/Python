#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:13:18 2024

@author: jack
"""
# https://www.wuzao.com/document/cvxpy/examples/dgp/power_control.html


import cvxpy as cp
import numpy as np

# 问题数据
n = 5                     # 收发器数量
sigma = 0.5 * np.ones(n)  # 接收器 i 的噪声功率
p_min = 0.1 * np.ones(n)  # 发射器 i 的最小功率
p_max = 5 * np.ones(n)    # 发射器 i 的最大功率
sinr_min = 0.2            # 每个接收器的信干噪比阈值

# 路径增益矩阵
G = np.array([[1.0, 0.1, 0.2, 0.1, 0.05],
            [0.1, 1.0, 0.1, 0.1, 0.05],
            [0.2, 0.1, 1.0, 0.2, 0.2],
            [0.1, 0.1, 0.2, 1.0, 0.1],
            [0.05, 0.05, 0.2, 0.1, 1.0]])
p = cp.Variable(shape=(n,), pos=True)
objective = cp.Minimize(cp.sum(p))

S_p = []
for i in range(n):
    S_p.append(cp.sum(cp.hstack(G[i, k]*p[k] for k in range(n) if i != k)))
S = sigma + cp.hstack(S_p)
signal_power = cp.multiply(cp.diag(G), p)
inverse_sinr = S/signal_power
constraints = [
    p >= p_min,
    p <= p_max,
    inverse_sinr <= (1/sinr_min),
]

problem = cp.Problem(objective, constraints)
print(problem.is_dgp())


problem.solve(gp=True)
print(f"problem.value = {problem.value}")


print(f"p.value = {p.value}")

print(f"inverse_sinr.value = {inverse_sinr.value}")

