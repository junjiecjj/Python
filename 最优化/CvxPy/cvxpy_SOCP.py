#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:33:44 2024
@author: jack
"""

#=================================================================
# 二阶锥规划 second-order cone program (SOCP)
# https://www.wuzao.com/document/cvxpy/examples/basic/socp.html
#=================================================================


# 引入库。
import cvxpy as cp
import numpy as np

# 生成一个随机可行的SOCP。
m = 3
n = 10
p = 5
n_i = 5
np.random.seed(2)
f = np.random.randn(n)
A = []
b = []
c = []
d = []
x0 = np.random.randn(n)

for i in range(m):
    A.append(np.random.randn(n_i, n))
    b.append(np.random.randn(n_i))
    c.append(np.random.randn(n))
    d.append(np.linalg.norm(A[i] @ x0 + b, 2) - c[i].T @ x0)
F = np.random.randn(p, n)
g = F @ x0

# 定义并求解CVXPY问题。
x = cp.Variable(n)
# 我们使用 cp.SOC(t, x) 来创建约束 ||x||_2 <= t 的二阶锥约束。
soc_constraints = [ cp.SOC(c[i].T @ x + d[i], A[i] @ x + b[i]) for i in range(m) ] + [F @ x == g]
prob = cp.Problem(cp.Minimize(f.T@x), soc_constraints )
prob.solve()

# 打印结果。
print("最优值为", prob.value)
print("一个解 x 为")
print(x.value)
for i in range(m):
    print("SOC 约束 %i 对偶变量的解" % i)
    print(soc_constraints[i].dual_value)






























