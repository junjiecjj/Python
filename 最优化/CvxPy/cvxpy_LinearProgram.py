#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:19:05 2024

@author: jack
"""



#=================================================================
#                           对偶变量
#=================================================================



import cvxpy as cp

# 创建两个标量优化变量。
x = cp.Variable()
y = cp.Variable()

# 创建两个约束。
constraints = [x + y == 1,
               x - y >= 1]

# 构建目标函数。
obj = cp.Minimize((x - y)**2)# 形成和解决问题。
prob = cp.Problem(obj, constraints)
prob.solve()

# 约束条件的最优对偶变量（拉格朗日乘子）存储在 constraint.dual_value 中。
print("最优的 (x + y == 1) 对偶变量：", constraints[0].dual_value)
print("最优的 (x - y >= 1) 对偶变量：", constraints[1].dual_value)
print("x - y 的值：", (x - y).value)






#=================================================================
#                  Linear program 线性规划
#=================================================================
# https://www.cvxpy.org/examples/basic/linear_program.html

# Import packages.
import cvxpy as cp
import numpy as np

# Generate a random non-trivial linear program.
m = 15
n = 10
np.random.seed(1)
s0 = np.random.randn(m)
lamb0 = np.maximum(-s0, 0)
s0 = np.maximum(s0, 0)
x0 = np.random.randn(n)
A = np.random.randn(m, n)
b = A @ x0 + s0
c = -A.T @ lamb0

# Define and solve the CVXPY problem.
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(c.T@x),
                 [A @ x <= b])
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(prob.constraints[0].dual_value)

















