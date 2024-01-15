#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 01:47:47 2024

@author: jack
"""

import cvxpy as cp
import numpy as np


# 创建一个由10个布尔值组成的向量。
x = cp.Variable(10, boolean=True)

# expr1 必须是布尔值。
# constr1 = (expr1 == x)

# 创建一个5行7列的矩阵，其值受限于整数。
Z = cp.Variable((5, 7), integer=True)

# expr2 必须是整数值。
# constr2 = (expr2 == Z)



# 一个复数变量。
x = cp.Variable(complex=True)
# 一个纯虚数参数。
p = cp.Parameter(imag=True)

print("p.is_imag() = ", p.is_imag())
print("(x + 2).is_real() = ", (x + 2).is_real())


#=================================================================
#   mixed-integer quadratic program (MIQP) 混合整数二次规划
# https://www.wuzao.com/document/cvxpy/examples/basic/mixed_integer_quadratic_program.html
#=================================================================


# 生成一个随机问题
np.random.seed(0)
m, n= 40, 25

A = np.random.rand(m, n)
b = np.random.randn(m)




# 构造一个CVXPY问题
x = cp.Variable(n, integer=True)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
prob = cp.Problem(objective)
prob.solve()



print("状态: ", prob.status)
print("最优值为", prob.value)
print("一个解 x 是")
print(x.value)





























































































































































































