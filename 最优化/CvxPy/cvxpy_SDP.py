#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 22:07:35 2024

@author: jack
"""


#=================================================================
# https://www.cvxpy.org/examples/basic/sdp.html
# cvxpy  Semidefinite program, SDP 半定规划问题
#=================================================================
# https://www.wuzao.com/document/cvxpy/tutorial/advanced/index.html
# 许多凸优化问题涉及将矩阵约束为正定或负定（例如，SDP问题）。 在CVXPY中，有两种方法可以实现这一点
# 创建一个正定矩阵的两种方式：
# (1)  # 创建一个100x100的对称且正定 变量。
#      X = cp.Variable((100, 100), PSD = True)


# (2) 下面的代码演示了如何将矩阵表达式约束为正定或负定（但不一定是对称的）。
#      expr1必须是正定的。
#      constr1 = (expr1 >> 0)

# # expr2 必须为半负定的。
#     constr2 = (expr2 << 0)

## 为了将矩阵表达式限制为对称的，只需写如下：
## expr 必须为对称的。
## constr = (expr == expr.T)
# 还可以使用 Variable((n, n), symmetric=True) 创建一个 n 行 n 列的变量，并约束其为对称的。 通过在属性中指定变量为对称的和添加约束 X == X.T 之间的区别在于， 属性会被解析为 DCP 信息，并将对称变量定义在对称矩阵的（较低维度的）向量空间上。



#=================================================================
# cvxpy  Semidefinite program (SDP)， 半定规划问题
# https://www.wuzao.com/document/cvxpy/examples/basic/sdp.html
#=================================================================


# Import packages.
import cvxpy as cp
import numpy as np

# 生成一个随机SDP。
n = 3
p = 3
np.random.seed(1)
C = np.random.randn(n, n)
A = []
b = []
for i in range(p):
    A.append(np.random.randn(n, n))
    b.append(np.random.randn())

# 定义并解决CVXPY问题。
# 创建一个对称矩阵变量。
X = cp.Variable((n,n), symmetric=True)
# 运算符 >> 表示矩阵不等式。
constraints = [X >> 0]
constraints += [ cp.trace(A[i] @ X) == b[i] for i in range(p) ]

prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
prob.solve()

# 打印结果。
print("最优值为", prob.value)
print("一个解 X 为")
print(X.value)

#=================================================================

#=================================================================




































