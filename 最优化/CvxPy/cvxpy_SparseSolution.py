#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:50:33 2024
@author: jack
"""
# https://www.wuzao.com/document/cvxpy/examples/applications/sparse_solution.html
##====================== 计算线性不等式集合的稀疏解 ==============================
import cvxpy as cp
import numpy as np

# 修正随机数生成器以便我们可以重复实验。
np.random.seed(1)

# 我们认为低于这个阈值的元素为零。
delta = 1e-8

# 问题维度（m个不等式在n维空间中）。
m = 100
n = 50

# 构造一组可行的不等式。
# （这个系统对于x0点是可行的。）
A  = np.random.randn(m, n)
x0 = np.random.randn(n)
b  = A@x0 + np.random.random(m)

# 创建变量。
x_l1 = cp.Variable(shape=n)
# 创建约束条件。
constraints = [A@x_l1 <= b]
# 构造目标函数。
obj = cp.Minimize(cp.norm(x_l1, 1))
# 构造并解决问题。
prob = cp.Problem(obj, constraints)
prob.solve()
print("状态: {}".format(prob.status))

# 解的非零元素个数（其基数或多样性）。
nnz_l1 = (np.absolute(x_l1.value) > delta).sum()
print('在R^{}中找到一个可行的x，它有{}个非零元素。'.format(n, nnz_l1))
print("最优目标函数值: {}".format(obj.value))



##=======================  迭代日志启发式算法  =====================================

# 进行15次迭代，为每次运行分配变量以保存非零个数（x的基数）。
NUM_RUNS = 15
nnzs_log = np.array(())

# 将W存储为正参数，以便简单修改问题。
W = cp.Parameter(shape=n, nonneg=True);
x_log = cp.Variable(shape=n)

# 初始权重。
W.value = np.ones(n)

# 设置问题。
obj = cp.Minimize( W.T@cp.abs(x_log) ) # 逐元素积的和
constraints = [A@x_log <= b]
prob = cp.Problem(obj, constraints)

# 进行问题的迭代，并解决和更新W。
for k in range(1, NUM_RUNS+1):
    # 解决问题。
    # ECOS求解器在此问题上存在已知的数值问题，所以要强制使用其他求解器。
    prob.solve(solver=cp.CVXOPT)

    # 检查错误。
    if prob.status != cp.OPTIMAL:
        raise Exception("求解器未收敛！")

    # 显示解向量中新的非零个数。
    nnz = (np.absolute(x_log.value) > delta).sum()
    nnzs_log = np.append(nnzs_log, nnz);
    print('迭代{}：在R^{}中找到了一个可行的x，其中非零个数为{}...'.format(k, n, nnz))

    # 逐元素调整权重并重新迭代
    W.value = np.ones(n)/(delta*np.ones(n) + np.absolute(x_log.value))














































































































































































































