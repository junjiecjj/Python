#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:48:07 2024

@author: jack
"""

#=================================================================
#        无线通信系统中的功率分配
# https://www.wuzao.com/document/cvxpy/examples/applications/maximise_minimum_SINR_BV4.20.html
#=================================================================



import cvxpy as cp
import numpy as np


def maxmin_sinr(G, P_max, P_received, sigma, Group, Group_max, epsilon = 0.001):
    # 根据路径增益矩阵的大小找到 n 和 m
    n, m = np.shape(G)
    # 检查输入的大小
    if m != np.size(P_max):
        print('错误：P_max 的维度与增益矩阵的维度不匹配\n')
        return '错误：P_max 的维度与增益矩阵的维度不匹配\n', np.nan, np.nan, np.nan
    if n != np.size(P_received):
        print('错误：P_received 的维度与增益矩阵的维度不匹配\n')
        return '错误：P_received 的维度与增益矩阵的维度不匹配', np.nan, np.nan, np.nan
    if n != np.size(sigma):
        print('错误：σ 的维度与增益矩阵的维度不匹配\n')
        return '错误：σ 的维度与增益矩阵的维度不匹配', np.nan, np.nan, np.nan

    delta = np.identity(n)
    S = np.multiply(G, delta)  # 信号功率矩阵
    I = G - S  # 干扰功率矩阵

    # 分组矩阵：按照发射机的数量分组
    num_groups = Group.shape[0]

    if num_groups != np.size(Group_max):
        print('错误：Group 矩阵中的组数与 Group_max 的维度不匹配\n')
        return ('错误：Group 矩阵中的组数与 Group_max 的维度不匹配', np.nan, np.nan, np.nan, np.nan)

    # 将组的最大功率归一化为范围 [0,1]
    Group_norm = Group/np.sum(Group,axis=1).reshape((num_groups,1))

    # 创建标量优化变量 p：n 个发射机的功率
    p = cp.Variable(shape=n)
    best = np.zeros(n)

    # 设置子级集的上下界
    u = 1e4
    l = 0

    # alpha 定义广义线性二次问题的子级集，在这种情况下，α 是最小 SINR 的倒数
    alpha = cp.Parameter(shape=1)

    # 设置双分度可行性检验的约束条件
    constraints = [I@p + sigma <= alpha*S@p, p <= P_max, p >= 0, G@p <= P_received, Group_norm@p <= Group_max]

    # 定义目标函数，在这个案例中只希望测试解的可行性
    obj = cp.Minimize(alpha)

    # 现在检查解是否位于 u 和 l 之间
    alpha.value = [u]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status != 'optimal':
        # 在这种情况下，等级集 u 在解的下方
        print('上下界之间无最优解\n')
        return '错误：上下界之间无最优解', np.nan, np.nan, np.nan

    alpha.value = [l]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status == 'optimal':
        # 在这种情况下，等级集 l 在解的下方
        print('上下界之间无最优解\n')
        return '错误：上下界之间无最优解', np.nan, np.nan, np.nan

    # 双分度算法开始
    maxLoop = int(1e7)
    for i in range(1, maxLoop):
        # 首先检查 u 是否在可行域内，l 是否不在可行域内，如果不是这样，循环在此结束
        # 将 α 设置为区间的中点
        alpha.value = np.atleast_1d((u + l)/2.0)

        # 根据指定的容差测试区间的大小
        if u-l <= epsilon:
            break

        # 形成并求解问题
        prob = cp.Problem(obj, constraints)
        prob.solve()

        # 如果问题是可行的，则 u -> α，如果不是，则 l -> α，当达到容差时，新的 α 可能超出界限，最佳值取最后一个可行值作为最优值
        if prob.status == 'optimal':
            u = alpha.value
            best = p.value
        else:
            l = alpha.value

        # 最终条件检查区间是否收敛到顺序 ε，即最优子级集的范围 <=ε
        if u - l > epsilon and i == (maxLoop-1):
            print("解未收敛到顺序 epsilon")

    return l, u, float(alpha.value), best


np.set_printoptions(precision=3)

# 在这种情况下，我们将使用一个信号权重为 0.6 和干扰权重为 0.1 的增益矩阵
G = np.array([[0.6,0.1,0.1,0.1,0.1],
              [0.1,0.6,0.1,0.1,0.1],
              [0.1,0.1,0.6,0.1,0.1],
              [0.1,0.1,0.1,0.6,0.1],
              [0.1,0.1,0.1,0.1,0.6]])

# 在这种情况下，m=n，但是这个问题可以推广到我们想要 n 个接收机和 m 个发射机的情况
n, m = G.shape

# 设置每个发射机和接收机的最大功率饱和级别
P_max = np.array([1.0]*n)

# 归一化的接收功率，总可能性是所有发射机的所有功率，所以是 1/n
P_received = np.array([4.0, 4.0, 4.0, 4.0, 4.0])/n

# 设置噪声水平
sigma = np.array([0.1,0.1,0.1,0.1,0.1])

# 分组矩阵：按照发射机的数量分组
Group = np.array([[1.0, 1.0, 0, 0, 0],[0, 0, 1.0, 1.0, 1.0]])

# 最大归一化功率组，组数乘以 1
Group_max = np.array([1.8, 1.8])

# 现在运行优化问题
l, u, alpha, best = maxmin_sinr(G, P_max, P_received, sigma, Group, Group_max)

print('最小 SINR = {:.4g}'.format(1/alpha))
print('功率 = {}'.format(best))






























































































































































































































































































