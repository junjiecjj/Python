#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 12:02:20 2023

@author: jack

这是Markov决策过程中的Batch Manufacture问题的程序：

"""

import numpy as np


c = 1
K = 5
n = 10
p = 0.5
alpha = 0.9

print(f"c  = {c},  K = {K}, n = {n},  p = {p}, alpha = {alpha}\n")
#===================================================================================
#=====    Discounted Problem
#===================================================================================
#========================== (一)值迭代 =========================================

# 方法一: 比较节约内存的方式
def DisCountedProblemValueIteration(c, K, n, p, alpha=1):
    Jk = np.zeros(n+1)   # 保存第k次的代价
    Jkp1 = np.zeros(n+1) # 保存第k+1次的代价
    threshold = 1e-10    # 停止迭代的误差阈值
    pro = 0
    k=1# 迭代次数
    policy = np.zeros(n+1)
    while 1:
        k+=1
        pro = K + alpha*(1-p)*Jk[0] + alpha*p*Jk[1]             #  u(i)  = 1 时的成本
        # print(f"pro = {pro}")
        for i in range(n):   # 状态为0,1...,n-1时的成本
            unpro = c*i + alpha*(1-p)*Jk[i] + alpha*p*Jk[i+1]  #  u(i)  = 0 时的成本
            Jkp1[i] = min(pro, unpro)
            if pro < unpro:
                policy[i] = 1   # 保存策略，生产
            else:
                policy[i] = 0  # 不生产
        Jkp1[n] = pro  # 求状态为n时的成本
        policy[n] = 1  # 当状态为n时策略必须为生产

        # 迭代终止条件，即前后两次的J_k(i)和J_{k+1}(i)向量相差很小
        if   np.all(np.abs(Jkp1 - Jk) <threshold):
            break
        # 将k+1次的值替换上次的指，继续下一次迭代，而不是创建矩阵,这样可以节约内存。
        Jk = Jkp1.copy()


    print(f"[Discounted Problem, 值迭代] alpha为{str(alpha)}时最终的成本为: \n{Jkp1}")
    print(f"[Discounted Problem, 值迭代] alpha为{str(alpha)}时最终的策略为: \n{policy}")
    print(f"[Discounted Problem, 值迭代] 迭代的次数为:{k}\n\n")
    return Jkp1[n], policy, Jkp1

#pron, policy, CovJ = DisCountedProblemValueIteration(c, K, n, p, alpha )


# 寻找最优阈值m
def DiscProbOptimalThreshold(Pro, CovCost):
    for m in  range(1,n):
        m1 = c*(m-1) + alpha*(1-p)*CovCost[m-1] + alpha*p*CovCost[m]
        m2 = c*m + alpha*(1-p)*CovCost[m] + alpha*p*CovCost[m+1]
        if m1 <= Pro and Pro <= m2:
            break
    print(f"Discounted Problem: alpha为{str(alpha)}时的最佳阈值为：{m}\n")
    return m


#OptThreshold = DiscProbOptimalThreshold(pron, CovJ)


# 方法二: 保存迭代过程中所有历史策略和成本的方式，比较浪费空间，为了查看迭代历史，这里采用方法2
def DisCountedProblemValueIteration1(c, K, n, p, alpha=1):
    Jk = np.zeros((1,n+1))   # 保存历史代价，包括最优代价
    Jkp1 = np.zeros( n+1)   # 保存第k+1次的代价
    threshold = 1e-10        # 停止迭代的误差阈值
    policy = np.ones( n+1)
    Policy = np.ones((1,n+1))
    while 1:
        pro = K + alpha*(1-p)*Jk[-1,0] + alpha*p*Jk[-1,1]             #  u(i)  = 1 时的成本
        # print(f"pro = {pro}")
        for i in range(n):   # 状态为0,1...,n-1时的成本
            unpro = c*i + alpha*(1-p)*Jk[-1,i] + alpha*p*Jk[-1,i+1]   #  u(i)  = 0 时的成本
            Jkp1[i] = min(pro, unpro)
            if pro < unpro:
                policy[i] = 1  # 保存策略，生产
            else:
                policy[i] = 0  # 保存策略，不生产
        Jkp1[n] = pro  # 求状态为n时的成本
        policy[n] = 1  # 当状态为n时策略必须为生产
        Jk = np.vstack([Jk,Jkp1])
        Policy = np.vstack([Policy,policy])
        # 迭代终止条件，即前后两次的J_k(i)和J_{k+1}(i)向量相差很小
        if   np.all(np.abs(Jk[-1,:]-Jk[-2,:]) <threshold):
            break

    print(f"[Discounted Problem, 值迭代] alpha为{str(alpha)}时最终的成本为: \n{Jk[-1]}")
    print(f"[Discounted Problem, 值迭代] alpha为{str(alpha)}时最终的策略为: \n{Policy[-1]}")
    print(f"[Discounted Problem, 值迭代] 迭代的次数为：Jk.shape = {Jk.shape}, Policy.shape = {Policy.shape}\n\n")
    return pro, Policy, Jk

Pro, Policy, CovJ = DisCountedProblemValueIteration1(c, K, n, p, alpha)



OptThreshold = DiscProbOptimalThreshold(Pro, CovJ[-1])



#========================== (二)策略迭代  ====================================================
# 根据当前策略u(i)求解成本J(i), 解法一:解方程组解法
def DicprobLineSover(c, K, policy, n, p, alpha):
    """
    c,K,n,p,alpha
    """

    A = np.zeros((n+1,n+1))
    b = np.zeros( n+1)
    for i in range(n):
        if policy[i] == 0:  # 不生产
            A[i,i] = 1-alpha*(1-p)  #  J(i)的系数
            A[i,i+1] = -alpha*p     #  J(i+1)的系数
            b[i] = c*i
        else:
            A[i,0] = -alpha*(1-p)   #  J(0)的系数
            A[i,1] = -alpha*p       #  J(1)的系数
            A[i,i] += 1
            b[i] = K

    A[n,0] =  -alpha*(1-p)        # 当前状态为n时策略必须为生产，J(0)的系数
    A[n,1] = -alpha*p             # 当前状态为n时策略必须为生产，J(1)的系数
    A[n,n] = 1
    b[n]   = K
    # print(f"A = \n{A}\nb = {b}")
    Jk = np.linalg.solve(A, b)

    return Jk


# 根据当前策略u(i)求解成本J(i),解法二:迭代解法
def DisProbCostFunction(c, K, policy, n, p, alpha):
    Jkp1 = np.ones(n+1)
    threshold = 1e-10
    k = 0
    while 1:
        Jk = Jkp1.copy()
        pro = K + alpha*(1-p)*Jk[0] + alpha*p*Jk[1]
        for i in range(n):
            action = policy[i]

            if action:  # process
                Jkp1[i] = pro

            else:  # not process
                Jkp1[i] = c*i + alpha*(1-p)*Jk[i] + alpha*p*Jk[i+1]
        Jkp1[n] = pro
        if np.all(np.abs(Jkp1-Jk) <= threshold):
            break
        k+=1

    return Jkp1


#  根据成本J(i)更新策略u(i), 方法一： 比较节约内存的方式
def DisProbPolicyIteration(c, K, n, p, alpha=1):
    threshold = 1e-10
    Policy = np.ones(n+1)
    NewPolicy = np.ones(n+1)
    k = 1
    Jk = np.zeros(n+1)
    while 1:
        k += 1
        # 更新成本，两种方法
        # Jkp1_ = DisProbCostFunction(c, K, Policy, n, p, alpha)
        Jkp1 = DicprobLineSover(c, K, Policy, n, p, alpha)

        # 更新策略
        pro = K + alpha*((1-p)*Jkp1[0] + p*Jkp1[1])
        # 计算 当前状态i 做出 动作act 后的值
        for i in range(n):
            unpro = c*i + alpha*( (1-p)*Jkp1[i] + p*Jkp1[i+1] )
            if pro<unpro: #  更新策略
                NewPolicy[i] = 1
            else:
                NewPolicy[i] = 0
        NewPolicy[n] = 1
        if np.all(Policy == NewPolicy) or np.all(np.abs(Jkp1 - Jk) <= threshold):
            break

        # 将当前策略替换，参与下次成本更新
        Policy = NewPolicy.copy()
        Jk = Jkp1.copy()


    print(f"[Discounted Problem, 策略迭代] alpha为{str(alpha)}时最终的成本J(i)为:\n{Jkp1}")
    print(f"[Discounted Problem, 策略迭代] alpha为{str(alpha)}时最终的策略u(i)为:\n{Policy}")
    print(f"[Discounted Problem, 策略迭代] 迭代的次数为:{k}\n\n")
    return NewPolicy, Jk

#NewPolicy, Jk = DisProbPolicyIteration(c, K, n, p, alpha)

#  根据成本J(i)更新策略u(i), 方法二： 记录历史迭代的成本和策略，空间需求大
def DisProbPolicyIteration1(c, K, n, p, alpha=1):
    threshold = 1e-10
    Policy = np.ones((1,n+1))
    NewPolicy = np.ones(n+1)

    Jk = np.zeros((1,n+1))
    while 1:

        # 更新成本，两种方法
        #Jkp1_ = DisProbCostFunction(c, K, Policy[-1], n, p, alpha)
        Jkp1 = DicprobLineSover(c, K, Policy[-1], n, p, alpha)
        #print(f"Jkp1 = \n{Jkp1}\n Jkp1_ = \n{Jkp1_}\n Jk = \n{Jk}")

        # 更新策略
        pro = K + alpha*((1-p)*Jkp1[0] + p*Jkp1[1])
        # 计算 当前状态i 做出 动作act 后的值
        for i in range(n):
            unpro = c*i + alpha*( (1-p)*Jkp1[i] + p*Jkp1[i+1] )
            if pro<unpro: #  更新策略
                NewPolicy[i] = 1
            else:
                NewPolicy[i] = 0
        NewPolicy[n] = 1
        # 保存历史迭代的策略和成本
        Policy = np.vstack([Policy,NewPolicy])
        Jk = np.vstack([Jk,Jkp1])
        if np.all(Policy[-1] == Policy[-2]) or np.all(np.abs(Jk[-1] - Jk[-2]) <= threshold):
            break

    print(f"[Discounted Problem, 策略迭代] alpha为{str(alpha)}时最终的成本J(i)为:\n{Jk[-1]}")
    print(f"[Discounted Problem, 策略迭代] alpha为{str(alpha)}时的最终策略u(i)为:\n{Policy[-1]}")
    print(f"[Discounted Problem, 策略迭代] 迭代的次数为:Jk.shape = {Jk.shape}, Policy.shape = {Policy.shape}\n\n")
    return Policy, Jk


PolicyC, Jk = DisProbPolicyIteration1(c, K, n, p, alpha)


#=======================================================================================
#====================== 解线性方程组 ========================



# # m代表系数矩阵。
# m = np.array([[1, -2, 1],
#               [0, 2, -8],
#               [-4, 5, 9]])

# # v代表常数列
# v = np.array([0, 8, -9])

# # 解线性代数。
# r = np.linalg.solve(m, v)

# print("结果：")
# name = ["X1", "X2", "X3"]
# for i in range(len(name)):
#     print(name[i] + "=" + str(r[i]))


# # # 1. 利用gekko的GEKKO求解
# # """利用gekko求解线性方程组"""
# # from gekko import GEKKO

# # m = GEKKO()  # 定义模型
# # x = m.Var()  # 定义模型变量，初值为0
# # y = m.Var()
# # z = m.Var()
# # m.Equations([10 * x - y - 2 * z == 72,
# #              -x + 10 * y - 2 * z == 83,
# #              -x - y + 5 * z == 42, ])  # 方程组
# # m.solve(disp=False)  # 求解
# # x, y, z = x.value, y.value, z.value
# # print(x,y,z)  # 打印结果


# # 2 . 利用scipy的linalg求解
# from scipy import linalg
# import numpy as np

# A = np.array([[10, -1, -2], [-1, 10, -2], [-1, -1, 5]])  # A代表系数矩阵
# b = np.array([72, 83, 42])  # b代表常数列
# x = linalg.solve(A, b)
# print(x)

# # 3. 利用scipy.optimize的root或fsolve求解
# from scipy.optimize import root, fsolve

# def f(X):
#     x = X[0]
#     y = X[1]
#     z = X[2]  # 切分变量

#     return [10 * x - y - 2 * z - 72,
#             -x + 10 * y - 2 * z - 83,
#             -x - y + 5 * z - 42]

# X0 = [1, 2, 3]  # 设定变量初值
# m1 = root(f, X0).x  # 利用root求解并给出结果
# m2 = fsolve(f, X0)  # 利用fsolve求解并给出结果

# print(m1)
# print(m2)


# # 4. 利用Numpy的linalg求解
# import numpy as np

# A = np.array([[10, -1, -2], [-1, 10, -2], [-1, -1, 5]])  # A为系数矩阵
# b = np.array([72, 83, 42])  # b为常数列
# inv_A = np.linalg.inv(A)  # A的逆矩阵
# x = inv_A.dot(b)  # A的逆矩阵与b做点积运算
# x = np.linalg.solve(A, b) # 5,6两行也可以用本行替代
# print(x)

# import numpy as np

# # A = np.mat([[10, -1, -2], [-1, 10, -2], [-1, -1, 5]])  # A为系数矩阵
# # b = np.mat([[72], [83], [42]])  # b为常数列
# A = np.mat("10, -1, -2; -1, 10, -2; -1, -1, 5")  # A为系数矩阵
# b = np.mat("72;83;42")  # b为常数列
# inv_A = np.linalg.inv(A)  # A的逆矩阵
# inv_A = A.I  # A的逆矩阵
# # x = inv_A.dot(b)  # A的逆矩阵与b做点积运算
# x = np.linalg.solve(A, b)
# print(x)

# # 5. 利用sympy的solve和nsolve求解
# # 5.1 利用solve求解所有精确解
# from sympy import symbols, Eq, solve

# x, y, z = symbols('x y z')
# eqs = [Eq(10 * x - y - 2 * z, 72),
#        Eq(-x + 10 * y - 2 * z, 83),
#        Eq(-x - y + 5 * z, 42)]
# print(solve(eqs, [x, y, z]))

# # 5.1 利用nsolve求解数值解
# from sympy import symbols, Eq, nsolve

# x, y, z = symbols('x y z')
# eqs = [Eq(10 * x - y - 2 * z, 72),
#        Eq(-x + 10 * y - 2 * z, 83),
#        Eq(-x - y + 5 * z, 42)]
# initialValue = [1, 2, 3]
# print(nsolve(eqs, [x, y, z], initialValue))

























