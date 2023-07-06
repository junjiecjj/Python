#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 09:58:09 2023

@author: jack
"""
import numpy as np

c = 1
K = 5
n = 10
p = 0.5



print(f"c  = {c},  K = {K}, n = {n},  p = {p}\n")
#===================================================================================
#=====     Average Cost Problem
#===================================================================================
#========================== (一)值迭代 =========================================

#  值迭代(value iteration),记录所有成本和决策的迭代历史
def AveCostProbValueIteration(c, K, n, p):
    threshold = 1e-10
    h = np.zeros(n+1)
    H = np.zeros((1,n+1))
    Lambda = []
    lambd = 0
    policy = np.zeros(n+1)
    Policy = np.zeros((1,n+1))
    while 1:
        # 利用h(n)=0来求出此次迭代的lamda的值
        lambd = K + (1-p)*H[-1,0] + p*H[-1,1]
        Lambda.append(lambd)
        for i in range(n):
            unpro = c*i + (1-p)*H[-1,i] + p*H[-1,i+1]
            #  h(i)取二者最小值，并且每一次迭代的h(i)都会被储存到一个矩阵中
            h[i] = min(lambd, unpro) - lambd
            if lambd < unpro:
                policy[i] = 1
            else:
                policy[i] = 0
        #  n状态下h的取值是固定的，u(n)必定为1， (n)必定为0;
        h[n] = 0
        policy[n] = 1

        H = np.vstack([H,h])
        Policy = np.vstack([Policy,policy])
        #  如果这次迭代的h与上次迭代中的h近似相等时，停止迭代，跳出while循环
        if np.all(np.abs(H[-1]-H[-2])<= threshold):
            break
    print(f"[Average Cost Problem,值迭代] 最终的H为:\n{H[-1]}")
    print(f"[Average Cost Problem,值迭代] 最终的成本lambda[-1]为:\n{Lambda[-1]}")
    print(f"[Average Cost Problem,值迭代] 最终的策略Policy[-1]为:\n{Policy[-1]} ")
    print(f"[Average Cost Problem,值迭代] 迭代的次数为：len(Lambda) = {len(Lambda)+1}, Policy.shape = {Policy.shape}\n\n")

    return H, Lambda, Policy

H, Lambda, Policy = AveCostProbValueIteration(c, K, n, p)

# 寻找最优阈值
def FindOptimalThresh(lambd, H):
    for m in range(1,n):
        m1 = c*(m-1) + (1-p)*H[m-1] + p*H[m]
        m2 = c*m + (1-p)*H[m] + p*H[m+1]
        if m1 <= lambd and lambd <= m2:
            break
    print(f"[Average Cost Problem, 值迭代] 的最佳阈值为：{m}\n")
    return m


OptThresholdCost = FindOptimalThresh(Lambda[-1], H[-1])


#========================== (二)策略迭代  ====================================================
# 根据当前策略u(i)求解h(i), 解方程组解法
def CostProbPolicyIterLineSolve(c, K, policy, n, p):
    A = np.zeros((n+2,n+2))
    b = np.zeros(n+2)
    for i in range(n):
        if policy[i] == 0:
            A[i,i] = 1 - (1-p)   # h(i)系数
            A[i,i+1] = -p        # h(i+1)系数
            A[i,n+1] = 1
            b[i] = c*i
        else:
            A[i,0] = -(1-p)     # h(0)系数
            A[i,1] =  -p        # h(1)系数
            A[i,i] += 1
            A[i,n+1] = 1
            b[i] = K
    A[n, 0] = -(1-p)
    A[n,1] = -p
    A[n,n] = 1
    A[n,n+1] = 1
    b[n] = K

    A[n+1, n] = 1

    h = np.linalg.solve(A, b)
    return h

#  根据h(i)更新策略u(i), ,记录所有成本和决策的迭代历史
def AveCostPolicyIteration(c, K, n, p):
    threshold = 1e-10
    policy = np.zeros(n+1)
    Policy = np.zeros((1,n+1))
    Policy[-1,-1] = 1
    H = np.zeros((1,n+1))
    Lambda = []
    lambd = 0
    k = 0
    while 1:
        # 根据策略求解h(i)
        h = CostProbPolicyIterLineSolve(c,K,Policy[-1],n,p)
        H = np.vstack([H, h[:-1]])
        Lambda.append(h[-1])

        # 根据求解出的h(i)更新策略
        pro = K + (1-p)*H[-1,0] + p*H[-1,1]
        for i in range(n):
            unpro = c*i + (1-p)*H[-1,i] + p*H[-1,i+1]
            if pro <= unpro:
                policy[i] = 1
            else:
                policy[i] = 0
        policy[n] = 1
        Policy = np.vstack([Policy,policy])
        if np.all(Policy[-1] == Policy[-2]) or np.all(np.abs(H[-1] - H[-2]) <= threshold):
            break
    print(f"[Average Cost Problem,策略迭代] 最终的H[-1]为:\n{H[-1]}")
    print(f"[Average Cost Problem,值迭代] 最终的成本lambda[-1]为:\n{Lambda[-1]}")
    print(f"[Average Cost Problem,策略迭代] 最终的策略Poliy[-1]为:\n{Policy[-1]} ")
    print(f"迭代的次数为: H.shape = {H.shape[0]}, Policy.shape = {Policy.shape}\n\n")
    return H, Lambda, Policy

Hc, LambdaC, PolicyC = AveCostPolicyIteration(c, K, n, p)



# 平稳分析(stationary analysis)的方法来求解 threshold
def  StationaryThresh(c,K,n,p):
    # 利用平稳分布pi来求解threshold
    f = np.zeros(n)
    for m in  range(0,n):
        #print(f"m = {m}")
        P = np.zeros((m+2, m+2))
        # 初始化概率转移矩阵PT
        for i in range(0, m+1):
            #print(f"i = {i}")
            P[i, i+1] = p
            P[i, i] = 1-p
        P[m+1, 0] = 1-p
        P[m+1, 1] = p
        #print(f"P = {P}")
        k = 1
        # 初始化向量v=(1,0,...,0)
        v = np.zeros((1, m+2))
        v[0, 0] = 1
        # 循环乘上概率转移矩阵PT使其收敛
        while 1:
            if  np.all(np.abs(v - v@P) < 1.0000e-6 ):
                break
            v = v @ P
        #print(f"v = {v}")
        for i in range(0, m+1):
            # 计算每个m对应的f(m)
            f[m] = f[m] + c*i*v[0, i]
        f[m] = f[m] + K*v[0,m+1]
    print(f"f =  {f}")
    # 找到使得f(m)最小的threshold
    idx = np.where(f == f.min())[0][0]
    threshold2 = f.min()
    print(f"[Average Cost Problem, 策略迭代, 平稳分析]的最佳阈值为:{idx+1}, 最低成本为:{threshold2}")
    return idx, threshold2


idx, threshold2 =  StationaryThresh(c,K,n,p)

























