#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 22:36:47 2021

@author: jack
"""
import math
import numpy as np

def softmax1D_U(x):
    # 
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def softmax1D(x):
    # 
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

def softmax(x, axis=1):
    """
    softmax函数实现
    
    参数：
    x --- 一个二维矩阵, m * n,其中m表示向量个数，n表示向量维度
    
    返回：
    softmax计算结果
    """
    # 计算每行的最大值
    row_max = x.max(axis=axis)
 
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max = row_max.reshape(-1, 1)
    x = x - row_max
 
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

def SoftMax(X,axis=1):
    """
    softmax函数实现
    
    参数：
    x --- 一个二维矩阵, m * n,其中m表示向量个数，n表示向量维度
    
    返回：
    softmax计算结果
    """
    assert(len(X.shape) == 2)
    row_max = np.max(X, axis=axis).reshape(-1, 1)
    X -= row_max
    X_exp = np.exp(X)
    s = X_exp / np.sum(X_exp, axis=axis, keepdims=True)

    return s


z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
 
z_exp = [math.exp(i) for i in z]  
 
print(z_exp)  # Result: [2.72, 7.39, 20.09, 54.6, 2.72, 7.39, 20.09] 
 
sum_z_exp = sum(z_exp)  
print(sum_z_exp)  # Result: 114.98 
# Result: [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]
 
softmax1 = [round(i / sum_z_exp, 3) for i in z_exp]
print(softmax1)  



scores = np.array([123, 456, 789])
print("scores1 = \n",scores)
scores -= np.max(scores)
print("scores2 = \n",scores)
p = np.exp(scores) / np.sum(np.exp(scores))
print(p) # [5.75274406e-290 2.39848787e-145 1.00000000e+000]



 
 
A = [[1, 1, 5, 3],
     [0.2, 0.2, 0.5, 0.1]]
A= np.array(A)
axis = 1  # 默认计算最后一维
 
# [1]使用自定义softmax
s1 = softmax(A, axis=axis)


s2 = SoftMax(A, axis=axis)

a = [[1,2,3],[-1,-2,-3]]
b = [[1,2,3]]
c = [1,2,3]
a = np.array(a)
b = np.array(b)
c = np.array(c)

print("SoftMax(a) = \n", SoftMax(a))
print("SoftMax(b) = \n", SoftMax(b))
print("SoftMax(c) = \n", SoftMax(c)) # error