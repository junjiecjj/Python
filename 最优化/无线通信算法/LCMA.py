#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 11:32:45 2025

@author: jack

https://www.zhihu.com/column/c_1598336162149687297


恒模约束问题求解
    https://zhuanlan.zhihu.com/p/608004371

    https://zhuanlan.zhihu.com/p/601068904

    https://zhuanlan.zhihu.com/p/600350273

基于流形优化的非凸问题求解
    https://zhuanlan.zhihu.com/p/601045018

    https://zhuanlan.zhihu.com/p/620536961

无线通信中的多目标优化问题及trade-off仿真
    https://zhuanlan.zhihu.com/p/608534375
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.datasets import make_spd_matrix


# 生成一个5x5的随机对称正定矩阵
n = 5
A = np.random.rand(n, n)
A = 0.5 * (A + A.T) # 将矩阵对称化
if np.all(np.linalg.eigvals(A) > 0):
    print("A是一个对称正定矩阵。")
else:
    print("A不是一个对称正定矩阵。")


n = 5
A = np.random.rand(n, n)
B =  A @ A.transpose()
# C = B+B.T # makesure symmetric
# test whether C is definite
# D = np.linalg.cholesky(C) # if there is no error, C is definite
if np.all(np.linalg.eigvals(B) > 0):
    print("B是一个对称正定矩阵。")
else:
    print("B不是一个对称正定矩阵。")


# 生成一个5x5的随机对称正定矩阵
n = 5
A = make_spd_matrix(n, random_state=42)
if np.all(np.linalg.eigvals(A) > 0):
    print("A是一个对称正定矩阵。")
else:
    print("A不是一个对称正定矩阵。")







m = 2
n = 3
A = np.random.rand(m, n) + 1j * np.random.rand(m, n)
B = A.conjugate()@A.T
print(f"{B}")

B1 = A@A.T.conjugate()
print(f"{B1}")




































































































































































































