#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 00:08:15 2023

@author: jack
"""

##=============================================================================
##                  拉格朗日插值法及python实现
##=============================================================================
##  方法一  导包的方法
from scipy.interpolate import lagrange

x = [1, 3, 5]
y = [2, 10, 1]
print(lagrange(x, y))
print(lagrange(x, y)(10))






import numpy
import matplotlib
import matplotlib.pyplot as plt

#  手撸代码
x = [1, 3, 5]
y = [2, 10, 1]


def lagrange_interpolate(x1):
    P = []
    L_n = 0
    # for循环计算每一个基函数  并将每个基函数存入list列表中
    for i in range(len(x)):
        fenzi = 1
        fenmu = 1
        for j in range(len(x)):
            if j != i:
                fenzi *= (x1 - x[j])
                fenmu *= (x[i] - x[j])
        P.append(fenzi / fenmu)

    #L_n存入每个点的结果
    for i in range(len(x)):
        L_n += y[i] * P[i]
    return round(L_n, 3)


x2 = numpy.linspace(-10, 10, 20, endpoint=False)
y2 = []
for i in range(len(x2)):
    y2.append(lagrange_interpolate(x2[i]))

plt.plot(x2, y2)
plt.scatter(x, y, marker='.')
plt.show()



import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


# 构造x,y 数组，x为自变量、y为因变量
x = np.arange(10)
y = np.sin(x)
# 使用 CubiSpline 构建分段三次样条插值函数cs，cs即为scipy.interpolate.PPoly类的对象
# 默认情况下，三次样条端点边界条件为‘not-a-knot’ （默认值）,曲线末端的第一段和第二段是相同的多项式。当没有关于边界条件的信息时，这是一个很好的默认值。
cs = CubicSpline(x, y)


# 构造插值点
xs = np.arange(-0.5, 9.6, 0.1)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(x, y, 'o', label='data')
ax.plot(xs, np.sin(xs), label='true')
ax.plot(xs, cs(xs), label="S-插值点")
# 一阶导数
ax.plot(xs, cs(xs, 1), label="S'-插值处的一阶导数")

ax.set_xlim(-0.5, 9.5)
ax.legend(loc='lower left', ncol=2)
plt.show()





























