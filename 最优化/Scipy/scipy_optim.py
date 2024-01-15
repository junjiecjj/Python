#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 12:47:59 2024

@author: jack
"""


# https://weihuang.blog.csdn.net/article/details/82834888?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-82834888-blog-122233430.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-82834888-blog-122233430.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=2

##====================================================================================
##             导入sympy包，用于求导，方程组求解等等
##====================================================================================

from scipy import optimize as op
import numpy as np
c = np.array([2, 3, -5])
A_ub = np.array([[-2, 5, -1],[1, 3, 1]])#注意是-2，5，-1
B_ub = np.array([-10, 12])
A_eq = np.array([[1, 1, 1]])
B_eq = np.array([7])
# 上限7是根据约束条件1和4得出的
x1 = (0,7)
x2 = (0,7)
x3 = (0,7)
res = op.linprog(-c, A_ub, B_ub, A_eq, B_eq, bounds=(x1,x2,x3))
print(res)


##====================================================================================
##             scipy包里面的minimize函数求解
##====================================================================================


from scipy.optimize import minimize
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

#目标函数：
def func(args):
    fun = lambda x: 10 - x[0]**2 - x[1]**2
    return fun

#约束条件，包括等式约束和不等式约束
def con(args):
    cons = ({'type': 'ineq', 'fun': lambda x: x[1]-x[0]**2},
            {'type': 'eq', 'fun': lambda x: x[0]+x[1]})
    return cons

#画三维模式图
def draw3D():
    fig = plt.figure()
    ax = Axes3D(fig)
    x_arange = np.arange(-5.0, 5.0)
    y_arange = np.arange(-5.0, 5.0)
    X, Y = np.meshgrid(x_arange, y_arange)
    Z1 = 10 - X**2 - Y**2
    Z2 = Y - X**2
    Z3 = X + Y
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap='rainbow')
    ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap='rainbow')
    ax.plot_surface(X, Y, Z3, rstride=1, cstride=1, cmap='rainbow')
    plt.show()

#画等高线图
def drawContour():
    x_arange = np.linspace(-3.0, 4.0, 256)
    y_arange = np.linspace(-3.0, 4.0, 256)
    X, Y = np.meshgrid(x_arange, y_arange)
    Z1 = 10 - X**2 - Y**2
    Z2 = Y - X**2
    Z3 = X + Y
    plt.xlabel('x')
    plt.ylabel('y')
    plt.contourf(X, Y, Z1, 8, alpha=0.75, cmap='rainbow')
    plt.contourf(X, Y, Z2, 8, alpha=0.75, cmap='rainbow')
    plt.contourf(X, Y, Z3, 8, alpha=0.75, cmap='rainbow')
    C1 = plt.contour(X, Y, Z1, 8, colors='black')
    C2 = plt.contour(X, Y, Z2, 8, colors='blue')
    C3 = plt.contour(X, Y, Z3, 8, colors='red')
    plt.clabel(C1, inline=1, fontsize=10)
    plt.clabel(C2, inline=1, fontsize=10)
    plt.clabel(C3, inline=1, fontsize=10)
    plt.show()


if __name__ == "__main__":
    args = ()
    args1 = ()
    cons = con(args1)
    x0 = np.array((1.0, 2.0))  #设置初始值，初始值的设置很重要，很容易收敛到另外的极值点中，建议多试几个值

    #求解#
    res = minimize(func(args), x0, method='SLSQP', constraints=cons)
    #####
    print(res.fun)
    print(res.success)
    print(res.x)

    # draw3D()
    drawContour()




##====================================================================================
#                        利用拉格朗日乘子法
##====================================================================================

import sympy

#设置变量
x1 = sympy.symbols("x1")
x2 = sympy.symbols("x2")
alpha = sympy.symbols("alpha")
beta = sympy.symbols("beta")

#构造拉格朗日等式
L = 10 - x1*x1 - x2*x2 + alpha * (x1*x1 - x2) + beta * (x1 + x2)

#求导，构造KKT条件
difyL_x1 = sympy.diff(L, x1)  #对变量x1求导
difyL_x2 = sympy.diff(L, x2)  #对变量x2求导
difyL_beta = sympy.diff(L, beta)  #对乘子beta求导
dualCpt = alpha * (x1 * x1 - x2)  #对偶互补条件

#求解KKT等式
aa = sympy.solve([difyL_x1, difyL_x2, difyL_beta, dualCpt], [x1, x2, alpha, beta])

#打印结果，还需验证alpha>=0和不等式约束<=0
for i in aa:
    if i[2] >= 0:
        if (i[0]**2 - i[1]) <= 0:
            print(i)



















