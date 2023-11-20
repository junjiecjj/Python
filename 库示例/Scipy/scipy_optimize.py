#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 00:26:03 2023

@author: jack

https://blog.csdn.net/qq_35516360/article/details/122066046

https://blog.csdn.net/nejssd/article/details/104901610

https://zhuanlan.zhihu.com/p/28155370

https://blog.csdn.net/m0_46778675/article/details/119983568
本文将要介绍几种方法去求解各种复杂的方程组，包括实数域和复数域的线性、非线性方程组，并对比这几种方法的优缺点。本文用到了numpy、scipy、sympy这三个科学计算包。


"""


##==========================================================================
##                    scipy fsolve 非线性方程组求解
##==========================================================================


# https://blog.csdn.net/qq_35516360/article/details/122066046
from math import sin, cos
from scipy import optimize


def f(x):
    x0, x1, x2 = x.tolist()
    return [
        5 * x1 + 3,
        4 * x0 ** 2 - 2 * sin(x1 * x2),
        x1 * x2 - 1.5
        ]


result = optimize.fsolve(f, [1, 1, 1])
print(result)
print(f(result))


# [-0.70622057 -0.6        -2.5       ]
# [0.0, -9.126033262418787e-14, 5.329070518200751e-15]



from math import sin, cos
from scipy import optimize


def f(x):
    x0, x1, x2 = x.tolist()
    return [
        5 * x1 + 3,
        4 * x0 ** 2 - 2 * sin(x1 * x2),
        x1 * x2 - 1.5
    ]


def j(x):
    x0, x1, x2 = x.tolist()
    return [
        [0, 5, 0],
        [8 * x0, -2 * x2 * cos(x1 * x2), -2 * x1 * cos(x1 * x2)],
        [0, x2, x1]
    ]


result = optimize.fsolve(f, [1, 1, 1], fprime=j)
print(result)
print(f(result))
# [-0.70622057 -0.6        -2.5       ]
# [0.0, -9.126033262418787e-14, 5.329070518200751e-15]


### https://zhuanlan.zhihu.com/p/28155370
import numpy as np
from scipy.optimize import fsolve


def f(x):
#转换为标准的浮点数列表
    x0, x1, x2 = x.tolist()
    return[5*x1+3,
           4*x0*x0 - 2*np.sin(x1*x2),
           x1*x2-1.5]

#f是计算的方程组误差函数，[1,1,1]是未知数的初始值
result = fsolve(f, [1,1,1])
#输出方程组的解
print(result)
#输出误差
print(f(result))


# https://blog.csdn.net/nejssd/article/details/104901610

# 二、非线性方程组。
# scipy和sympy不但可以解线性方程(组)，还可以求解非线性方程(组)，但是也有各自的优缺点。

# 1.scipy求解
# scipy.optimize里面有两个函数可以数值求解方程组，分别是root和solve，这两个函数会找到方程组的一个近似解。下面通过例子介绍这两个函数的使用。

from scipy.optimize import root

def f1(x):
   return [x[0]+x[0]*x[1]-2,x[0]-x[1]-2]

print(root(f1,[0,-1]).x)#初始猜测值[0,-1]
print(root(f1,[0,0]).x)#初始猜测值[0,0]

# 此外，在上面的基础上我们可以设置参数jac来提高运算速度（尤其是计算量很大时效果很明显）。
from numpy import array,mat,sin,cos,exp
from scipy.optimize import root

def f(x):
    eqs=[]
    eqs.append(x[0]*x[1]+x[1]*x[2]+sin(x[0])*exp(x[1])+x[1])
    eqs.append(x[0]*x[1]-exp(x[0])*x[1]+1)
    eqs.append(x[1]*x[2]+exp(x[1])*x[2]+1)
    return eqs

def jac1(x):#方程组对应的雅可比矩阵
    return mat([[x[1]+cos(x[0])*exp(x[1]), x[0]+x[2]+sin(x[0])*exp(x[1])+1, x[1]],
                [x[1]-exp(x[0])*x[1], x[0]-exp(x[0]), 0],
                [0 ,x[2]+exp(x[1])*x[2], x[1]+exp(x[1])]])

print(root(f,[0,0,0]).x)
print(root(f,[0,0,0],jac=jac1).x)#加上参数jac加快运算速度


# 再来看一个复数域的例子
from scipy.optimize import root

def f1(x):
    return [x[0]*(1j)+x[0]*x[1]+1,x[0]+x[1]-1j]

print(root(f1,[1,1],method='krylov').x)
print(root(f1,[1,1],method='krylov',tol=1e-10).x)#设置能够允许的误差为10的-10次方



# (2).fslove
# fsolve的用法和root很类似，但是它的功能不如root全面，fsolve其实就是用hybr算法求解, 因此它不能解复数域的方程组。下图可以看出fsolve和root的差别：
from scipy.optimize import fsolve
from numpy import array,mat

def f1(x):
  return [x[0]+x[0]*x[1]-2,x[0]-x[1]-2]

def jac1(x):#方程组对应的雅可比矩阵
  return mat([[1+x[1],x[0]],[1,-1]])

print(fsolve(f1,[0,-1]))#初始猜测值[0,-1]
print(fsolve(f1,[0,-1],fprime=jac1))#初始猜测值[0,-1],并设置参数prime
print(fsolve(f1,[0,0]))#初始猜测值[0,0]
print(fsolve(f1,[0,0],fprime=jac1))#初始猜测值[0,0],并设置参数prime



# 2.sympy求解
# sympy中的solve函数可以严格求解某些方程组，nsolve可以数值求近似解。
# (1).solve
# 直接看一个复数域的例子：

from sympy import symbols,Eq,solve
x0,x1=symbols('x0 x1')
eqs=[Eq(x0*(1j)+x0*x1,-5),Eq(x0+x1,1j)]
print(solve(eqs,[x0,x1]))

# (2).nsolve
# nsolve函数需要指定一个初始猜测解。
from sympy import symbols,Eq,nsolve
x0,x1=symbols('x0 x1')
eqs=[Eq(x0*(1j)+x0*x1,-5),Eq(x0+x1,1j)]
print(nsolve(eqs,[x0,x1],[1,1]))#初始猜测解设为[1，1]



# https://blog.csdn.net/m0_46778675/article/details/119983568
# 2. 利用scipy.optimize的fsolve求解
# optimize库中的fsolve函数可以用来对非线性方程组进行求解。
from scipy.optimize import fsolve

def f(X):
    x = X[0]
    y = X[1]
    return [x ** 2 / 4 + y ** 2 - 1,
            (x - 0.2) ** 2 - y - 3]

X0 = [0, 0]
result = fsolve(f, X0)
print(result)

# 3. 利用scipy.optimize的root求解
from scipy.optimize import fsolve, root

def f(X):
    x = X[0]
    y = X[1]
    return [x ** 2 / 4 + y ** 2 - 1,
            (x - 0.2) ** 2 - y - 3]

X0 = [10, 10]
result1 = fsolve(f, X0)
result2 = root(f, X0)

print(result2)


# 4. 利用scipy.optimize的leastsq求解
from scipy.optimize import leastsq

def f(X):
    x = X[0]
    y = X[1]
    return [x ** 2 / 4 + y ** 2 - 1,
            (x - 0.2) ** 2 - y - 3]

X0 = [10, 10]
h = leastsq(f, X0)
print(h)


# 5. 利用sympy的solve和nsolve求解
# 5.1 利用solve求解所有精确解
from sympy import symbols, Eq, solve, nsolve

x, y = symbols('x y')
eqs = [Eq(x ** 2 / 4 + y ** 2, 1),
       Eq((x - 0.2) ** 2 - y, 3)]

print(solve(eqs, [x, y]))
# 可以看出，用这种方法能够得到方程组的全部解，并且为精确解，缺点是求解时间较长。


# 5.2 利用nsolve求解数值解

from sympy import symbols, Eq, nsolve
x, y = symbols('x y')
eqs = [Eq(x ** 2 / 4 + y ** 2, 1),
       Eq((x - 0.2) ** 2 - y, 3)]

X0 = [3, 4]
print(nsolve(eqs, [x, y], X0))
# nsolve为数值求解，需要指定一个初始值，初始值会影响最终得到哪一个解（如果有多解的话），而且初始值设的不好，则可能找不到解。
# scipy.optimize.root求解速度快，但只能得到靠近初始值的一个解。对形式简单、有求根公式的方程，sympy.solve能够得到所有严格解，但当方程组变量较多时，它求起来会很慢。而且对于不存在求根公式的复杂方程，sympy.solve无法求解。



# https://blog.csdn.net/nejssd/article/details/104901610
##==========================================================================
##                  线性方程组可以用numpy去求解。
##==========================================================================


import numpy as np
a = np.mat('1,2,3;2,4,8;9,6,3')
b = np.mat('1;1;3')
c = np.linalg.solve(a,b)
print(c)
# [[-0.5]
#  [ 1.5]
#  [-0.5]]



import numpy as np
a = np.mat('1,-1j; 1j,-1')
b = np.mat('1; 1')
c = np.linalg.solve(a,b)
print(c)
# [[ 0.5-0.5j]
#  [-0.5+0.5j]]

























































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































