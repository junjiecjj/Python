#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 12:47:59 2024

@author: jack
"""


# https://weihuang.blog.csdn.net/article/details/82834888?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-82834888-blog-122233430.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-82834888-blog-122233430.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=2


#%% 利用Python求解带约束的最优化问题
##====================================================================================
##             scipy包里面的minimize函数求解
##====================================================================================
# min: 10 - x1**2 - x2**2,
# s.t.: x2 >= x1**2
#       x1 + x2 = 0

from scipy.optimize import minimize
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

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

#目标函数：
def func(x):
    fun = lambda x: 10 - x[0]**2 - x[1]**2
    return fun

#约束条件，包括等式约束和不等式约束
def con(x):
    cons = ({'type': 'ineq', 'fun': lambda x: x[1]-x[0]**2},
            {'type': 'eq', 'fun': lambda x: x[0]+x[1]})
    return cons

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

    draw3D()
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
difyL_x1 = sympy.diff(L, x1)  # 对变量x1求导
difyL_x2 = sympy.diff(L, x2)  # 对变量x2求导
difyL_beta = sympy.diff(L, beta)  # 对乘子beta求导
dualCpt = alpha * (x1 * x1 - x2)  # 对偶互补条件

#求解KKT等式
aa = sympy.solve([difyL_x1, difyL_x2, difyL_beta, dualCpt], [x1, x2, alpha, beta])

#打印结果，还需验证alpha>=0和不等式约束<=0
for i in aa:
    if i[2] >= 0:
        if (i[0]**2 - i[1]) <= 0:
            print(i)

"""
https://blog.csdn.net/qq_35516360/article/details/122066046
https://blog.csdn.net/nejssd/article/details/104901610
https://zhuanlan.zhihu.com/p/28155370
https://blog.csdn.net/m0_46778675/article/details/119983568
本文将要介绍几种方法去求解各种复杂的方程组，包括实数域和复数域的线性、非线性方程组，并对比这几种方法的优缺点。本文用到了numpy、scipy、sympy这三个科学计算包。
"""
#%% 求解线性方程组
##====================================================================================
##             导入sympy包，用于求导，方程组求解等等
##====================================================================================

import scipy
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
res = scipy.optimize.linprog(-c, A_ub, B_ub, A_eq, B_eq, bounds=(x1,x2,x3))
print(res)


# https://blog.csdn.net/nejssd/article/details/104901610
##==========================================================================
##                  numpy 求解线性方程组
##==========================================================================

import numpy as np
a = np.mat('1,2,3; 2,4,8; 9,6,3')
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

##==========================================================================
##                  scipy 求解线性方程组
##==========================================================================

import scipy
c = [-1, 4]
A = [[-3, 1], [1, 2]]
b = [6, 4]
x0_bounds = (None, None)
x1_bounds = (-3, None)
res = scipy.optimize.linprog(c, A_ub = A, b_ub = b, bounds = [x0_bounds, x1_bounds])
res.fun
# -22.0
res.x
# array([10., -3.])
res.message
'Optimization terminated successfully. (HiGHS Status 7: Optimal)'

##==========================================================================
##                  scipy lstsq
##==========================================================================

import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt


x = np.array([1, 2.5, 3.5, 4, 5, 7, 8.5])
y = np.array([0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6])

M = x[:, np.newaxis]**[0, 2]

p, res, rnk, s = lstsq(M, y)

plt.plot(x, y, 'o', label='data')
xx = np.linspace(0, 9, 101)
yy = p[0] + p[1]*xx**2
plt.plot(xx, yy, label = f'{p}, least squares fit, $y = a + bx^2$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(framealpha=1, shadow=True)
plt.grid(alpha=0.25)
plt.show()

#%% 非线性方程组求解
##==========================================================================
##                    scipy fsolve 非线性方程组求解
##==========================================================================

##>>>>>>>>>>>>>>>>>>>
# https://blog.csdn.net/qq_35516360/article/details/122066046
from math import sin, cos

def f(x):
    x0, x1, x2 = x.tolist()
    return [
        5 * x1 + 3,
        4 * x0 ** 2 - 2 * sin(x1 * x2),
        x1 * x2 - 1.5
        ]

result = scipy.optimize.fsolve(f, [1, 1, 1])
print(result)
print(f(result))
# [-0.70622057 -0.6        -2.5       ]
# [0.0, -9.126033262418787e-14, 5.329070518200751e-15]

##>>>>>>>>>>>>>>>>>>>

def func(x):
    return [x[0] * np.cos(x[1]) - 4,
            x[1] * x[0] - x[1] - 5]
x = scipy.optimize.fsolve(func, [1, 1])
print(x)
# array([6.50409711, 0.90841421])
print(np.isclose(func(x), [0.0, 0.0]))  # func(root) should be almost 0.0.
# array([ True,  True])


##>>>>>>>>>>>>>>>>>>>

def f(x):
    x0, x1, x2 = x.tolist()
    return [ 5 * x1 + 3, 4 * x0 ** 2 - 2 * np.sin(x1 * x2), x1 * x2 - 1.5 ]

def j(x):
    x0, x1, x2 = x.tolist()
    return [ [0, 5, 0], [8 * x0, -2 * x2 * np.cos(x1 * x2), -2 * x1 * np.cos(x1 * x2)], [0, x2, x1] ]

result = scipy.optimize.fsolve(f, [1, 1, 1], fprime = j)
print(result)
print(f(result))
# [-0.70622057 -0.6        -2.5       ]
# [0.0, -9.126033262418787e-14, 5.329070518200751e-15]

##>>>>>>>>>>>>>>>>>>>
### https://zhuanlan.zhihu.com/p/28155370
import numpy as np

def f(x):
#转换为标准的浮点数列表
    x0, x1, x2 = x.tolist()
    return[5*x1+3, 4*x0*x0 - 2*np.sin(x1*x2), x1*x2-1.5]

# f是计算的方程组误差函数，[1, 1, 1]是未知数的初始值
result = scipy.optimize.fsolve(f, [1, 1, 1])
# 输出方程组的解
print(result)
# 输出误差
print(f(result))

# fsolve 的用法和 root 很类似，但是它的功能不如 root 全面，fsolve 其实就是用 hybr 算法求解, 因此它不能解复数域的方程组。下图可以看出fsolve和root的差别：
from scipy.optimize import fsolve
from numpy import array, mat

def f1(x):
  return [x[0] + x[0] * x[1] - 2, x[0] - x[1] - 2]

def jac1(x): # 方程组对应的雅可比矩阵
  return mat([ [1 + x[1], x[0]], [1, -1]])

print(scipy.optimize.fsolve(f1,[0,-1]))                     # 初始猜测值[0,-1]
print(scipy.optimize.fsolve(f1,[0,-1], fprime = jac1))      # 初始猜测值[0,-1],并设置参数prime
print(scipy.optimize.fsolve(f1,[0,0]))                      # 初始猜测值[0,0]
print(scipy.optimize.fsolve(f1,[0,0], fprime = jac1))       # 初始猜测值[0,0],并设置参数prime


# optimize库中的fsolve函数可以用来对非线性方程组进行求解。
# from scipy.optimize import fsolve

def f(X):
    x = X[0]
    y = X[1]
    return [x ** 2 / 4 + y ** 2 - 1,
            (x - 0.2) ** 2 - y - 3]

X0 = [0, 0]
result = scipy.optimize.fsolve(f, X0)
print(result)

# 3. 利用scipy.optimize的root求解
import scipy

def f(X):
    x = X[0]
    y = X[1]
    return [x ** 2 / 4 + y ** 2 - 1,
            (x - 0.2) ** 2 - y - 3]

X0 = [0, 0]
result2 = scipy.optimize.root(f, X0).x
print(result2)

##==========================================================================
##                    scipy.optimize.leastsq 非线性方程组求解
##==========================================================================

# 4. 利用scipy.optimize的 leastsq 求解
from scipy.optimize import leastsq

def f(X):
    x = X[0]
    y = X[1]
    return [x ** 2 / 4 + y ** 2 - 1, (x - 0.2) ** 2 - y - 3]

X0 = [0, 0]
h = leastsq(f, X0)
print(h)
# (array([-1.29613389, -0.76158337]), 2)

from scipy.optimize import leastsq
def func(x):
    return 2*(x-3)**2+1
print(leastsq(func, 0))
# (array([2.99999999]), 1)

##==========================================================================
##                    scipy root 非线性方程组求解
##==========================================================================

# scipy.optimize里面有两个函数可以数值求解方程组，分别是root和solve，这两个函数会找到方程组的一个近似解。下面通过例子介绍这两个函数的使用。

# from scipy.optimize import root

def f1(x):
   return [x[0] + x[0] * x[1] - 2, x[0] - x[1] - 2]

print(scipy.optimize.root(f1, [0,-1]).x)#初始猜测值[0,-1]
print(scipy.optimize.root(f1, [0,0]).x)#初始猜测值[0,0]

# 此外，在上面的基础上我们可以设置参数jac来提高运算速度（尤其是计算量很大时效果很明显）。
# from numpy import array,mat,sin,cos,exp
import numpy as np
# from scipy.optimize import root

def f(x):
    eqs=[]
    eqs.append(x[0]*x[1]+x[1]*x[2]+np.sin(x[0])*np.exp(x[1])+x[1])
    eqs.append(x[0]*x[1]-np.exp(x[0])*x[1]+1)
    eqs.append(x[1]*x[2]+np.exp(x[1])*x[2]+1)
    return eqs

def jac1(x):#方程组对应的雅可比矩阵
    return np.mat([[x[1] + np.cos(x[0]) * np.exp(x[1]), x[0] + x[2] + np.sin(x[0]) * np.exp(x[1]) + 1, x[1]],
                [x[1] - np.exp(x[0])*x[1], x[0] - np.exp(x[0]), 0],
                [0 , x[2] + np.exp(x[1]) * x[2], x[1] + np.exp(x[1])]])

print(scipy.optimize.root(f, [0,0,0]).x)
print(scipy.optimize.root(f, [0,0,0], jac = jac1).x)#加上参数jac加快运算速度

# 再来看一个复数域的例子
from scipy.optimize import root

def f1(x):
    return [x[0]*(1j)+x[0]*x[1]+1,x[0]+x[1]-1j]

print(root(f1, [1, 1], method = 'krylov').x)
print(root(f1, [1, 1], method = 'krylov', tol = 1e-10).x)#设置能够允许的误差为10的-10次方

##==========================================================================
##                    scipy sympy 非线性方程组求解
##==========================================================================

# 2.sympy求解
# sympy中的solve函数可以严格求解某些方程组，nsolve可以数值求近似解。
# (1).solve
# 直接看一个复数域的例子：

from sympy import symbols,Eq,solve
x0,x1=symbols('x0 x1')
eqs=[Eq(x0*(1j)+x0*x1,-5),Eq(x0+x1,1j)]
print(solve(eqs, [x0,x1]))

# (2).nsolve
# nsolve函数需要指定一个初始猜测解。
from sympy import symbols,Eq,nsolve
x0,x1=symbols('x0 x1')
eqs=[Eq(x0*(1j)+x0*x1,-5),Eq(x0+x1,1j)]
print(nsolve(eqs,[x0,x1],[1,1]))#初始猜测解设为[1，1]

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

#%%==========================================================================
##                    scipy minimize
##==========================================================================

#%%====================================================================================
##  Unconstrained minimization of multivariate scalar functions (minimize)
##====================================================================================

#%% >>>>>>>>>>>>>>>>>>>>> Nelder-Mead Simplex algorithm (method='Nelder-Mead')
import numpy as np
from scipy.optimize import minimize

def rosen(x):
    """The Rosenbrock function"""
    return np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosen, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})

print(res.x)


#%% >>>>>>>>>>>>>>>>>>>>>
def rosen_with_args(x, a, b):
    """The Rosenbrock function with additional arguments"""
    return sum(a*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0) + b

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosen_with_args, x0, method='nelder-mead', args=(0.5, 1.), options={'xatol': 1e-8, 'disp': True})

print(res.x)

#%% >>>>>>>>>>>>>>>>>>>>>
def rosen_with_args(x, a, *, b):  # b is a keyword-only argument
    return sum(a*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0) + b

def wrapped_rosen_without_args(x):
    return rosen_with_args(x, 0.5, b=1.)  # pass in `a` and `b`

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(wrapped_rosen_without_args, x0, method='nelder-mead', options={'xatol': 1e-8,})
print(res.x)

from functools import partial
partial_rosen = partial(rosen_with_args, a=0.5, b=1.)
res = minimize(partial_rosen, x0, method='nelder-mead', options={'xatol': 1e-8,})
print(res.x)


#%% >>>>>>>>>>>>>>>>>>>>> Broyden-Fletcher-Goldfarb-Shanno algorithm (method='BFGS')
def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosen, x0, method = 'BFGS', jac = rosen_der, options = {'disp': True})
print(res.x)

#######
def f(x):
   return -expensive(x[0])**2

def df(x):
    return -2 * expensive(x[0]) * dexpensive(x[0])

def expensive(x):
    # this function is computationally expensive!
    expensive.count += 1  # let's keep track of how many times it runs
    return np.sin(x)
expensive.count = 0

def dexpensive(x):
    return np.cos(x)
res = minimize(f, 0.5, jac = df)

print(res.fun)
# -0.9999999999999174
print(res.nfev, res.njev)
# 6, 6
print(expensive.count)
# 12

def f_and_df(x):
    expensive_value = expensive(x[0])
    return (-expensive_value**2,  # objective function
            -2*expensive_value*dexpensive(x[0]))  # gradient

expensive.count = 0  # reset the counter
res = minimize(f_and_df, 0.5, jac=True)
print(res.fun)
# -0.9999999999999174
print(expensive.count)
# 6

#%% >>>>>>>>>>>>>>>>>>>>>  Newton-Conjugate-Gradient algorithm (method='Newton-CG')
# Full Hessian example:
def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H
x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosen, x0, method='Newton-CG', jac=rosen_der, hess=rosen_hess, options={'xtol': 1e-8, 'disp': True})
print(res.x)


# Hessian product example:
def rosen_hess_p(x, p):
    x = np.asarray(x)
    Hp = np.zeros_like(x)
    Hp[0] = (1200*x[0]**2 - 400*x[1] + 2)*p[0] - 400*x[0]*p[1]
    Hp[1:-1] = -400*x[:-2]*p[:-2]+(202+1200*x[1:-1]**2-400*x[2:])*p[1:-1] -400*x[1:-1]*p[2:]
    Hp[-1] = -400*x[-2]*p[-2] + 200*p[-1]
    return Hp

res = minimize(rosen, x0, method='Newton-CG', jac=rosen_der, hessp=rosen_hess_p, options={'xtol': 1e-8, 'disp': True})
print(res.x)

#%% >>>>>>>>>>>>>>>>>>>>>   Trust-Region Newton-Conjugate-Gradient Algorithm (method='trust-ncg')

# Trust-Region Newton-Conjugate-Gradient Algorithm (method='trust-ncg')
res = minimize(rosen, x0, method='trust-ncg', jac=rosen_der, hess=rosen_hess, options={'gtol': 1e-8, 'disp': True})
print(res.x)


res = minimize(rosen, x0, method='trust-ncg', jac=rosen_der, hessp=rosen_hess_p, options={'gtol': 1e-8, 'disp': True})
print(res.x)




#%% >>>>>>>>>>>>>>>>>>>>>  Trust-Region Truncated Generalized Lanczos / Conjugate Gradient Algorithm (method='trust-krylov')

res = minimize(rosen, x0, method='trust-krylov', jac=rosen_der, hess=rosen_hess, options={'gtol': 1e-8, 'disp': True})
print(res.x)


res = minimize(rosen, x0, method='trust-krylov', jac=rosen_der, hessp=rosen_hess_p, options={'gtol': 1e-8, 'disp': True})
print(res.x)


#%% >>>>>>>>>>>>>>>>>>>>>  Trust-Region Nearly Exact Algorithm (method='trust-exact')

res = minimize(rosen, x0, method='trust-exact', jac=rosen_der, hess=rosen_hess, options={'gtol': 1e-8, 'disp': True})
print(res.x)

##====================================================================================
# Constrained minimization of multivariate scalar functions (minimize)
##====================================================================================
# The minimize function provides several algorithms for constrained minimization, namely 'trust-constr' , 'SLSQP', 'COBYLA', and 'COBYQA'. They require the constraints to be defined using slightly different structures. The methods 'trust-constr' and 'COBYQA' require the constraints to be defined as a sequence of objects LinearConstraint and NonlinearConstraint. Methods 'SLSQP' and 'COBYLA', on the other hand, require constraints to be defined as a sequence of dictionaries, with keys type, fun and jac.
#%% >>>>>>>>>>>>>>>>>>>>> Trust-Region Constrained Algorithm (method='trust-constr')

import scipy

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H

def cons_f(x):
    return [x[0]**2 + x[1], x[0]**2 - x[1]]

def cons_J(x):
    return [[2*x[0], 1], [2*x[0], -1]]

def cons_H(x, v):
    return v[0]*np.array([[2, 0], [0, 0]]) + v[1]*np.array([[2, 0], [0, 0]])

def cons_H_sparse(x, v):
    return v[0]*scipy.optimize.csc_matrix([[2, 0], [0, 0]]) + v[1]*scipy.optimize.csc_matrix([[2, 0], [0, 0]])

def cons_H_linear_operator(x, v):
    def matvec(p):
        return np.array([p[0]*2*(v[0]+v[1]), 0])
    return scipy.optimize.LinearOperator((2, 2), matvec=matvec)

bounds = scipy.optimize.Bounds([0, -0.5], [1.0, 2.0])
linear_constraint = scipy.optimize.LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])


nonlinear_constraint = scipy.optimize.NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H)

nonlinear_constraint = scipy.optimize.NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H_sparse)

nonlinear_constraint = scipy.optimize.NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=cons_H_linear_operator)

nonlinear_constraint = scipy.optimize.NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess=scipy.optimize.BFGS())

nonlinear_constraint = scipy.optimize.NonlinearConstraint(cons_f, -np.inf, 1, jac=cons_J, hess='2-point')

nonlinear_constraint = scipy.optimize.NonlinearConstraint(cons_f, -np.inf, 1, jac='2-point', hess=scipy.optimize.BFGS())


x0 = np.array([0.5, 0])
res = scipy.optimize.minimize(rosen, x0, method = 'trust-constr', jac = rosen_der, hess = rosen_hess, constraints = [linear_constraint, nonlinear_constraint], options = {'verbose': 1}, bounds = bounds)
print(res.x)
# [0.41494531 0.17010937]

res = scipy.optimize.minimize(rosen, x0, method = 'trust-constr', jac = rosen_der, hessp = rosen_hess_p, constraints = [linear_constraint, nonlinear_constraint], options = {'verbose': 1}, bounds = bounds)
print(res.x)

from scipy.optimize import SR1
res = scipy.optimize.minimize(rosen, x0, method = 'trust-constr', jac = "2-point", hess = SR1(), constraints = [linear_constraint, nonlinear_constraint], options = {'verbose': 1}, bounds = bounds)
print(res.x)


#%% >>>>>>>>>>>>>>>>>>>>>  Sequential Least SQuares Programming (SLSQP) Algorithm (method='SLSQP')

ineq_cons = {'type': 'ineq',
             'fun' : lambda x: np.array([1 - x[0] - 2*x[1],
                                         1 - x[0]**2 - x[1],
                                         1 - x[0]**2 + x[1]]),
             'jac' : lambda x: np.array([[-1.0, -2.0],
                                         [-2*x[0], -1.0],
                                         [-2*x[0], 1.0]])}
eq_cons = {'type': 'eq',
           'fun' : lambda x: np.array([2*x[0] + x[1] - 1]),
           'jac' : lambda x: np.array([2.0, 1.0])}


x0 = np.array([0.5, 0])
res = minimize(rosen, x0, method='SLSQP', jac=rosen_der, constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': True}, bounds=bounds)
print(res.x)


##====================================================================================
#  Global optimization
##====================================================================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize

def eggholder(x):
    return (-(x[1] + 47) * np.sin(np.sqrt(abs(x[0]/2 + (x[1]  + 47)))) -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1]  + 47)))))

bounds = [(-512, 512), (-512, 512)]
x = np.arange(-512, 513)
y = np.arange(-512, 513)
xgrid, ygrid = np.meshgrid(x, y)
xy = np.stack([xgrid, ygrid])

fig = plt.figure(figsize = (8, 6), constrained_layout = True)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45, -45)
ax.plot_surface(xgrid, ygrid, eggholder(xy), cmap='terrain')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('eggholder(x, y)')
plt.show()


results = {}
results['shgo'] = optimize.shgo(eggholder, bounds)
print(results['shgo'])

results = {}
results['DA'] = optimize.dual_annealing(eggholder, bounds)
print(results['DA'])

results = {}
results['DE'] = optimize.differential_evolution(eggholder, bounds)
print(results['DE'])

results = {}
results['shgo_sobol'] = optimize.shgo(eggholder, bounds, n=200, iters=5, sampling_method='sobol')
print(results['shgo_sobol'])


fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(eggholder(xy), interpolation='bilinear', origin='lower', cmap='gray')
ax.set_xlabel('x')
ax.set_ylabel('y')

def plot_point(res, marker='o', color=None):
    ax.plot(512+res.x[0], 512+res.x[1], marker=marker, color=color, ms=10)

plot_point(results['DE'], color='c')  # differential_evolution - cyan
plot_point(results['DA'], color='w')  # dual_annealing.        - white
plot_point(results['shgo'], color='r', marker='+')
plot_point(results['shgo_sobol'], color='r', marker='x')

# plot them all (with a smaller marker size)
for i in range(results['shgo_sobol'].xl.shape[0]):
    ax.plot(512 + results['shgo_sobol'].xl[i, 0], 512 + results['shgo_sobol'].xl[i, 1], 'ro', ms=4)

ax.set_xlim([-4, 514*2])
ax.set_ylim([-4, 514*2])
plt.show()


##====================================================================================
#  Least-squares minimization (least_squares)
##====================================================================================

from scipy.optimize import least_squares

def model(x, u):
    return x[0] * (u ** 2 + x[1] * u) / (u ** 2 + x[2] * u + x[3])

def fun(x, u, y):
    return model(x, u) - y

def jac(x, u, y):
    J = np.empty((u.size, x.size))
    den = u ** 2 + x[2] * u + x[3]
    num = u ** 2 + x[1] * u
    J[:, 0] = num / den
    J[:, 1] = x[0] * u / den
    J[:, 2] = -x[0] * num * u / den ** 2
    J[:, 3] = -x[0] * num / den ** 2
    return J


u = np.array([4.0, 2.0, 1.0, 5.0e-1, 2.5e-1, 1.67e-1, 1.25e-1, 1.0e-1, 8.33e-2, 7.14e-2, 6.25e-2])
y = np.array([1.957e-1, 1.947e-1, 1.735e-1, 1.6e-1, 8.44e-2, 6.27e-2, 4.56e-2, 3.42e-2, 3.23e-2, 2.35e-2, 2.46e-2])
x0 = np.array([2.5, 3.9, 4.15, 3.9])
res = least_squares(fun, x0, jac = jac, bounds=(0, 100), args=(u, y), verbose=1)


print(res.x)

import matplotlib.pyplot as plt
u_test = np.linspace(0, 5)
y_test = model(res.x, u_test)
plt.plot(u, y, 'o', markersize=4, label='data')
plt.plot(u_test, y_test, label='fitted model')
plt.xlabel("u")
plt.ylabel("y")
plt.legend(loc='lower right')
plt.show()


##====================================================================================
#  Univariate function minimizers (minimize_scalar)
##====================================================================================

from scipy.optimize import minimize_scalar
f = lambda x: (x - 2) * (x + 1)**2
res = minimize_scalar(f, method='brent')
print(res.x)
print(f(res.x))


from scipy.special import j1
res = minimize_scalar(j1, bounds=(4, 7), method='bounded')
print(res.x)
print(f(res.x))

##====================================================================================
#    Root finding
##====================================================================================

import numpy as np
from scipy.optimize import root
def func(x):
    return x + 2 * np.cos(x)
sol = root(func, 0.3)
print(sol.x)
# array([-1.02986653])
print(sol.fun)
# array([ -6.66133815e-16])

def func2(x):
    f = [x[0] * np.cos(x[1]) - 4,
         x[1]*x[0] - x[1] - 5]
    df = np.array([[np.cos(x[1]), -x[0] * np.sin(x[1])], [x[1], x[0] - 1]])
    return f, df
sol = root(func2, [1, 1], jac = True, method = 'lm')
print(sol.x)
# array([ 6.50409711,  0.90841421])


import numpy as np
from scipy.optimize import root
from numpy import cosh, zeros_like, mgrid, zeros

# parameters
nx, ny = 75, 75
hx, hy = 1./(nx-1), 1./(ny-1)

P_left, P_right = 0, 0
P_top, P_bottom = 1, 0

def residual(P):
   d2x = zeros_like(P)
   d2y = zeros_like(P)

   d2x[1:-1] = (P[2:]   - 2*P[1:-1] + P[:-2]) / hx/hx
   d2x[0]    = (P[1]    - 2*P[0]    + P_left)/hx/hx
   d2x[-1]   = (P_right - 2*P[-1]   + P[-2])/hx/hx

   d2y[:,1:-1] = (P[:,2:] - 2*P[:,1:-1] + P[:,:-2])/hy/hy
   d2y[:,0]    = (P[:,1]  - 2*P[:,0]    + P_bottom)/hy/hy
   d2y[:,-1]   = (P_top   - 2*P[:,-1]   + P[:,-2])/hy/hy

   return d2x + d2y + 5*cosh(P).mean()**2

# solve
guess = zeros((nx, ny), float)
sol = root(residual, guess, method='krylov', options={'disp': True})
#sol = root(residual, guess, method='broyden2', options={'disp': True, 'max_rank': 50})
#sol = root(residual, guess, method='anderson', options={'disp': True, 'M': 10})
print('Residual: %g' % abs(residual(sol.x)).max())

# visualize
import matplotlib.pyplot as plt
x, y = mgrid[0:1:(nx*1j), 0:1:(ny*1j)]
plt.pcolormesh(x, y, sol.x, shading='gouraud')
plt.colorbar()
plt.show()

##====================================================================================
#  Linear programming (linprog)
##====================================================================================

import numpy as np
from scipy.optimize import linprog
c = np.array([-29.0, -45.0, 0.0, 0.0])
A_ub = np.array([[1.0, -1.0, -3.0, 0.0], [-2.0, 3.0, 7.0, -3.0]])
b_ub = np.array([5.0, -10.0])
A_eq = np.array([[2.0, 8.0, 1.0, 0.0], [4.0, 4.0, 0.0, 1.0]])
b_eq = np.array([60.0, 60.0])
x0_bounds = (0, None)
x1_bounds = (0, 5.0)
x2_bounds = (-np.inf, 0.5)  # +/- np.inf can be used instead of None
x3_bounds = (-3.0, None)
bounds = [x0_bounds, x1_bounds, x2_bounds, x3_bounds]
result = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = bounds)
print(result.message)

x1_bounds = (0, 6)
bounds = [x0_bounds, x1_bounds, x2_bounds, x3_bounds]
result = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = bounds)
print(result.message)

x = np.array(result.x)
obj = result.fun
print(c @ x)
# -505.97435889013434  # may vary
print(obj)
# -505.97435889013434  # may vary

print(b_ub - (A_ub @ x).flatten())  # this is equivalent to result.slack
# [ 6.52747190e-10, -2.26730279e-09]  # may vary
print(b_eq - (A_eq @ x).flatten())  # this is equivalent to result.con
# [ 9.78840831e-09, 1.04662945e-08]]  # may vary
print([0 <= result.x[0], 0 <= result.x[1] <= 6.0, result.x[2] <= 0.5, -3.0 <= result.x[3]])
# [True, True, True, True]































































































































































































































































































































































































































































































































































































































































































































































































































































