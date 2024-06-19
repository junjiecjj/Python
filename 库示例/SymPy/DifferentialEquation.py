#!/usr/bin/env python3
#!-*-coding=utf-8-*-
#########################################################################

# 此程序的功能是： Sympy和Numpy、scipy

"""
https://blog.csdn.net/handsomeswp/article/details/111061087

https://zhuanlan.zhihu.com/p/60509430

https://blog.csdn.net/lanchunhui/article/details/49979411

https://zhuanlan.zhihu.com/p/111573239

https://www.cnblogs.com/sunshine-blog/p/8477523.html

https://vlight.me/2018/04/01/Numerical-Python-Symbolic-Computing/

https://blog.csdn.net/qq_42818403/article/details/120613079
https://blog.csdn.net/weixin_45870904/article/details/113080181

https://www.cnblogs.com/youcans/p/14912966.html
"""
#########################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, time, math
import scipy
import sympy as sy
from IPython.display import display, Latex
sy.init_printing()  #  这里我们还调用了 sympy.init_printing 函数，它用于配置 SymPy 打印系统以显示良好格式化的数学表达式。在 IPython 中，它会使用 MathJax JavaScript 库来渲染 SymPy 表达式。


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##****************   解方程 使用 solveset 求解方程。 ********************

#**************** 解一元一次方程 ********************
# https://zhuanlan.zhihu.com/p/60509430
"""
我们来求解这个一元一次方程组。(题目来源于人教版七年级数学上) [公式]
"""
x = sy.Symbol('x')
print(sy.solve(6*x + 6*(x-2000)-150000, x))

#**************** 解二元一次方程组 ********************

x,y = sy.symbols('x y')
print(sy.solve([x + y-10, 2*x+y-16], [x,y]))


#对多个方程求解，使用linsolve。方程的解为x=-1，y=3
s_expr = sy.linsolve([x+2*y-5,2*x+y-1], (x,y))
print(s_expr)


#**************** 解三元一次方程组 ********************
x,y,z = sy.symbols('x y, z')

s_expr = sy.solve([x+y+z-12, x+2*y+5*z-22, x - 4*y], [x,y,z])
print(s_expr)

#**************** 解一元二次方程组 ********************
# sympy.solve 基于方程等于零的假设给出解。当包含多个符号时，必须指出需要求解的符号
x,y = sy.symbols('x y')
a,b,c = sy.symbols('a b c')
expr= a*x**2 + b*x + c
s_expr = sy.solve( expr, x)
print(s_expr)

# 注意：在 SymPy 中，我们用 Eq(左边表达式, 右边表达式) 表示左边表达式与右边表达式相等。
print(sy.solveset(sy.Eq(x**2 - x, 0), x, domain = sy.Reals))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##****************  常微分方程：解析解 ********************
## https://blog.csdn.net/m0_37816922/article/details/135424455



"""
sympy.solvers.ode.dsolve(eq, func=None, hint='default', simplify=True, ics=None, xi=None, eta=None, x0=0, n=6, **kwargs)
各参数含义为
    eq 即常微分方程表达式，是唯一必须输入从桉树
    func 一个单变量函数，即dsolve的求解对象，一般无需指明，dsolve会自动从eq中检测待求解函数。
    hint 准备采用的求解方法，默认采用classify_ode返回的方法。
    simplify 为True时，通过odesimp()函数进行简化
    ics 微分方程的边界条件
    xi, eta 是两个极小函数
    x0 幂级数展开点
    n 幂级数所对应的因变量的指数

"""

##  二阶微分方程
# https://zhuanlan.zhihu.com/p/111573239
f = sy.symbols('f', cls = sy.Function)
diffeq = sy.Eq(f(x).diff(x, 2) - 2*f(x).diff(x) + f(x), sy.sin(x))
print(diffeq)
print(sy.dsolve(diffeq, f(x)))


## https://blog.csdn.net/m0_37816922/article/details/135424455
from sympy import Function, dsolve, Derivative
from sympy.abc import x
from sympy import print_latex
f = Function('f')
ff = dsolve(Derivative(f(x), x, x) + 9*f(x), f(x))
# print_latex(ff)
display(Latex(f"$$ {sy.latex(ff)}$$"))

## https://blog.csdn.net/m0_37816922/article/details/135424455
from sympy import Function, dsolve, Derivative
from sympy.abc import x
from sympy import print_latex
f = Function('f')
ff = dsolve(f(x).diff(x, 2) + 9*f(x), f(x))
# print_latex(ff)
display(Latex(f"$$ {sy.latex(ff)}$$"))

## https://blog.csdn.net/m0_37816922/article/details/135424455
# sympy.solvers.ode中提供了allhints元组，给出了所有hint的可选方法，共计45个，下面随机挑选两个进行测试。
x,y = sy.symbols('x y')
f = sy.symbols('f', cls = sy.Function)
eq = sy.sin(x) * sy.cos(f(x)) + sy.cos(x) * sy.sin(f(x)) * f(x).diff(x)
f1 = dsolve(eq, hint='1st_exact')
f2 = dsolve(eq, hint='almost_linear')




# https://blog.csdn.net/2302_76305195/article/details/135869216
## 带初值的一阶常微分方程
import sympy as sp
x=sp.var('x')
y=sp.Function('y')
eq = y(x).diff(x)+2*y(x)-2*x*x-2*x
s=sp.dsolve(eq,y(x), ics={y(0):1})
s=sp.simplify(s)
print(s)

# https://blog.csdn.net/2302_76305195/article/details/135869216
## 带初值的二阶微分方程
import sympy as sp
x=sp.var('x')
y=sp.Function('y')
eq=y(x).diff(x,2)-2*y(x).diff(x)+y(x)-sp.exp(x)
con={y(0):1,y(x).diff(x).subs(x,0):-1}
s=sp.dsolve(eq, ics=con)
print(s)

# 带初值的二阶微分方程
import sympy as sp
t=sp.var('t')
y=sp.Function('y')
u=sp.exp(-t)*sp.cos(t)
eq=y(t).diff(t,4)+10*y(t).diff(t,3)+35*y(t).diff(t,2)+50*y(t).diff(t)+24*y(t)-u.diff(t,2)
con={y(0):0,y(t).diff(t).subs(t,0):-1,y(t).diff(t,2).subs(t,0):1,y(t).diff(t,3).subs(t,0):1}
s=sp.dsolve(eq,ics=con)
s=sp.expand(s)
print(s)


#
# x`t = y+1;
# y`t = x+1
# x(0) = -2
# y(0) = 0
import sympy as sp
t = sp.var('t')
y = sp.Function('y')
x = sp.Function('x')
eq = (x(t).diff(t,1)-y(t)-1, y(t).diff(t,1)-x(t)-1 )
con = {x(0):-2, y(0):0}
s = sp.dsolve(eq,  ics=con)
print(s)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##****************  常微分方程组：解析解 ********************
## https://blog.csdn.net/m0_37816922/article/details/135424455
from sympy import Eq, symbols
from sympy.abc import t

x, y = symbols('x, y', cls=Function)
eq = (Eq(Derivative(x(t),t), 12*t*x(t) + 8*y(t)), Eq(Derivative(y(t),t), 21*x(t) + 7*t*y(t)))
xys = dsolve(eq)
display(Latex(f"$$ {xys}$$"))





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##****************  常微分方程的 数值解 ********************
# https://blog.csdn.net/2302_76305195/article/details/135869216

from scipy.integrate import odeint
import numpy as np
import pylab as plt
import sympy as sp

dy = lambda y,x:-2*y+2*x*x +2*x
xx = np.linspace(0, 3, 31)#自变量的取值
s = odeint(dy, 1, xx)#y的取值
print('x={}\n对应的数值解y={}'.format(xx, s.flatten()))
plt.plot(xx,s,'*')

x=sp.var('x')
y=sp.Function('y')
eq=y(x).diff(x)+2*y(x)-2*x*x-2*x
s2=sp.dsolve(eq,ics={y(0):1})
sx=sp.lambdify(x,s2.args[1],'numpy')

plt.plot(xx,sx(xx)) #观察吻合度
plt.show()



from scipy.integrate import odeint
import numpy as np
import pylab as plt
yx=lambda y,x:[y[1],np.sqrt(1+y[1]**2)/5/(1-x)]
x0=np.arange(0,1,0.00001)
y0=odeint(yx,[0,0],x0)

plt.rc('font',size=16)
plt.plot(x0,y0[:,0])
plt.show()
















