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

##  二阶常系数微分方程(显含时间变量)
# https://zhuanlan.zhihu.com/p/111573239
f = sy.symbols('f', cls = sy.Function)
diffeq = sy.Eq(f(x).diff(x, 2) - 2*f(x).diff(x) + f(x), sy.sin(x))
eq = f(x).diff(x, 2) - 2*f(x).diff(x) + f(x) - sy.sin(x)
print(diffeq)
s = sy.dsolve(eq, f(x))
print(s)

##  二阶常系数齐次微分方程(不显含时间变量)
## https://blog.csdn.net/m0_37816922/article/details/135424455
from sympy import Function, dsolve, Derivative
from sympy.abc import x
from sympy import print_latex
f = Function('f')
ff = dsolve(Derivative(f(x), x, x) + 9*f(x), f(x))
# print_latex(ff)
display(Latex(f"$$ {sy.latex(ff)}$$"))

##  二阶常系数微分方程(显含时间变量)
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



# 一阶微分方程（一阶）(显含时间变量)
# https://blog.csdn.net/2302_76305195/article/details/135869216
## 带初值的一阶 （非齐次）线性微分方程
import sympy as sp
x=sp.var('x')
y=sp.Function('y')
eq = y(x).diff(x)+2*y(x)-2*x*x-2*x
s=sp.dsolve(eq,y(x), ics={y(0):1})
s=sp.simplify(s)
print(s)

# https://blog.csdn.net/2302_76305195/article/details/135869216
## 带初值的二阶常系数非齐次微分方程(显含时间变量)
import sympy as sp
x=sp.var('x')
y=sp.Function('y')
eq=y(x).diff(x,2)-2*y(x).diff(x)+y(x)-sp.exp(x)
con={y(0):1,y(x).diff(x).subs(x,0):-1}
s=sp.dsolve(eq, ics=con)
print(s)

# 带初值的高阶常系数微分方程（n 阶）(显含时间变量)
import sympy as sp
t = sp.var('t')
y = sp.Function('y')
u = sp.exp(-t)*sp.cos(t)
eq = y(t).diff(t,4)+10*y(t).diff(t,3)+35*y(t).diff(t,2)+50*y(t).diff(t)+24*y(t)-u.diff(t,2)
con = {y(0):0,y(t).diff(t).subs(t,0):-1,y(t).diff(t,2).subs(t,0):1,y(t).diff(t,3).subs(t,0):1}
s = sp.dsolve(eq, ics=con)
s = sp.expand(s)
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

"""
python对于常微分方程的数值求解是基于一阶方程进行的，高阶微分方程必须化成一阶方程组，通常采用龙格-库塔方法. scipy.integrate模块的odeint模块的odeint函数求常微分方程的数值解，其基本调用格式为：
sol=odeint(func,y0,t)
    func是定义微分方程的函数或匿名函数
    y0: array：　　初始条件 y0，对于常微分方程组 y0 则为数组向量
    t: array：　　求解函数值对应的时间点的序列。序列的第一个元素是与初始条件 t0 对应的初始时间 t0; 时间序列必须是单调递增或单调递减的，允许重复值。
    返回值sol是对应于序列t中元素的数值解，如果微分方程组中有n个函数，返回值会是n列的矩阵，第i(i=1,2···,n)列对应于第i个函数的数值解.
"""
# https://blog.csdn.net/2302_76305195/article/details/135869216
## 1
from scipy.integrate import odeint
import numpy as np
import pylab as plt
import sympy as sp

dy = lambda y,x:-2*y+2*x*x +2*x
xx = np.linspace(0, 3, 31)#自变量的取值
s = odeint(dy, 1, xx)#y的取值
print('x={}\n对应的数值解y={}'.format(xx, s.flatten()))
plt.plot(xx,s,'*')

x = sp.var('x')
y = sp.Function('y')
eq = y(x).diff(x)+2*y(x)-2*x*x-2*x
s2 = sp.dsolve(eq, ics={y(0):1})
sx = sp.lambdify(x, s2.args[1],'numpy')

plt.plot(xx, sx(xx)) #观察吻合度
plt.show()


## 2
from scipy.integrate import odeint
import numpy as np
import pylab as plt
yx=lambda y,x:[y[1], np.sqrt(1+y[1]**2)/5/(1-x)]
x0=np.arange(0,1,0.00001)
y0=odeint(yx,[0,0],x0)
plt.rc('font',size=16)
plt.plot(x0, y0[:,0], '--*')

plt.show()


## 3 (不显含时间变量)
# https://blog.csdn.net/qq_38981614/article/details/113915904
def pend(y, t, b, c):
    theta, omega = y
    dydt = [omega, -b*omega - c*np.sin(theta)]
    return dydt
b = 0.25
c = 5.0
y0 = [np.pi - 0.1, 0.0] #初始条件
t = np.linspace(0, 10, 101)
sol = odeint(pend, y0, t, args=(b, c))
plt.plot(t, sol[:, 0], 'b', label='theta(t)')
plt.plot(t, sol[:, 1], 'g', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

## 4
## # https://blog.csdn.net/qq_38981614/article/details/113915904
initial = 100 # 定义原方程初始值
def f1(y, x):
    return 12*x**2
x = np.linspace(0, 10, 100)     #定义自变量
result1 = odeint(f1, initial, x) #调用求解
plt.plot(x, result1)               #求解值
plt.plot(x, 4*x**3+initial,'--k') #真实值
plt.show()

## 5 (显含时间变量)
initial = 100 #原初值
derivative1 = 0 #一阶初值
def f2(y, x):
    y1, v = y
    return np.array([v, 24*x]) #返回值列阵 先一阶 再 二阶 由低到高
x = np.linspace(0, 10, 100)    #定义自变量
result2 = odeint(f2, (initial, derivative1), x)  #初值对应 先原方程 再 一阶 由低到高
plt.plot(x, result2[:,0])
plt.plot(x, 4*x**3+initial, '--k')
plt.show()

## 6
# https://blog.csdn.net/weixin_45870904/article/details/113080181
from scipy.integrate import odeint
from numpy import arange
dy = lambda y,x: -2*y+x**2+2*x
x = arange(1, 10.5, 0.5)
sol = odeint(dy, 2, x)
print("x={}\n对应的数值解y={}".format(x, sol.T))

## 7 (不显含时间变量)
# https://blog.csdn.net/weixin_45870904/article/details/113080181
from scipy.integrate import odeint
from sympy.abc import t
import numpy as np
import matplotlib.pyplot as plt
#定义一个方程组（微分方程组）
def pfun(y, x):
    y1, v = y    #让'y'成为一个[y1',y2']的向量 所以将等式左边都化为1阶微分是很重要的
    return np.array([v, -2*y1-2*v]) #返回的是等式右边的值
x=np.arange(0,10,0.1) #创建自变量序列
soli=odeint(pfun, [0.0, 1.0], x) #求数值解
plt.rc('font',size=16)
plt.rc('font',family='SimHei')
plt.plot(x,soli[:,0], 'r*', label="数值解")

x = sp.var('x')
y = sp.Function('y')
eq = y(x).diff(x,2) + 2*y(x).diff(x,1) + 2*y(x)
s2 = sp.dsolve(eq, ics={y(0):0, y(x).diff(x,1).subs(x,0):1})
sx = sp.lambdify(x, s2.args[1],'numpy')
plt.plot(xx, sx(xx), 'g',label="符号解曲线") #观察吻合度

# plt.plot(x,np.exp(-x)*np.sin(x),'g',label="符号解曲线")
plt.legend()
plt.show()


## 8  Lorenz的混沌效应
# https://blog.csdn.net/weixin_45870904/article/details/113080181
from scipy.integrate import odeint
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
def lorenz(w,t): #定义微分方程组
    sigma = 10
    rho = 28
    beta = 8/3
    x, y, z = w
    return np.array([sigma*(y-x), rho*x-y-x*z, x*y-beta*z])

t = np.arange(0,50,0.01) #建立自变量序列（也就是时间点）
sol1 = odeint(lorenz, [0.0, 1.0, 0.0], t) #第一个初值问题求解
sol2 = odeint(lorenz, [0.0, 1.001, 0.0], t) #第二个初值问题求解
#画图代码 （可忽略）
plt.rc('font',size=16)
plt.rc('text', usetex=False)
#第一个图的各轴的定义
ax1=plt.subplot(121,projection='3d')
ax1.plot(sol1[:,0],sol1[:,1],sol1[:,2],'r')
ax1.set_xlabel('$x$');ax1.set_ylabel('$y$');ax1.set_zlabel('$z$')
ax2=plt.subplot(122,projection='3d')
ax2.plot(sol1[:,0]-sol2[:,0],sol1[:,1]-sol2[:,1],  sol1[:,2]-sol2[:,2],'g')
ax2.set_xlabel('$x$');ax2.set_ylabel('$y$');ax2.set_zlabel('$z$')
plt.show()
print("sol1=",sol1,'\n\n',"sol1-sol2",sol1-sol2)





#%% https://www.cnblogs.com/youcans/p/14912966.html
# 1. 求解微分方程初值问题(scipy.integrate.odeint)
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np
import matplotlib.pyplot as plt

def dy_dt(y, t):  # 定义函数 f(y,t)
    return np.sin(t**2)

y0 = [1]  # y0 = 1 也可以
t = np.arange(-10,10,0.01)  # (start,stop,step)
y = odeint(dy_dt, y0, t)  # 求解微分方程初值问题

# 绘图
plt.plot(t, y)
plt.title("scipy.integrate.odeint")
plt.show()



# 2. 求解微分方程组初值问题(scipy.integrate.odeint)
from scipy.integrate import odeint    # 导入 scipy.integrate 模块
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导数函数, 求 W=[x,y,z] 点的导数 dW/dt
def lorenz(W,t,p,r,b):  # by youcans
    x, y, z = W  # W=[x,y,z]
    dx_dt = p*(y-x)  # dx/dt = p*(y-x), p: sigma
    dy_dt = x*(r-z) - y  # dy/dt = x*(r-z)-y, r:rho
    dz_dt = x*y - b*z  # dz/dt = x*y - b*z, b;beta
    return np.array([dx_dt,dy_dt,dz_dt])

t = np.arange(0, 30, 0.01)  # 创建时间点 (start,stop,step)
paras = (10.0, 28.0, 3.0)  # 设置 Lorenz 方程中的参数 (p,r,b)

# 调用ode对lorenz进行求解, 用两个不同的初始值 W1、W2 分别求解
W1 = (0.0, 1.00, 0.0)  # 定义初值为 W1
track1 = odeint(lorenz, W1, t, args=(10.0, 28.0, 3.0))  # args 设置导数函数的参数
W2 = (0.0, 1.01, 0.0)  # 定义初值为 W2
track2 = odeint(lorenz, W2, t, args=paras)  # 通过 paras 传递导数函数的参数

# 绘图
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(track1[:,0], track1[:,1], track1[:,2], color='magenta') # 绘制轨迹 1
ax.plot(track2[:,0], track2[:,1], track2[:,2], color='deepskyblue') # 绘制轨迹 2
ax.set_title("Lorenz attractor by scipy.integrate.odeint")
plt.show()



# 3. 求解二阶微分方程初值问题(scipy.integrate.odeint)
# Second ODE by scipy.integrate.odeint
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np
import matplotlib.pyplot as plt

# 导数函数，求 Y=[u,v] 点的导数 dY/dt
def deriv(Y, t, a, w):
    u, v = Y  # Y=[u,v]
    dY_dt = [v, -2*a*v-w*w*u]
    return dY_dt

t = np.arange(0, 20, 0.01)  # 创建时间点 (start,stop,step)
# 设置导数函数中的参数 (a, w)
paras1 = (1, 0.6)  # 过阻尼：a^2 - w^2 > 0
paras2 = (1, 1)  # 临界阻尼：a^2 - w^2 = 0
paras3 = (0.3, 1)  # 欠阻尼：a^2 - w^2 < 0

# 调用ode对进行求解, 用两个不同的初始值 W1、W2 分别求解
Y0 = (1.0, 0.0)  # 定义初值为 Y0=[u0,v0]
Y1 = odeint(deriv, Y0, t, args=paras1)  # args 设置导数函数的参数
Y2 = odeint(deriv, Y0, t, args=paras2)  # args 设置导数函数的参数
Y3 = odeint(deriv, Y0, t, args=paras3)  # args 设置导数函数的参数
# W2 = (0.0, 1.01, 0.0)  # 定义初值为 W2
# track2 = odeint(lorenz, W2, t, args=paras)  # 通过 paras 传递导数函数的参数

# 绘图
plt.plot(t, Y1[:, 0], 'r-', label='u1(t)')
plt.plot(t, Y2[:, 0], 'b-', label='u2(t)')
plt.plot(t, Y3[:, 0], 'g-', label='u3(t)')
plt.plot(t, Y1[:, 1], 'r:', label='v1(t)')
plt.plot(t, Y2[:, 1], 'b:', label='v2(t)')
plt.plot(t, Y3[:, 1], 'g:', label='v3(t)')
plt.axis([0, 20, -0.8, 1.2])
plt.legend(loc='best')
plt.title("Second ODE by scipy.integrate.odeint")
plt.show()
















































































