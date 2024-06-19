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
##                                        基础

# 定义一个 符号
x,y = sy.symbols("x y")
print(type(x))

# 变量 x 现在表示一个抽象的数学符号 x，默认情况下知道的信息非常少。此时，x 可以表示例如实数、整数、复数、函数以及大量其他可能。在很多情况下，用这个抽象的、未指定的 Symbol 对象表示一个数学符号就足够了，但有时需要给 SymPy 更多的信息，以确切知道 Symbol 对象代表具体什么类型的符号。这可以帮助 SymPy 更有效地操纵分析表达式。我们可以添加各种假设，以缩小符号类型潜在的可能。下表总结了可以与 Symbol 类实例关联的一些常用假设的选择。

# 假设的关键字参数	属性	描述
# real, imaginary	is_real, is_imaginary	指定一个符号代表实数或虚数
# positive, negative	is_positive, is_negative	指定一个符号代表正数或负数
# integer	is_integer	该符号代表整数
# odd, even	is_odd, is_even	该符号代表奇数或偶数
# prime	is_prime	该符号代表质数
# finite, infinite	is_finite, is_infinite	该符号代表有限或无限

z = sy.Symbol("z", imaginary=True)
print(z.is_real)

# 可以利用 symbols 函数依次新建类似的多个变量
vars = sy.symbols('x_1:5')
print(vars)

# 新建符号变量时可以指定其定义域，比如指定x>0
x = sy.symbols('x', positive = True)

# 符号表达式
f = x**2+3*x-5
print(f"f(x)={f}")
display(Latex(f"$$f(x)={sy.latex(f)}$$"))

# 符号计算往往也需要得到数值结果的，【evalf】函数便可以胜任这项工作，其参数subs通过字典的形式，一一指定自变量。
xx = 3
print(f.evalf(subs={x:1, y:1, }))
sy.pprint(f"f({xx})={f.subs({x:xx})}")
print(f"f({xx})={f.subs({x:xx})}")


sy.N(1+np.pi)
# Out[5]: 4.14159265358979

sy.N(sy.pi, 50) # SymPy 的多精度浮点数能够用来计算高达 50 位
# Out[6]: 3.1415926535897932384626433832795028841971693993751

(x + 1/np.pi).evalf(10)
# Out[7]: x + 0.3183098862


expr = sy.sin(sy.pi * x * sy.exp(x))
[expr.subs(x, xx).evalf(3) for xx in range(0, 10)]
# 但是，此方法相当低效，SymPy 使用函数 sympy.lambdify 为执行这类操作提供了更高效的方法。该函数将一组自由符号和一个表达式作为参数，并生成一个可高效评估表达式数值的函数。生成函数的参数数量与传递给 sympy.lambdify 第一个参数的自由符号的数量相同。
expr_func = sy.lambdify(x, expr, 'numpy')
xvalues = np.arange(0, 10)
expr_func(xvalues)

##========================================= 利用 lambdify 函数将 SymPy 表达式转换为 NumPy 可使用的函数
# 如果进行简单的计算，使用 subs 和 evalf 是可行的，但要获得更高的精度，则需要使用更加有效的方法。例如，要保留小数点后 1000 位，则使用 SymPy 的速度会很慢。这时，您就需要使用 NumPy 库。
# lambdify 函数的功能就是可以将 SymPy 表达式转换为 NumPy 可以使用的函数，然后用户可以利用 NumPy 计算获得更高的精度。
a = np.pi / 3
x = sy.symbols('x')
expr = sy.sin(x)
f = sy.lambdify(x, expr, 'numpy')
f(a)
# 0.8660254037844386
expr.subs(x, a)

a,b = sy.symbols('a,bx')
expr=a**2+b**2
f = sy.lambdify([a,b],expr)
print(f(2,3))
# 13


from sympy.utilities.lambdify import implemented_function
f = implemented_function('f', lambda x: x+1)
lam_f = sy.lambdify(x, f(x))
print(lam_f(4))


x = sy.symbols('x')
f = sy.lambdify([x], x + 1)
print(f(1))

x,y = sy.symbols('x, y')
f = sy.lambdify([x, y], x + y)
print(f(1,2))


x,y,z = sy.symbols('x, y, z')
f = sy.lambdify([x, (y, z)], x + y + z)
print(f(1, (2, 3)))


# 采用符号变量的 subs 方法进行替换操作，
x = sy.symbols('x')
expr = sy.cos(x) + 1
print(expr.subs(x, 0))


xx = 3
yy = 4
f1 = sy.sqrt(x**2+y**2)
display(Latex(f"$$f_1(x,y)={sy.latex(f1)}$$"))
sy.pprint(f"f1({xx},{yy})={f1.subs({x:xx, y:yy})}")


a,x,t = sy.symbols("a x t")
f = sy.cos(x)
print(f"f(t)={f.subs({x:a**t})}")

# # 构造分数 1/2
y = sy.Rational(1,2) + sy.sqrt(2)
print(f"y={y}")
print(f"y={y.evalf()}")


y = x**2 + sy.Rational(1,2)
sy.pprint(y)
value_x = 1
print(f"y={y.subs({x:value_x}).evalf()}")
print(f"y={y.evalf(subs=({x:value_x}))}")

## 将字符串转换为 SymPy 表达式,注意：sympify 是符号化，与另一个函数 simplify （化简）拼写相近，不要混淆。
str_expr = 'x**2 + 2*x + 1'
expr = sy.sympify(str_expr)
print(expr)

# sympy：多项式运算
x,y = sy.symbols('x y')
expr = sy.poly(x*(x**2+x-1)**2)
print(expr)


# 可以使用符号变量的 evalf 方法将其转换为指定精度的数值解，例如：
# pi = sy.Symbol('pi')
# pi = 3.141592697
# pi =  pi.evalf(3) # pi 保留 3 位有效数字


#**************** 数学符号与表达式 ********************
print(math.sqrt(8))
print(sy.sqrt(8))


x = sy.Symbol('x')
y = sy.Symbol('y')
k, m, n = sy.symbols('k m n')
print(3*x+y**3)



#**************** 折叠与展开表达式 ********************
"""
factor() 函数可以折叠表达式，而 expand() 函数可以展开表达式，
比如表达式： [公式] ，折叠之后应该是 [公式] 。我们来看具体的代码：
"""
x,y = sy.symbols('x y')
expr=x**4+x*y+8*x
f_expr = sy.factor(expr)
e_expr = sy.expand(f_expr)
print(f_expr)
print(e_expr)

from fractions import Fraction
x = sy.Symbol('x')
y = sy.Symbol('y')
expr = sy.expand((x+(2*x*y)** Fraction(1, 2)+y)*(x-(2*x*y)** Fraction(1, 2)+y))


#**************** 表达式化简 ********************
"""
simplify () 函数可以对表达式进行化简。有一些表达式看起来会比较复杂，
就拿人教版初二上的一道多项式的乘法为例，简化 [公式] 。
"""
x,y = sy.symbols('x y')
expr = (2*x)**3*(-5*x*y**2)
s_expr = sy.simplify(expr)
print(s_expr)


expr = sy.simplify(sy.sin(x)**2 + sy.cos(x)**2)
print(expr)

alpha  = sy.symbols('alpha')
print(sy.simplify(2*sy.sin(alpha )*sy.cos(alpha)))

print("**************** 多项式和有理函数化简 ******************** ")
x_1 = sy.symbols('x_1')
print(sy.expand((x_1 + 1)**2))


print("**************** factor (因式分解) ******************** ")

print(sy.factor(x**3 - x**2 + x - 1))

print("**************** collect (合并同类项) ******************** ")

expr = x*y + x - 3 + 2*x**2 - z*x**2 + x**3
print(sy.collect(expr, x))

print("**************** cancel (有理分式化简) ******************** ")
print(sy.cancel((x**2 + 2*x + 1)/(x**2 + x)))


print("**************** apart (部分分式展开), 使用 apart 函数可以将分式展开，例如：******************** ")
expr = sy.apart(4*x**3 + 21*x**2 + 10*x + 12)/(x**4 + 5*x**3 + 5*x**2 + 4*x)
print(expr)
print(sy.apart(expr))


expr = sy.together(1/ (y*x + y)+1 / (1+x))
print(expr)

# 这里我们将要考虑的最后一种数学化简是重写分数。函数 sympy.apart 和 sympy.together 分别是将分数分解成多个部分和将多个分数组合成一个分数。
print(sy.cancel(y/(y*x + y)))


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



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##**************** 微积分 Calculus ********************
"""
微积分是大学高等数学里非常重要的学习内容，比如求极限、导数、微分、不定积分、定积分等都是可以使用 Sympy 来运算的。
求极限 Sympy 是使用 limit (表达式，变量，极限值) 函数来求极限的，比如我们要求
lim_{x->0}{sin(x)/x}
的值。
"""
# 极限
x, y, z = sy.symbols('x y z')
expr = sy.sin(x)/x
l_expr = sy.limit(expr, x, 0)
print(l_expr)


#  求导
x,y = sy.symbols('x y')
expr=sy.sin(x)*sy.exp(x)
diff_expr=sy.diff(expr, x)
diff_expr2=sy.diff(expr,x,2)
print(diff_expr)
print(diff_expr2)

# 求一阶导数
print(sy.diff(sy.cos(x), x))

# 求 3 阶导数
print(sy.diff(x**4, x, 3))

#我们也可以用 符号变量的 diff 方法 求微分，例如：
expr = sy.cos(x)
print(expr.diff(x, 2))

# 多元函数求偏导函数
expr = sy.exp(x*y*z)
print(sy.diff(expr, x))


#**************** 求不定积分 ********************c

x,y = sy.symbols('x y')
expr = sy.exp(x)*sy.sin(x) + sy.exp(x)*sy.cos(x)
i_expr = sy.integrate(expr,x)
print(i_expr)


#===========
x, a, s, t = sy.symbols('x, a, s, t')
expr = (x-a)**2 * (1/sy.sqrt(2 * sy.pi)) * sy.exp(-x**2/2)
i_expr = sy.integrate(expr, (x, s, t))
print(i_expr)

display(Latex(f"$$f(x)={sy.latex(i_expr)}$$"))
sy.simplify(i_expr)

#===========
x, s,  t = sy.symbols('x, s, t')
expr = x**2 * (1/sy.sqrt(2 * sy.pi)) * sy.exp(-x**2/2)
i_expr=sy.integrate(expr,(x, s, t))
print(i_expr)

display(Latex(f"$$f(x)={sy.latex(i_expr)}$$"))
print(sy.simplify(i_expr))


#===========
x, t = sy.symbols('x,  t')
expr =  (1/sy.sqrt(2 * sy.pi)) * sy.exp(-x**2/2)
i_expr = sy.integrate(expr,(x, -sy.oo, t))
# print(i_expr)
display(Latex(f"$$f(x)={sy.latex(i_expr)}$$"))


sy.simplify(i_expr)
print(sy.simplify(i_expr))

display(Latex(f"$$f_1(x,y)={sy.latex(f1)}$$"))


#===========
x, t = sy.symbols('x,  t')
expr =  sy.sin(x) /x
i_expr = sy.integrate(expr,(x, -sy.oo, t))
# print(i_expr)
display(Latex(f"$$f(x)={sy.latex(i_expr)}$$"))


sy.simplify(i_expr)
print(sy.simplify(i_expr))

display(Latex(f"$$f_1(x,y)={sy.latex(f1)}$$"))


#c **************** 求定积分 ********************
"""
Sympy 同样是使用 integrate () 函数来做定积分的求解，只是语法不同：integrate (表达式，（变量，下区间，上区间))，我们来看如果求解
"""
x,y = sy.symbols('x y')
expr = sy.sin(x**2)
i_expr=sy.integrate(expr, (x, -np.inf, np.inf))
print(i_expr)



x  = sy.symbols('x')
expr =  x * sy.exp(-2*x) + x * 0.25 * sy.exp(-x/2)
i_expr = sy.integrate(expr, (x, 0, np.inf))
print(i_expr)


x  = sy.symbols('x')
expr =  x**2 * sy.exp(-2*x) + x**2 * 0.25 * sy.exp(-x/2)
i_expr = sy.integrate(expr, (x, 0, np.inf))
print(i_expr)


#求 [公式] 的定积分：注意：在 SymPy 中，我们用 'sy.oo' 表示 [公式] 。
print(sy.integrate(sy.exp(-x), (x, 0, sy.oo)))

# 使用 limit 函数求极限，例如：
print(sy.limit(sy.sin(x)/x, x, 0))
print(sy.limit(1/x, x, 0, '+'))

##================================================
## 求函数 [公式] 在 [公式] 的二重积分：
print(sy.integrate(sy.exp(-x**2 - y**2), (x, -np.inf, np.inf), (y, -np.inf, np.inf)))

print("**************** series (级数展开) ******************** ")
expr = sy.sin(x)
print(expr.series(x, 0, 4))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##****************   线性代数 矩阵运算********************
# 构造矩阵
print(f"sy.Matrix([[1, -1], [3, 4], [0, 2]])")

# 构造列向量
print(sy.Matrix([1, 2, 3]))


# 构造行向量
print(sy.Matrix([[1], [2], [3]]).T)


# 构造单位矩阵
sy.eye(4)

# 构造零矩阵
sy.zeros(4)

# 构造壹矩阵
sy.ones(4)

# 构造对角矩阵
sy.diag(1, 2, 3, 4)

#矩阵转置用矩阵变量的 T 方法。例如：
a = sy.Matrix([[1, -1], [3, 4], [0, 2]])
print(a)

# 求矩阵 a 的转置
print(a.T)

# 求矩阵 M 的 2 次幂
M = sy.Matrix([[1, 3], [-2, 3]])
print(M**2)

# 求矩阵 M 的逆
print(M**-1)

#用矩阵变量的 det 方法可以求其行列式：
M = sy.Matrix([[1, 0, 1], [2, -1, 3], [4, 3, 2]])
print(M.det())


#用矩阵变量的 eigenvals 和 charpoly 方法求其特征值和特征多项式。
M = sy.Matrix([[3, -2,  4, -2], [5,  3, -3, -2], [5, -2,  2, -2], [5, -2, -3,  3]])
print(M.eigenvals())
# {3: 1, -2: 1, 5: 2}
lamda = sy.symbols('lamda')
p = M.charpoly(lamda)
sy.factor(p)

print("**************** 矩阵运算 矩阵乘法******************** ")

a_11,a_12,a_21,a_22 = sy.symbols('a_11 a_12 a_21 a_22')
b_11,b_12,b_21,b_22 = sy.symbols('b_11 b_12 b_21 b_22')
a = sy.Matrix([[a_11,a_12], [a_21,a_22]])
b = sy.Matrix([[b_11,b_12], [b_21,b_22]])
print(a*b)



a1 = sy.Matrix([[0,1], [1,0]])
b1 = sy.Matrix([[0,-1j], [1j,0]])
print(a1*b1)

print("**************** 矩阵张量积 ******************** ")

from sympy.physics.quantum import TensorProduct
a_11,a_12,a_21,a_22 = sy.symbols('a_11 a_12 a_21 a_22')
b_11,b_12,b_21,b_22 = sy.symbols('b_11 b_12 b_21 b_22')
A = sy.Matrix([[a_11,a_12],[a_21,a_22]])
B = sy.Matrix([[b_11,b_12],[b_21,b_22]])
C = TensorProduct(A,B)

display(Latex(f"$${sy.latex(C)}$$"))


#%% 可以利用 laplace_transform 函数进行 Laplace 变换，例如：
#  Laplace (拉普拉斯)变换
from sympy.abc import t, s
expr = sy.sin(t)
sy.laplace_transform(expr, t, s)

#利用 inverse_laplace_transform 函数进行逆 Laplace 变换：
expr = 1/(s - 1)
sy.inverse_laplace_transform(expr, s, t)
# 利用 SymPy 画函数图像 使用 plot 函数绘制二维函数图像，例如：
from sympy.plotting import plot
from sympy.abc import x
plot(x**2, (x, -2, 2))


from sympy import plot_implicit
from sympy import Eq
from sympy.abc import x, y
plot_implicit(Eq(x**2 + y**2, 1))




# 使用 SymPy 画出三维函数图像，例如： ********************

from sympy.plotting import plot3d
from sympy.abc import x, y
from sympy import exp
plot3d(x*exp(-x**2 - y**2), (x, -3, 3), (y, -2, 2))

#如果是链式求导，sympy 该怎么做呢 ********************
from sympy import *
r, t = symbols('r t') # r (radius), t (angle theta)
f = symbols('f', cls = Function)
x = r * cos(t)
y = r * sin(t)
g = f(x, y)
Derivative(g, r, 1).doit()



#%% https://zhuanlan.zhihu.com/p/83822118
print("**************** 求解二元一次方程组 ******************** ")

import sympy as sym
from sympy import sin,cos
x,y = sym.symbols('x, y')
print(sym.solve([x + y - 1,x - y -3],[x,y]))

print("**************** 测试不定积分 ******************** ")
x = sym.symbols('x')
a = sym.Integral(cos(x))
# 积分之后的结果
a.doit()
# 显示等式
sym.Eq(a, a.doit())



print("**************** 测试定积分 ******************** ")
e = sym.Integral(cos(x), (x, 0, sym.pi/2))
e
# 计算得到结果
e.doit()

print("**************** 求极限 ******************** ")
n = sym.Symbol('n')
s = ((n+3)/(n+2))**n

#无穷为两个小写o
sym.limit(s, x, sym.oo)


print("**************** 测试三角函数合并 ******************** ")
theta = symbols('theta')
a = cos(theta) * cos(theta) - sin(theta)*sin(theta)
Eq(a)


print("**************** 三角函数简化 ******************** ")
#可以调用 simplify 函数进行结果的简化，简直是太好用了！！！！

sym.simplify(a)


print("**************** 多元表达式 ******************** ")
x, y = sym.symbols('x y')
f = (x + 2)**2 + 5*y
sym.Eq(f)
#传入数值
f.evalf(subs = {x:1,y:2})

print("**************** 拓展 Latex 格式 ******************** ")
delta_t = sym.symbols('delta_t')
dt = sym.symbols('delta_t')
# 定义矩阵T
T = sym.Matrix(
    [[1, 0, 0, 0],
     [1, dt, dt**2, dt**3],
     [0, 1, 0, 0],
     [0, 1, 2*dt, 3*dt**2]])
T
# 计算行列式
sym.det(T)
# 矩阵求逆
T_inverse = T.inv()
# 逆矩阵
T_inverse

print("**************** 运动学公式自动推导 ******************** ")
# 定义符号
theta_1, theta_2,theta_3, l_2, l_3 = sym.symbols('theta_1, theta_2, theta_3, l_2, l_3')
theta_1
def RZ(theta):
    '''绕Z轴旋转'''
    return sym.Matrix(
    [[cos(theta), -sin(theta), 0, 0],
     [sin(theta), cos(theta), 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]])

def RX(gamma):
    '''绕X轴旋转'''
    return sym.Matrix([
        [1, 0, 0, 0],
        [0, cos(gamma), -sin(gamma), 0],
        [0, sin(gamma), cos(gamma), 0],
        [0, 0, 0, 1]])

def DX(x):
    '''绕X轴平移'''
    return sym.Matrix(
    [[1, 0, 0, x],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]])

def DZ(x):
    '''绕Z轴'''
    return sym.Matrix(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, x],
     [0, 0, 0, 1]])



#按照变换顺序，依次相乘，得到总的变换矩阵
T04 = RZ(theta_1)*RX(-sym.pi/2)*RZ(theta_2)*DX(l_2)*RZ(theta_3)*DX(l_3)

#公式简化，最终得到了三自由度机械臂， 正向运动学的结果
T04 = sym.simplify(T04)
T04



print("**************** 导出 Latex ******************** ")

#运算得到的结果可以直接插到论文里面，不用自己再手敲一遍 latex.

#直接导出结果的 latex 字符

print(latex(T_inverse))



import latexify

@latexify.with_latex
def f(x):
    return (3 * x + 5) / 2

print(f)



#=======================================================================================
#====================== 计算二项式 ========================



# SciPy 有两种方法来计算二项式系数。第一个函数称为 scipy.special.binom()。此函数通常有效地处理大值。
# 例如，

import scipy.special
print(scipy.special.binom(10,5))





# 返回二项式系数的第二个函数称为 scipy.special.comb()。
# 例如，

import scipy.special
print(scipy.special.comb(10,5))






# math 模块中的 comb() 函数返回给定值的组合，该组合本质上与二项式系数具有相同的公式。此方法是对 Python 3.8 及更高版本的最新版本的补充。
# 例如，

import math
print(math.comb(10,5))



# 我们可以使用 math 模块中的 fact() 函数来实现计算二项式系数的数学公式。
# 请参考下面的代码。

from math import factorial as fact

def binomial(n, r):
    return fact(n) // fact(r) // fact(n - r)

print(binomial(10,5))


#=======================================================================================
#====================== 解线性方程组 ========================



# m代表系数矩阵。
m = np.array([[1, -2, 1],
              [0, 2, -8],
              [-4, 5, 9]])

# v代表常数列
v = np.array([0, 8, -9])

# 解线性代数。
r = np.linalg.solve(m, v)

print("结果：")
name = ["X1", "X2", "X3"]
for i in range(len(name)):
    print(name[i] + "=" + str(r[i]))


# 1. 利用gekko的GEKKO求解
"""利用gekko求解线性方程组"""
from gekko import GEKKO

m = GEKKO()  # 定义模型
x = m.Var()  # 定义模型变量，初值为0
y = m.Var()
z = m.Var()
m.Equations([10 * x - y - 2 * z == 72,
             -x + 10 * y - 2 * z == 83,
             -x - y + 5 * z == 42, ])  # 方程组
m.solve(disp=False)  # 求解
x, y, z = x.value, y.value, z.value
print(x,y,z)  # 打印结果


# 2 . 利用scipy的linalg求解
from scipy import linalg
import numpy as np

A = np.array([[10, -1, -2], [-1, 10, -2], [-1, -1, 5]])  # A代表系数矩阵
b = np.array([72, 83, 42])  # b代表常数列
x = linalg.solve(A, b)
print(x)

# 3. 利用scipy.optimize的root或fsolve求解
from scipy.optimize import root, fsolve

def f(X):
    x = X[0]
    y = X[1]
    z = X[2]  # 切分变量

    return [10 * x - y - 2 * z - 72,
            -x + 10 * y - 2 * z - 83,
            -x - y + 5 * z - 42]

X0 = [1, 2, 3]  # 设定变量初值
m1 = root(f, X0).x  # 利用root求解并给出结果
m2 = fsolve(f, X0)  # 利用fsolve求解并给出结果

print(m1)
print(m2)


# 4. 利用Numpy的linalg求解
import numpy as np

A = np.array([[10, -1, -2], [-1, 10, -2], [-1, -1, 5]])  # A为系数矩阵
b = np.array([72, 83, 42])  # b为常数列
inv_A = np.linalg.inv(A)  # A的逆矩阵
x = inv_A.dot(b)  # A的逆矩阵与b做点积运算
x = np.linalg.solve(A, b) # 5,6两行也可以用本行替代
print(x)

import numpy as np

# A = np.mat([[10, -1, -2], [-1, 10, -2], [-1, -1, 5]])  # A为系数矩阵
# b = np.mat([[72], [83], [42]])  # b为常数列
A = np.mat("10, -1, -2; -1, 10, -2; -1, -1, 5")  # A为系数矩阵
b = np.mat("72;83;42")  # b为常数列
inv_A = np.linalg.inv(A)  # A的逆矩阵
inv_A = A.I  # A的逆矩阵
# x = inv_A.dot(b)  # A的逆矩阵与b做点积运算
x = np.linalg.solve(A, b)
print(x)

# 5. 利用sympy的solve和nsolve求解
# 5.1 利用solve求解所有精确解
from sympy import symbols, Eq, solve

x, y, z = symbols('x y z')
eqs = [Eq(10 * x - y - 2 * z, 72),
       Eq(-x + 10 * y - 2 * z, 83),
       Eq(-x - y + 5 * z, 42)]
print(solve(eqs, [x, y, z]))

# 5.1 利用nsolve求解数值解
from sympy import symbols, Eq, nsolve

x, y, z = symbols('x y z')
eqs = [Eq(10 * x - y - 2 * z, 72),
       Eq(-x + 10 * y - 2 * z, 83),
       Eq(-x - y + 5 * z, 42)]
initialValue = [1, 2, 3]
print(nsolve(eqs, [x, y, z], initialValue))

















