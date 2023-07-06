#!/usr/bin/env python3
#!-*-coding=utf-8-*-
#########################################################################
# File Name: mathmatic.py
# Author: 陈俊杰
# Created Time: 2021年07月23日 星期五 08时27分03秒

# mail: 2716705056@qq.com
# 此程序的功能是： Sympy和Numpy、scipy

"""
https://blog.csdn.net/handsomeswp/article/details/111061087

https://zhuanlan.zhihu.com/p/60509430

https://blog.csdn.net/lanchunhui/article/details/49979411

https://zhuanlan.zhihu.com/p/111573239

https://www.cnblogs.com/sunshine-blog/p/8477523.html
"""
#########################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, time, math
import scipy
import sympy as sy
from IPython.display import display, Latex
sy.init_printing()

x,y = sy.symbols("x y")
print("***********  print(type(x))  *******************")
print(type(x))




x,y=sy.symbols("x y")
f=x**2+3*x-5
print("**************** 1 ******************** ")
print(f"f(x)={f}")

print("**************** 2 ******************** ")
display(Latex(f"$$f(x)={sy.latex(f)}$$"))

xx=3
print("**************** 3 ******************** ")
print(f"f({xx})={f.subs({x:xx})}")

print("**************** 4 先 init_printing() 再 pprint() 就能以更易懂的方式输出符号表达式。 ******************** ")
sy.pprint(f"f({xx})={f.subs({x:xx})}")


yy=4
f1=sy.sqrt(x**2+y**2)
print("**************** 5 ******************** ")
display(Latex(f"$$f_1(x,y)={sy.latex(f1)}$$"))
print("**************** 6 ******************** ")
sy.pprint(f"f1({xx},{yy})={f1.subs({x:xx, y:yy})}")


a,x,t = sy.symbols("a x t")
f = sy.cos(x)

print("**************** 7 ******************** ")
print(f"f(t)={f.subs({x:a**t})}")


print("**************** 8 ******************** ")
y = sy.Rational(1,2) + sy.sqrt(2)
print(f"y={y}")
print(f"y={y.evalf()}")


print("**************** 9 ******************** ")
y = x**2 + sy.Rational(1,2)
sy.pprint(y)
value_x = 1
print(f"y={y.subs({x:value_x}).evalf()}")

print("**************** https://zhuanlan.zhihu.com/p/60509430 ******************** ")
print("**************** 数学符号与表达式 ******************** ")
print(math.sqrt(8))
print(sy.sqrt(8))


x = sy.Symbol('x')
y = sy.Symbol('y')
k, m, n = sy.symbols('k m n')
print(3*x+y**3)



print("**************** 折叠与展开表达式 ******************** ")
"""
factor() 函数可以折叠表达式，而 expand() 函数可以展开表达式，
比如表达式： [公式] ，折叠之后应该是 [公式] 。我们来看具体的代码：
"""
x,y = sy.symbols('x y')
expr=x**4+x*y+8*x
f_expr=sy.factor(expr)
e_expr=sy.expand(f_expr)
print(f_expr)
print(e_expr)


print("**************** 表达式化简 ******************** ")
"""
simplify () 函数可以对表达式进行化简。有一些表达式看起来会比较复杂，
就拿人教版初二上的一道多项式的乘法为例，简化 [公式] 。
"""
x,y = sy.symbols('x y')
expr=(2*x)**3*(-5*x*y**2)
s_expr=sy.simplify(expr)
print(s_expr)


print("**************** 解一元一次方程 ******************** ")
"""
我们来求解这个一元一次方程组。(题目来源于人教版七年级数学上) [公式]
"""
x = sy.Symbol('x')
print(sy.solve(6*x + 6*(x-2000)-150000,x))

print("**************** 解二元一次方程组 ******************** ")

x,y = sy.symbols('x y')
print(sy.solve([x + y-10,2*x+y-16],[x,y]))


print("**************** 解三元一次方程组 ******************** ")
x,y = sy.symbols('x y')
a,b,c=sy.symbols('a b c')
expr=a*x**2 + b*x + c
s_expr=sy.solve( expr, x)
print(s_expr)


print("**************** 微积分 Calculus ******************** ")
"""
微积分是大学高等数学里非常重要的学习内容，比如求极限、导数、微分、不定积分、定积分等都是可以使用 Sympy 来运算的。
求极限 Sympy 是使用 limit (表达式，变量，极限值) 函数来求极限的，比如我们要求
lim_{x->0}{sin(x)/x}
的值。
"""
x, y, z = sy.symbols('x y z')
expr = sy.sin(x)/x
l_expr=sy.limit(expr, x, 0)
print(l_expr)


print("**************** 求导 ******************** ")
x,y = sy.symbols('x y')
expr=sy.sin(x)*sy.exp(x)
diff_expr=sy.diff(expr, x)
diff_expr2=sy.diff(expr,x,2)
print(diff_expr)
print(diff_expr2)


print("**************** 求不定积分 ******************** ")

x,y = sy.symbols('x y')
expr=sy.exp(x)*sy.sin(x) + sy.exp(x)*sy.cos(x)
i_expr=sy.integrate(expr,x)
print(i_expr)

print("**************** 求定积分 ******************** ")
"""
Sympy 同样是使用 integrate () 函数来做定积分的求解，只是语法不同：integrate (表达式，（变量，下区间，上区间))，我们来看如果求解
"""
x,y = sy.symbols('x y')
expr=sy.sin(x**2)
i_expr=sy.integrate(expr, (x, -np.inf,np.inf))
print(i_expr)

print("**************** https://zhuanlan.zhihu.com/p/111573239 ******************** ")
from sympy import *
print(sy.sin(sy.pi))


print("**************** 新建符号 ******************** ")
# 新建符号 x, y
x, y = symbols('x y')
#还有一个更简洁的方法是，利用 SymPy 的 abc 子模块导入所有拉丁、希腊字母：
from sympy.abc import x, y
x = symbols('x', positive = True)
vars = symbols('x_1:5')

print(vars[0])

x, y, z = symbols('x y z')
y = expand((x + 1)**2) # expand() 是展开函数
print(y)
z = Rational(1, 2) # 构造分数 1/2
print(z)

print("**************** 替换 ******************** ")
x = symbols('x')
expr = cos(x) + 1
print(expr.subs(x, 0))


print("**************** 利用 sympify 函数可以将字符串表达式转换为 SymPy 表达式。 ******************** ")
# 注意：sympify 是符号化，与另一个函数 simplify （化简）拼写相近，不要混淆。

str_expr = 'x**2 + 2*x + 1'
expr = sympify(str_expr)
print(expr)

print("****************转换为指定精度的数值解 ******************** ")

pi.evalf(3) # pi 保留 3 位有效数字
print(pi)

print("**************** 替换 ******************** ")

a = np.pi / 3
x = symbols('x')
expr = sin(x)
f = lambdify(x, expr, 'numpy')
print(f(a))
print(expr.subs(x, pi/3))


print("**************** 使用 simplify (化简) ******************** ")
print(simplify(sin(x)**2 + cos(x)**2))

alpha  = symbols('alpha ')
print(simplify(2*sin(alpha )*cos(alpha)))

print("**************** 多项式和有理函数化简 ******************** ")
x_1 = symbols('x_1')
print(expand((x_1 + 1)**2))


print("**************** factor (因式分解) ******************** ")

print(factor(x**3 - x**2 + x - 1))

print("**************** collect (合并同类项) ******************** ")

expr = x*y + x - 3 + 2*x**2 - z*x**2 + x**3
print(collect(expr, x))

print("**************** cancel (有理分式化简) ******************** ")
print(cancel((x**2 + 2*x + 1)/(x**2 + x)))


print("**************** apart (部分分式展开), 使用 apart 函数可以将分式展开，例如：******************** ")
expr = (4*x**3 + 21*x**2 + 10*x + 12)/(x**4 + 5*x**3 + 5*x**2 + 4*x)
print(expr)


print("**************** 微积分符号计算 ******************** ")


print("**************** 一元函数求导函数 ******************** ")
# 求一阶导数
print(diff(cos(x), x))

# 求 3 阶导数
print(diff(x**4, x, 3))

#我们也可以用 符号变量的 diff 方法 求微分，例如：
expr = cos(x)
print(expr.diff(x, 2))

print("**************** 多元函数求偏导函数 ******************** ")
expr = exp(x*y*z)
print(diff(expr, x))

print("**************** integrate (积分) ******************** ")
# 求不定积分
print(integrate(cos(x), x))

#求 [公式] 的定积分：注意：在 SymPy 中，我们用 'oo' 表示 [公式] 。
print(integrate(exp(-x), (x, 0, oo)))

# 求函数 [公式] 在 [公式] 的二重积分：
#integrate(exp(-x**2 - y**2), (x, -oo, oo), (y, -oo, oo))

print("**************** limit (求极限) ******************** ")
# 使用 limit 函数求极限，例如：
print(limit(sin(x)/x, x, 0))

print(limit(1/x, x, 0, '+'))
print("**************** series (级数展开) ******************** ")
expr = sin(x)
print(expr.series(x, 0, 4))

print("**************** 解方程 使用 solveset 求解方程。 ******************** ")
Eq(x**2 - x, 0)
print(solveset(Eq(x**2 - x, 0), x, domain = S.Reals))

print("**************** 求解微分方程 ******************** ")
f = symbols('f', cls = Function)
diffeq = Eq(f(x).diff(x, 2) - 2*f(x).diff(x) + f(x), sin(x))
print(diffeq)
print(dsolve(diffeq, f(x)))

print("**************** 矩阵运算******************** ")
# 构造矩阵
print(f"Matrix([[1, -1], [3, 4], [0, 2]])")

# 构造列向量
print(Matrix([1, 2, 3]))


# 构造行向量
print(Matrix([[1], [2], [3]]).T)


# 构造单位矩阵
eye(4)

# 构造零矩阵
zeros(4)

# 构造壹矩阵
ones(4)

# 构造对角矩阵
diag(1, 2, 3, 4)

#矩阵转置用矩阵变量的 T 方法。例如：
a = Matrix([[1, -1], [3, 4], [0, 2]])
a

# 求矩阵 a 的转置
a.T


# 求矩阵 M 的 2 次幂
M = Matrix([[1, 3], [-2, 3]])
M**2

# 求矩阵 M 的逆
M**-1

#用矩阵变量的 det 方法可以求其行列式：
M = Matrix([[1, 0, 1], [2, -1, 3], [4, 3, 2]])
M.det()


#用矩阵变量的 eigenvals 和 charpoly 方法求其特征值和特征多项式。
M = Matrix([[3, -2,  4, -2], [5,  3, -3, -2], [5, -2,  2, -2], [5, -2, -3,  3]])
M
M.eigenvals()
{3: 1, -2: 1, 5: 2}
lamda = symbols('lamda')
p = M.charpoly(lamda)
factor(p)

print("**************** 矩阵运算 矩阵乘法******************** ")

a_11,a_12,a_21,a_22 = sy.symbols('a_11 a_12 a_21 a_22')
b_11,b_12,b_21,b_22 = sy.symbols('b_11 b_12 b_21 b_22')
a = sy.Matrix([[a_11,a_12], [a_21,a_22]])
b = sy.Matrix([[b_11,b_12], [b_21,b_22]])
a*b




a1 = sy.Matrix([[0,1], [1,0]])
b1 = sy.Matrix([[0,-1j], [1j,0]])
a1*b1

print("**************** 矩阵张量积 ******************** ")

from sympy.physics.quantum import TensorProduct
a_11,a_12,a_21,a_22 = sy.symbols('a_11 a_12 a_21 a_22')
b_11,b_12,b_21,b_22 = sy.symbols('b_11 b_12 b_21 b_22')
A = sy.Matrix([[a_11,a_12],[a_21,a_22]])
B = sy.Matrix([[b_11,b_12],[b_21,b_22]])
C = TensorProduct(A,B)

display(Latex(f"$${sy.latex(C)}$$"))


#可以利用 laplace_transform 函数进行 Laplace 变换，例如：
#  Laplace (拉普拉斯)变换
from sympy.abc import t, s
expr = sin(t)
laplace_transform(expr, t, s)

#利用 inverse_laplace_transform 函数进行逆 Laplace 变换：
expr = 1/(s - 1)
inverse_laplace_transform(expr, s, t)
print("**************** 利用 SymPy 画函数图像 使用 plot 函数绘制二维函数图像，例如：******************** ")
from sympy.plotting import plot
from sympy.abc import x
plot(x**2, (x, -2, 2))


from sympy import plot_implicit
from sympy import Eq
from sympy.abc import x, y
plot_implicit(Eq(x**2 + y**2, 1))

print("**************** 输出运算结果的 Latex 代码 ******************** ")
print(latex(integrate(sqrt(x), x)))


print("**************** 使用 SymPy 画出三维函数图像，例如： ******************** ")

from sympy.plotting import plot3d
from sympy.abc import x, y
from sympy import exp
plot3d(x*exp(-x**2 - y**2), (x, -3, 3), (y, -2, 2))

print("**************** 如果是链式求导，sympy 该怎么做呢 ******************** ")
from sympy import *
r, t = symbols('r t') # r (radius), t (angle theta)
f = symbols('f', cls = Function)
x = r * cos(t)
y = r * sin(t)
g = f(x, y)
Derivative(g, r, 1).doit()



# https://zhuanlan.zhihu.com/p/83822118
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

















