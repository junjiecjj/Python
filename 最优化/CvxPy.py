#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 00:08:14 2022

@author: jack

与matlab中cvx的工具包类似，用于求解凸优化问题。cvx与cvxpy都是由CIT的Stephen Boyd教授课题组开发。
cvx用于matlab的包，cvxpy是用于python的包。下载、安装及学习地址如下：

cvxpy是解决凸优化问题的，在使用之前要确保目标函数是一个凸优化问题(包括其中的变量范围设置，参数设置等)

https://blog.csdn.net/geekwill/article/details/78836054


CVXPY
CVX是由Michael Grant和Stephen Boyd开发的用于构造和解决严格的凸规划(DCP)的建模系统，建立在Löfberg (YALMIP)， Dahl和Vandenberghe (CVXOPT)的工作上。

CVX支持的问题类型
Linear programs (LPs)
Quadratic programs (QPs)
Second-order cone programs (SOCPs)
Semidefinite programs (SDPs)
还可以解决更复杂的凸优化问题，包括

不可微函数，如L1范数
使用CVX方便地表述和解决约束范数最小化、熵最大化、行列式最大化，etc.
支持求解mixed integer
disciplined convex programs (MIDCPs)
CVX使用的注意事项
CVX不是用来检查你的问题是否凸的工具。如果在将问题输入CVX之前不能确定问题是凸的，那么您使用工具的方法不正确，您的努力很可能会失败。
CVX不是用来解决大规模问题的。它是一个很好的工具，用于试验和原型化凸优化问题。
CVX本身并不解决任何优化问题。它只将问题重新表述成一种形式(SDP和SOCP)，需要搭配solver求解。
在CVX中，目标函数必须是凸的，约束函数必须是凸的或仿射的，diag 和trace 是仿射函数。 CVX使用了扩展值函数(即凸函数在其域外取值为
∞ \infin∞ ，凹函数取值为− ∞ -\infin−∞)。



https://blog.csdn.net/qq_21747841/article/details/78457395

https://blog.csdn.net/geekwill/article/details/78836054

https://www.cvxpy.org/tutorial/functions/index.html

https://blog.csdn.net/fan_h_l/article/details/81981715

https://zhuanlan.zhihu.com/p/410478494

https://www.zhihu.com/question/59378236

https://blog.csdn.net/weixin_43464554/article/details/121280322

https://zhuanlan.zhihu.com/p/114849004

"""



"""

https://blog.csdn.net/abc200941410128/article/details/111246026

scipy库是个功能很强大的包，可以通过调用optimize.linprog函数解决简单的线性规划：

scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
bounds=None, method=‘simplex’, callback=None, options=None)

c指的应该是要求最大值的函数的系数数组，A_ub是应该是不等式未知量的系数矩阵，【注意：来这不等式指的是<=的不等式，那如果是>=，就需要乘个负号】。A_eq就是其中等式的未知量系数矩阵了。B_ub就是不等式的右边了，B_eq就是等式右边了。bounds的话，指的就是每个未知量的范围了。


"""
from scipy import optimize as op
import numpy as np
c=np.array([2,3,-5])
A_ub=np.array([[-2,5,-1],[1,3,1]])#注意是-2，5，-1
B_ub=np.array([-10,12])
A_eq=np.array([[1,1,1]])
B_eq=np.array([7])
# 上限7是根据约束条件1和4得出的
x1=(0,7)
x2=(0,7)
x3=(0,7)
res=op.linprog(-c,A_ub,B_ub,A_eq,B_eq,bounds=(x1,x2,x3))
print(res)


# https://weihuang.blog.csdn.net/article/details/82834888?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-82834888-blog-122233430.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-82834888-blog-122233430.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=2
#导入sympy包，用于求导，方程组求解等等
from sympy import *

#设置变量
x1 = symbols("x1")
x2 = symbols("x2")
alpha = symbols("alpha")
beta = symbols("beta")

#构造拉格朗日等式
L = 10 - x1*x1 - x2*x2 + alpha * (x1*x1 - x2) + beta * (x1 + x2)

#求导，构造KKT条件
difyL_x1 = diff(L, x1)  #对变量x1求导
difyL_x2 = diff(L, x2)  #对变量x2求导
difyL_beta = diff(L, beta)  #对乘子beta求导
dualCpt = alpha * (x1 * x1 - x2)  #对偶互补条件

#求解KKT等式
aa = solve([difyL_x1, difyL_x2, difyL_beta, dualCpt], [x1, x2, alpha, beta])

#打印结果，还需验证alpha>=0和不等式约束<=0
for i in aa:
    if i[2] >= 0:
        if (i[0]**2 - i[1]) <= 0:
            print(i)



#====================================================
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





#========================================================================================
# https://blog.csdn.net/weixin_43464554/article/details/121280322
# Import packages.
import cvxpy as cp
import numpy as np

# Generate data.
m = 20
n = 15
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Define and solve the CVXPY problem.
x = cp.Variable(n)
cost = cp.sum_squares(A @ x - b)
prob = cp.Problem(cp.Minimize(cost))
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("The optimal x is")
print(x.value)
print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)


#==================================================================================
# https://zhuanlan.zhihu.com/p/410478494

import cvxpy as cp
import numpy as np

# Problem data.
m = 30
n = 20
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Construct the problem.
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A@x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()

print("status:", prob.status)
# The optimal value for x is stored in `x.value`.
print(x.value)
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(constraints[0].dual_value)


#========================================================================================
# https://zhuanlan.zhihu.com/p/114849004

import cvxpy as cp

# Create two scalar optimization variables.
# 在CVXPY中变量有标量(只有数值大小)，向量，矩阵。
# 在CVXPY中有常量(见下文的Parameter)
x = cp.Variable() # 定义变量x,定义变量y。两个都是标量
y = cp.Variable()
# Create two constraints.
# 定义两个约束式
constraints = [x + y == 1,
              x - y >= 1]
# 优化的目标函数
obj = cp.Minimize(cp.square(x - y))
# 把目标函数与约束传进Problem函数中
prob = cp.Problem(obj, constraints)
prob.solve()  # Returns the optimal value.
print("status:", prob.status)
print("optimal value", prob.value) # 最优值
print("optimal var", x.value, y.value) # x与y的解
# 状态域被赋予'optimal'，说明这个问题被成功解决。
# 最优值是针对所有满足约束条件的变量x,y中目标函数的最小值
# prob.solve()返回最优值，同时更新prob.status,prob.value,和所有变量的值。

import cvxpy as cvx
# Solving a problem with different solvers.
x = cvx.Variable(2)
obj = cvx.Minimize(x[0] + cvx.norm(x, 1))
constraints = [x >= 2]
prob = cvx.Problem(obj, constraints)

# Solve with ECOS.
prob.solve(solver=cvx.ECOS)
print("optimal value with ECOS:", prob.value)

# Solve with ECOS_BB.
prob.solve(solver=cvx.ECOS_BB)
print("optimal value with ECOS_BB:", prob.value)

# Solve with CVXOPT.
prob.solve(solver=cvx.CVXOPT)
print("optimal value with CVXOPT:", prob.value)

# Solve with SCS.
prob.solve(solver=cvx.SCS)
print("optimal value with SCS:", prob.value)

# Solve with GLPK.
prob.solve(solver=cvx.GLPK)
print("optimal value with GLPK:", prob.value)

# Solve with GLPK_MI.
prob.solve(solver=cvx.GLPK_MI)
print("optimal value with GLPK_MI:", prob.value)

# Solve with GUROBI.
prob.solve(solver=cvx.GUROBI)
print("optimal value with GUROBI:", prob.value)

# Solve with MOSEK.
prob.solve(solver=cvx.MOSEK)
print("optimal value with MOSEK:", prob.value)

# Solve with Elemental.
prob.solve(solver=cvx.ELEMENTAL)
print("optimal value with Elemental:", prob.value)

# Solve with CBC.
prob.solve(solver=cvx.CBC)
print("optimal value with CBC:", prob.value)

# Use the installed_solvers utility function to get a list of the solvers your installation of CVXPY //supports.

print(cvx.installed_solvers())

#========================================================================================
# https://zhuanlan.zhihu.com/p/114849004
import numpy as np
np.random.seed(1)
n = 10
mu = np.abs(np.random.randn(n, 1))
Sigma = np.random.randn(n, n)
Sigma = Sigma.T.dot(Sigma)
# Long only portfolio optimization.
import cvxpy as cp


w = cp.Variable(n)
gamma = cp.Parameter(nonneg=True)
ret = mu.T*w
risk = cp.quad_form(w, Sigma)
prob = cp.Problem(cp.Maximize(ret - gamma*risk),
               [cp.sum(w) == 1,
                w >= 0])
# Compute trade-off curve.
SAMPLES = 100
risk_data = np.zeros(SAMPLES)
ret_data = np.zeros(SAMPLES)
gamma_vals = np.logspace(-2, 3, num=SAMPLES)
for i in range(SAMPLES):
    gamma.value = gamma_vals[i]
    prob.solve()
    risk_data[i] = cp.sqrt(risk).value
    ret_data[i] = ret.value
# Plot long only trade-off curve.
import matplotlib.pyplot as plt

markers_on = [29, 40]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(risk_data, ret_data, 'g-')
for marker in markers_on:
    plt.plot(risk_data[marker], ret_data[marker], 'bs')
    ax.annotate(r"$\gamma = %.2f$" % gamma_vals[marker], xy=(risk_data[marker]+.08, ret_data[marker]-.03))
for i in range(n):
    plt.plot(cp.sqrt(Sigma[i,i]).value, mu[i], 'ro')
plt.xlabel('Standard deviation')
plt.ylabel('Return')
plt.show()





#=================================================================
# https://zhuanlan.zhihu.com/p/410478494
# cvxpy最小二乘法求解过程
#=================================================================

import cvxpy as cp
import numpy as np

# Problem data.
m = 30
n = 20
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

# Construct the problem.
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A@x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve() # result就是最小二乘解
# The optimal value for x is stored in `x.value`.

print("status:", prob.status)
print(x.value) # x就是对应最优解的x
# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
print(constraints[0].dual_value)


#=================================================================
# https://www.cvxpy.org/examples/basic/sdp.html
# cvxpy  Semidefinite program
#=================================================================


# Import packages.
import cvxpy as cp
import numpy as np

# Generate a random SDP.
n = 3
p = 3
np.random.seed(1)
C = np.random.randn(n, n)
A = []
b = []
for i in range(p):
    A.append(np.random.randn(n, n))
    b.append(np.random.randn())

# Define and solve the CVXPY problem.
# Create a symmetric matrix variable.
X = cp.Variable((n,n), symmetric=True)
# The operator >> denotes matrix inequality.
constraints = [X >> 0]
constraints += [
    cp.trace(A[i] @ X) == b[i] for i in range(p)
]
prob = cp.Problem(cp.Minimize(cp.trace(C @ X)),
                  constraints)
prob.solve()

# Print result.
print("The optimal value is", prob.value)
print("A solution X is")
print(X.value)

#=================================================================
# https://www.cvxpy.org/examples/applications/water_filling_BVex5.2.html
# cvxpy  Water Filling in Communications
#=================================================================

import numpy as np
import cvxpy as cp

def water_filling(n, a, sum_x=1):
    '''
    Boyd and Vandenberghe, Convex Optimization, example 5.2 page 145
    Water-filling.

    This problem arises in information theory, in allocating power to a set of
    n communication channels in order to maximise the total channel capacity.
    The variable x_i represents the transmitter power allocated to the ith channel,
    and log(α_i+x_i) gives the capacity or maximum communication rate of the channel.
    The objective is to minimise -∑log(α_i+x_i) subject to the constraint ∑x_i = 1
    '''

    # Declare variables and parameters
    x = cp.Variable(shape=n)
    alpha = cp.Parameter(n, nonneg=True)
    alpha.value = a

    # Choose objective function. Interpret as maximising the total communication rate of all the channels
    obj = cp.Maximize(cp.sum(cp.log(alpha + x)))

    # Declare constraints
    constraints = [x >= 0, cp.sum(x) - sum_x == 0]

    # Solve
    prob = cp.Problem(obj, constraints)
    prob.solve()
    if(prob.status=='optimal'):
        return prob.status, prob.value, x.value
    else:
        return prob.status, np.nan, np.nan


# As an example, we will solve the water filling problem with 3 buckets, each with different α
np.set_printoptions(precision=3)
buckets = 3
alpha = np.array([0.8, 1.0, 1.2])


stat, prob, x = water_filling(buckets, alpha)
print('Problem status: {}'.format(stat))
print('Optimal communication rate = {:.4g} '.format(prob))
print('Transmitter powers:\n{}'.format(x))

import matplotlib
import matplotlib.pylab as plt


matplotlib.rcParams.update({'font.size': 14})

axis = np.arange(0.5,buckets+1.5,1)
index = axis+0.5
X = x.copy()
Y = alpha + X

# to include the last data point as a step, we need to repeat it
A = np.concatenate((alpha,[alpha[-1]]))
X = np.concatenate((X,[X[-1]]))
Y = np.concatenate((Y,[Y[-1]]))

plt.xticks(index)
plt.xlim(0.5,buckets+0.5)
plt.ylim(0,1.5)
plt.step(axis,A,where='post',label =r'$\alpha$',lw=2)
plt.step(axis,Y,where='post',label=r'$\alpha + x$',lw=2)
plt.legend(loc='lower right')
plt.xlabel('Bucket Number')
plt.ylabel('Power Level')
plt.title('Water Filling Solution')
plt.show()


#=================================================================
# https://www.cvxpy.org/examples/applications/max_entropy.html
# cvxpy  Entropy maximization
#=================================================================


import cvxpy as cp
import numpy as np

# Make random input repeatable.
np.random.seed(0)

# Matrix size parameters.
n = 20
m = 10
p = 5

# Generate random problem data.
tmp = np.random.rand(n)
A = np.random.randn(m, n)
b = A.dot(tmp)
F = np.random.randn(p, n)
g = F.dot(tmp) + np.random.rand(p)

# Entropy maximization.
x = cp.Variable(shape=n)
obj = cp.Maximize(cp.sum(cp.entr(x)))
constraints = [A*x == b,
               F*x <= g ]
prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.CVXOPT, verbose=True)

# Print result.
print("\nThe optimal value is:", prob.value)
print('\nThe optimal solution is:')
print(x.value)



#=================================================================
# https://www.cvxpy.org/examples/applications/Channel_capacity_BV4.57.html
# cvxpy  Capacity of a Communication Channel
#=================================================================

import cvxpy as cp
import numpy as np
import math
from scipy.special import xlogy


def channel_capacity(n, m, P, sum_x=1):
    '''
    Boyd and Vandenberghe, Convex Optimization, exercise 4.57 page 207
    Capacity of a communication channel.

    We consider a communication channel, with input X(t)∈{1,..,n} and
    output Y(t)∈{1,...,m}, for t=1,2,... .The relation between the
    input and output is given statistically:
    p_(i,j) = ℙ(Y(t)=i|X(t)=j), i=1,..,m  j=1,...,n

    The matrix P ∈ ℝ^(m*n) is called the channel transition matrix, and
    the channel is called a discrete memoryless channel. Assuming X has a
    probability distribution denoted x ∈ ℝ^n, i.e.,
    x_j = ℙ(X=j), j=1,...,n

    The mutual information between X and Y is given by
    ∑(∑(x_j p_(i,j)log_2(p_(i,j)/∑(x_k p_(i,k)))))
    Then channel capacity C is given by
    C = sup I(X;Y).
    With a variable change of y = Px this becomes
    I(X;Y)=  c^T x - ∑(y_i log_2 y_i)
    where c_j = ∑(p_(i,j)log_2(p_(i,j)))
    '''

    # n is the number of different input values
    # m is the number of different output values
    if n*m == 0:
        print('The range of both input and output values must be greater than zero')
        return 'failed', np.nan, np.nan

    # x is probability distribution of the input signal X(t)
    x = cp.Variable(shape=n)

    # y is the probability distribution of the output signal Y(t)
    # P is the channel transition matrix
    y = P@x

    # I is the mutual information between x and y
    c = np.sum(np.array((xlogy(P, P) / math.log(2))), axis=0)
    I = c@x + cp.sum(cp.entr(y) / math.log(2))

    # Channel capacity maximised by maximising the mutual information
    obj = cp.Maximize(I)
    constraints = [cp.sum(x) == sum_x,x >= 0]

    # Form and solve problem
    prob = cp.Problem(obj,constraints)
    prob.solve()
    if prob.status=='optimal':
        return prob.status, prob.value, x.value
    else:
        return prob.status, np.nan, np.nan


np.set_printoptions(precision=3)
n = 2
m = 2
P = np.array([[0.75,0.25],
             [0.25,0.75]])
stat, C, x = channel_capacity(n, m, P)
print('Problem status: ',stat)
print('Optimal value of C = {:.4g}'.format(C))
print('Optimal variable x = \n', x)


#=================================================================
# 非线性规划
# https://blog.csdn.net/qq_20144897/article/details/125395218
#=================================================================


# 决策变量
n = 1
z = 8
x = cp.Variable(n,integer = True)
y = cp.Variable(z,integer = True)
prob = cp.Problem(cp.Minimize(x),
                  [x - 2 * y[0] == 1,
                   x - 3 * y[1] == 0,
                   x - 4 * y[2] == 1,
                   x - 5 * y[3] == 4,
                   x - 6 * y[4] == 3,
                   x - 7 * y[5] == 4,
                   x - 8 * y[6] == 1,
                   x - 9 * y[7] == 0,
                   x >= 0,y >= 0])
# 求解
ans = prob.solve() #solver='GLPK_MI'
# 输出结果
print("目标函数最小值:", ans)
print(x.value)

#==================================================================

#'CVXOPT', 'ECOS', 'ECOS_BB', 'GLOP', 'GLPK', 'GLPK_MI', 'OSQP', 'PDLP', 'SCIPY', 'SCS'
# 决策变量
n = 5
x = cp.Variable(n,integer = True)
# 约束1
A1 = np.array([[1,1,1,1,1],
               [1,2,2,1,6],
               [2,1,6,0,0],
               [0,0,1,1,5]])
b1 = np.array([400,800,200,200])
prob = cp.Problem(cp.Minimize(x[0]**2+x[1]**2+3*x[2]**2+4*x[3]**2+2*x[4]**2-8*x[0]-2*x[1]-3*x[2]-x[3]-2*x[4]),
                  [A1 @ x <= b1,
                   x >= 0,
                   x <= 99])
# 求解
ans = prob.solve()#solver='CPLEX'
# 输出结果
print("目标函数最小值:", ans)
print(x.value)

#==================================================================

# 决策变量
n = 100
x = cp.Variable(n,nonneg=True)
constraints = [np.arange(100,0,-1) @ x <= 1000,x >= 0]
for i in range(0,4):
    constraints += [np.arange(1,i+2,1) @ x[0:(i+1)] <= 10 * (i+1)]
prob = cp.Problem(cp.Maximize(cp.sum(x ** 0.5)),constraints)
# 求解
ans = prob.solve() #  solver='CPLEX'
# 输出结果
print("目标函数最大值:", ans)
print(x.value)


#==================================================================
n = 3
x = cp.Variable(n)
prob = cp.Problem(cp.Maximize(2*x[0]+3*x[0]**2+3*x[1]+x[1]**2+x[2]),
                  [x[0]+2*x[0]**2+x[1]+2*x[1]**2+x[2] <= 10,
                   x[0]+x[0]**2+x[1]+x[1]**2-x[2] <= 50,
                   2*x[0]+x[0]**2+2*x[1]+x[2] <= 40,
                   x[0]**2+x[2] == 2,
                   x[0]+2*x[1] >= 1,
                   x[0] >= 0])
# 求解
ans = prob.solve(solver='CPLEX') # solver='CPLEX'
# 输出结果
print("目标函数最大值:", ans)
print(x.value)




#==================================================================






#==================================================================






























































































































































































