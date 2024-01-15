#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 00:08:14 2022

@author: jack

## 官方网址
https://www.cvxpy.org/examples/index.html
## 中文教程
https://www.wuzao.com/document/cvxpy/index.html




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

scipy.optimize.linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method=‘simplex’, callback=None, options=None)

c指的应该是要求最大值的函数的系数数组，A_ub是应该是不等式未知量的系数矩阵，【注意：来这不等式指的是<=的不等式，那如果是>=，就需要乘个负号】。A_eq就是其中等式的未知量系数矩阵了。B_ub就是不等式的右边了，B_eq就是等式右边了。bounds的话，指的就是每个未知量的范围了。


"""

import cvxpy as cp
## 查看已经安装的优化器
print(cp.installed_solvers())
# ['CLARABEL', 'ECOS', 'ECOS_BB', 'GUROBI', 'MOSEK', 'OSQP', 'SCIPY', 'SCS']


 # 要获取描述二分法的详细输出，可以将关键字参数 verbose=True 传递给求解方法（ problem.solve(qcp=True, verbose=True) ）。

##====================================================================================
##      最小二乘问题
##====================================================================================

import cvxpy as cp
import numpy

# Problem data.
m = 10
n = 5
numpy.random.seed(1)
A = numpy.random.randn(m, n)
b = numpy.random.randn(m)

# Construct the problem.
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)

print("Optimal value", prob.solve())
print("Optimal var")
print(x.value) # A numpy ndarray.


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
objective = cp.Minimize(cp.sum_squares(A @ x - b))
constraints = [0 <= x, x <= 1]
prob = cp.Problem(objective, constraints)
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("The optimal x is")
print(x.value)
print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)



#=================================================================
#                   https://zhuanlan.zhihu.com/p/410478494
#                       cvxpy最小二乘法求解过程
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




##====================================================================================
##                                   LASSO  问题
##====================================================================================

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Problem data.
n = 15
m = 10
np.random.seed(1)
A = np.random.randn(n, m)
b = np.random.randn(n)
# gamma must be nonnegative due to DCP rules.
gamma = cp.Parameter(nonneg=True)

# Construct the problem.
x = cp.Variable(m)
error = cp.sum_squares(A @ x - b)
obj = cp.Minimize(error + gamma*cp.norm(x, 1))
prob = cp.Problem(obj)

# Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
sq_penalty = []
l1_penalty = []
x_values = []
gamma_vals = np.logspace(-4, 6)
for val in gamma_vals:
    gamma.value = val
    prob.solve()
    # Use expr.value to get the numerical value of
    # an expression in the problem.
    sq_penalty.append(error.value)
    l1_penalty.append(cp.norm(x, 1).value)
    x_values.append(x.value)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig, axs = plt.subplots(2, 1, figsize = (8, 8), constrained_layout = True)

# Plot trade-off curve.
axs[0].plot(l1_penalty, sq_penalty)
axs[0].set_xlabel(r'$\|x\|_1$', fontsize=16)
axs[0].set_ylabel(r'$\|Ax-b\|^2$', fontsize=16)
axs[0].set_title('Trade-Off Curve for LASSO', fontsize=16)

# Plot entries of x vs. gamma.
for i in range(m):
    axs[1].plot(gamma_vals, [xi[i] for xi in x_values])
axs[1].set_xlabel(r'$\gamma$', fontsize=16)
axs[1].set_ylabel(r'$x_{i}$', fontsize=16)
axs[1].set_xscale('log')
axs[1].set_title( "Entries of x vs. " + r'$\gamma$', fontsize=16)

plt.tight_layout()
out_fig = plt.gcf()
out_fig.savefig('/home/jack/snap/lasso.eps',   bbox_inches = 'tight')
plt.close()


##====================================================================================
##                Disciplined quasiconvex programming (DQCP)
##====================================================================================
import cvxpy as cp

x = cp.Variable()
y = cp.Variable(pos=True)
objective_fn = -cp.sqrt(x) / y
problem = cp.Problem(cp.Minimize(objective_fn), [cp.exp(x) <= y])
problem.solve(qcp=True,  solver= cp.SCS)
assert problem.is_dqcp()
print("Optimal value: ", problem.value)
print("x: ", x.value)
print("y: ", y.value)


##  调用 object.is_dqcp() 来检查问题、约束条件或目标函数是否符合DQCP规则
import cvxpy as cp

# 变量的符号会影响曲率分析。
x = cp.Variable(nonneg=True)
concave_fractional_fn = x * cp.sqrt(x)
constraint = [cp.ceil(x) <= 10]
problem = cp.Problem(cp.Maximize(concave_fractional_fn), constraint)
assert concave_fractional_fn.is_quasiconcave()
assert constraint[0].is_dqcp()
assert problem.is_dqcp()

w = cp.Variable()
fn = w * cp.sqrt(w)
problem = cp.Problem(cp.Maximize(fn))
assert not fn.is_dqcp()
assert not problem.is_dqcp()

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




##====================================================================================
##                                 查看系统有哪些优化器
##====================================================================================
import cvxpy as cp
# Solving a problem with different solvers.
x = cp.Variable(2)
obj = cp.Minimize(x[0] + cp.norm(x, 1))
constraints = [x >= 2]
prob = cp.Problem(obj, constraints)

# Solve with ECOS.
prob.solve(solver = cp.ECOS)
print("optimal value with ECOS:", prob.value)

# Solve with ECOS_BB.
prob.solve(solver = cp.ECOS_BB)
print("optimal value with ECOS_BB:", prob.value)

# Solve with CVXOPT.
prob.solve(solver = cp.CVXOPT)
print("optimal value with CVXOPT:", prob.value)

# Solve with SCS.
prob.solve(solver = cp.SCS)
print("optimal value with SCS:", prob.value)

# Solve with GLPK.
prob.solve(solver = cp.GLPK)
print("optimal value with GLPK:", prob.value)

# Solve with GLPK_MI.
prob.solve(solver = cp.GLPK_MI)
print("optimal value with GLPK_MI:", prob.value)

# Solve with GUROBI.
prob.solve(solver = cp.GUROBI)
print("optimal value with GUROBI:", prob.value)

# Solve with MOSEK.
prob.solve(solver = cp.MOSEK)
print("optimal value with MOSEK:", prob.value)

# Solve with Elemental.
prob.solve(solver = cp.ELEMENTAL)
print("optimal value with Elemental:", prob.value)

# Solve with CBC.
prob.solve(solver = cp.CBC)
print("optimal value with CBC:", prob.value)

# Use the installed_solvers utility function to get a list of the solvers your installation of CVXPY //supports.

print(cp.installed_solvers())



#========================================================================================
# https://zhuanlan.zhihu.com/p/114849004
import numpy as np
import cvxpy as cp


np.random.seed(1)
n = 10
mu = np.abs(np.random.randn(n, 1))
Sigma = np.random.randn(n, n)
Sigma = Sigma.T.dot(Sigma)
# Long only portfolio optimization.

w = cp.Variable(n)
gamma = cp.Parameter(nonneg=True)
ret = mu.T@w
risk = cp.quad_form(w, Sigma)
prob = cp.Problem(cp.Maximize(ret - gamma*risk), [cp.sum(w) == 1, w >= 0])


# Compute trade-off  curve.
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

plt.tight_layout()
out_fig = plt.gcf()
out_fig.savefig('/home/jack/snap/lasso.eps',   bbox_inches = 'tight')
plt.close()









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






























































































































































































