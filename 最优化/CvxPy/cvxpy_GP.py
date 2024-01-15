#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:33:44 2024
@author: jack
"""


#=================================================================
# disciplined geometric programming (DGP)， 标准几何规划
# https://www.wuzao.com/document/cvxpy/examples/dgp/dgp_fundamentals.html
#=================================================================
import cvxpy as cp

# Only elementwise positive constants are allowed in DGP.
c = cp.Constant(1.0)
print(c, c.log_log_curvature)

c = cp.Constant([1.0, 2.0])
print(c, c.log_log_curvature)

c = cp.Constant([1.0, 0.0])
print(c, c.log_log_curvature)

c = cp.Constant(-2.0)
print(c, c.log_log_curvature)


# Variables and parameters must be positive, ie, they must be constructed with the option `pos=True`
v = cp.Variable(pos=True)
print(v, v.log_log_curvature)

v = cp.Variable()
print(v, v.log_log_curvature)

p = cp.Parameter(pos=True)
print(p, p.log_log_curvature)

p = cp.Parameter()
print(p, p.log_log_curvature)

## The following functions are special cases of log-log affine functions:
    # Exponential function
    # f = cp.exp(x)
    # f(x) = exp(x_1) + exp(x_2) + ldots + exp(x_n)

    # Power function
    # f = cp.power(x, p)
    # f(x) = x_1^p_1 cdot x_2^p_2 ldots x_n^p_n


##=============================================================
x = cp.Variable(shape=(3,), pos=True, name="x")
c = 2.0
a = [0.5, 2.0, 1.8]

monomial = c * x[0] ** a[0] * x[1] ** a[1] * x[2] ** a[2]
# Monomials are not convex.
assert not monomial.is_convex()

# They are, however, log-log affine.
print(monomial, ":", monomial.log_log_curvature)
assert monomial.is_log_log_affine()

##=============================================================
# 多个单项式函数的和是对数-对数凸的；在几何规划的背景下，这样的函数被称为多项式。有一些函数不是多项式但仍然是对数-对数凸的。

x = cp.Variable(pos=True, name="x")
y = cp.Variable(pos=True, name="y")

constant = cp.Constant(2.0)
monomial = constant * x * y
posynomial = monomial + (x ** 1.5) * (y ** -1)
reciprocal = posynomial ** -1
unknown = reciprocal + posynomial

print(constant, ":", constant.log_log_curvature)
print(monomial, ":", monomial.log_log_curvature)
print(posynomial, ":", posynomial.log_log_curvature)
print(reciprocal, ":", reciprocal.log_log_curvature)
print(unknown, ":", unknown.log_log_curvature)

##=============================================================
# 对数-对数曲率规则集
x = cp.Variable(pos=True, name="x")
y = cp.Variable(pos=True, name="y")

monomial = 2.0 * x * y
posynomial = monomial + (x ** 1.5) * (y ** -1)

print(monomial, "is dgp?", monomial.is_dgp())
print(posynomial, "is dgp?", posynomial.is_dgp())

##=============================================================
# DGP 问题
x = cp.Variable(pos=True, name="x")
y = cp.Variable(pos=True, name="y")
z = cp.Variable(pos=True, name="z")

objective_fn = x * y * z
constraints = [ 4 * x * y * z + 2 * x * z <= 10,
               x <= 2*y,
               y <= 2*x,
               z >= 1 ]

assert objective_fn.is_log_log_concave()
assert all(constraint.is_dgp() for constraint in constraints)
problem = cp.Problem(cp.Maximize(objective_fn), constraints)

print(problem)
print("Is this problem DGP?", problem.is_dgp())


# 解决DGP问题
problem.solve(gp=True)
print("最优值:", problem.value)
print(x, ":", x.value)
print(y, ":", y.value)
print(z, ":", z.value)
print("对偶值: ", list(c.dual_value for c in constraints))






























































































































































































































