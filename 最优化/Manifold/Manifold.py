#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:11:52 2024

@author: jack
https://pymanopt.org/docs/stable/quickstart.html#installation


https://blog.csdn.net/weixin_39274659/article/details/115735867
https://blog.csdn.net/weixin_53463894/article/details/135337883?spm=1001.2101.3001.6650.7&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-7-135337883-blog-115735867.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-7-135337883-blog-115735867.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=12

https://blog.csdn.net/2401_85133351/article/details/139909336



# # 1 计算高斯曲率

# from sympy import symbols, Function, diff, simplify

# x, y = symbols('x y')
# g = Function('g')(x, y)
# R = Function('R')(x, y)

# K = R / g

# print(simplify(K))


# # 2 解爱因斯坦场方程
# import numpy as np
# from scipy.integrate import odeint

# def einstein_eqn(y, x):
#     G, T = y
#     dGdx = 8 * np.pi * G * T
#     return [dGdx, 0]

# x = np.linspace(0, 1, 100)
# y0 = [0, 1]
# y = odeint(einstein_eqn, y0, x)

# print(y)


# # 3 计算测地线
# from sympy import symbols, Function, diff, simplify

# x, y, z, t = symbols('x y z t')
# g = Function('g')(x, y, z)

# dxdt = diff(x, t)
# dydt = diff(y, t)
# dzdt = diff(z, t)

# geodesic_eqn = simplify(dxdt**2 + dydt**2 + dzdt**2 - g)
# print(geodesic_eqn)


"""


import autograd.numpy as anp
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers





#%% A Simple Example
anp.random.seed(42)

dim = 3
manifold = pymanopt.manifolds.Sphere(dim)

matrix = anp.random.normal(size=(dim, dim))
matrix = 0.5 * (matrix + matrix.T)

@pymanopt.function.autograd(manifold)
def cost(point):
    return -point @ matrix @ point

problem = pymanopt.Problem(manifold, cost)

optimizer = pymanopt.optimizers.SteepestDescent()
result = optimizer.run(problem)

eigenvalues, eigenvectors = anp.linalg.eig(matrix)
dominant_eigenvector = eigenvectors[:, eigenvalues.argmax()]

print("Dominant eigenvector:", dominant_eigenvector)
print("Pymanopt solution:", result.point)






















