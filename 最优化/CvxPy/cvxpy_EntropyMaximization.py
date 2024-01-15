#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:54:49 2024

@author: jack
"""
# https://www.wuzao.com/document/cvxpy/examples/applications/max_entropy.html


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
b = A@tmp
F = np.random.randn(p, n)
g = F@tmp + np.random.rand(p)


# Entropy maximization.
x = cp.Variable(shape=n)
obj = cp.Maximize(cp.sum(cp.entr(x)))
constraints = [A@x == b,
               F@x <= g ]
prob = cp.Problem(obj, constraints)
prob.solve( )

# Print result.
print("\nThe optimal value is:", prob.value)
print('\nThe optimal solution is:')
print(x.value)






































