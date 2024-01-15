#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:02:08 2024

@author: jack
"""

import cvxpy as cp

# Problem data.
A_wall = 100
A_flr = 10
alpha = 0.5
beta = 2
gamma = 0.5
delta = 2

h = cp.Variable(pos=True, name="h")
w = cp.Variable(pos=True, name="w")
d = cp.Variable(pos=True, name="d")

volume = h * w * d
wall_area = 2 * (h * w + h * d)
flr_area = w * d
hw_ratio = h/w
dw_ratio = d/w
constraints = [
    wall_area <= A_wall,
    flr_area <= A_flr,
    hw_ratio >= alpha,
    hw_ratio <= beta,
    dw_ratio >= gamma,
    dw_ratio <= delta
]
problem = cp.Problem(cp.Maximize(volume), constraints)
print(problem)
assert not problem.is_dcp()
assert problem.is_dgp()
problem.solve(gp=True)
print(f"problem.value = {problem.value}")
print(f"h.value = {h.value}")
print(f"w.value = {w.value}")
print(f"d.value = {d.value}")


