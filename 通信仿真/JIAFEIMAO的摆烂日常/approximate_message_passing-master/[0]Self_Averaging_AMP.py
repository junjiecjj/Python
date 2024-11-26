#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:23:32 2024

@author: jack
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import linear_model
from tqdm import tqdm_notebook as tqdm

import ampy
# %matplotlib inline

np.random.seed(0)


alpha = 0.8
N = 4096
M = int(N * alpha)

A = np.random.normal(0.0, 1.0/ N**0.5, (M, N))
x_0 = np.random.normal(0.0, 1.0, N)
x_0 = np.random.binomial(1.0, 0.1, N) * x_0
y = A @ x_0 + np.random.normal(0.0, 1e-2, M)

lasso_cv = linear_model.LassoCV(cv=5, n_jobs=2, tol=1e-5)
lasso_cv.fit(A, y)


lasso = linear_model.Lasso(alpha=lasso_cv.alpha_, tol=1e-12, max_iter=1e5)
lasso.fit(A, y)

fig = plt.figure(figsize=(15, 4))
ax = fig.add_subplot(121)
ax.plot(x_0, "o", c="b", alpha =0.5, label="true")
ax.plot(lasso.coef_, "x", c="r", label="estimated")
ax.grid()
ax.legend()

ax = fig.add_subplot(122)
ax.plot(x_0, lasso.coef_, "o", c="b", alpha=0.5)
ax.set_xlabel("true")
ax.set_ylabel("estimated")
ax.grid()
ax.legend()

fig.tight_layout()


lasso = linear_model.Lasso(alpha=lasso_cv.alpha_, tol=1e-5, max_iter=1e5, warm_start=True)
l_list = lasso_cv.alpha_ * np.logspace(3, 0, base=10.0, num=100)
solution_lasso_list = []
lasso.alpha = l_list[0]
lasso.fit(A, y)
t1 = time.time()
for l in tqdm(l_list[1:]):
    lasso.alpha = l
    lasso.fit(A, y)
t2 = time.time()
print(t2 - t1, "sec")

dumping = 0.9

tol = 1e-2
max_iteration = 100
l_list = M * lasso_cv.alpha_ * np.logspace(3, 0, base=10.0, num=100)
amp_solver = ampy.SelfAveragingAMPSolver.SelfAveragingAMPSolver(A, y, l_list[0], dumping)

amp_solver.d = 1.0
amp_solver.solve(max_iteration=max_iteration, tolerance=tol)
amp_solver.d = dumping
solution_list = []
t1 = time.time()
for i, l in tqdm(enumerate(l_list[1:]), total=len(l_list[1:])):
#     print(i)
    amp_solver.l = l
    amp_solver.solve(max_iteration=max_iteration, tolerance=tol, message=False)
    solution_list.append(amp_solver.r.copy())
#     print()
t2 = time.time()
print(t2 - t1, "sec")


dumping = 0.9

tol = 1e-2
max_iteration = 100
l_list = M * lasso_cv.alpha_ * np.logspace(3, 0, base=10.0, num=100)
amp_solver = ampy.SelfAveragingAMPSolver.SelfAveragingAMPSolver(A, y, l_list[0], dumping)

amp_solver.d = 1.0
amp_solver.solve(max_iteration=max_iteration, tolerance=tol)
amp_solver.d = dumping
solution_list = []
t1 = time.time()
for i, l in tqdm(enumerate(l_list[1:]), total=len(l_list[1:])):
#     print(i)
    amp_solver.l = l
    amp_solver.solve(max_iteration=max_iteration, tolerance=tol, message=False)
    solution_list.append(amp_solver.r.copy())
#     print()
t2 = time.time()
print(t2 - t1, "sec")















































































































