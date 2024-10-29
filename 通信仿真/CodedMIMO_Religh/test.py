#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:28:04 2024

@author: jack
"""
import scipy
import numpy as np

# Gauss-Seidel迭代算法
def gauss_seidel(A, b, x0, max_iter=1000, tol=1e-6):
    L = np.tril(A, k=-1)
    D = np.diag(np.diag(A))
    U = np.triu(A, k=1)
    M = scipy.linalg.pinv(D + L)
    x = x0
    for i in range(max_iter):
        x_new = M @ (b - U@x)  #np.dot(M, b - np.dot(U, x))
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x

def gauss_seidel1(A, b, max_iter=1000, tol=1e-6):
    D = np.diag(np.diag(A))
    U = -np.triu(A, k=1)
    L = -np.tril(A, k=-1)
    x_hat = np.diag(scipy.linalg.pinv(D))

    for i in range(max_iter):
        x_new = scipy.linalg.pinv(D - U) @ (L @ x_hat + b)
        if np.linalg.norm(x_new - x_hat) < tol:
            return x_new
        x_hat = x_new
    return x

# # 示例1
# A = np.array([[4, 1], [1, 3]])
# b = np.array([1, 2])
# x0 = np.array([0, 0])
# x = gauss_seidel(A, b, x0)
# print("Solution:", x)


A = np.array([[10, 2, 1], [1,5, 1], [2, 3, 10]])
b = np.array([7, -8, 6])
x0 = np.array([0, 0, 0])
x = gauss_seidel(A, b, x0)
print("Solution:", x)

x1 = gauss_seidel1(A, b)
print("Solution:", x1)

