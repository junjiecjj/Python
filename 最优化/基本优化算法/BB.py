#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 10:23:59 2025

@author: jack

Barzilai－Borwein（BB）方法是一种改进的梯度下降算法，通过构造对角矩阵近似海森矩阵（Hessian）的逆矩阵，以自适应<步长>提升收敛效率。

https://mp.weixin.qq.com/s?__biz=MzI0NDMyODUxMA==&mid=2247488073&idx=1&sn=4cf7412140c6ecebeadf6a507a319cda&chksm=e8d6c148304c61f1c8885e2930c7bbdd953ac1a25f31127b9d826594d9ef11ebc5448d01fd02&mpshare=1&scene=1&srcid=04185B5WNPP3OY28mQ3aPXkX&sharer_shareinfo=6bcc6fb425d19159a71470510f8310dc&sharer_shareinfo_first=6bcc6fb425d19159a71470510f8310dc&exportkey=n_ChQIAhIQXsWeDxwRI%2FES8ZmuUiHUcRKfAgIE97dBBAEAAAAAADaQNC%2FWmNAAAAAOpnltbLcz9gKNyK89dVj0lF7hQNkRq9wfw6ZxUf%2Brmxiwdcr2e8FNwQXyNEfmBDoEe%2BKPJ7dsJLCJjdtiCI0fdP%2F6PfkbfwxurHlAG0lr3iMiH%2F%2FU7o6Dw2ndshBnRyDe687XtRbS6hScPBXmW4pntW9L0H2AgpxhspXNihOWvq99anOLLOL0R8KcX64PtibI7DJLstJBahDbihVPMQu3hmMstcd7NfQYTrqVK%2F%2FHVLAgN88hsR7m%2Bc5p8sKI00yUHds117w5D3%2BhQOEu%2BqcfScRSvJV9RH64j2PUBBntv%2Fr2dyX0fVlAoJww4ukZ8ty41J7PkqKzz6z2j8jWk2V34c6yLYDWfdgY&acctmode=0&pass_ticket=bzhxxhICaN6VRF8u0S3Gz1wpdST8vV%2B3l0uf%2FCTMnBe1WQ03qHFztcHQ95GAzGLF&wx_header=0&poc_token=HJK3AWijUah5rNo5BkoTj0T9WHqUgnU4N32epu5a
"""


import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（解决中文乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei']              # Windows系统
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
plt.rcParams['axes.unicode_minus'] = False                # 解决负号显示问题

# 定义二次函数 f(x,y) = x² + 10y² 及其梯度
def f(x, y):
    return x ** 2 + 10 * y ** 2

def gradient(x, y):
    return np.array([2 * x, 20 * y])

# BB方法实现
def bb_method(f, grad, x0, max_iter=100, tol=1e-6, alpha_m=1e-6, alpha_M=1e6):
    """
    Barzilai-Borwein 方法优化器

    参数:
        f: 目标函数
        grad: 梯度函数
        x0: 初始点
        max_iter: 最大迭代次数
        tol: 收敛容忍度
        alpha_m: 最小步长
        alpha_M: 最大步长

    返回:
        x_history: 迭代路径
        f_history: 函数值历史
    """
    x = x0.copy()
    x_history = [x.copy()]
    f_history = [f(*x)]

    # 初始步长设为传统梯度下降的推荐步长
    alpha = 0.1

    for k in range(1, max_iter):
        # 计算当前梯度
        g = grad(*x)

        # 更新迭代点
        x_new = x - alpha * g

        # 保存历史记录
        x_history.append(x_new.copy())
        f_history.append(f(*x_new))

        # 检查收敛
        if np.linalg.norm(x_new - x) < tol:
            break

        # 计算BB步长所需的量
        s = x_new - x
        y = grad(*x_new) - g

        # 计算BB1步长
        alpha_bb1 = np.dot(s, s) / np.dot(s, y)

        # 计算BB2步长
        alpha_bb2 = np.dot(s, y) / np.dot(y, y)

        # 交替使用BB1和BB2步长
        if k % 2 == 0:
            alpha = alpha_bb1
        else:
            alpha = alpha_bb2

        # 步长截断
        alpha = max(alpha_m, min(alpha, alpha_M))

        # 更新x
        x = x_new

    return np.array(x_history), np.array(f_history)
# 传统梯度下降法作为对比
def gradient_descent(f, grad, x0, max_iter=100, tol=1e-6, alpha=0.1):
    x = x0.copy()
    x_history = [x.copy()]
    f_history = [f(*x)]

    for k in range(max_iter):
        g = grad(*x)
        x_new = x - alpha * g

        x_history.append(x_new.copy())
        f_history.append(f(*x_new))

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new

    return np.array(x_history), np.array(f_history)

# 设置初始点
x0 = np.array([-10.0, -1.0])

# 运行BB方法
x_hist_bb, f_hist_bb = bb_method(f, gradient, x0)

x_hist_gd, f_hist_gd = gradient_descent(f, gradient, x0)

# 绘制结果
plt.figure(figsize=(12, 6))

# 绘制等高线
x = np.linspace(-11, 11, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.plot(x_hist_gd[:, 0], x_hist_gd[:, 1], 'r-', label='梯度下降法', linewidth=2)
plt.plot(x_hist_bb[:, 0], x_hist_bb[:, 1], 'k-', label='BB方法', linewidth=2)
plt.scatter(x0[0], x0[1], c='blue', marker='o', s=100, label='初始点')
plt.scatter(0, 0, c='red', marker='*', s=200, label='最优解')

plt.xlabel('x')
plt.ylabel('y')
plt.title('二次函数优化: BB方法 vs 梯度下降法')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()

# 打印收敛信息
print(f"BB方法收敛于{f_hist_bb[-1]:.6f}, 迭代次数:{len(x_hist_bb)}")
print(f"梯度下降法收敛于{f_hist_gd[-1]:.6f}, 迭代次数:{len(x_hist_gd)}")
