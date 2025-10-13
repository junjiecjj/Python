#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 13:33:56 2025

@author: jack
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cvxpy as cp

# 全局设置字体大小
plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.family"] = "SimSun"
plt.rcParams['font.size'] = 18               # 设置全局字体大小
plt.rcParams['axes.titlesize'] = 18          # 设置坐标轴标题字体大小
plt.rcParams['axes.labelsize'] = 18          # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 18         # 设置 x 轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 18         # 设置 y 轴刻度字体大小
plt.rcParams['axes.unicode_minus'] = False   # 用来显示负号
plt.rcParams["figure.figsize"] = [8, 6]      # 调整生成的图表最大尺寸
# plt.rcParams['figure.dpi'] = 300           # 每英寸点数
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.linewidth'] = 2          # 线条宽度
plt.rcParams['lines.color'] = 'blue'
plt.rcParams['lines.markersize'] = 6         # 标记大小
# plt.rcParams['figure.facecolor'] = 'lightgrey'   # 设置图形背景色为浅灰色
plt.rcParams['figure.facecolor'] = 'white'         # 设置图形背景色为浅灰色
plt.rcParams['axes.edgecolor'] = 'black'           # 设置坐标轴边框颜色为黑色
plt.rcParams['legend.fontsize'] = 18
np.random.seed(42)


#%%
def align_eigenvectors(U_c, Psi_c):
    """
    通过计算列向量间的内积来对齐U_c和Psi_c的特征向量顺序
    参数:
        U_c: Σc的特征向量矩阵
        Psi_c: Rc的特征向量矩阵
    返回:
        Psi_c_aligned: 对齐后的Psi_c
        mapping: 列映射关系
    """
    N = U_c.shape[1]
    # 计算相关系数矩阵
    correlation_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            # 计算归一化内积 (a·b*)/(|a|·|b|)
            a = U_c[:, i]
            b = Psi_c[:, j]
            correlation = np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
            correlation_matrix[i, j] = correlation

    reorder = correlation_matrix.argmax(axis = 1)
    Psi_c_aligned = Psi_c[:, reorder]

    return Psi_c_aligned, reorder


#%% 按行展开, 是错的，对不上
M = 2
N = 3
T = 4
I = np.eye(T)
H = np.arange(M*N).reshape(M, N)
X = np.random.randn(N, T)
Y = H@X

y = Y.flatten('C')

Hhat = np.kron(I, H)
x = X.flatten('C')

yhat = Hhat @ x

#%% 按列展开,对的, Eq.(12)
M = 2
N = 3
T = 4

Hs = np.random.randn(M, N) + 1j * np.random.randn(M, N)
Xs = np.random.randn(N, T) + 1j * np.random.randn(N, T)
Ys = Hs@Xs

ys = Ys.conj().T.flatten('F')
I = np.eye(M)
Xhat = np.kron(I, Xs.conj().T)
hs = Hs.conj().T.flatten('F')

yhat = Xhat @ hs

#%% Verified Eq.(7)-(11);

M = 4
T = 100
N = 4
PT = 1
sigma_c2 = 1
I_N = np.eye(N)

Hc = np.random.randn(M, N) + 1j * np.random.randn(M, N)
Sigma_C = Hc.conj().T @ Hc

Rc = cp.Variable((N, N), hermitian = True)

constraints = [0 << Rc,
               cp.trace(Rc) <= PT,
              ]

prob = cp.Problem(cp.Maximize(cp.log_det(Hc@Rc@Hc.conj().T + I_N)), constraints)
prob.solve()

if prob.status=='optimal':
     print(f"{prob.value}")
     print(f"{Rc.value}")

Rc = Rc.value
Lambda_c_hat, U_c = np.linalg.eig(Sigma_C)
Lambda_c_hat = np.abs(Lambda_c_hat)

Lambda_c2, Psi_c = np.linalg.eig(Rc)
Lambda_c2 = np.abs(Lambda_c2)

# U_c 和 Psi_c的列的顺序不同，但是值确实是一样的，也就是列的顺序打乱了
Psi_c_aligned, reorder = align_eigenvectors(U_c, Psi_c)
Lambda_C2 = Lambda_c2[reorder]

from WaterFilling import water_filling, plot_waterfilling

optimal_powers, water_level = water_filling(sigma_c2, Lambda_c_hat, PT)
print(f"最优功率分配: {optimal_powers}")
print(f"实际使用功率: {np.sum(optimal_powers):.4f}")
plot_waterfilling(sigma_c2/Lambda_c_hat, optimal_powers, water_level)

## Lambda_C2 == optimal_powers, Eq.(10-11) checked, amazing !!!

print(f"Lambda_C2 = {Lambda_C2}")
print(f"optimal_powers = {optimal_powers}")


#%% Verified Eq.(12);

M = 4
T = 100
N = 4


## 这里不管Hs的分布
Hs = np.random.randn(M, N) + 1j * np.random.randn(M, N)
Xs = np.random.randn(N, T) + 1j * np.random.randn(N, T)
Ys = Hs@Xs

ys1 = Ys.conj().T.flatten('F')
I = np.eye(M)
Xhat = np.kron(I, Xs.conj().T)
hs = Hs.conj().T.flatten('F')

ys = Xhat @ hs  #  Eq.(12), ys == ys1


#%% Verified Eq.(12)-(16);


C = np.array([[1,0.5,0.3],[0.5,1,0.3],[0.3,0.3,1]])

L = np.linalg.cholesky(C)
U = L.T

R = np.random.randn(100000, 3)
Rc = R@U

X = Rc[:,0]
Y = Rc[:,1]
Z = Rc[:,2]

C_hat = np.cov(Rc.T)
print("相关系数矩阵=\n", C_hat)


#%%
from PSDmatrix import generate_psd_hermitian_method1

M = 4
T = 100
N = 4
PT = 1
sigma_s2 = 1
# Sigma_S = np.array([[1,0.5,0.3],[0.5,1,0.3],[0.3,0.3,1]])

Sigma_S = generate_psd_hermitian_method1(N, seed=None)

Rs = cp.Variable((N, N), hermitian = True)

constraints = [0 << Rs,
               cp.trace(Rs) <= PT,
              ]

prob = cp.Problem(cp.Maximize(cp.trace(Hc@Rc@Hc.conj().T + I_N)), constraints)
prob.solve()

if prob.status=='optimal':
     print(f"{prob.value}")
     print(f"{Rc.value}")

Rc = Rc.value
Lambda_c_hat, U_c = np.linalg.eig(Sigma_C)
Lambda_c_hat = np.abs(Lambda_c_hat)

Lambda_c2, Psi_c = np.linalg.eig(Rc)
Lambda_c2 = np.abs(Lambda_c2)





































































































































































































































































































