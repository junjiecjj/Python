#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 17:29:40 2025

@author: jack
"""


# Bk4_Ch24_01_A
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
from scipy.stats import zscore
from sklearn.metrics.pairwise import cosine_similarity
plt.rcParams['image.cmap'] = 'RdBu_r'
#  Bk4_Ch15_02_A
PRECISION = 3

def svd(X):
    full_matrices = True
    U, s, Vt = np.linalg.svd(X, full_matrices = full_matrices)
    # Put the vector singular values into a padded matrix
    if full_matrices:
        S = np.zeros(X.shape)
        np.fill_diagonal(S, s)
    else:
        S = np.diag(s)
    # Rounding for display
    return np.round(U, PRECISION), np.round(S, PRECISION), np.round(Vt.T, PRECISION)

def visualize_3M(X, U, S, V, title_X, title_U, title_S, title_V, fig_height = 5):
    # Run SVD, as defined above
    # U, S, V = svd(X)
    # all_ = np.r_[X.flatten(order='C'), U.flatten(order='C'), S.flatten(order='C'), V.flatten(order='C')]
    # all_max = max(all_.max(),all_.min())
    # all_min = -max(all_.max(),all_.min())
    # all_max = 6
    # all_min = -6
    # Visualization
    fig, axs = plt.subplots(1, 7, figsize=(20, fig_height))

    plt.sca(axs[0])
    ax = sns.heatmap(X, cmap='rainbow', linewidths=.05, annot=True, fmt='.2f', cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_X)

    plt.sca(axs[1])
    plt.title('=')
    plt.axis('off')

    plt.sca(axs[2])
    ax = sns.heatmap(U, cmap='rainbow', linewidths=.05, annot=True, fmt='.2f', cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_U)

    plt.sca(axs[3])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[4])
    ax = sns.heatmap(S, cmap='rainbow', linewidths=.05, annot=True, fmt='.2f', cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_S)

    plt.sca(axs[5])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[6])
    ax = sns.heatmap(V.T, cmap='rainbow', linewidths=.05, annot=True, fmt='.2f', cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_V)
    return X, U, S, V

def visualize_2M(X, Q, R, title_X, title_Q, title_R):
    # QR decomposition， complete version
    # Q, R = np.linalg.qr(X, mode = 'complete')

    fig, axs = plt.subplots(1, 5, figsize=(12, 5))

    plt.sca(axs[0])
    ax = sns.heatmap(X, cmap='rainbow', linewidths=.05, annot=True, fmt='.2f', cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_X)

    plt.sca(axs[1])
    plt.title('=')
    plt.axis('off')

    plt.sca(axs[2])
    ax = sns.heatmap(Q, cmap='rainbow', linewidths=.05, annot=True, fmt='.2f', cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_Q)

    plt.sca(axs[3])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[4])
    ax = sns.heatmap(R, cmap='rainbow', linewidths=.05, annot=True, fmt='.2f', cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_R)
    return

# A copy from Seaborn
iris = load_iris()
X = iris.data
y = iris.target
feature_names = ['Sepal length, x1','Sepal width, x2',
                 'Petal length, x3','Petal width, x4']
# Convert X array to dataframe
X_df = pd.DataFrame(X, columns = feature_names)

#%% 原始数据矩阵 X
# X = X_df.to_numpy();
X = np.array([[2.1, 1.1, 3.1], [0.9, 0.8, 2.6], [5.1, 1.9, 7], [3.3, 4.4, -2.2]])
EX = X.mean(axis = 0)
VarX = X.var(axis = 0)
StdX = X.std(axis = 0)

# Gram matrix, G
G = X.T@X
## 余弦相似度矩阵 (cosine similarity matrix)C, C != rho
D = np.diag(np.linalg.norm(X, ord = 2, axis=0))
C = scipy.linalg.inv(D) @ G @ scipy.linalg.inv(D)
# D_norm = np.diag(np.sqrt(np.diag(G)))
# C = scipy.linalg.inv(D_norm)@G@scipy.linalg.inv(D_norm)
## or use package cosine_similarity
C1 = cosine_similarity(X.T)

## SVD 分解
Ux, Sx, Vx = svd(X)
visualize_3M(X, Ux, Sx, Vx.T, r'$X$', r'$U_x$', r'$S_x$', r'$V_x^T$', fig_height = 5)

### QR 分解:获得正交系
Q, R = np.linalg.qr(X, mode = 'reduced')
visualize_2M(X, Q, R, r'$X$', r'$Q_X$', r'$R_X$', )

## 特征值分解
Lambda_X, V_x = np.linalg.eigh(G)
Lambda_X = np.diag(Lambda_X)
visualize_3M(G, V_x, Lambda_X, V_x.T, r'$G$', r'$V_x$', r'$\Lambda_x$', r'$V_x^T$', fig_height = 5)

print(f"Sx.T@Sx = \n{Sx.T @ Sx}")
print(f"Lambda_X = \n{Lambda_X}")

## Cholesky 分解:找到列向量的坐标
L_G = np.linalg.cholesky(G)
R_G = L_G.T
visualize_2M(G, L_G, L_G.T, r'$G$', r'$L_G$', r'$L_G^T$' )



#%% 中心化数据矩阵, X_c
# X_c = X_df.sub(X_df.mean()).to_numpy();
# 中心化
Xc = X - X.mean(axis = 0)  # == X_c

# 计算协方差矩阵
# SIGMA = np.array(X_df.cov()) # == XXT
SIGMA = np.matrix(Xc.T) * np.matrix(Xc) / (len(Xc)-1)
Sigma = np.cov(X.T)  #  == SIGMA


## SVD 分解
Uc, Sc, Vc = svd(Xc)
visualize_3M(Xc, Uc, Sc, Vc.T, r'$X_c$', r'$U_c$', r'$S_c$', r'$V_c^T$', fig_height = 5)

### QR 分解:获得正交系
Qc, Rc = np.linalg.qr(Xc, mode = 'reduced')
visualize_2M(Xc, Qc, Rc, r'$X_c$', r'$Q_C$', r'$R_C$', )

## 特征值分解
Lambda_C, V_c = np.linalg.eigh(SIGMA)
Lambda_C = np.diag(Lambda_C)
visualize_3M(SIGMA, V_c, Lambda_C, V_c.T, r'$\Sigma$', r'$V_c$', r'$\Lambda_C$', r'$V_c^T$', fig_height = 5)

n = len(X)
print(f"Sc.T@Sc/(n-1) = \n{Sc.T @ Sc/(n-1)}")
print(f"Lambda_C = \n{Lambda_C}")

## Cholesky 分解:找到列向量的坐标
L_C = np.linalg.cholesky(SIGMA)
R_C = L_C.T
visualize_2M(SIGMA, L_C, L_C.T, r'$\Sigma$', r'$L_C$', r'$L_C^T$' )


#%% 标准化数据矩阵 ZX
# Z_x = zscore(X_df).to_numpy();
Z_x = zscore(X)
Zx = (X - EX) / StdX        # == Zx

#  相关性系数矩阵
# RHO = np.array(X_df.corr()) # == rho
# rho = np.corrcoef(X.T)
rho1 = np.cov(Z_x.T)
S = np.diag(StdX)
S1 = np.diag(np.sqrt(np.diag(SIGMA)))
SPS = S@rho1@S  # == Sigma


## SVD 分解
Uz, Sz, Vz = svd(Zx)
visualize_3M(Zx, Uz, Sz, Vz.T, r'$Z_x$', r'$U_z$', r'$S_z$', r'$V_z^T$', fig_height = 5)

### QR 分解:获得正交系
Qz, Rz = np.linalg.qr(Zx, mode = 'reduced')
visualize_2M(Zx, Qz, Rz, r'$Z_x$', r'$Q_Z$', r'$R_Z$', )

## 特征值分解
Lambda_Z, V_z = np.linalg.eigh(rho1)
Lambda_Z = np.diag(Lambda_Z)
visualize_3M(rho1, V_z, Lambda_Z, V_z.T, r'$\rho$', r'$V_z$', r'$\Lambda_Z$', r'$V_Z^T$', fig_height = 5)

n = len(X)
print(f"Sz.T@Sz/(n-1) = \n{Sz.T @ Sz/(n-1)}")
print(f"Lambda_Z = \n{Lambda_Z}")

## Cholesky 分解:找到列向量的坐标
L_Z = np.linalg.cholesky(rho1)
R_Z = L_Z.T
visualize_2M(rho1, L_Z, L_Z.T, r'$\rho$', r'$L_Z$', r'$L_Z^T$' )






































































































































































































































































