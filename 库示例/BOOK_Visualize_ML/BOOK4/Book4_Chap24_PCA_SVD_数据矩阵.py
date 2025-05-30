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

def visualize_svd(X, title_X, title_U, title_S, title_V, fig_height=5):
    # Run SVD, as defined above
    U, S, V = svd(X)
    all_ = np.r_[X.flatten(order='C'), U.flatten(order='C'), S.flatten(order='C'), V.flatten(order='C')]

    all_max = max(all_.max(),all_.min())
    all_min = -max(all_.max(),all_.min())
    # all_max = 6
    # all_min = -6
    # Visualization
    fig, axs = plt.subplots(1, 7, figsize=(22, fig_height))

    plt.sca(axs[0])
    ax = sns.heatmap(X, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_X)

    plt.sca(axs[1])
    plt.title('=')
    plt.axis('off')

    plt.sca(axs[2])
    ax = sns.heatmap(U, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_U)

    plt.sca(axs[3])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[4])
    ax = sns.heatmap(S, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_S)

    plt.sca(axs[5])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[6])
    ax = sns.heatmap(V.T, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(title_V)
    return X, U, S, V

# A copy from Seaborn
iris = load_iris()
X = iris.data
y = iris.target
feature_names = ['Sepal length, x1','Sepal width, x2',
                 'Petal length, x3','Petal width, x4']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

#%% 原始数据矩阵 X
X = X_df.to_numpy();

# Gram matrix, G
G = X.T@X
## 余弦相似度矩阵 (cosine similarity matrix)C, C != rho
D = np.diag(np.linalg.norm(X, ord = 2, axis=0))
C = scipy.linalg.inv(D) @ G @ scipy.linalg.inv(D)
# D_norm = np.diag(np.sqrt(np.diag(G)))
# C = scipy.linalg.inv(D_norm)@G@scipy.linalg.inv(D_norm)
## or use package cosine_similarity
C1 = cosine_similarity(X.T)

visualize_svd(X, r'$X$', r'$U_x$', r'$S_x$', r'$V_x^T$', fig_height=5)

#%% Demean, centralize, X_c
EX = X.mean(axis = 0)
VarX = X.var(axis = 0)
StdX = X.std(axis = 0)
X_c = X_df.sub(X_df.mean()).to_numpy();
# 中心化
Xc = X - X.mean(axis = 0)  # == X_c

#%% 标准化数据矩阵 ZX
Z_x = zscore(X_df).to_numpy();
Zx = (X - EX) / StdX        # == Zx


#%% Cosine similarity matrix, C


# from numpy.linalg import inv

S_norm = np.diag(np.sqrt(np.diag(G)))
# scaling matrix, diagnal element is the norm of x_j

C1 = np.linalg.inv(S_norm)@G@np.linalg.inv(S_norm)

#%% centroid of data matrix, E(X)
E_X = X_df.mean().to_frame().T


#%% covariance matrix, Sigma

SIGMA = X_df.cov()
# 计算协方差矩阵
# cov_x = np.cov(X)
cov_xt = np.cov(X.T)

#%% correlation matrix, P
RHO = X_df.corr()

# 计算相关性系数矩阵
# corr_x = np.corrcoef(X)
corr_xt = np.corrcoef(X.T)



#%% Bk4_Ch24_01_B
#%% QR decomposition

Q, R = np.linalg.qr(X_df, mode = 'reduced')

#%%  Bk4_Ch24_01_C
#%% Cholesky decomposition


L_G = np.linalg.cholesky(G)
R_G = L_G.T

#%% Cholesky decompose covariance matrix, SIGMA
L_Sigma = np.linalg.cholesky(SIGMA)

R_Sigma = L_Sigma.T

#%% Bk4_Ch24_01_D
#%% eigen decompose G
from numpy.linalg import eig

Lambs_G, V_G = np.linalg.eig(G)
Lambs_G = np.diag(Lambs_G)

#%% eigen decompose Sigma, covariance matrix
Lambs_sigma, V_sigma = np.linalg.eig(SIGMA)
Lambs_sigma = np.diag(Lambs_sigma)

#%% eigen decompose P, correlation matrix
Lambs_P, V_P = np.linalg.eig(RHO)
Lambs_P = np.diag(Lambs_P)

#%% Bk4_Ch24_01_E
#%% SVD, original data X


U_X,S_X_,V_X = np.linalg.svd(X_df, full_matrices=False)
V_X = V_X.T

# full_matrices=True
# indices_diagonal = np.diag_indices(4)
# S_X = np.zeros_like(X_df)
# S_X[indices_diagonal] = S_X_

# full_matrices=False
S_X = np.diag(S_X_)

#%% SVD, original data Xc

U_Xc, S_Xc, V_Xc = np.linalg.svd(X_c, full_matrices=False)
V_Xc = V_Xc.T
S_Xc = np.diag(S_Xc)

#%% SVD, z scores
U_Z, S_Z, V_Z = np.linalg.svd(Z_X, full_matrices = False)
V_Z = V_Z.T
S_Z = np.diag(S_Z)

#%%
# A copy from Seaborn
iris = load_iris()
X = iris.data
y = iris.target

feature_names = ['Sepal length, x1','Sepal width, x2', 'Petal length, x3','Petal width, x4']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

#  Original data, X
X = X_df.to_numpy()


#%% Gram matrix, G
G = X.T@X

# 计算协方差矩阵
cov_x = np.cov(X.T)

# 计算相关性系数矩阵
corr_x = np.corrcoef(X.T)

#%% QR 分解:获得正交系
# %% QR decomposition
from numpy.linalg import qr
Q, R = qr(X, mode = 'reduced')

#%% 24.3 Cholesky 分解:找到列向量的坐标

from numpy.linalg import cholesky as chol

L_G = chol(G)
R_G = L_G.T

# Cholesky decompose covariance matrix, SIGMA
L_Sigma = chol(cov_x)
R_Sigma = L_Sigma.T

L_rho = chol(corr_x)
R_rho = L_Sigma.T


#%% 24.4 特征值分解:获得行空间和零空间
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'RdBu_r'
import seaborn as sns
PRECISION = 3

from numpy.linalg import eig

Lambs_G, V_G = eig(G)
Lambs_G = np.diag(Lambs_G)

##### 1: V_G @ Lambs_G @ V_G.T == G


all_max = 6
all_min = -6
# Visualization
fig, axs = plt.subplots(1, 7, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(G, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title("$G$")

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(V_G, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title("$V_x$")

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(Lambs_G, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title("$\Lambda$")

plt.sca(axs[5])
plt.title('@')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(V_G.T, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title("$V_x.T$")


#####2: eigen decompose Sigma, covariance matrix
Lambs_sigma, V_sigma = eig(cov_x)
Lambs_sigma = np.diag(Lambs_sigma)


all_max = 6
all_min = -6
# Visualization
fig, axs = plt.subplots(1, 7, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(cov_x, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title("$\sigma$")

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(V_sigma, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title("$V_{c}$")

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(Lambs_sigma, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title("$\Lambda$")

plt.sca(axs[5])
plt.title('@')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(V_sigma.T, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title("$V_c^T$")


########3:  eigen decompose P, correlation matrix
Lambs_P, V_P = eig(corr_x)
Lambs_P = np.diag(Lambs_P)


#%% 24.5 SVD 分解:获得四个空间

#%% SVD, original data X
from numpy.linalg import svd

U_X,S_X_,V_X = np.linalg.svd(X, full_matrices=False)
V_X = V_X.T

# full_matrices=True
# indices_diagonal = np.diag_indices(4)
# S_X = np.zeros_like(X_df)
# S_X[indices_diagonal] = S_X_

# full_matrices=False
S_X = np.diag(S_X_)

#%% SVD, centralized data Xc
X_c = X - X.mean(axis = 0)

U_Xc, S_Xc, V_Xc = np.linalg.svd(X_c, full_matrices=False)
V_Xc = V_Xc.T
S_Xc = np.diag(S_Xc)

#%% SVD, z scores
from scipy.stats import zscore

Z_X = zscore(X)

U_Z, S_Z, V_Z = np.linalg.svd(Z_X, full_matrices = False)
V_Z = V_Z.T
S_Z = np.diag(S_Z)

















































































































































































































































































