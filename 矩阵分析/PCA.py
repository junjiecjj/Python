#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:01:17 2024

@author: jack
"""

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

#########################################################################
##  特征值分解手动计算PCA

iris = load_iris()
print(iris.data.shape) # 150, 4
X = iris.data
# 中心化
Xc = X - X.mean(axis = 0)
# 计算协方差矩阵
XXT = np.matrix(Xc.T) * np.matrix(Xc) / (len(Xc)-1)

print(f"np.cov(X.T) = \n{np.cov(X.T)}")  # == XXT

# 求协方差矩阵的特征值和特征向量
eigVals, eigVects = np.linalg.eig(np.mat(XXT))
print("特征值: ", eigVals)
print(f"特征向量: \n{eigVects}", ) # 和 pca.components_.T 完全一样 (可能差正负号)
# 特征值:  [4.22824171 0.24267075 0.0782095  0.02383509]
# 特征向量:
# [[ 0.36138659 -0.65658877 -0.58202985  0.31548719]
#  [-0.08452251 -0.73016143  0.59791083 -0.3197231 ]
#  [ 0.85667061  0.17337266  0.07623608 -0.47983899]
#  [ 0.3582892   0.07548102  0.54583143  0.75365743]]

#>>>>>>>>>>>>>>>> 中心化数据在特征向量的投影
Zc = Xc @ eigVects[:,:2]
print(f"np.cov(Zc.T) = \n{np.cov(Zc.T)}")
# np.cov(Zc.T) = # 投影后数据的协方差矩阵是对角阵，就是特征值
# [[ 4.22824171e+00 -8.01395836e-16]
#  [-8.01395836e-16  2.42670748e-01]]

### 工具包自动计算
pcamodel = PCA(n_components=2,  svd_solver='full') # whiten = False, 投影后不利用特征值的均方根归一化
pcamodel.fit(X)
print(pcamodel.explained_variance_)
# [4.22824171 0.24267075]
print(pcamodel.components_.T)
# [[ 0.36138659  0.65658877]
#  [-0.08452251  0.73016143]
#  [ 0.85667061 -0.17337266]
#  [ 0.3582892  -0.07548102]]
X_pca1 = pcamodel.transform(X)
# X_pca1 = pcamodel.fit_transform(X)
# X_pca1 == Zc

#>>>>>>>>>>>>>>>> 中心化数据在特征向量的投影
Zc_n = Zc @ np.linalg.inv(np.diag(np.sqrt(eigVals[:2])))  # 这个才是和下面的pca.transform(X)完全一样的结果(可能某列差正负号)
print(f"np.cov(Zc_n.T) = \n{np.cov(Zc_n.T)}")
# [[ 1.00000000e+00 -6.00904408e-16]
 # [-6.00904408e-16  1.00000000e+00]]
### 工具包自动计算
# iris = load_iris()
# X = iris.data
pca = PCA(n_components = 2, whiten = True, svd_solver = 'full') # whiten = True, 投影后利用特征值的均方根归一化
pca.fit(X) # 先求出X的均值(mean_)、协方差矩阵的特征值特征值(explained_variance_,)、协方差矩阵的特征向量(pca.components_.T)，作为标准
print(pca.explained_variance_)
# [4.22824171 0.24267075]
print(pca.components_.T)
# [[ 0.36138659  0.65658877]
#  [-0.08452251  0.73016143]
#  [ 0.85667061 -0.17337266]
#  [ 0.3582892  -0.07548102]]
# 数据变换
X_pca = pca.transform(X)
print(X_pca.shape) # 150, 2
print(np.cov(X_pca.T))
# (150, 2)
# [[1.00000000e+00 9.14950022e-16]
#  [9.14950022e-16 1.00000000e+00]]

X_reco = pca.inverse_transform(X_pca)
print(f"(X_reco - X).max() = {(X_reco - X).max()}, (X_reco - X).min() = {(X_reco - X).min()}")

# 原始数据在特征向量的投影
Z = X @ eigVects[:,:2]
print(f"np.cov(Z.T) = \n{np.cov(Z.T)}")
# np.cov(Z.T) = # 投影后数据的协方差矩阵是对角阵，就是特征值
# [[ 4.22824171e+00 -6.25538223e-16]
#  [-6.25538223e-16  2.42670748e-01]]

Z_n = Z @ np.linalg.inv(np.diag(np.sqrt(eigVals[:2])))
print(np.cov(Z_n.T))
# [[ 1.0000000e+00 -5.6728177e-16]
#  [-5.6728177e-16  1.0000000e+00]]

#########################################################################
### 数据 X_pca 在 [v1, v2] 中散点图
fig, ax = plt.subplots(1, 1, figsize = (6, 6), constrained_layout = True)
ax.scatter(X[:, 0], X[:, 1], alpha = 0.5, marker = '.')
ax.set_xlabel('$z_1$')
ax.set_ylabel('$z_2$')
plt.show()

Z_df = pd.DataFrame(X[:,:2], columns = ['z_1', 'z_2'])
sns.jointplot(data = Z_df, x = 'z_1', y = 'z_2', kind = 'kde', fill = True, ax = ax)
ax.set_aspect('equal')
ax.set_xlabel('$z_1$')
ax.set_ylabel('$z_2$')
plt.show()

### 数据 X_pca 在 [v1, v2] 中散点图
fig, ax = plt.subplots(1, 1, figsize = (6, 6), constrained_layout = True)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha = 0.5, marker = '.')
plt.axvline(x=0, color='r', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('$z_1$')
plt.ylabel('$z_2$')
plt.axis('scaled')
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
plt.show()

####  数据 X_pca 在 [v1, v2] 中散点图
# fig, ax = plt.subplots()
Z_df = pd.DataFrame(X_pca, columns = ['z_1', 'z_2'])
sns.jointplot(data = Z_df, x = 'z_1', y = 'z_2', kind = 'kde', fill = True, xlim = (-5,5), ylim = (-5,5), ax = ax)
ax.set_aspect('equal')
plt.xlabel('$z_1$')
plt.ylabel('$z_2$')
plt.show()

#########################################################################
##  Xc的SVD分解(奇异值)和协方差矩阵的特征值分解(特征值)的关系
# Load the iris data
# iris_sns = sns.load_dataset("iris")
iris = load_iris()
X = iris.data

# X = np.array(iris_sns[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
Xc = X - X.mean(axis =  0)
# 协方差矩阵
Sigma = Xc.T @ Xc / (X.shape[0] - 1)
SIGMA = np.cov(X.T) # SIGMA == Sigma

# 协方差矩阵的特征值分解
Lambda_, V = np.linalg.eig(SIGMA)
Lambda = np.diag(Lambda_)

# Xc的SVD分解(奇异值)
U, s, Vt = np.linalg.svd(Xc, full_matrices = True)
SS = np.zeros(X.shape)
np.fill_diagonal(SS, s)
S = np.diag(s)

# Vt.T 和 V完全一样，可能差个符号或者顺序
print(f"V = \n{V}\nVt.T = \n{Vt.T}")

# Xc的(奇异值)和Sigma的特征值的关系
Lambda_reproduced = S**2/(len(X) - 1)
print(Lambda_reproduced - Lambda)

# Xc的特征向量和Sigma的特征向量的关系，某列可能差正负号.
print(np.abs(V) - np.abs(Vt.T))

# 矩阵 U 每一列数据相当于 Z 对应列向量的标准化：
Z = Xc@V
US = U @ SS
Z_US = np.abs(Z) - np.abs(US)
print(f" {Z_US.min()}, {Z_US.max()}")

## 把原始数据 X 或中心化数据 Xc投影到 V 中结果不一样。从统计角度来看，差异主要体现在质心位置，而投影得到的数据协方差矩阵相同。
ones = np.array([1]*X.shape[0])[:,None]
Z = X@V
Zc = Xc@V  # = X@V - 1^T X@V/n
print(Zc.mean(axis = 0))  # == [0,0,0,0]
# print(np.cov(Z.T) - np.cov(Zc.T)) # 两者相等，因为求协方差本身就需要中心化，只不过Zc已经中心化，求Z的协方差之前先进行中心化
Zc1 = X@V - ones.T @X@V/X.shape[0]  ## == Zc

# 对Z直接求Z^T@Z与Zc.T@Zc的关系
Zsigma = Z.T @ Z / (Z.shape[0] - 1)
Zcsigma = Zc.T @ Zc / (Zc.shape[0] - 1)

# Zcsigma1 == Zcsigma
Zcsigma1 = Z.T @ Z / (Z.shape[0] - 1) -  V.T@X.T@ones@ones.T@X@V/(Z.shape[0] - 1)/Z.shape[0] # + V.T@X.T@ones@ones.T@ones@ones.T@X@V/(Z.shape[0] - 1)/Z.shape[0]**2


Ex = ones.T @ X
Zsigma_hat = Zcsigma +  V.T @ (Ex.T) @ Ex @ V / Zc.shape[0]/(Zc.shape[0] - 1)

# Zsigma_hat == Zsigma
# print(Zsigma_hat - Zsigma)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 可视化:协方差矩阵的特征值分解 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
all_max = 2
all_min = -2
#%% 协方差矩阵特征值
iris = load_iris()
X = iris.data
Xc = X - X.mean(axis =  0)
# 协方差矩阵
Sigma = Xc.T @ Xc / (X.shape[0] - 1)
# SIGMA = np.cov(X.T) # SIGMA == Sigma

# 协方差矩阵的特征值分解
Lambda_, V = np.linalg.eig(Sigma)
LAMBDA = np.diag(Lambda_)

fig, axs = plt.subplots(1, 7, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(SIGMA,cmap='RdBu_r',vmax = all_max,vmin = all_min,  cbar=False)
ax.set_aspect("equal")
plt.title(r'$\Sigma$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(V,cmap='RdBu_r',vmax = all_max,vmin = all_min,  cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$V$')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(LAMBDA,cmap='RdBu_r',vmax = all_max,vmin = all_min,  cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\Lambda$')

plt.sca(axs[5])
plt.title('@')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(V.T,cmap='RdBu_r', vmax = all_max,vmin = all_min,  cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$V^T$')
plt.show()

### Sigma = lambda1 * v_1@v_1^T + lambda2 * v_2@v_2^T + lambda3 * v_3@v_3^T + lambda4 * v_4@v_4^T
j = 0; v_j = V[:,j].reshape(-1,1)
l1_v1v1T = Lambda_[j] * v_j @ v_j.T

j = 1; v_j = V[:,j].reshape(-1,1)
l2_v2v2T = Lambda_[j] * v_j @ v_j.T

j = 2; v_j = V[:,j].reshape(-1,1)
l3_v3v3T = Lambda_[j] * v_j @ v_j.T

j = 3; v_j = V[:,j].reshape(-1,1)
l4_v4v4T = Lambda_[j] * v_j @ v_j.T

Sigma_hat = l1_v1v1T + l2_v2v2T + l3_v3v3T + l4_v4v4T

fig, axs = plt.subplots(1, 11, figsize=(18, 3))
plt.sca(axs[0])
ax = sns.heatmap(Sigma_hat,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\hat{\Sigma}$')

plt.sca(axs[1])
plt.title(r'$=$')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(Sigma, cmap='RdBu_r',vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\Sigma$')

plt.sca(axs[3])
plt.title('=')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(l1_v1v1T,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
idx = 0
plt.title('$\u03BB_' + str(idx + 1) + 'v_' + str(idx + 1) + ' @ v_'  + str(idx + 1) + '^T$')

plt.sca(axs[5])
plt.title('+')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(l2_v2v2T,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
idx = 1
plt.title('$\u03BB_' + str(idx + 1) + 'v_' + str(idx + 1) + ' @ v_'  + str(idx + 1) + '^T$')

plt.sca(axs[7])
plt.title('+')
plt.axis('off')

plt.sca(axs[8])
ax = sns.heatmap(l3_v3v3T,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
idx = 2
plt.title('$\u03BB_' + str(idx + 1) + 'v_' + str(idx + 1) + ' @ v_'  + str(idx + 1) + '^T$')

plt.sca(axs[9])
plt.title('+')
plt.axis('off')

plt.sca(axs[10])
ax = sns.heatmap(l4_v4v4T,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
idx = 3
plt.title('$\u03BB_' + str(idx + 1) + 'v_' + str(idx + 1) + ' @ v_'  + str(idx + 1) + '^T$')
plt.show()

######### Sigma - Sigma_hat
fig, axs = plt.subplots(1, 3, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(Sigma, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\Sigma$')

plt.sca(axs[1])
ax = sns.heatmap(Sigma_hat, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\hat{\Sigma}$')

plt.sca(axs[2])
ax = sns.heatmap(Sigma - Sigma_hat, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$E = X - \hat{X}$')

#%%######################################################### Xc的SVD分解(奇异值) #####################################################
# Repeatability
np.random.seed(1)
# Generate random matrix
X = np.random.randn(6, 4)

# manipulate X and reduce rank to 3
# X[:,3] = X[:,0] + X[:,1]

U, s, VT = np.linalg.svd(X, full_matrices = True)
SS = np.zeros(X.shape)
np.fill_diagonal(SS, s)
S = np.diag(s)
V = VT.T

all_max = 2
all_min = -2

####################  X = U @ S @ V^T
fig, axs = plt.subplots(1, 7, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(X, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$X$')

plt.sca(axs[1])
plt.title('=')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(U,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$U$')

plt.sca(axs[3])
plt.title('@')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(SS,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$S$')

plt.sca(axs[5])
plt.title('@')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(VT,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$V^T$')

#################### 四幅热图叠加还原原始图像
#### X = s_1 * u1 @ v_1^T + s_2 * u2 @ v_2^T + s_3 * u3 @ v_3^T + s_3 * u3 @ v_3^T

j=0
s1u1v1 = S[j,j]*U[:,j][:, None]@V[:,j][None, :];
j=1
s2u2v2 = S[j,j]*U[:,j][:, None]@V[:,j][None, :];
j=2
s3u3v3 = S[j,j]*U[:,j][:, None]@V[:,j][None, :];
j=3
s4u4v4 = S[j,j]*U[:,j][:, None]@V[:,j][None, :];
X_hat = s1u1v1 + s2u2v2 + s3u3v3 + s4u4v4

fig, axs = plt.subplots(1, 11, figsize=(18, 3))

plt.sca(axs[0])
ax = sns.heatmap(X_hat,cmap='RdBu_r',vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\hat{X}$')

plt.sca(axs[1])
plt.title(r'$=$')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(X, cmap='RdBu_r',vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$X$')

plt.sca(axs[3])
plt.title('=')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(s1u1v1,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$s_1*u_1@v_1^T$')

plt.sca(axs[5])
plt.title('+')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(s2u2v2,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$s_2*u_2@v_2^T$')

plt.sca(axs[7])
plt.title('+')
plt.axis('off')

plt.sca(axs[8])
ax = sns.heatmap(s3u3v3,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$s_3*u_3@v_3^T$')

plt.sca(axs[9])
plt.title('+')
plt.axis('off')

plt.sca(axs[10])
ax = sns.heatmap(s4u4v4,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$s_4*u_4@v_4^T$')
plt.show()

######### X - X_hat
fig, axs = plt.subplots(1, 3, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(X, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$X$')

plt.sca(axs[1])
ax = sns.heatmap(X_hat, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\hat{X}$')

plt.sca(axs[2])
ax = sns.heatmap(X - X_hat, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$E = X - \hat{X}$')

################## 数据还原和误差
Z = X @ V
#  projection,  X 向 v_j 映射结果为 s_j u_j
for j in [0, 1, 2, 3]:
    fig, axs = plt.subplots(1, 9, figsize=(12, 3))
    v_j = V[:,j]
    v_j = np.matrix(v_j).T
    s_j = S[j,j]
    s_j = np.matrix(s_j)
    u_j = U[:,j]
    u_j = np.matrix(u_j).T

    plt.sca(axs[0])
    ax = sns.heatmap(Z[:,j].reshape(-1,1),cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title(f'Z_{j}')

    plt.sca(axs[1])
    plt.title('=')
    plt.axis('off')

    plt.sca(axs[2])
    ax = sns.heatmap(X, cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title('X')

    plt.sca(axs[3])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[4])
    ax = sns.heatmap(v_j,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title('v_'+ str(j+1))

    plt.sca(axs[5])
    plt.title('=')
    plt.axis('off')

    plt.sca(axs[6])
    ax = sns.heatmap(s_j,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title('s_'+ str(j+1))

    plt.sca(axs[7])
    plt.title('@')
    plt.axis('off')

    plt.sca(axs[8])
    ax = sns.heatmap(u_j,cmap='RdBu_r', vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
    ax.set_aspect("equal")
    plt.title('u_'+ str(j+1))

######## X = Z_1 @ v_1^T + Z_2 @ v_2^T + Z_3 @ v_4^T + Z_4 v_4^T
# Tensor products, 计算张量积,以及绘制还原原始数据过程热图
Z1_outer_v1 = np.outer(Z[:,0][:, None], V[:,0][:, None]);
Z2_outer_v2 = np.outer(Z[:,1][:, None], V[:,1][:, None]);
Z3_outer_v3 = np.outer(Z[:,2][:, None], V[:,2][:, None]);
Z4_outer_v4 = np.outer(Z[:,3][:, None], V[:,3][:, None]);

X_hat1 = Z1_outer_v1 + Z2_outer_v2 + Z3_outer_v3 + Z4_outer_v4

fig, axs = plt.subplots(1, 11, figsize=(18, 3))
plt.sca(axs[0])
ax = sns.heatmap(X_hat1,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\hat{X}_1$')

plt.sca(axs[1])
plt.title('$=$')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(X, cmap='RdBu_r',vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$X$')

plt.sca(axs[3])
plt.title('=')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(Z1_outer_v1,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title('$Z_1v_1^T$')

plt.sca(axs[5])
plt.title('+')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(Z2_outer_v2,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$Z_2v_2^T$')

plt.sca(axs[7])
plt.title('+')
plt.axis('off')

plt.sca(axs[8])
ax = sns.heatmap(Z3_outer_v3,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$Z_3v_3^T$')

plt.sca(axs[9])
plt.title('+')
plt.axis('off')

plt.sca(axs[10])
ax = sns.heatmap(Z4_outer_v4,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$Z_4v_4^T$')
plt.show()

######### X - X_hat1
fig, axs = plt.subplots(1, 3, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(X, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$X$')

plt.sca(axs[1])
ax = sns.heatmap(X_hat1, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\hat{X}_1$')

plt.sca(axs[2])
ax = sns.heatmap(X - X_hat1, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$E = X - \hat{X}_1$')

######## 张量积: X = X @ (v1@v1^T) + X @ (v2@v2^T) + X @ (v3@v3^T) + X @ (v4@v4^T)
# Tensor products, 计算张量积,以及绘制还原原始数据过程热图
Z1_outer_v11 = X @ np.outer(V[:,0][:, None], V[:,0][:, None]);
Z2_outer_v22 = X @ np.outer(V[:,1][:, None], V[:,1][:, None]);
Z3_outer_v33 = X @ np.outer(V[:,2][:, None], V[:,2][:, None]);
Z4_outer_v44 = X @ np.outer(V[:,3][:, None], V[:,3][:, None]);

X_hat2 = Z1_outer_v11 + Z2_outer_v22 + Z3_outer_v33 + Z4_outer_v44

fig, axs = plt.subplots(1, 11, figsize=(18, 3))
plt.sca(axs[0])
ax = sns.heatmap(X_hat2,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\hat{X}_2$')

plt.sca(axs[1])
plt.title(r'$=$')
plt.axis('off')

plt.sca(axs[2])
ax = sns.heatmap(X, cmap='RdBu_r',vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$X$')

plt.sca(axs[3])
plt.title('=')
plt.axis('off')

plt.sca(axs[4])
ax = sns.heatmap(Z1_outer_v11,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$X@v_1@v_1^T$')

plt.sca(axs[5])
plt.title('+')
plt.axis('off')

plt.sca(axs[6])
ax = sns.heatmap(Z2_outer_v2,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$X@v_2@v_2^T$')

plt.sca(axs[7])
plt.title('+')
plt.axis('off')

plt.sca(axs[8])
ax = sns.heatmap(Z3_outer_v3,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$X@v_3@v_3^T$')

plt.sca(axs[9])
plt.title('+')
plt.axis('off')

plt.sca(axs[10])
ax = sns.heatmap(Z4_outer_v4,cmap='RdBu_r',vmax = all_max,vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$X@v_4@v_4^T$')
plt.show()

######### X - X_hat2
fig, axs = plt.subplots(1, 3, figsize=(12, 3))

plt.sca(axs[0])
ax = sns.heatmap(X, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$X$')

plt.sca(axs[1])
ax = sns.heatmap(X_hat2, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$\hat{X}_2$')

plt.sca(axs[2])
ax = sns.heatmap(X - X_hat2, cmap='RdBu_r', vmax = all_max, vmin = all_min, cbar_kws={"orientation": "horizontal"})
ax.set_aspect("equal")
plt.title(r'$E = X - \hat{X}_2$')























































































































































