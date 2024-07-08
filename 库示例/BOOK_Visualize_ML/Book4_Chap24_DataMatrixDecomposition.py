





# Bk4_Ch24_01_A

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# A copy from Seaborn
iris = load_iris()

X = iris.data
y = iris.target

feature_names = ['Sepal length, x1','Sepal width, x2',
                 'Petal length, x3','Petal width, x4']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

#%% Original data, X

X = X_df.to_numpy();

#%% Gram matrix, G
G = X.T@X

#%% Cosine similarity matrix, C
from sklearn.metrics.pairwise import cosine_similarity
C = cosine_similarity(X.T)
from numpy.linalg import inv

S_norm = np.diag(np.sqrt(np.diag(G)))
# scaling matrix, diagnal element is the norm of x_j

C = inv(S_norm)@G@inv(S_norm)

#%% centroid of data matrix, E(X)
E_X = X_df.mean().to_frame().T

#%% Demean, centralize, X_c
X_c = X_df.sub(X_df.mean())



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

#%% Normalize data, Z_X

from scipy.stats import zscore

Z_X = zscore(X_df)

#%% Bk4_Ch24_01_B
#%% QR decomposition
from numpy.linalg import qr

Q, R = qr(X_df, mode = 'reduced')

#%%  Bk4_Ch24_01_C
#%% Cholesky decomposition
from numpy.linalg import cholesky as chol

L_G = chol(G)
R_G = L_G.T

#%% Cholesky decompose covariance matrix, SIGMA
L_Sigma = chol(SIGMA)

R_Sigma = L_Sigma.T

#%% Bk4_Ch24_01_D
#%% eigen decompose G
from numpy.linalg import eig

Lambs_G, V_G = eig(G)
Lambs_G = np.diag(Lambs_G)

#%% eigen decompose Sigma, covariance matrix
Lambs_sigma, V_sigma = eig(SIGMA)
Lambs_sigma = np.diag(Lambs_sigma)

#%% eigen decompose P, correlation matrix
Lambs_P, V_P = eig(RHO)
Lambs_P = np.diag(Lambs_P)

#%% Bk4_Ch24_01_E
#%% SVD, original data X

from numpy.linalg import svd

U_X,S_X_,V_X = svd(X_df, full_matrices=False)
V_X = V_X.T

# full_matrices=True
# indices_diagonal = np.diag_indices(4)
# S_X = np.zeros_like(X_df)
# S_X[indices_diagonal] = S_X_

# full_matrices=False
S_X = np.diag(S_X_)

#%% SVD, original data Xc

U_Xc, S_Xc, V_Xc = svd(X_c, full_matrices=False)
V_Xc = V_Xc.T
S_Xc = np.diag(S_Xc)

#%% SVD, z scores
U_Z, S_Z, V_Z = svd(Z_X, full_matrices = False)
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

U_X,S_X_,V_X = svd(X, full_matrices=False)
V_X = V_X.T

# full_matrices=True
# indices_diagonal = np.diag_indices(4)
# S_X = np.zeros_like(X_df)
# S_X[indices_diagonal] = S_X_

# full_matrices=False
S_X = np.diag(S_X_)

#%% SVD, centralized data Xc
X_c = X - X.mean(axis = 0)

U_Xc, S_Xc, V_Xc = svd(X_c, full_matrices=False)
V_Xc = V_Xc.T
S_Xc = np.diag(S_Xc)

#%% SVD, z scores
from scipy.stats import zscore

Z_X = zscore(X)

U_Z, S_Z, V_Z = svd(Z_X, full_matrices = False)
V_Z = V_Z.T
S_Z = np.diag(S_Z)

















































































































































































































































































