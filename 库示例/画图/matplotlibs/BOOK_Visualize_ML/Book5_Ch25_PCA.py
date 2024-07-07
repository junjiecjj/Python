#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 15:19:05 2024

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

mean = [1, 2]
# center of data
cov = [[1, 1], [1, 1.5]]
# covariance matrix

X = np.random.multivariate_normal(mean, cov, 500)

fig, ax = plt.subplots()
plt.scatter(X[:, 0], X[:, 1], alpha = 0.5, marker = '.')

plt.axvline(x=mean[0], color='r', linestyle='--')
plt.axhline(y=mean[1], color='r', linestyle='--')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('scaled')

ax.set_xlim([-3,5])
ax.set_ylim([-2,6])

X_df = pd.DataFrame(X, columns=['x_1', 'x_2'])

sns.jointplot(data=X_df,x = 'x_1', y = 'x_2', kind = 'kde', fill = True, xlim = (-3,5), ylim = (-2,6))

ax.set_aspect('equal')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')


#%% PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

fig, ax = plt.subplots()

plt.scatter(X[:, 0], X[:, 1], alpha = 0.5, marker = '.')

plt.axvline(x=mean[0], color='r', linestyle='--')
plt.axhline(y=mean[1], color='r', linestyle='--')

# plot first principal component, PC1

PC1_x = pca.components_[0,0]
PC1_y = pca.components_[0,1]

ax.quiver(mean[0],mean[1],PC1_x,PC1_y, angles='xy', scale_units='xy',scale=1/3, edgecolor='none', facecolor= 'b')

# plot second principal component, PC2

PC2_x = pca.components_[1,0]
PC2_y = pca.components_[1,1]

ax.quiver(mean[0],mean[1], PC2_x,PC2_y, angles='xy', scale_units='xy',scale=1/3, edgecolor='none', facecolor= 'r')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('scaled')

ax.set_xlim([-3,5])
ax.set_ylim([-2,6])

# convert X to Z

Z = pca.transform(X)

Z_df = pd.DataFrame(Z, columns=['z_1', 'z_2'])

fig, ax = plt.subplots()

sns.kdeplot(data=Z_df)
sns.rugplot(data=Z_df)

fig, ax = plt.subplots()
plt.scatter(Z[:, 0], Z[:, 1], alpha = 0.5, marker = '.')

plt.axvline(x=0, color='r', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')

plt.xlabel('$z_1$')
plt.ylabel('$z_2$')
plt.axis('scaled')

ax.set_xlim([-5,5])
ax.set_ylim([-5,5])

sns.jointplot(data=Z_df,x = 'z_1', y = 'z_2', kind = 'kde', fill = True, xlim = (-5,5), ylim = (-5,5))

ax.set_aspect('equal')
plt.xlabel('$z_1$')
plt.ylabel('$z_2$')

#%% dimension reduction

pca_PC1 = PCA(n_components=1)
pca_PC1.fit(X)

z1 = pca_PC1.transform(X)
x1_proj = pca_PC1.inverse_transform(z1)
fig, ax = plt.subplots()
plt.scatter(X[:, 0], X[:, 1], alpha = 0.5, marker = '.')
# plot first principal component, PC1

PC1_x = pca_PC1.components_[0,0]
PC1_y = pca_PC1.components_[0,1]

ax.quiver(mean[0], mean[1], PC1_x, PC1_y, angles='xy', scale_units='xy', scale=1/3, edgecolor='none', facecolor= 'b')
plt.scatter(x1_proj[:, 0], x1_proj[:, 1], alpha=0.5, c = 'k', marker = 'x')

plt.axvline(x=mean[0], color='r', linestyle='--')
plt.axhline(y=mean[1], color='r', linestyle='--')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('scaled')

ax.set_xlim([-3,5])
ax.set_ylim([-2,6])



# #%%
# from sklearn.datasets import load_iris
# import numpy as np


# iris = load_iris()

# print(iris.data.shape) # 150,4
# X = iris.data
# # 中心化
# X = X - X.mean(axis=0)
# # 计算协方差矩阵
# XXT = np.matrix(X.T) * np.matrix(X) / (len(X)-1)

# # 求特征值和特征向量
# eigVals, eigVects = np.linalg.eig(np.mat(XXT))
# print("特征值: ", eigVals)
# print("特征向量: ", eigVects)



# # 工具包自动计算

# from sklearn.decomposition import PCA

# pca = PCA(n_components=2, whiten='True',svd_solver='full')
# iris = load_iris()
# X = iris.data
# pca.fit(X)
# print(pca.explained_variance_)
# # [4.22824171 0.24267075]
# print(pca.components_.T)
# # [[ 0.36138659  0.65658877]
# #  [-0.08452251  0.73016143]
# #  [ 0.85667061 -0.17337266]
# #  [ 0.3582892  -0.07548102]]

# # 数据变换
# X1 = pca.transform(iris.data.T)
# print(X1.shape) # 150,2




#%% 3D特征值分解投影
# 方式一
plt.subplot(projection='3d')

# # 方式二
# # get current axes
# plt.figure(figsize=(14, 10))
# axes = plt.gca(projection='3d')

x = np.linspace(0, 10, 20)
y = np.linspace(2, 8, 20)
z = 2*x + 5*y + 3
plt.plot(x, y, z)



# https://www.cnblogs.com/shanger/p/13201139.html
plt.subplot(projection='3d')
# 三维平面, 要求X,Y都是二维的
x = np.linspace(0, 10, 20)
y = np.linspace(2, 8, 20)
X,Y = np.meshgrid(x, y)
Z = 2*X + 5*Y + 3

plt.figure(figsize=(14, 10))
axes = plt.gca(projection='3d')
axes.plot_surface(X, Y, Z, color = 'b', alpha = 0.2)

axes.plot(x, y, z, c='k', lw=2,)





#%% Xc的SVD分解(奇异值)和协方差矩阵的特征值分解(特征值)的关系
# Load the iris data
iris_sns = sns.load_dataset("iris")
X = np.array(iris_sns[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
Xc = X - X.mean(axis =  0)

Sigma = Xc.T @ Xc / (X.shape[0] - 1)
SIGMA = np.cov(X.T)


LAMBDA_, V = np.linalg.eig(SIGMA)
LAMBDA = np.diag(LAMBDA_)


U, s, Vt = np.linalg.svd(Xc, full_matrices = True)
S = np.diag(s)
Lambda_reproduced = S**2/(len(X) - 1)


print(Lambda_reproduced - LAMBDA)
print(V - Vt.T)














































































































































































