#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:57:02 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247488850&idx=1&sn=d3533dc33db7843fe8dee0be659afad1&chksm=c16ff99f18f19f7c7fb75d8ad8ed174e4c9727a6c5fa706f1686371f3fa5d1ed629ce8bdddb8&mpshare=1&scene=1&srcid=01041RiX8zCqb8uBLR9t2poQ&sharer_shareinfo=dc8f45010ed3822e4dc2ecb73c42733f&sharer_shareinfo_first=dc8f45010ed3822e4dc2ecb73c42733f&exportkey=n_ChQIAhIQYhwweveq%2BMlh%2FcC44AwWrRKfAgIE97dBBAEAAAAAAD8HIJI7Fh0AAAAOpnltbLcz9gKNyK89dVj0HDmc%2BhTTygq62KW%2FEFgyr3LpPZR%2BR6pyAodSGOX8P%2BP4rVIDRJEZgGeNfZyEpDeUC9FzNnOX42Y7F7Numty3fT8CtUQpkgFp%2BZefZAOgm%2FC4Ty%2BTUSbn9lrDBqalS98nVLammAzsu2THyRRchdTcaiXdlDoWWxOS0u3EGS5ggnF6QvpSaRXYj5Tjh3UE%2Bq%2FCFzeMHCGhD957kG43YTdImrwa5tu0jEinTVbgtAEMQBCsYpRns0q3EmSniaaL%2BaDxJZOytLs8EM7Ne5wIra7%2BglYrvR7o1oM4TqbLto0yc%2F8N46idWYchKgoPyIRRr%2Bj7rQkPZBCJZTlA&acctmode=0&pass_ticket=GP%2F6KZGaN7pUSVyaTVxFGZ%2B31HEa0eTbh6TOEoYpTyKdLxQO%2BPiIGvJyralsnZKz&wx_header=0#rd

"""

#%%>>>>>>>>>>>>>> 主成分分析

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. 生成虚拟数据集
np.random.seed(42)
n_samples = 10000
X = np.dot(np.random.rand(n_samples, 3), [[2, 1, 1], [1, 2, 1], [1, 1, 2]]) + np.random.normal(size=(n_samples, 3))

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 主成分方向
components = pca.components_

# 3. 绘图
fig, ax = plt.subplots(1, 3, figsize=(18, 6), dpi=120)

# 原始数据散点图 (前两维投影)
ax[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c='cyan', alpha=0.7, edgecolors='k')
ax[0].set_title("Original Data (First 2 Features)")
ax[0].set_xlabel("Feature 1")
ax[0].set_ylabel("Feature 2")

# 投影后数据散点图
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c='magenta', alpha=0.7, edgecolors='k')
ax[1].set_title("Data After PCA")
ax[1].set_xlabel("Principal Component 1")
ax[1].set_ylabel("Principal Component 2")

# 主成分方向的可视化
for i, (comp, var) in enumerate(zip(components, pca.explained_variance_)):
    ax[2].arrow(0, 0, comp[0] * var, comp[1] * var,
                color=f'C{i}', width=0.02, head_width=0.1, label=f'PC{i+1}')
ax[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c='lightgreen', alpha=0.7, edgecolors='k')
ax[2].legend()
ax[2].set_title("Principal Components in Original Space")
ax[2].set_xlabel("Feature 1")
ax[2].set_ylabel("Feature 2")

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>> 线性判别分析



import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 第一步：生成合成数据集
X, y = make_classification(n_samples=2000, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, n_classes=2, random_state=42)

# 分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 第二步：应用LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# 将数据投影到LDA组件上
X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)

# 第三步：预测并评估
y_pred = lda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"LDA 分类准确率: {accuracy:.2f}")

# 第四步：可视化
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Original data distribution
for label, color in zip(np.unique(y), ['red', 'blue']):
    ax[0].scatter(X[y == label, 0], X[y == label, 1], c=color, label=f"Class {label}", alpha=0.6)
ax[0].set_title("Original Data Distribution")
ax[0].set_xlabel("Feature 1")
ax[0].set_ylabel("Feature 2")
ax[0].legend()

# LDA projected data
for label, color in zip(np.unique(y_train), ['red', 'blue']):
    ax[1].hist(X_train_lda[y_train == label], bins=15, alpha=0.6, color=color, label=f"Class {label}")
ax[1].set_title("LDA Projected Data")
ax[1].set_xlabel("LDA Component 1")
ax[1].set_ylabel("Frequency")
ax[1].legend()

# Decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax[2].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
for label, color in zip(np.unique(y), ['red', 'blue']):
    ax[2].scatter(X[y == label, 0], X[y == label, 1], c=color, label=f"Class {label}", edgecolor='k')
ax[2].set_title("LDA Decision Boundary")
ax[2].set_xlabel("Feature 1")
ax[2].set_ylabel("Feature 2")
ax[2].legend()

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>> 因子分析

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
import seaborn as sns

# 1. 生成虚拟数据
np.random.seed(42)

# 创建 8 个变量，其中部分变量具有潜在因子相关性
n_samples = 3000
n_features = 8

# 模拟两个潜在因子
latent_factors = np.random.normal(size=(n_samples, 2))

# 定义因子载荷矩阵（8个变量由2个潜在因子主导）
loadings = np.array([
    [0.9, 0.1],  # 强依赖于因子1
    [0.8, 0.2],
    [0.85, 0.15],
    [0.1, 0.9],  # 强依赖于因子2
    [0.2, 0.85],
    [0.25, 0.8],
    [0.5, 0.5],  # 混合依赖
    [0.4, 0.6]
])

# 生成观测数据
X = np.dot(latent_factors, loadings.T) + np.random.normal(size=(n_samples, n_features)) * 0.1
columns = [f"Var{i+1}" for i in range(n_features)]
data = pd.DataFrame(X, columns=columns)

# 2. 因子分析
fa = FactorAnalysis(n_components=2, random_state=42)
fa.fit(data)

# 提取因子载荷矩阵和因子得分
factor_loadings = fa.components_.T
factor_scores = fa.transform(data)

# 3. 可视化
plt.figure(figsize=(12, 6))

# 因子载荷热力图
plt.subplot(1, 2, 1)
sns.heatmap(factor_loadings, annot=True, cmap="YlGnBu", xticklabels=["Factor1", "Factor2"], yticklabels=columns)
plt.title("Factor Loadings Heatmap")
plt.xlabel("Factors")
plt.ylabel("Variables")

# 因子得分散点图
plt.subplot(1, 2, 2)
plt.scatter(factor_scores[:, 0], factor_scores[:, 1], c="tomato", alpha=0.7, edgecolor="k")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
plt.title("Factor Scores Scatter Plot")
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>> 独立成分分析

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Step 1: 数据生成
np.random.seed(42)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# 独立信号生成
s1 = np.sin(2 * time)  # 正弦信号
s2 = np.sign(np.sin(3 * time))  # 方波信号
s3 = np.random.normal(size=n_samples)  # 高斯噪声

# 合并信号
S = np.c_[s1, s2, s3]
S /= S.std(axis=0)  # 标准化

# 混合信号生成
A = np.array([[1, 1, 0.5], [0.5, 2, 1], [1.5, 1, 2]])  # 混合矩阵
X = np.dot(S, A.T)  # 观测信号

# Step 2: 应用 ICA
ica = FastICA(n_components=3, random_state=42)
S_estimated = ica.fit_transform(X)  # 分离出的信号
A_estimated = ica.mixing_  # 估计的混合矩阵

# Step 3: 绘图分析
fig, axes = plt.subplots(3, 2, figsize=(12, 8), constrained_layout=True)

# 原始独立信号
axes[0, 0].plot(time, S[:, 0], color='red')
axes[0, 0].set_title('Original Signal 1')
axes[0, 1].plot(time, S[:, 1], color='blue')
axes[0, 1].set_title('Original Signal 2')

# 混合信号
axes[1, 0].plot(time, X[:, 0], color='green')
axes[1, 0].set_title('Mixed Signal 1')
axes[1, 1].plot(time, X[:, 1], color='purple')
axes[1, 1].set_title('Mixed Signal 2')

# 分离信号
axes[2, 0].plot(time, S_estimated[:, 0], color='orange')
axes[2, 0].set_title('Recovered Signal 1')
axes[2, 1].plot(time, S_estimated[:, 1], color='cyan')
axes[2, 1].set_title('Recovered Signal 2')

plt.suptitle('Independent Component Analysis (ICA) Results', fontsize=16)
plt.show()




#%%>>>>>>>>>>>>>> 奇异值分解

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 创建虚拟数据集
np.random.seed(42)
m, n = 10, 8  # 矩阵尺寸
A = np.random.rand(m, n) * 100  # 随机生成 m x n 矩阵

# 奇异值分解
U, S, VT = np.linalg.svd(A, full_matrices=False)  # 计算 SVD
Sigma = np.diag(S)  # 将奇异值构造成对角矩阵

# 重构矩阵（验证 SVD 是否正确）
A_reconstructed = U @ Sigma @ VT

# 数据降维 (取前两个奇异值对应的方向)
A_projected = A @ VT.T[:, :2]  # 投影到 2D 空间

# 绘制分析图形
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 图 1: 原始矩阵和重构矩阵的热力图
sns.heatmap(A, annot=True, fmt=".1f", cmap="coolwarm", ax=axes[0], cbar=False)
axes[0].set_title("Original Matrix (A)")

sns.heatmap(A_reconstructed, annot=True, fmt=".1f", cmap="coolwarm", ax=axes[1], cbar=False)
axes[1].set_title("Reconstructed Matrix (UΣVᵀ)")

# 新窗口用于数据降维后的结果
plt.figure(figsize=(8, 6))
plt.scatter(A_projected[:, 0], A_projected[:, 1], c='r', label="Projected Data", edgecolor='k', s=100)
plt.title("Data Projection onto 2D Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.legend()
plt.grid()

plt.show()




#%%>>>>>>>>>>>>>> 多维尺度分析

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs

# 生成虚拟数据集
np.random.seed(42)
data, labels = make_blobs(n_samples=10000, centers=4, n_features=5, random_state=42)

# 计算距离矩阵
distance_matrix = pairwise_distances(data, metric='euclidean')

# MDS降维
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
low_dim_data = mds.fit_transform(distance_matrix)

# 绘图
fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=120)

# 图1：MDS降维后的散点图
scatter = ax[0].scatter(low_dim_data[:, 0], low_dim_data[:, 1], c=labels, cmap='viridis', s=50, edgecolor='k')
ax[0].set_title("MDS Projection (2D)", fontsize=14)
ax[0].set_xlabel("Component 1")
ax[0].set_ylabel("Component 2")
plt.colorbar(scatter, ax=ax[0], label="Cluster Labels")

# 图2：距离矩阵的热力图
im = ax[1].imshow(distance_matrix, cmap='hot', interpolation='nearest')
ax[1].set_title("Distance Matrix Heatmap", fontsize=14)
ax[1].set_xlabel("Sample Index")
ax[1].set_ylabel("Sample Index")
plt.colorbar(im, ax=ax[1], label="Distance")

plt.tight_layout()
plt.show()




#%%>>>>>>>>>>>>>> t-SNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. 生成虚拟数据集
n_samples = 1000
n_features = 50
n_clusters = 5
random_state = 42

X, y = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=random_state)

# 2. 使用PCA降维到3D空间（便于对比）
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# 3. 使用t-SNE降维到2D空间
tsne = TSNE(n_components=2, random_state=random_state, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X)

# 4. KMeans聚类
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
y_kmeans = kmeans.fit_predict(X)

# 5. 图形可视化
fig = plt.figure(figsize=(16, 8))

# 原始PCA 3D降维
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='rainbow', s=10)
ax1.set_title('PCA 3D Visualization')
ax1.set_xlabel('PCA1')
ax1.set_ylabel('PCA2')
ax1.set_zlabel('PCA3')
legend1 = ax1.legend(*scatter.legend_elements(), title="Classes", loc="best")
ax1.add_artist(legend1)

# t-SNE 2D降维
ax2 = fig.add_subplot(1, 2, 2)
scatter = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='rainbow', s=10)
ax2.set_title('t-SNE 2D Visualization')
ax2.set_xlabel('t-SNE1')
ax2.set_ylabel('t-SNE2')
legend2 = ax2.legend(*scatter.legend_elements(), title="Classes", loc="best")
ax2.add_artist(legend2)

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>> UMAP


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_blobs
import umap

# 第一步：生成合成数据集
# Swiss Roll 数据集（非线性结构）
X1, color1 = make_swiss_roll(n_samples=10000, noise=0.2)
# Blobs 数据集（簇状结构）
X2, color2 = make_blobs(n_samples=2000, centers=3, cluster_std=1.0, random_state=42)

# Option 1: Add a third feature to X2 to match the dimensions of X1
X2_new = np.hstack([X2, np.zeros((X2.shape[0], 1))])  # Add a column of zeros to X2
X = np.vstack([X1, X2_new])

# 合并数据集以增加复杂性
color = np.concatenate([color1, color2])  # If color2 is 1D

# 第二步：应用 UMAP 进行降维
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(X)

# 第三步：可视化
fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=120)

# 1. 原始高维结构（Swiss Roll）
axs[0].scatter(X1[:, 0], X1[:, 2], c=color1, cmap='Spectral', s=5)
axs[0].set_title("Swiss Roll in 3D (Projection)")
axs[0].set_xlabel("X 轴")
axs[0].set_ylabel("Z 轴")

# 2. UMAP 2D 嵌入
scatter = axs[1].scatter(embedding[:, 0], embedding[:, 1], c=color, cmap='Spectral', s=5)
axs[1].set_title("UMAP 2D Embedding")
axs[1].set_xlabel("UMAP1")
axs[1].set_ylabel("UMAP2")

# 3. UMAP 嵌入的密度图
sns.kdeplot(x=embedding[:, 0], y=embedding[:, 1], cmap="Reds", fill=True, ax=axs[2])
axs[2].set_title("Density of UMAP Embedding")
axs[2].set_xlabel("UMAP1")
axs[2].set_ylabel("UMAP2")

# 为 UMAP 图添加颜色条
cbar = fig.colorbar(scatter, ax=axs[1], orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label("Color")

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>> 核PCA

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons

# 1. 生成非线性分布数据集
np.random.seed(42)
X, y = make_moons(n_samples=2000, noise=0.05)

# 2. 原始数据可视化
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 3. 核PCA降维
kernel_pca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kernel_pca.fit_transform(X)

# 4. 投影后数据可视化
plt.subplot(1, 3, 2)
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='plasma', edgecolor='k')
plt.title('Kernel PCA Projection')
plt.xlabel('PC1')
plt.ylabel('PC2')

# 5. 累计方差贡献率分析
# 核PCA没有显式的解释方差，因此直接基于特征值来估计
lambdas = kernel_pca.lambdas_
explained_variance_ratio = lambdas / np.sum(lambdas)
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.subplot(1, 3, 3)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', color='red')
plt.title('Cumulative Variance Ratio')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')

# 6. 总体展示
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>> 自编码器


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成虚拟数据集
np.random.seed(42)
torch.manual_seed(42)

# 数据：二维正态分布
data = np.random.randn(1000, 2) * 2
data = torch.tensor(data, dtype=torch.float32)

# 2. 定义自编码器
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# 3. 初始化模型
input_dim = 2
latent_dim = 2
model = Autoencoder(input_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
epochs = 100
losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    x_hat, z = model(data)
    loss = criterion(x_hat, data)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 5. 可视化分析
model.eval()
with torch.no_grad():
    x_hat, z = model(data)

# 创建子图
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# 图1：原始数据 vs 重建数据
axs[0].scatter(data[:, 0], data[:, 1], color='blue', alpha=0.5, label='Original Data')
axs[0].scatter(x_hat[:, 0], x_hat[:, 1], color='red', alpha=0.5, label='Reconstructed Data')
axs[0].set_title('Original vs Reconstructed Data')
axs[0].legend()
axs[0].grid()

# 图2：潜在空间分布
axs[1].scatter(z[:, 0], z[:, 1], color='green', alpha=0.7)
axs[1].set_title('Latent Space Distribution')
axs[1].grid()

plt.tight_layout()
plt.show()












