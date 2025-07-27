#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 13:54:49 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247491429&idx=1&sn=69c43dce778c85ed62413dba3b228823&chksm=c1a88632879cbca92cf86b713953e63d56e3b99f31f3122f50fe0e626ed527b828691f1f4169&mpshare=1&scene=1&srcid=0726cslxYtgq247xCIdEbJxQ&sharer_shareinfo=2c385bb5b0e1e78132fadf2098856d9a&sharer_shareinfo_first=2c385bb5b0e1e78132fadf2098856d9a&exportkey=n_ChQIAhIQ%2F40v8dN6dEjKZO4Se6MQrBKfAgIE97dBBAEAAAAAANxJIfK4j4IAAAAOpnltbLcz9gKNyK89dVj0I0yC6TgiYy8%2FuA0xH5QLpEno%2BqlobNDYpE3S0%2FGFG02kKbqkIk9qAlBFNbHLKz0g2cvPB1T2F%2Bl6pvwIkJyMA2sg9x%2F7UPSNgs8FyO3bB9dlJeLlC9BU4Lr1YKZXKjQyLUWjTvxVgKD3aHdLr2u9gwxjF5LBy7G9qrEHwx9lSxKN%2FTXb1dufAUo%2FiHM1DzvJ805Wi6jxUnxccS8oe1BUTKXZo5U1m%2FS7ZTpRUciSLJf9Vj0tJUGLLgLNpelu87g0lylRyNZGdHaLzgHPSuQ1cShciAD0ZKY5WRwUv0IW1WCMKY8OmfWSbijN3ij8M1UYxklSmDc2ORI8&acctmode=0&pass_ticket=5urxjUGyuDgeR428NsUHY2kWCCcYEJw8ScY862UIvTpp8ROUfV2Mm5Bseiqf2y0C&wx_header=0#rd


"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

# --- 加载 MNIST 数据集 ---
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data  # 数据矩阵 (70000, 784)
y = mnist.target.astype(int)  # 数字标签 (70000,)

# 数据概览
print(f"数据形状: {X.shape}, 标签数量: {np.unique(y)}")

# --- 数据预处理 ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 标准化

# --- PCA降维到2D ---
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# --- 可视化1: 2D分布 ---
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
plt.title("PCA 2D Visualization of MNIST", fontsize=16)
plt.xlabel("Principal Component 1", fontsize=14)
plt.ylabel("Principal Component 2", fontsize=14)
plt.colorbar(scatter, label='Digit Class')
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# --- PCA降维到784维中的前64维 ---
n_components = 64
pca_high = PCA(n_components=n_components)
X_pca_high = pca_high.fit_transform(X_scaled)
X_reconstructed = pca_high.inverse_transform(X_pca_high)  # 用64维重建数据

# --- 可视化2: 原图 vs PCA还原图 ---
n_samples = 10
fig, axes = plt.subplots(2, n_samples, figsize=(15, 4))
for i in range(n_samples):
    # 原始图像
    axes[0, i].imshow(X.iloc[i].values.reshape(28, 28), cmap='gray')
    axes[0, i].axis("off")
    axes[0, i].set_title(f"Original {y[i]}")

    # 还原图像
    axes[1, i].imshow(X_reconstructed[i].reshape(28, 28), cmap='viridis')
    axes[1, i].axis("off")
    axes[1, i].set_title("Reconstructed")

fig.suptitle("MNIST Original Images vs. PCA Reconstructed Images (64 Components)", fontsize=16)
plt.tight_layout()
plt.show()

# --- 可视化3: 主成分贡献率 ---
explained_variance = pca_high.explained_variance_ratio_
plt.figure(figsize=(12, 6))
plt.plot(range(1, n_components + 1), np.cumsum(explained_variance), marker='o', color='b')
plt.title("Cumulative Explained Variance by Principal Components", fontsize=16)
plt.xlabel("Number of Principal Components", fontsize=14)
plt.ylabel("Cumulative Variance Ratio", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
plt.axhline(y=0.95, color='r', linestyle="--", label="95% Variance Explained")
plt.legend(fontsize=12)
plt.show()


"""
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

"""
https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247488019&idx=1&sn=23903ce599082f1fa29c8a7644cd39b4&chksm=c106c7917a44e4e18934f76519027524d9c2c291f5df514e49a2b692f2667d4260d71ddaf3cb&mpshare=1&scene=1&srcid=0104fS5QKBqt1fxgfRezvXhn&sharer_shareinfo=50eaf452ab5c7c1ec727c58b5a367941&sharer_shareinfo_first=50eaf452ab5c7c1ec727c58b5a367941&exportkey=n_ChQIAhIQ7ixRZ3uXs64INTSGX5ZfaBKfAgIE97dBBAEAAAAAAIjgANjcEdAAAAAOpnltbLcz9gKNyK89dVj0IrIqS5D5BrOfWpqyiS63Pm9njjcH4l%2FommmDhIXe3kTGc76Pm3Ovsp4oU2kyqIjS1q1gh65LBClPzZJl7Hxaoc7Ce%2Ftl82LLb3Axs6DFdLTSyjDgtpvVmlt7ho0ep79Gl39JMcEhdAKMjF2P%2FOWN2VCu3Pdtgx951uC1Xo97Szqse0dvSrDNTYOWiLnFxZ36e6oXtt%2FdJilw2dlXEQ8LqMZENjEMX0X%2BNoZxt1o%2BA6db1XBvjeM4Zfz7lKnnRH8WlWFo5%2BWDWzs%2BoLuo27AP8dlCjjXo98POcK%2FwTj3kWGBjfbATPChsPGas9txeMtqOMRLcfR6tanzP&acctmode=0&pass_ticket=A2I%2Fxr1k0Mrwa6Jl7O1epsjZ7FvUYLmAgqs7Pva2A6oae%2BYIYk%2BV%2FbJ%2FjMIruNw9&wx_header=0#rd

"""

#%%>>>>>>>>>>>>>> 1. 主成分分析 (PCA)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成一个三维数据集
mean = [0, 0, 0]
cov = [[1, 0.8, 0.5], [0.8, 1, 0.3], [0.5, 0.3, 1]]  # 协方差矩阵，定义数据的相关性
data = np.random.multivariate_normal(mean, cov, 5000)

# 使用PCA将数据降维到2D
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

# 创建一个图形并绘制多个子图
fig = plt.figure(figsize=(16, 8))

# 第一个子图：三维散点图（原始数据）
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', marker='o', alpha=0.6)
ax1.set_title('3D Scatter Plot of Original Data', fontsize=15)
ax1.set_xlabel('X axis')
ax1.set_ylabel('Y axis')
ax1.set_zlabel('Z axis')

# 第二个子图：二维散点图（PCA降维后的数据）
ax2 = fig.add_subplot(222)
ax2.scatter(data_2d[:, 0], data_2d[:, 1], c='b', marker='o', alpha=0.6)
ax2.set_title('2D Scatter Plot after PCA', fontsize=15)
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')

# 第三个子图：解释方差比例的柱状图
explained_variance = pca.explained_variance_ratio_
ax3 = fig.add_subplot(224)
ax3.bar(range(1, len(explained_variance) + 1), explained_variance, color='g', alpha=0.7)
ax3.set_title('Explained Variance Ratio of Principal Components', fontsize=15)
ax3.set_xlabel('Principal Component')
ax3.set_ylabel('Variance Explained')

# 调整布局
plt.tight_layout()
plt.show()
