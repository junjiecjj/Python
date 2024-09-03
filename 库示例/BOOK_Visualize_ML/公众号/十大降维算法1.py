#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 22:24:47 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247486289&idx=1&sn=4d4e92cfb61ff84ebcc25f299e5f8580&chksm=c16717d176f1dc4e5ad2fe15477f1ec59761d71a743a3199e43ceb21cf637bc1833e74db7a8b&mpshare=1&scene=1&srcid=0903da3Zom3MRbZXcazs395A&sharer_shareinfo=8ff48d6025ae07f2f9c730734fcd5e1a&sharer_shareinfo_first=8ff48d6025ae07f2f9c730734fcd5e1a&exportkey=n_ChQIAhIQSlhfMNCSsXo5GC1de3IdfBKfAgIE97dBBAEAAAAAAHipEtmRbTEAAAAOpnltbLcz9gKNyK89dVj0xjg7YzdlbOhIQ4A1GffFxiznr%2F1KQNoI%2BcsS%2Fl7zNYPWQonPCYF0tsgReS5j8nX2KfALORQZ4G8LuQ1jYiMJpMqTtQXqd%2B%2BlxK1v9Rj3xrJEBDbHgKS2PNGyjmHvXs2WIrp8Lbuc0z%2FTXqGI0O%2Bsvh3ZcmRvGpWVWcicBPQoOZ2Vo03%2F38HGynRxnXR4iHth37su7t%2FpiAAQ1tFiwT%2BLsx%2FQ0Akhf%2FVK7ucT95AVxBoZCH2Qup6xnIdLeeMrPRezN0C%2FL2DjeyRwJ1NWTtVAbEwIp0X7sGx2uRiS6n13oDfM8N%2BmpTyzTpuFueoItVQEger%2BNAKt3K1S&acctmode=0&pass_ticket=1Y22VQujACDvTbXEm4ZbVYZwFf8ilrSqEGOvRkeew%2BVeKJdG1A9ZiQ1gVqFlbK%2B1&wx_header=0#rd


"""






#%%>>>>>>>>>>>>>>>>>>>>>>> 1. 主成分分析 (PCA)

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



#%%>>>>>>>>>>>>>>>>>>>>>>> 2. 线性判别分析 (LDA)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D

# 生成三分类的虚拟数据集
X, y = make_classification(n_samples=1000,
                           n_features=3,
                           n_informative=3,
                           n_redundant=0,
                           n_clusters_per_class=1,
                           n_classes=3,
                           class_sep=2,
                           random_state=42)

# 初始化LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# 获取LDA的权重向量
weights = lda.coef_

# 创建图形
fig = plt.figure(figsize=(18, 6))

# 1. 原始数据的三维散点图
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
colors = ['r', 'g', 'b']
markers = ['o', '^', 's']
for class_value in np.unique(y):
    ax1.scatter(X[y == class_value, 0],
                X[y == class_value, 1],
                X[y == class_value, 2],
                c=colors[class_value],
                marker=markers[class_value],
                label=f'Class {class_value}')
ax1.set_title('3D Scatter Plot of Original Data')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_zlabel('Feature 3')
ax1.legend()

# 2. LDA降维后的二维散点图
ax2 = fig.add_subplot(1, 3, 2)
for class_value in np.unique(y):
    ax2.scatter(X_lda[y == class_value, 0],
                X_lda[y == class_value, 1],
                c=colors[class_value],
                marker=markers[class_value],
                label=f'Class {class_value}',
                edgecolor='k',
                alpha=0.7)
ax2.set_title('2D Scatter Plot after LDA')
ax2.set_xlabel('LD1')
ax2.set_ylabel('LD2')
ax2.legend()

# 3. 可视化类间距离和LDA的投影方向
ax3 = fig.add_subplot(1, 3, 3)

# 计算类中心
class_centers = lda.means_

# 绘制类间距离（类中心之间的距离）
for i in range(len(class_centers)):
    for j in range(i + 1, len(class_centers)):
        ax3.plot([class_centers[i, 0], class_centers[j, 0]],
                 [class_centers[i, 1], class_centers[j, 1]],
                 'k--', linewidth=1)

# 绘制LDA的投影方向
for i in range(weights.shape[0]):
    ax3.arrow(0, 0,
              weights[i, 0]*3,
              weights[i, 1]*3,
              color='k',
              width=0.05,
              head_width=0.2)
    ax3.text(weights[i, 0]*3.1, weights[i, 1]*3.1,
             f'W{i+1}', color='k', fontsize=12)

ax3.set_title('Inter-class Distances and LDA Directions')
ax3.set_xlabel('LD1')
ax3.set_ylabel('LD2')
ax3.grid(True)

# 调整布局并显示图形
plt.tight_layout()
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>> 3. 独立成分分析 (ICA)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# 设置随机种子
np.random.seed(0)

# 生成虚拟信号数据
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# 生成3个独立的信号：正弦波、方波和噪声
s1 = np.sin(2 * time)  # 正弦波
s2 = np.sign(np.sin(3 * time))  # 方波
s3 = np.random.normal(size=n_samples)  # 高斯噪声

# 将信号组合成矩阵
S = np.c_[s1, s2, s3]

# 将信号标准化到范围内
S /= S.std(axis=0)

# 生成混合数据（线性混合）
A = np.array([[1, 1, 0.5], [0.5, 2, 1], [1.5, 1, 2.5]])  # 混合矩阵
X = np.dot(S, A.T)  # 生成混合信号

# 使用FastICA从混合信号中分离独立成分
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  # 重构后的信号
A_ = ica.mixing_  # 分离出的混合矩阵

# 绘制原始信号、混合信号和分离出的信号
plt.figure(figsize=(15, 10))

# 原始信号
plt.subplot(3, 1, 1)
plt.title("Original Signals")
colors = ['red', 'blue', 'green']
for i, signal in enumerate(S.T):
    plt.plot(time, signal, color=colors[i], label=f"Signal {i+1}")
plt.legend(loc='upper right')

# 混合信号
plt.subplot(3, 1, 2)
plt.title("Mixed Signals")
colors = ['orange', 'purple', 'brown']
for i, signal in enumerate(X.T):
    plt.plot(time, signal, color=colors[i], label=f"Mixed {i+1}")
plt.legend(loc='upper right')

# 分离出的信号
plt.subplot(3, 1, 3)
plt.title("ICA Recovered Signals")
colors = ['cyan', 'magenta', 'yellow']
for i, signal in enumerate(S_.T):
    plt.plot(time, signal, color=colors[i], label=f"Recovered {i+1}")
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 4. 特征选择 (Feature Selection)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 生成虚拟数据集
X, y = make_classification(n_samples=200, n_features=10, n_informative=5, n_redundant=2, n_classes=2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 初始化分类器（SVM）
svc = SVC(kernel="linear")

# 使用RFE进行特征选择
rfe = RFE(estimator=svc, n_features_to_select=5)
X_rfe = rfe.fit_transform(X_scaled, y)

# 使用PCA进行降维到2D用于可视化
pca = PCA(n_components=2)
X_pca_before = pca.fit_transform(X_scaled)
X_pca_after = pca.fit_transform(X_rfe)

# 绘制特征选择前后的PCA可视化
plt.figure(figsize=(12, 6))

# 特征选择前的PCA
plt.subplot(1, 2, 1)
plt.scatter(X_pca_before[y==0, 0], X_pca_before[y==0, 1], color='red', label='Class 0', alpha=0.6)
plt.scatter(X_pca_before[y==1, 0], X_pca_before[y==1, 1], color='blue', label='Class 1', alpha=0.6)
plt.title('PCA Before Feature Selection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# 特征选择后的PCA
plt.subplot(1, 2, 2)
plt.scatter(X_pca_after[y==0, 0], X_pca_after[y==0, 1], color='red', label='Class 0', alpha=0.6)
plt.scatter(X_pca_after[y==1, 0], X_pca_after[y==1, 1], color='blue', label='Class 1', alpha=0.6)
plt.title('PCA After Feature Selection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# 显示图像
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 5. 奇异值分解 (SVD)
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 生成一个随机矩阵（比如：50x50）
original_matrix = np.random.rand(50, 50)

# 对矩阵执行 SVD 分解
U, S, VT = np.linalg.svd(original_matrix, full_matrices=False)

# 定义重构矩阵的奇异值数量
k_values = [5, 10, 20, 50]

# 创建图像
fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(20, 5))

# 显示原始矩阵
axes[0].imshow(original_matrix, cmap='viridis')
axes[0].set_title("Original Matrix", fontsize=14)
axes[0].axis('off')

# 根据不同的奇异值数量重构矩阵并绘制热力图
for i, k in enumerate(k_values):
    # 使用前 k 个奇异值重构矩阵
    S_k = np.zeros((k, k))
    np.fill_diagonal(S_k, S[:k])

    # 重构矩阵
    reconstructed_matrix = U[:, :k] @ S_k @ VT[:k, :]

    # 显示重构的矩阵
    axes[i + 1].imshow(reconstructed_matrix, cmap='viridis')
    axes[i + 1].set_title(f"Reconstructed Matrix (k={k})", fontsize=14)
    axes[i + 1].axis('off')

# 调整布局
plt.tight_layout()
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>> 6. t-SNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 生成虚拟数据集
n_samples = 1000
X1, y1 = make_blobs(n_samples=n_samples, centers=4, cluster_std=1.0, random_state=42)
X2, y2 = make_moons(n_samples=n_samples, noise=0.1, random_state=42)

# 将两个数据集结合起来
X = np.vstack([X1, X2])
y = np.hstack([y1, y2 + 4])

# 使用PCA进行初步降维
pca = PCA(n_components=2)  # 将组件数改为2
X_pca = pca.fit_transform(X)

# 使用t-SNE降维到2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_pca)

# 绘图
plt.figure(figsize=(14, 7))

# t-SNE降维结果图
plt.subplot(1, 2, 1)
palette = sns.color_palette("hsv", len(np.unique(y)))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette=palette, s=60, legend='full')
plt.title('t-SNE visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

# 原始数据集的PCA降维前后的对比图
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=10, alpha=0.6)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=60, alpha=0.8, edgecolors='k')
plt.title('Original vs t-SNE reduced')
plt.xlabel('Feature 1 / t-SNE Component 1')
plt.ylabel('Feature 2 / t-SNE Component 2')

# 调整图例
plt.legend(title='Class')
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 7. 核主成分分析 (Kernel PCA)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

# 设置随机种子
np.random.seed(42)

# 生成虚拟数据集1: make_blobs (三簇高斯分布点)
n_samples = 1500
X1, y1 = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.0, random_state=42)
X1 = StandardScaler().fit_transform(X1)

# 生成虚拟数据集2: make_moons (两个互锁的半月形)
X2, y2 = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
X2 = StandardScaler().fit_transform(X2)

# 将两个数据集合并
X = np.vstack([X1, X2])
y = np.hstack([y1, y2 + 3])

# 使用PCA先做一次初步降维 (为了将UMAP结果与PCA进行比较)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 使用UMAP进行降维
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X)

# 设置颜色
colors = sns.color_palette("hsv", 5)  # 生成5种鲜艳的颜色

# 创建图形
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# 第一幅图：原始数据分布 (数据集1)
axs[0, 0].scatter(X1[:, 0], X1[:, 1], c=y1, cmap='viridis', s=5, alpha=0.8)
axs[0, 0].set_title("Original Data (make_blobs)")
axs[0, 0].set_xlabel("Feature 1")
axs[0, 0].set_ylabel("Feature 2")

# 第二幅图：原始数据分布 (数据集2)
axs[0, 1].scatter(X2[:, 0], X2[:, 1], c=y2, cmap='plasma', s=5, alpha=0.8)
axs[0, 1].set_title("Original Data (make_moons)")
axs[0, 1].set_xlabel("Feature 1")
axs[0, 1].set_ylabel("Feature 2")

# 第三幅图：PCA降维后的数据分布
sc = axs[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='Spectral', s=5, alpha=0.8)
axs[1, 0].set_title("PCA Reduction")
axs[1, 0].set_xlabel("PCA Component 1")
axs[1, 0].set_ylabel("PCA Component 2")

# 第四幅图：UMAP降维后的数据分布
sc = axs[1, 1].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='Spectral', s=5, alpha=0.8)
axs[1, 1].set_title("UMAP Reduction")
axs[1, 1].set_xlabel("UMAP Component 1")
axs[1, 1].set_ylabel("UMAP Component 2")

# 调整布局并显示图像
plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 8. 因子分析 (Factor Analysis)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

# 生成虚拟数据集
n_samples = 1000
n_features = 6
n_components = 2

X, _ = make_blobs(n_samples=n_samples, centers=5, n_features=n_features, random_state=42)

# 数据预处理，将数据缩放到0到1之间
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 使用NMF进行降维
nmf = NMF(n_components=n_components, random_state=42)
X_nmf = nmf.fit_transform(X_scaled)

# 可视化原始高维数据集（只选取前两个特征进行2D可视化）
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='blue', edgecolor='k', s=50, cmap='rainbow')
plt.title('Original High-Dimensional Data (2D projection)', fontsize=14)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.grid(True)

# 可视化降维后的数据
plt.subplot(1, 2, 2)
plt.scatter(X_nmf[:, 0], X_nmf[:, 1], c='red', edgecolor='k', s=50, cmap='rainbow')
plt.title('Data after NMF (2 components)', fontsize=14)
plt.xlabel('Component 1', fontsize=12)
plt.ylabel('Component 2', fontsize=12)
plt.grid(True)

# 显示所有图像
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 9. 多维尺度分析 (MDS)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_moons, make_circles
from mpl_toolkits.mplot3d import Axes3D

# 生成非线性数据集
n_samples = 1000
X_moons, y_moons = make_moons(n_samples=n_samples, noise=0.05, random_state=42)
X_circles, y_circles = make_circles(n_samples=n_samples, factor=0.3, noise=0.05, random_state=42)

# 将数据集叠加在一起以形成一个更复杂的高维数据集
X = np.vstack([X_moons, X_circles])
y = np.hstack([y_moons, y_circles + 2])

# 添加一个额外的维度
X = np.hstack([X, np.sin(X[:, 0:1])])

# 使用标准PCA进行降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 使用核PCA进行降维
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X)

# 创建图形
fig = plt.figure(figsize=(18, 6))

# 原始数据的3D分布
ax1 = fig.add_subplot(131, projection='3d')
scatter1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='Spectral', edgecolor='k')
ax1.set_title('Original High-Dimensional Data', fontsize=14)
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('X3')

# 标准PCA降维结果
ax2 = fig.add_subplot(132)
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='Spectral', edgecolor='k')
ax2.set_title('PCA Projection', fontsize=14)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')

# 核PCA降维结果
ax3 = fig.add_subplot(133)
scatter3 = ax3.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='Spectral', edgecolor='k')
ax3.set_title('Kernel PCA Projection', fontsize=14)
ax3.set_xlabel('KPC1')
ax3.set_ylabel('KPC2')

# 添加颜色条
cbar = fig.colorbar(scatter3, ax=[ax1, ax2, ax3], orientation='horizontal', fraction=0.05, pad=0.1)
cbar.set_label('Class Label', fontsize=12)

plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 10. 随机投影 (Random Projection)





















