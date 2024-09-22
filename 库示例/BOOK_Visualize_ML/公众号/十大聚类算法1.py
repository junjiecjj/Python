#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 13:11:55 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247486377&idx=1&sn=1dac4ea7b6df45cd372014f8b0eb4e29&chksm=c14c7cc33e77c12268ab26ce693e93de84f91d867c95917058b050412c3c26305212ff01afd7&mpshare=1&scene=1&srcid=0905TJdTrQxgCM6BYCub9PrN&sharer_shareinfo=d1ff18d877a1789e10be14fdafa97019&sharer_shareinfo_first=d1ff18d877a1789e10be14fdafa97019&exportkey=n_ChQIAhIQhwYJhOC2PM65Ra%2BTVFou2RKfAgIE97dBBAEAAAAAAJYaIoF7njIAAAAOpnltbLcz9gKNyK89dVj0deoMVTSPQIodvmxZBcw2A6wo14b9tF9QcYj5EcvpPoeT3FX4LLEEd47kEx9WBhIYOtlfmW83tvPJVG0%2F3i27U4TVj2ly9k92mgNULgzxqbu4nOZVu%2BC7eioVh%2BHOhAa1TgTP5ZzTF03ZItfx8xkukojsnV8HQMviSmBvCU4%2BkjkWPhMv%2B8tK4QzD%2BlBIDMo%2F8q1CNAvsswosp%2BGIRw487Lj8SkY2uSzDWqBHujscOni5ZsWvLbeGdHOo6e5V6AnSufPNjahWDKDxa2fClXRUiny18wOyVKR6CLKoZjZIpIuMZAeXM8SVHeQBEdg%2FFgq7bsWSbMbZzfS9&acctmode=0&pass_ticket=mPiZrlADS%2BbDL8zAtawFZaG1BRr5nnrnEdd9%2BlnKMuXFkqNkS0NWJSf5lf%2B3Sfpr&wx_header=0#rd


"""

#%%>>>>>>>>>>>>>>>>>>>>>>>>> 1. K-Means 聚类

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# 生成虚拟数据集
np.random.seed(42)
X, y = make_blobs(n_samples=1000, centers=5, cluster_std=1.0, random_state=42)

# K-Means聚类
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 计算轮廓系数
sil_score = silhouette_score(X, y_kmeans)

# 绘制聚类结果
plt.figure(figsize=(18, 8))

# 子图1：原始数据分布
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'rainbow', edgecolors = 'k')
plt.title("Original Data Distribution", fontsize=14)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# 子图2：K-Means聚类结果
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='rainbow', edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', edgecolors='k')
plt.title(f"K-Means Clustering (k=5)\nSilhouette Score = {sil_score:.2f}", fontsize=14)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# 子图3：每个簇的数据点数量
plt.subplot(1, 3, 3)
unique, counts = np.unique(y_kmeans, return_counts=True)
plt.bar(unique, counts, color='dodgerblue', edgecolor='k')
plt.xticks(unique)
plt.title("Number of Points per Cluster", fontsize=14)
plt.xlabel("Cluster")
plt.ylabel("Number of Points")

# 显示图像
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>>> 2. 层次聚类

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 设置随机种子
np.random.seed(42)

# 生成虚拟数据集
n_samples = 1000
n_features = 5
n_clusters = 3

X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, cluster_std=1.0)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用PCA将数据降维至2维，方便可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 使用层次聚类
linked = linkage(X_scaled, 'ward')

# 聚类分配
max_d = 7.5  # 最大距离，调整该值可以改变聚类数量
clusters = fcluster(linked, max_d, criterion='distance')

# 设置颜色列表，确保颜色鲜艳
colors = sns.color_palette("hsv", n_clusters)

# 创建图形
plt.figure(figsize=(16, 8))

# 绘制PCA后的散点图
plt.subplot(1, 2, 1)
for i in range(1, n_clusters + 1):
    plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1], label=f'Cluster {i}', color=colors[i-1])

plt.title('PCA of Hierarchical Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

# 绘制层次聚类树状图
plt.subplot(1, 2, 2)
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=False,
           color_threshold=max_d,
           above_threshold_color='grey',
           truncate_mode='lastp',
           p=n_clusters)

plt.axhline(y=max_d, color='r', linestyle='--')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')

# 显示图形
plt.tight_layout()
plt.show()




#%%>>>>>>>>>>>>>>>>>>>>>>>>> 3. DBSCAN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 生成虚拟数据集
n_samples = 1500
X1, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
X2, _ = make_blobs(n_samples=n_samples, centers=3, random_state=42, cluster_std=1.0)
X2 = StandardScaler().fit_transform(X2)

# 合并数据集
X = np.vstack((X1, X2))

# 运行DBSCAN算法
dbscan = DBSCAN(eps=0.3, min_samples=10)
labels = dbscan.fit_predict(X)

# 获取聚类标签数
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# 创建子图
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 图1：原始数据分布
axs[0].scatter(X[:, 0], X[:, 1], c='gray', edgecolor='k', s=30)
axs[0].set_title("Original Data Distribution")
axs[0].set_xlabel("Feature 1")
axs[0].set_ylabel("Feature 2")

# 图2：DBSCAN聚类结果
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'  # 噪声点标记为黑色
    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    axs[1].scatter(xy[:, 0], xy[:, 1], c=[col], edgecolor='k', s=30)

axs[1].set_title(f"DBSCAN Clustering Result\nEstimated clusters: {n_clusters}")
axs[1].set_xlabel("Feature 1")
axs[1].set_ylabel("Feature 2")

# 图3：聚类后的数据点密度图
axs[2].scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', edgecolor='k', s=30)
axs[2].set_title("Density of Clustered Points")
axs[2].set_xlabel("Feature 1")
axs[2].set_ylabel("Feature 2")

# 调整布局
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>>> 4. 高斯混合模型

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse

# 生成虚拟数据集
np.random.seed(42)
n_samples = 1500

# 使用make_blobs生成有4个中心的二维数据集
X, y_true = make_blobs(n_samples=n_samples, centers=4, cluster_std=1.0, random_state=42)

# 添加一些随机噪声
X = np.dot(X, np.random.RandomState(42).randn(2, 2))

# 用高斯混合模型进行聚类
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
gmm.fit(X)
y_gmm = gmm.predict(X)

# 创建一个图形，包含四个子图
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# 子图1：原始数据集的散点图，标注真实类别
axs[0, 0].scatter(X[:, 0], X[:, 1], c=y_true, s=40, cmap='viridis', zorder=2)
axs[0, 0].set_title("Original Data with True Labels")
axs[0, 0].set_xlabel("Feature 1")
axs[0, 0].set_ylabel("Feature 2")

# 子图2：GMM聚类结果的散点图
axs[0, 1].scatter(X[:, 0], X[:, 1], c=y_gmm, s=40, cmap='viridis', zorder=2)
axs[0, 1].set_title("GMM Clustered Data")
axs[0, 1].set_xlabel("Feature 1")
axs[0, 1].set_ylabel("Feature 2")

# 子图3：GMM的预测概率分布（软分配）
prob_density = gmm.predict_proba(X)
axs[1, 0].scatter(X[:, 0], X[:, 1], c=prob_density.max(axis=1), s=40, cmap='viridis', zorder=2)
axs[1, 0].set_title("GMM Predicted Probabilities")
axs[1, 0].set_xlabel("Feature 1")
axs[1, 0].set_ylabel("Feature 2")

# 子图4：在散点图上绘制GMM的高斯椭圆
def draw_ellipse(position, covariance, ax, color):
    """在给定位置和协方差矩阵处绘制高斯椭圆。"""
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    ell = Ellipse(position, width, height, edgecolor=color, facecolor='none')
    ax.add_patch(ell)

# 绘制GMM的椭圆
axs[1, 1].scatter(X[:, 0], X[:, 1], c=y_gmm, s=40, cmap='viridis', zorder=2)
for pos, covar, color in zip(gmm.means_, gmm.covariances_, ['red', 'green', 'blue', 'purple']):
    draw_ellipse(pos, covar, axs[1, 1], color)
axs[1, 1].set_title("GMM with Gaussian Ellipses")
axs[1, 1].set_xlabel("Feature 1")
axs[1, 1].set_ylabel("Feature 2")

plt.tight_layout()
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>>>> 5. 均值漂移

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from itertools import cycle

# 生成虚拟数据集
centers = [[1, 1], [5, 5], [8, 1], [8, 8]]
cluster_std = [0.4, 0.5, 0.3, 0.7]  # 每个簇的标准差不同，增加复杂度
X, _ = make_blobs(n_samples=1000, centers=centers, cluster_std=cluster_std, random_state=42)

# 标准化数据集
X = StandardScaler().fit_transform(X)

# 应用均值漂移算法
mean_shift = MeanShift(bin_seeding=True)
mean_shift.fit(X)
labels = mean_shift.labels_
cluster_centers = mean_shift.cluster_centers_
n_clusters = len(np.unique(labels))

# 颜色配置
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

# 创建图像
plt.figure(figsize=(18, 8))

# 图1：原始数据分布
plt.subplot(1, 3, 1)
plt.title('Original Data Distribution')
for k, col in zip(range(n_clusters), colors):
    my_members = (labels == k)
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='yellow', marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 图2：基于密度的均值漂移算法的结果
plt.subplot(1, 3, 2)
plt.title('Mean Shift Clustering')
for k, col in zip(range(n_clusters), colors):
    my_members = (labels == k)
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='yellow', marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 图3：簇分布密度的2D直方图
plt.subplot(1, 3, 3)
plt.title('Cluster Density 2D Histogram')
plt.hist2d(X[:, 0], X[:, 1], bins=50, cmap='jet')
plt.colorbar(label='Density')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=100, c='white', edgecolor='black', marker='x')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>>>> 6. 模糊C均值

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import skfuzzy as fuzz
from scipy.interpolate import griddata

# 生成虚拟数据集
n_samples = 1500
centers = [[2, 2], [8, 3], [5, 8]]
cluster_std = [1.0, 1.0, 1.0]
X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=42)

# 模糊C均值聚类
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, 3, 2, error=0.005, maxiter=1000, init=None)

# 聚类结果分类
cluster_labels = np.argmax(u, axis=0)

# 为绘制等高线图准备网格数据
x = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
y = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
X_grid, Y_grid = np.meshgrid(x, y)
grid_points = np.c_[X_grid.ravel(), Y_grid.ravel()]

# 使用griddata对模糊隶属度进行插值
Z = np.zeros((X_grid.shape[0], X_grid.shape[1], 3))

for j in range(3):
    # 插值每个网格点的隶属度
    Z[:, :, j] = griddata(X, u[j], grid_points, method='linear').reshape(X_grid.shape)

# 绘图
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# 原始数据的散点图
ax[0].scatter(X[:, 0], X[:, 1], c='gray', marker='o', s=30, edgecolor='k', alpha=0.5)
ax[0].set_title('Original Data', fontsize=15)
ax[0].set_xlabel('Feature 1')
ax[0].set_ylabel('Feature 2')

# 模糊隶属度的等高线图
for j in range(3):
    ax[1].contourf(X_grid, Y_grid, Z[:, :, j], alpha=0.8, levels=np.linspace(0, 1, 11))
ax[1].scatter(X[:, 0], X[:, 1], c='gray', marker='o', s=30, edgecolor='k', alpha=0.5)
ax[1].set_title('Fuzzy Membership Contours', fontsize=15)
ax[1].set_xlabel('Feature 1')
ax[1].set_ylabel('Feature 2')

# 聚类结果的散点图
colors = ['r', 'g', 'b']
for i in range(3):
    ax[2].scatter(X[cluster_labels == i, 0], X[cluster_labels == i, 1], c=colors[i], marker='o', s=50, edgecolor='k', label=f'Cluster {i+1}')
ax[2].set_title('Clustered Data (FCM)', fontsize=15)
ax[2].set_xlabel('Feature 1')
ax[2].set_ylabel('Feature 2')
ax[2].legend()

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>>> 7. 期望最大化算法

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

# 设置随机种子，保证结果可复现
np.random.seed(42)

# 生成虚拟数据集
n_samples = 1500

# 高斯分布1
mean1 = [-5, 0]
cov1 = [[3, 1], [1, 2]]
data1 = np.random.multivariate_normal(mean1, cov1, int(0.4 * n_samples))

# 高斯分布2
mean2 = [0, 10]
cov2 = [[2, -1], [-1, 2]]
data2 = np.random.multivariate_normal(mean2, cov2, int(0.3 * n_samples))

# 高斯分布3
mean3 = [10, 5]
cov3 = [[1, 0.5], [0.5, 1]]
data3 = np.random.multivariate_normal(mean3, cov3, int(0.3 * n_samples))

# 合并数据集
X = np.vstack((data1, data2, data3))

# 可视化初始数据
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=10, color='purple', alpha=0.6)
plt.title('Initial Data Distribution')
plt.xlabel('X1')
plt.ylabel('X2')

# 使用期望最大化算法（EM算法）
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)


# 获取初始的高斯分布参数
def plot_gaussian_ellipse(mean, cov, ax, color='black'):
    # 计算椭圆的宽度和高度
    v, w = np.linalg.eigh(cov)
    v = 2. * np.sqrt(2.) * np.sqrt(v)  # width and height of ellipse

    # 计算椭圆的角度
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])  # 计算旋转角度
    angle = np.degrees(angle)  # 从弧度转换为度

    # 创建并添加椭圆
    ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=angle, color=color, alpha=0.4)
    ell.set_clip_box(ax.bbox)
    ax.add_patch(ell)


# 可视化EM算法拟合的高斯分布
ax2 = plt.subplot(2, 2, 2)
plt.scatter(X[:, 0], X[:, 1], s=10, color='purple', alpha=0.6)
for i in range(gmm.n_components):
    plot_gaussian_ellipse(gmm.means_[i], gmm.covariances_[i], ax2, color='blue')
plt.title('Gaussian Mixture Model - Initial Fitting')
plt.xlabel('X1')
plt.ylabel('X2')

# 预测类别
labels = gmm.predict(X)

# 可视化聚类结果
ax3 = plt.subplot(2, 2, 3)
colors = ['red', 'green', 'blue']
for i in range(gmm.n_components):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], s=10, color=colors[i], alpha=0.6)
    plot_gaussian_ellipse(gmm.means_[i], gmm.covariances_[i], ax3, color=colors[i])
plt.title('Final Clustering Result')
plt.xlabel('X1')
plt.ylabel('X2')

# 可视化EM算法收敛过程的对数似然值
ax4 = plt.subplot(2, 2, 4)
try:
    # 检查 gmm.lower_bound_ 是否是一个可迭代对象
    if hasattr(gmm, 'lower_bound_') and isinstance(gmm.lower_bound_, np.ndarray):
        n_iter = np.arange(1, len(gmm.lower_bound_) + 1)
        plt.plot(n_iter, gmm.lower_bound_, marker='o', color='orange', linestyle='--')
    else:
        plt.text(0.5, 0.5, 'Log Likelihood Data Unavailable', horizontalalignment='center', verticalalignment='center')
except AttributeError:
    plt.text(0.5, 0.5, 'Log Likelihood Data Unavailable', horizontalalignment='center', verticalalignment='center')
plt.title('EM Algorithm Convergence')
plt.xlabel('Iterations')
plt.ylabel('Log Likelihood')

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>>>> 8. 谱聚类


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

# 生成虚拟数据集
n_samples = 1000
n_features = 2
centers = 4
cluster_std = [1.0, 2.5, 0.5, 1.5]
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=42)

# 应用 Spectral Clustering
spectral = SpectralClustering(n_clusters=4, affinity='rbf',
                              gamma=1.0, random_state=42)
y_spectral = spectral.fit_predict(X)

# 计算相似度矩阵
similarity_matrix = np.exp(-pairwise_distances(X, metric='sqeuclidean'))
similarity_matrix = normalize(similarity_matrix, norm='l1', axis=1)

# 设置绘图
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# 原始数据集的散点图
axs[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
axs[0, 0].set_title('Original Dataset with True Labels', fontsize=16)
axs[0, 0].set_xlabel('Feature 1')
axs[0, 0].set_ylabel('Feature 2')

# Spectral Clustering 结果的散点图
axs[0, 1].scatter(X[:, 0], X[:, 1], c=y_spectral, cmap='rainbow', s=50)
axs[0, 1].set_title('Spectral Clustering Results', fontsize=16)
axs[0, 1].set_xlabel('Feature 1')
axs[0, 1].set_ylabel('Feature 2')

# 相似度矩阵的热力图
cax = axs[1, 0].imshow(similarity_matrix, cmap='hot', aspect='auto')
fig.colorbar(cax, ax=axs[1, 0])
axs[1, 0].set_title('Similarity Matrix (Heatmap)', fontsize=16)
axs[1, 0].set_xlabel('Sample Index')
axs[1, 0].set_ylabel('Sample Index')

# 聚类后的样本数量柱状图
unique, counts = np.unique(y_spectral, return_counts=True)
axs[1, 1].bar(unique, counts, color=['red', 'green', 'blue', 'orange'])
axs[1, 1].set_title('Number of Samples per Cluster', fontsize=16)
axs[1, 1].set_xlabel('Cluster Label')
axs[1, 1].set_ylabel('Number of Samples')

# 调整布局
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>>>> 9. Birch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# 生成虚拟数据集（高维数据）
n_samples = 1500
n_features = 10  # 生成更多的特征
random_state = 170
X, y = make_blobs(n_samples=n_samples, n_features=n_features, random_state=random_state, centers=6, cluster_std=1.0)

# 通过 PCA 将数据降维到3维
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# 应用Birch聚类算法
birch_model = Birch(threshold=1.5, n_clusters=6)  # 设置 n_clusters 参数
y_pred = birch_model.fit_predict(X)

# 创建图形
fig = plt.figure(figsize=(18, 9))

# 2D Scatter plot of original data
ax1 = fig.add_subplot(231)
ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', s=10)
ax1.set_title('Original Data (2D)')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

# 2D Scatter plot of PCA reduced data
ax2 = fig.add_subplot(232)
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='rainbow', s=10)
ax2.set_title('PCA Reduced Data (2D)')
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')

# 2D Scatter plot of Birch clusters
ax3 = fig.add_subplot(233)
ax3.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='rainbow', s=10)
ax3.set_title('Birch Clustering (2D)')
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')

# 3D Scatter plot of original data
ax4 = fig.add_subplot(234, projection='3d')
ax4.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='rainbow', s=10)
ax4.set_title('Original Data (3D)')
ax4.set_xlabel('Principal Component 1')
ax4.set_ylabel('Principal Component 2')
ax4.set_zlabel('Principal Component 3')

# 3D Scatter plot of Birch clusters
ax5 = fig.add_subplot(235, projection='3d')
ax5.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_pred, cmap='rainbow', s=10)
ax5.set_title('Birch Clustering (3D)')
ax5.set_xlabel('Principal Component 1')
ax5.set_ylabel('Principal Component 2')
ax5.set_zlabel('Principal Component 3')

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>>>> 10. Affinity Propagation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
import matplotlib.cm as cm

# 设置随机种子
np.random.seed(42)

# 生成虚拟数据集
n_samples = 500
centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
cluster_std = [0.2, 0.3, 0.2, 0.3]
X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std)

# 计算相似性矩阵（负欧氏距离）
similarity = -pairwise_distances(X, metric = 'euclidean')

# 使用Affinity Propagation进行聚类
af = AffinityPropagation(affinity='precomputed', random_state=42)
af.fit(similarity)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

# 获取不同的聚类中心和标签
n_clusters = len(cluster_centers_indices)

# 设置颜色映射
cmap = cm.get_cmap('hsv')
colors = cmap(np.linspace(0, 1, n_clusters))

# 创建一个图形
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 绘制第一个图形：聚类结果的散点图
for k, col in zip(range(n_clusters), colors):
    class_members = (labels == k)
    cluster_center = X[cluster_centers_indices[k]]
    axes[0].plot(X[class_members, 0], X[class_members, 1], '.', color=col, markersize=10, label=f'Cluster {k+1}')
    axes[0].plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=20)
    for x in X[class_members]:
        axes[0].plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col, alpha=0.5)

axes[0].set_title(f'Affinity Propagation Clustering\nwith {n_clusters} Clusters')
axes[0].legend()

# 绘制第二个图形：簇中心距离的热力图
distances = pairwise_distances(X[cluster_centers_indices], metric='euclidean')
im = axes[1].imshow(distances, interpolation='nearest', cmap='plasma')
axes[1].set_title('Cluster Center Distance Heatmap')
plt.colorbar(im, ax=axes[1])

# 设置图形整体标题
plt.suptitle('Affinity Propagation Clustering Analysis', fontsize=16)

# 展示图形
plt.show()









