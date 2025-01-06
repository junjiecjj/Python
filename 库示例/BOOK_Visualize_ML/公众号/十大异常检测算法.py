#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:53:04 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247489093&idx=1&sn=55ba4e96eacec20beaf7c276ec242d9a&chksm=c1b5356074f995bcbad69287356fd8b192e3498a3b8e2c078876f7f36ede3f6c407bbdea1f1e&mpshare=1&scene=1&srcid=0104fJojN1vkcvzhIGzPEU9c&sharer_shareinfo=5cdf67f5509bad536be33ffb7f6356c2&sharer_shareinfo_first=5cdf67f5509bad536be33ffb7f6356c2&exportkey=n_ChQIAhIQ8mBVQZWgbiZfjsdSCcJclBKfAgIE97dBBAEAAAAAAJebIkI7r1kAAAAOpnltbLcz9gKNyK89dVj0OsmwpKEX2QpQts6IBw8AMe238LSPuuWC5bgZUjvDpDd%2BhN%2FoVgHZQdDMPp%2FeHFFdKouaVx5cTLmuIUtJCaWHUVrUbpYS2QCt%2BtXKdDSIWuWxsme%2BhhZfs8av6xJkPY9MCygGQ2VfUjzSb2kL695eiHIn7dOcDDyz8BaWy0IrYAU%2BB5NBCQgUe64ptr1nW0iZRpBC75zdh3%2F6MbfbeDLq4jNul9o4hHpgsu2daKuUyrx5A0D%2FOd2tF5vUrJaiEDlYONnTaS%2BsRJCAabTLKTLRlPX%2By7Rv5Xe9sIn0P603ueX67wp8vap8I2M0AAMA53p7JlR0nXKqCADa&acctmode=0&pass_ticket=zhJD1H7riob2BcD9aDjXdj6TBb%2BsaHD1yreqYpiBI3gwyhlFeEVunYKxX170nGYf&wx_header=0#rd

"""



#%%>>>>>>>>>>>>>> 高斯混合模型
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# 1. 生成数据
np.random.seed(42)
n_samples = 500
centers = [[-5, 0], [0, 5], [5, -5]]
X, y_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=1.5)

# 2. 拟合 GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)
y_gmm = gmm.predict(X)

# 3. 绘制图形
plt.figure(figsize=(12, 8))

# (a) 数据分布及聚类结果
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=10, label='True Clusters')
plt.title("True Data Distribution")
plt.legend()

# (b) GMM 聚类结果
plt.subplot(2, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='rainbow', s=10, label='GMM Clusters')
plt.title("GMM Cluster Assignments")
plt.legend()

# (c) 密度等高线
x = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
X_grid, Y_grid = np.meshgrid(x, y)
Z = -np.exp(gmm.score_samples(np.c_[X_grid.ravel(), Y_grid.ravel()]))
Z = Z.reshape(X_grid.shape)

plt.subplot(2, 2, 3)
plt.contourf(X_grid, Y_grid, Z, levels=20, cmap='coolwarm')
plt.colorbar(label="Density")
plt.title("Density Contour Plot")

# (d) 某一簇分布的单独分析
k = 1
X_cluster_k = X[y_gmm == k]
plt.subplot(2, 2, 4)
plt.scatter(X_cluster_k[:, 0], X_cluster_k[:, 1], c='blue', label=f'Cluster {k}')
plt.title(f"Distribution of Cluster {k}")
plt.legend()

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>> 主成分分析

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 生成虚拟数据
np.random.seed(42)
n_samples = 300
n_features = 5

# 数据生成
data = np.dot(np.random.rand(n_features, n_features), np.random.randn(n_features, n_samples)).T

# 数据标准化（中心化）
data_centered = data - np.mean(data, axis=0)

# 使用PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_centered)

# 计算累计方差贡献率
explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# 图像绘制
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 1. 降维后的二维散点图
scatter = ax[0].scatter(data_pca[:, 0], data_pca[:, 1], c=np.arange(n_samples), cmap='viridis')
ax[0].set_title('2D Projection using PCA')
ax[0].set_xlabel('Principal Component 1')
ax[0].set_ylabel('Principal Component 2')
fig.colorbar(scatter, ax=ax[0], label='Sample Index')

# 2. 累计方差贡献图
ax[1].plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', color='r')
ax[1].set_title('Cumulative Explained Variance')
ax[1].set_xlabel('Number of Principal Components')
ax[1].set_ylabel('Explained Variance Ratio')
ax[1].grid(True)

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>> 局部异常因子

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# 1. 创建虚拟数据
np.random.seed(42)
# 正常数据
X_inliers = np.random.normal(loc=0.0, scale=1.0, size=(200, 2))
# 异常数据
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.vstack((X_inliers, X_outliers))

# 2. 计算LOF
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
LOF_scores = -clf.negative_outlier_factor_

# 3. 绘制图像
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# (a) 数据分布
ax[0].scatter(X_inliers[:, 0], X_inliers[:, 1], color='blue', label='Inliers')
ax[0].scatter(X_outliers[:, 0], X_outliers[:, 1], color='red', label='Outliers')
ax[0].set_title("Data Distribution", fontsize=14)
ax[0].legend()

# (b) LOF分数图
colors = np.where(y_pred == 1, 'blue', 'red')
scatter = ax[1].scatter(X[:, 0], X[:, 1], c=LOF_scores, cmap='viridis', edgecolor='k')
ax[1].set_title("LOF Scores", fontsize=14)
fig.colorbar(scatter, ax=ax[1], label="LOF Score")

# 展示图像
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>> 一类支持向量机
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs

# 1. 生成虚拟数据
np.random.seed(42)
X_inliers = 0.5 * np.random.randn(1000, 2)  # 正常数据
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))  # 异常数据
X = np.concatenate([X_inliers, X_outliers])

# 2. 创建并训练 One-Class SVM 模型
model = OneClassSVM(kernel="rbf", gamma=0.1, nu=0.1)
model.fit(X_inliers)  # 只用正常数据训练模型

# 3. 预测
y_pred_train = model.predict(X)
X_inliers_pred = X[y_pred_train == 1]
X_outliers_pred = X[y_pred_train == -1]

# 4. 创建网格以进行决策边界绘制
xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
grid = np.c_[xx.ravel(), yy.ravel()]
decision_scores = model.decision_function(grid)
decision_boundary = decision_scores.reshape(xx.shape)

# 5. 可视化
plt.figure(figsize=(12, 8))

# 数据分布图
plt.subplot(1, 2, 1)
plt.scatter(X_inliers[:, 0], X_inliers[:, 1], c="blue", label="Inliers", alpha=0.7)
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c="red", label="Outliers", alpha=0.7)
plt.title("Data Distribution")
plt.legend()

# 决策边界和异常点检测结果
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, decision_boundary, levels=np.linspace(decision_boundary.min(), 0, 10), cmap="coolwarm", alpha=0.8)
plt.scatter(X_inliers_pred[:, 0], X_inliers_pred[:, 1], c="blue", label="Detected Inliers", alpha=0.7)
plt.scatter(X_outliers_pred[:, 0], X_outliers_pred[:, 1], c="red", label="Detected Outliers", alpha=0.7)
plt.title("One-Class SVM Decision Boundary")
plt.legend()

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>> 随机森林检测器
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.inspection import permutation_importance
from matplotlib.colors import ListedColormap

# 生成虚拟数据集
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0,
                           random_state=42, n_clusters_per_class=1, class_sep=1.5)

# 创建随机森林分类器
clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
clf.fit(X, y)

# 特征重要性可视化
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))

# 1. 特征重要性条形图
plt.subplot(1, 2, 1)
plt.bar(range(X.shape[1]), importances[indices], color='orange', align='center')
plt.xticks(range(X.shape[1]), [f"Feature {i+1}" for i in indices])
plt.title("Feature Importances", fontsize=14)
plt.xlabel("Feature")
plt.ylabel("Importance")

# 2. 决策边界图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=ListedColormap(['#FF0000', '#0000FF']))
plt.title("Decision Boundary", fontsize=14)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# 图像整体显示
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>> K-均值聚类

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 生成虚拟数据集
np.random.seed(42)
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# 创建子图
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = ['red', 'green', 'blue']

# 初始簇分配
kmeans = KMeans(n_clusters=3, init='random', n_init=1, max_iter=1, random_state=42)
kmeans.fit(X)
axes[0].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.6)
axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', s=200, alpha=0.75, marker='X')
axes[0].set_title('Initial Clusters')

# 动态迭代过程
kmeans = KMeans(n_clusters=3, init='random', n_init=1, max_iter=10, random_state=42)
for i, center in enumerate(kmeans.fit(X).cluster_centers_):
    axes[1].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.6)
    axes[1].scatter(center[0], center[1], c=colors[i], s=200, alpha=0.75, marker='X')
axes[1].set_title('Dynamic Iteration')

# 最终簇分布
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
axes[2].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.6)
axes[2].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', s=200, alpha=0.75, marker='X')
axes[2].set_title('Final Clusters')

# 调整布局
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>> 马氏距离

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

# 1. 生成虚拟数据集
np.random.seed(42)
mu = np.array([5, 10])  # 均值
cov = np.array([[3, 1], [1, 2]])  # 协方差矩阵

# 生成数据点
data = np.random.multivariate_normal(mu, cov, size=200)
df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])

# 2. 计算马氏距离
mean_vector = np.mean(data, axis=0)
cov_matrix = np.cov(data, rowvar=False)
inv_cov_matrix = np.linalg.inv(cov_matrix)

df['Mahalanobis'] = df.apply(
    lambda row: mahalanobis(row[:2], mean_vector, inv_cov_matrix), axis=1)

# 设置阈值：95% 置信区间
threshold = np.sqrt(chi2.ppf(0.95, df=2))

# 3. 图形分析
plt.figure(figsize=(14, 7))

# 3.1 散点图与等高线
plt.subplot(1, 2, 1)
x, y = np.meshgrid(
    np.linspace(df['Feature1'].min() - 1, df['Feature1'].max() + 1, 100),
    np.linspace(df['Feature2'].min() - 1, df['Feature2'].max() + 1, 100),
)
z = np.array([mahalanobis([i, j], mean_vector, inv_cov_matrix) for i, j in zip(x.ravel(), y.ravel())])
z = z.reshape(x.shape)

plt.scatter(df['Feature1'], df['Feature2'], c='blue', alpha=0.6, label='Data Points')
plt.contour(x, y, z, levels=[threshold], colors='red', linestyles='dashed', linewidths=1.5)
plt.title('Scatter Plot with Mahalanobis Distance Contour')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# 3.2 热力图
plt.subplot(1, 2, 2)
plt.contourf(x, y, z, levels=20, cmap='Spectral', alpha=0.75)
plt.colorbar(label='Mahalanobis Distance')
plt.scatter(df['Feature1'], df['Feature2'], c='blue', alpha=0.6)
plt.title('Mahalanobis Distance Heatmap')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>> 热图分析

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 生成虚拟数据集
np.random.seed(42)
data = np.random.normal(loc=0, scale=1, size=(20, 6))
columns = [f"Feature_{i+1}" for i in range(data.shape[1])]
df = pd.DataFrame(data, columns=columns)

# 计算相关性矩阵
correlation_matrix = df.corr()

# 创建热图
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 热图1：数据值分布
sns.heatmap(df, cmap="YlGnBu", ax=axes[0], cbar=True, annot=True, fmt=".2f")
axes[0].set_title("Feature Value Heatmap", fontsize=14)

# 热图2：相关性矩阵
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=axes[1], cbar=True, fmt=".2f")
axes[1].set_title("Correlation Heatmap", fontsize=14)

# 调整布局
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>> 神经网络自编码器

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成虚拟数据集
n_samples = 1000
data_dim = 2
data = np.random.rand(n_samples, data_dim) * 10 - 5  # 生成范围在[-5, 5]的二维点
data = torch.tensor(data, dtype=torch.float32)

# 2. 自编码器定义
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

latent_dim = 2
autoencoder = Autoencoder(data_dim, latent_dim)

# 3. 训练
criterion = nn.MSELoss()  # 损失函数
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

epochs = 200
batch_size = 64
losses = []

for epoch in range(epochs):
    epoch_loss = 0
    # 小批量训练
    for i in range(0, n_samples, batch_size):
        batch = data[i:i+batch_size]
        optimizer.zero_grad()
        recon, _ = autoencoder(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    losses.append(epoch_loss / n_samples)
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / n_samples:.4f}")

# 4. 数据分析与可视化
with torch.no_grad():
    recon_data, latent_data = autoencoder(data)

# 原始数据和重建数据可视化
plt.figure(figsize=(12, 6))

# 子图1: 原始数据
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c='blue', label="Original Data", alpha=0.6)
plt.title("Original Data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()

# 子图2: 重建数据
plt.subplot(1, 2, 2)
plt.scatter(recon_data[:, 0], recon_data[:, 1], c='orange', label="Reconstructed Data", alpha=0.6)
plt.title("Reconstructed Data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()

plt.tight_layout()
plt.show()

# 潜在空间分布分析
plt.figure(figsize=(8, 6))
plt.scatter(latent_data[:, 0], latent_data[:, 1], c='green', label="Latent Space", alpha=0.6)
plt.title("Latent Space Representation")
plt.xlabel("Z1")
plt.ylabel("Z2")
plt.legend()
plt.show()



#%%>>>>>>>>>>>>>> 密度峰值聚类


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score

# 1. 生成虚拟数据
from sklearn.datasets import make_moons, make_blobs

# 两种不同形状的聚类
data1, _ = make_moons(n_samples=300, noise=0.07)
data2, _ = make_blobs(n_samples=200, centers=[[3, 3], [-3, -3]], cluster_std=0.8)
X = np.vstack((data1, data2))

# 2. 计算距离矩阵
distances = pairwise_distances(X)
dc = np.percentile(distances, 2)  # 设置dc为距离的2%分位数

# 3. 计算局部密度
rho = np.sum(np.exp(-(distances / dc) ** 2), axis=1)

# 4. 计算每个点到更高密度点的最小距离
delta = np.zeros_like(rho)
max_distance = np.max(distances)
for i in range(len(rho)):
    higher_density = np.where(rho > rho[i])[0]
    delta[i] = np.min(distances[i, higher_density]) if higher_density.size > 0 else max_distance

# 5. 决策图选择聚类中心
decision_values = rho * delta
cluster_centers = np.argsort(decision_values)[-4:]  # 选择前4个点作为聚类中心

# 6. 分配类别
labels = -np.ones(len(X), dtype=int)
for center_idx, cluster_id in zip(cluster_centers, range(len(cluster_centers))):
    labels[center_idx] = cluster_id

# 分配剩余点
for i in np.argsort(-rho):  # 按密度降序
    if labels[i] == -1:  # 如果未分配
        labels[i] = labels[np.argmin(distances[i, labels >= 0])]

# 7. 计算轮廓系数
silhouette_avg = silhouette_score(X, labels)

# 8. 可视化
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# 决策图
ax[0].scatter(rho, delta, c='blue', s=20, label="Points")
ax[0].scatter(rho[cluster_centers], delta[cluster_centers], c='red', s=80, label="Cluster Centers")
ax[0].set_xlabel('Density (ρ)')
ax[0].set_ylabel('Distance (δ)')
ax[0].set_title('Decision Graph')
ax[0].legend()

# 聚类结果
for cluster_id in np.unique(labels):
    cluster_points = X[labels == cluster_id]
    ax[1].scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_id}', s=10)
ax[1].scatter(X[cluster_centers, 0], X[cluster_centers, 1], c='black', marker='x', s=100, label='Centers')
ax[1].set_title('Cluster Assignments')
ax[1].legend()

# 轮廓系数
ax[2].bar(range(len(np.unique(labels))), silhouette_score(X, labels, metric='euclidean') * np.ones(len(np.unique(labels))), color='green')
ax[2].set_title('Silhouette Coefficient')
ax[2].set_xlabel('Cluster')
ax[2].set_ylabel('Silhouette Score')

plt.tight_layout()
plt.show()









