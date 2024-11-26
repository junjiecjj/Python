#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:24:05 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247488361&idx=1&sn=e3168e6925645769570bd9d2c618b595&chksm=c0e5c9aff79240b92940339d6ff5cab57828a024e48339f98a7297b2e88e6f697935285c5fd5&cur_album_id=3445855686331105280&scene=190#rd


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 设置随机种子，生成消费者数据
np.random.seed(42)
n_samples = 10000

# 模拟消费者数据：年龄、收入、购买频率
age = np.random.randint(18, 70, n_samples)
income = np.random.normal(50000, 15000, n_samples).astype(int)
purchase_frequency = np.random.randint(1, 20, n_samples)

# 创建数据框
data = pd.DataFrame({
    'Age': age,
    'Income': income,
    'PurchaseFrequency': purchase_frequency
})

# 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 层次聚类
linkage_matrix = linkage(data_scaled, method='ward')  # Ward方法最常用

# 从层次聚类中提取实际的聚类标签
n_clusters = 5  # 假设分为5个聚类
labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')  # 提取聚类标签

# 图形1: 聚类结果的二维投影
# 使用PCA将数据降维至2D
pca = PCA(n_components = 2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(12, 8))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c = labels, cmap = 'Spectral', s = 50, alpha = 0.7)
plt.title('Cluster Visualization (PCA Projection)', fontsize=18)
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.colorbar(label='Cluster ID', shrink=0.8)
plt.grid(True)
plt.show()

# 图形2: 树状图（Dendrogram）
plt.figure(figsize=(16, 10))
dendrogram(linkage_matrix, truncate_mode = 'lastp', p = 10, leaf_rotation = 90, leaf_font_size = 12, show_contracted = True)
plt.title('Dendrogram for Hierarchical Clustering', fontsize=18)
plt.xlabel('Cluster Size', fontsize=12)
plt.ylabel('Distance (Ward\'s Method)', fontsize=12)
plt.show()























