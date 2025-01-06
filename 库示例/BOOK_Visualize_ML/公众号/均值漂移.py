#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:52:23 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247489198&idx=1&sn=68ef3137514da18d0e154f58ce1ed088&chksm=c16a7b420c6ee1eee0bd2b48847cf32c00d3076dcaa801a884e5699115c582dacc6a3dd97925&mpshare=1&scene=1&srcid=01045H5dKjYLotTEECCbFk6c&sharer_shareinfo=6dbd9dd0b5d6902b6c1325f8f6046a18&sharer_shareinfo_first=6dbd9dd0b5d6902b6c1325f8f6046a18&exportkey=n_ChQIAhIQ%2FZsrb5NJCQ9qz5cJqZ3HOhKfAgIE97dBBAEAAAAAAKimI55RtXkAAAAOpnltbLcz9gKNyK89dVj08iR4aRKI37FmUGZmkajJZQrZ1novFU6fneeMY%2Fw1p0dE%2By3BjYGxCLmiHwzEL%2B8gjUP2Y3KXU%2FLpXAcaQbi%2FtCCxUm4fBpSxNJcckopybRfHKCinjSufKHVeW4U3uXSxCsASYdFmJXp%2FFZnJD0H64EpGjPrwOFk7sZ4DDOGaXK%2BOTbgqbH5lXeZjuUxm7zNbUgNHkKLDcBCYPxS7Ebdno%2Bd3WPudBzfYxfaIzqT30YHFOl1om8dhIcJzvUQg4tpk9Yr1MrXX0PxUaMDue2h%2FzhbGzOG%2BuZhXVb13eGJ6DQrSS%2BYowrC2g8pzpo5Uczk7ysRJ%2BGFsjVMh&acctmode=0&pass_ticket=dM5IO5J%2FME4kuUeKLbh08g0refeLNNm03jRCFzGm0OOs61voXPFT7vws7QgJpDXD&wx_header=0#rd

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

# 设置绘图风格
sns.set(style="whitegrid")

# 1. 数据生成
centers = [[2, 3], [8, 8], [1, 10], [9, 1]]  # 指定聚类中心
cluster_std = [1.0, 0.8, 1.5, 0.5]  # 每个簇的标准差

# 生成样本数据
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=cluster_std, random_state=42)

# 2. 估计带宽并执行均值漂移聚类
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

n_clusters_ = len(np.unique(labels))

print(f"Number of clusters: {n_clusters_}")

# 3. 绘制结果图形

# 图 1: 数据分布和聚类结果
plt.figure(figsize=(14, 7))
unique_labels = np.unique(labels)
colors = sns.color_palette("bright", n_colors=n_clusters_)

for k, col in zip(unique_labels, colors):
    cluster_members = labels == k
    plt.scatter(X[cluster_members, 0], X[cluster_members, 1], c=[col], label=f"Cluster {k}", s=30, alpha=0.7, edgecolor='k')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=300, alpha=0.8, label='Centers', marker='X')
plt.title("Mean Shift Clustering Results", fontsize=16)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

# 图 2: 聚类后的密度估计
plt.figure(figsize=(14, 7))
sns.kdeplot(x=X[:, 0], y=X[:, 1], fill=True, cmap="viridis", alpha=0.8)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=300, label='Cluster Centers', marker='X')
plt.title("Density Estimation and Cluster Centers", fontsize=16)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

# 显示图形
plt.show()
