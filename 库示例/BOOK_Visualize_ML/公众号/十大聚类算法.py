#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 21:38:59 2024

@author: jack
"""

# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247484122&idx=1&sn=f8698083658f761b9e9235c8fc3993ef&chksm=c0e5d81cf792510a92dd44e6f00d06a5ead83f5455ce74318e16c31aa4c0bbca324db8bc648c&cur_album_id=3445855686331105280&scene=190#rd



#%% 1. K-means
from sklearn.cluster import KMeans
import numpy as np

# 示例数据
X = np.random.rand(100, 2)

# K-means 聚类
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# 聚类结果
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print(labels)
print(centroids)










#%% 2. 层次聚类（Hierarchical Clustering）


from sklearn.cluster import AgglomerativeClustering
import numpy as np

# 示例数据
X = np.random.rand(100, 2)

# 层次聚类
hierarchical = AgglomerativeClustering(n_clusters=3)
labels = hierarchical.fit_predict(X)

print(labels)





#%% 3. DBSCAN（Density-Based Spatial Clustering of Applications with Noise）

from sklearn.cluster import DBSCAN
import numpy as np

# 示例数据
X = np.random.rand(100, 2)

# DBSCAN 聚类
dbscan = DBSCAN(eps=0.1, min_samples=5)
labels = dbscan.fit_predict(X)

print(labels)






#%% 4. OPTICS（Ordering Points To Identify the Clustering Structure）

from sklearn.cluster import OPTICS
import numpy as np
import matplotlib.pyplot as plt

# 示例数据
X = np.random.rand(100, 2)

# OPTICS 聚类
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
labels = optics.fit_predict(X)
reachability = optics.reachability_[optics.ordering_]
ordering = optics.ordering_

# 可视化
plt.figure(figsize=(10, 7))
space = np.arange(len(X))
plt.plot(space, reachability, 'k.', alpha=0.5)
plt.xlabel('Sample index')
plt.ylabel('Reachability distance')
plt.title('OPTICS Reachability Plot')
plt.show()

print(labels)






#%% 5. 均值漂移（Mean Shift）
from sklearn.cluster import MeanShift
import numpy as np

# 示例数据
X = np.random.rand(100, 2)

# 均值漂移聚类
mean_shift = MeanShift()
labels = mean_shift.fit_predict(X)

print(labels)







#%% 6. GMM（Gaussian Mixture Model，高斯混合模型）

from sklearn.mixture import GaussianMixture
import numpy as np

# 示例数据
X = np.random.rand(100, 2)

# 高斯混合模型
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
labels = gmm.predict(X)

print(labels)






#%% 7. 谱聚类（Spectral Clustering）

from sklearn.cluster import SpectralClustering
import numpy as np

# 示例数据
X = np.random.rand(100, 2)

# 谱聚类
spectral = SpectralClustering(n_clusters=3)
labels = spectral.fit_predict(X)

print(labels)






#%% 8. 模糊C均值（Fuzzy C-Means）

from fcmeans import FCM
import numpy as np

# 示例数据
X = np.random.rand(100, 2)

# 模糊C均值聚类
fcm = FCM(n_clusters=3)
fcm.fit(X)
labels = fcm.predict(X)

print(labels)






#%% 9. BIRCH（Balanced Iterative Reducing and Clustering using Hierarchies）


from sklearn.cluster import Birch
import numpy as np

# 示例数据
X = np.random.rand(100, 2)

# BIRCH 聚类
birch = Birch(n_clusters=3)
labels = birch.fit_predict(X)

print(labels)






#%% 10. Affinity Propagation

from sklearn.cluster import AffinityPropagation
import numpy as np

# 示例数据
X = np.random.rand(100, 2)

# Affinity Propagation 聚类
affinity_propagation = AffinityPropagation()
labels = affinity_propagation.fit_predict(X)

print(labels)












