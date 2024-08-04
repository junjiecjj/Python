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


# https://mp.weixin.qq.com/s?__biz=MzAwNTkyNTUxMA==&mid=2247487881&idx=1&sn=0ae8bff4e93e31052fe25672ce5b6b38&chksm=9b146e60ac63e7760bb830418ee0a6538cb622892bacca926997aaf6af48752bfa87951ce40e&cur_album_id=3256084713219047427&scene=190#rd


#%% K均值聚类

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 生成样本数据
X, y = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=0)

# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 绘制结果图
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()




#%% 层次聚类（Hierarchical Clustering）
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# 生成样本数据
X, y = make_blobs(n_samples=500, centers=3, random_state=0, cluster_std=0.60)

# 进行层次聚类
Z = linkage(X, method='ward')

# 绘制树状图
plt.figure(figsize=(10, 7))
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample index or (Cluster size)")
plt.ylabel("Distance")
plt.show()



#%% DBSCAN（Density-Based Spatial Clustering of Applications with Noise）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# 生成样本数据
X, y = make_moons(n_samples=500, noise=0.1, random_state=0)

# 使用DBSCAN进行聚类
dbscan = DBSCAN(eps=0.2, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

# 绘制结果图
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, s=50, cmap='viridis')

# 标记噪声点
core_samples_mask = np.zeros_like(y_dbscan, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
noise_mask = (y_dbscan == -1)
plt.scatter(X[noise_mask, 0], X[noise_mask, 1], c='red', s=50, label='Noise')

plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()





#%% 高斯混合模型（Gaussian Mixture Model，GMM）



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# 生成样本数据
X, y = make_blobs(n_samples=500, centers=3, random_state=42, cluster_std=0.60)

# 使用GMM进行聚类
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

# 绘制结果图
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=200, alpha=0.75, marker='X')  # 标记出中心点
plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()




#%% 密度峰值聚类（Density Peaks Clustering）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist, squareform

# 生成样本数据
X, y = make_blobs(n_samples=500, centers=3, random_state=42, cluster_std=0.60)

# 计算距离矩阵
distances = squareform(pdist(X))

# 计算局部密度
dc = np.percentile(distances, 2)  # 截断距离
rho = np.sum(np.exp(-(distances / dc) ** 2), axis=1) - 1

# 计算最小距离
delta = np.zeros_like(rho)
delta[rho.argsort()[0]] = np.max(distances)
for i in range(1, len(rho)):
    delta[rho.argsort()[i]] = np.min(distances[rho.argsort()[i], np.where(rho > rho[rho.argsort()[i]])[0]])

# 选择簇中心
rho_delta = rho * delta
centers = np.argsort(-rho_delta)[:3]

# 分配簇
labels = np.zeros(len(X), dtype=int) - 1
for i, center in enumerate(centers):
    labels[center] = i

for i in range(len(X)):
    if labels[rho.argsort()[i]] == -1:
        labels[rho.argsort()[i]] = labels[np.argmin(distances[rho.argsort()[i], centers])]

# 绘制结果图
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.scatter(X[centers, 0], X[centers, 1], c='red', s=200, alpha=0.75, marker='X')  # 标记出中心点
plt.title("Density Peaks Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()



#%% 谱聚类（Spectral Clustering）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering

# 生成样本数据
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)

# 使用谱聚类进行聚类
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
labels = spectral.fit_predict(X)

# 绘制结果图
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.title("Spectral Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()






#%% OPTICS（Ordering Points To Identify the Clustering Structure）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs

# 生成样本数据
X, y = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)

# 使用OPTICS进行聚类
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
optics.fit(X)
labels = optics.labels_

# 绘制结果图
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.title("OPTICS Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()




#%% 自组织映射（Self-Organizing Maps，SOM）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from minisom import MiniSom

# 加载Iris数据集
data = load_iris()
X = data.data
y = data.target

# 标准化数据
X = (X - X.mean(axis=0)) / X.std(axis=0)

# 初始化和训练SOM
som = MiniSom(x=7, y=7, input_len=4, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, 100)

# 绘制结果图
plt.figure(figsize=(10, 10))
for i, x in enumerate(X):
    w = som.winner(x)
    plt.text(w[0] + 0.5, w[1] + 0.5, str(y[i]),
             color=plt.cm.rainbow(y[i] / 2.),
             fontdict={'weight': 'bold', 'size': 11})
plt.title("Self-Organizing Map of Iris Data")
plt.grid()
plt.show()












#%% Affinity Propagation


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import load_sample_image
from sklearn.metrics import pairwise_distances

# 加载样本图片
china = load_sample_image("flower.jpg")
image = np.array(china, dtype=np.float64) / 255

# 获取图片的宽、高和深度
w, h, d = original_shape = tuple(image.shape)
# 将图片数据转为二维数据
image_array = np.reshape(image, (w * h, d))

# 计算相似度矩阵（欧氏距离）
similarity = -pairwise_distances(image_array, metric='sqeuclidean')

# 使用 Affinity Propagation 进行聚类
affinity_propagation = AffinityPropagation(affinity='precomputed', random_state=42)
affinity_propagation.fit(similarity)
labels = affinity_propagation.predict(similarity)

# 获取聚类中心
cluster_centers_indices = affinity_propagation.cluster_centers_indices_
cluster_centers = image_array[cluster_centers_indices]

# 重新构造图片
reconstructed_image = cluster_centers[labels].reshape(w, h, d)

# 绘制原图和聚类后的图片
fig, ax = plt.subplots(1, 2, figsize=(15, 8))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(reconstructed_image)
ax[1].set_title("Clustered Image with Affinity Propagation")
ax[1].axis('off')

plt.show()



#%% Mean Shift Clustering


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle

# 加载样本图片
china = load_sample_image("flower.jpg")
image = np.array(china, dtype=np.float64) / 255

# 降低图像的维度以加快算法运行速度
w, h, d = original_shape = tuple(image.shape)
image_array = np.reshape(image, (w * h, d))

# 对数据进行降维
image_array_sample = shuffle(image_array, random_state=0)[:w*h]

# 使用Mean Shift进行聚类
bandwidth = 0.1
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(image_array_sample)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

# 绘制结果图
plt.figure(figsize=(10, 10))
plt.imshow(np.reshape(labels, (w, h)), cmap=plt.cm.get_cmap("tab20", cluster_centers.shape[0]))
plt.title("Mean Shift Clustering of Image")
plt.axis("off")
plt.show()












#%%















#%%


















































































































