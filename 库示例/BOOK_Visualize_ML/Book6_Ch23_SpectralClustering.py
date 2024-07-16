

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html

# sklearn.cluster.SpectralClustering() 谱聚类算法
# sklearn.datasets.make_circles() 创建环形样本数据
# sklearn.preprocessing.StandardScaler().fit_transform() 标准化数据；通过减去均值然后除以标准差，处理后数据符合标准正态分布

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  谱聚类


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from scipy.linalg import sqrtm as sqrtm
from sklearn.metrics import pairwise_distances

# 数据
np.random.seed(0)
# 样本数据的数量
n_samples = 500;

# 生成环形数据
dataset = datasets.make_circles(n_samples=n_samples, factor = 0.5, noise = 0.05)

# X 特征数据，y 标签数据
X, y = dataset
# 标准化数据集
X = StandardScaler().fit_transform(X) # (500, 2)

# 图 3. 样本点平面位置
fig, ax = plt.subplots(figsize=(8, 8))
plt.scatter(X[:,0],X[:,1])
ax.set_aspect('equal', adjustable='box')
# plt.savefig('散点图.svg')

# 图 4. 成对欧氏距离矩阵 D
# 计算成对距离矩阵
# D = np.linalg.norm(X[:, np.newaxis, :] - X, axis=2)
# 请尝试使用
# scipy.spatial.distance_matrix()
D = pairwise_distances(X, X, metric = 'euclidean', )
# 可视化成对距离矩阵
plt.figure(figsize=(8, 8))
sns.heatmap(D, square = True, cmap = 'Blues',xticklabels = [], yticklabels = []) # annot=True, fmt=".3f",

# plt.savefig('成对距离矩阵_heatmap.svg')
################ 相似度矩阵
# 自定义高斯核函数
def gaussian_kernel(distance, sigma=1.0):
    return np.exp(- (distance ** 2) / (2 * sigma ** 2))

S = gaussian_kernel(D,3)
# 参数sigma设为3

# 图 6. 成对相似度矩阵 S
plt.figure(figsize=(8,8))
sns.heatmap(S, square = True, cmap = 'viridis', vmin = 0, vmax = 1,xticklabels = [], yticklabels = [])
# plt.savefig('亲近度矩阵_heatmap.svg')

# 图 7. 相似度对称矩阵 S 无向图
# 创建无向图
S_copy = np.copy(S)
np.fill_diagonal(S_copy, 0)
# 用邻接矩阵创建无向图
G = nx.Graph(S_copy, nodetype=int)

# 添加节点和边
for i in range(len(X)):
    G.add_node(i, pos=(X[i, 0], X[i, 1]))
# 取出节点位置
pos = nx.get_node_attributes(G, 'pos')
# 增加节点属性
node_labels = {i: chr(ord('a') + i) for i in range(len(G.nodes))}
edge_weights = [G[i][j]['weight'] for i, j in G.edges]
# 可视化图
fig, ax = plt.subplots(figsize = (6,6))
nx.draw_networkx(G, pos, with_labels = False, node_size = 38,
                 node_color = 'blue', font_color = 'black',
                 edge_cmap = plt.cm.viridis, edge_color = edge_weights,
                 width = 1, alpha = 0.5)

ax.set_aspect('equal', adjustable='box')
ax.axis('off')
# plt.savefig('成对距离矩阵_无向图.svg')


# 图 8. 度矩阵 G
G = np.diag(S.sum(axis = 1))
# 可视化度矩阵
plt.figure(figsize=(8,8))
sns.heatmap(G, square = True,
            cmap = 'viridis',
            # linecolor = 'k',
            # linewidths = 0.05,
            mask = 1-np.identity(len(G)),
            vmin = S.sum(axis = 1).min(),
            vmax = S.sum(axis = 1).max(),
            # annot=True, fmt=".3f",
            xticklabels = [], yticklabels = [])
# plt.savefig('度矩阵_heatmap.svg')

# 归一化对称拉普拉斯矩阵
G_inv_sqr = sqrtm(np.linalg.inv(G))

L_s = G_inv_sqr @ (G - S) @ G_inv_sqr

# 图 9. 归一化拉普拉斯矩阵 Ls
plt.figure(figsize=(8,8))
sns.heatmap(L_s, square = True, cmap = 'plasma', xticklabels = [], yticklabels = [])
# plt.savefig('拉普拉斯矩阵_heatmap.svg')

# 特征值分解拉普拉斯矩阵
eigenValues_s, eigenVectors_s = np.linalg.eigh(L_s)
# 特征值分解

# 图 10. 拉普拉斯矩阵 L 特征值分解得到的特征值从小到大排序，前 50
idx_s = eigenValues_s.argsort() # [::-1]
eigenValues_s = eigenValues_s[idx_s]
eigenVectors_s = eigenVectors_s[:,idx_s]
fig, ax = plt.subplots(figsize = (6,6))
plt.plot(range(1,51),eigenValues_s[0:50])
ax.set_xlim(1,50)
ax.set_ylim(0,1)
# plt.savefig('特征值.svg')

# 图 11. 前两个特征向量构成的散点图，特征值从小到大排序
fig, ax = plt.subplots(figsize = (6,6))
plt.scatter(eigenVectors_s[:,0], eigenVectors_s[:,1])
# plt.savefig('散点图，投影后.svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
















