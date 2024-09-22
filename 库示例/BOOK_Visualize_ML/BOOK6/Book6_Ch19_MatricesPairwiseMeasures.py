

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html

# sklearn.metrics.pairwise.euclidean_distances() 计算成对欧氏距离矩阵
# sklearn.metrics.pairwise_distances() 计算成对距离矩阵
# metrics.pairwise.linear_kernel() 计算线性核成对亲近度矩阵
# metrics.pairwise.manhattan_distances() 计算成对城市街区距离矩阵
# metrics.pairwise.paired_cosine_distances(X,Q) 计算 X 和 Q 样本数据矩阵成对余弦距离矩阵
# metrics.pairwise.paired_euclidean_distances(X,Q) 计算 X 和 Q 样本数据矩阵成对欧氏距离矩阵
# metrics.pairwise.paired_manhattan_distances(X,Q) 计算 X 和 Q 样本数据矩阵成对城市街区距离矩阵
# metrics.pairwise.polynomial_kernel() 计算多项式核成对亲近度矩阵
# metrics.pairwise.rbf_kernel() 计算 RBF 核成对亲近度矩阵
# metrics.pairwise.sigmoid_kernel() 计算 sigmoid 核成对亲近度矩阵

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 将成对距离矩阵转化为完全图

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist

# 12个坐标点
points = np.array([[1,6], [4,6], [1,5], [6,0], [3,8], [8,3], [4,1], [3,5], [9,2], [5,9], [4,9], [8,4]])

# 计算成对距离矩阵
# D = np.linalg.norm(points[:, np.newaxis, :] - points, axis = 2)
D =  pairwise_distances(points, points, metric = 'euclidean', )
# D2 = pdist(points, 'euclidean')

# 创建无向图
G = nx.Graph()

# 请思考如何避免使用 for 循环
# 添加节点和边
for i in range(12):
    # 使用pos属性保存节点的坐标信息
    G.add_node(i, pos = (points[i, 0], points[i, 1]))
    for j in range(i + 1, 12):
        # 将距离作为边的权重
        G.add_edge(i, j, weight = D[i, j])

# 增加节点/边属性
pos = nx.get_node_attributes(G, 'pos')
labels = {i: chr(ord('a') + i) for i in range(len(G.nodes))}
edge_labels = {(i, j): f'{D[i, j]:.2f}' for i, j in G.edges}
edge_weights = [G[i][j]['weight'] for i, j in G.edges]

# 可视化图
fig, ax = plt.subplots(figsize = (16, 16))
nx.draw_networkx(G, pos, with_labels = True, labels = labels, node_size = 800, font_size = 20, node_color = 'grey', font_color = 'black', edge_vmin = 0, edge_vmax = 10, edge_cmap = plt.cm.RdYlBu, edge_color = edge_weights, width = 1, alpha = 0.7)

nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, font_size = 20, font_color = 'k')

ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.grid()
ax.set_aspect('equal', adjustable='box')
# plt.savefig('成对距离矩阵_无向图.svg')
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 将成对距离矩阵转化为完全图，设定阈值
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# 12个坐标点
points = np.array([[1,6],[4,6],[1,5],[6,0],
                   [3,8],[8,3],[4,1],[3,5],
                   [9,2],[5,9],[4,9],[8,4]])

# 计算成对距离矩阵
D = np.linalg.norm(points[:, np.newaxis, :] - points, axis=2)

# 设定阈值
threshold = 6
D_threshold = D
# 超过阈值置零
D_threshold[D_threshold > threshold] = 0

# 可视化成对距离矩阵
plt.figure(figsize=(8,8))
sns.heatmap(D_threshold, square = True, cmap = 'RdYlBu', vmin = 0, vmax = 10,
            # annot=True, fmt=".3f",
            xticklabels = [], yticklabels = [])
# plt.savefig('成对距离矩阵_heatmap, 设定阈值.svg')

# 用邻接矩阵创建无向图
G_threshold = nx.Graph(D_threshold, nodetype = int)

# 添加节点和边
for i in range(12):
    G_threshold.add_node(i, pos=(points[i, 0], points[i, 1]))

# 取出节点位置
pos = nx.get_node_attributes(G_threshold, 'pos')

# 增加节点属性
node_labels = {i: chr(ord('a') + i) for i in range(len(G_threshold.nodes))}
edge_weights = [G_threshold[i][j]['weight'] for i, j in G_threshold.edges]

# 可视化图
fig, ax = plt.subplots(figsize = (16,16))
nx.draw_networkx(G_threshold, pos,
                 with_labels=True,
                 labels=node_labels,
                 node_size=800, font_size = 20,
                 edge_vmin = 0, edge_vmax = 10,
                 node_color='grey',
                 font_color='black',
                 edge_color=edge_weights,
                 edge_cmap=plt.cm.RdYlBu,
                 width=2, )
edge_labels = {(i, j): f'{D[i, j]:.2f}' for i, j in G.edges if D_threshold[i,j] > 0}
nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels, font_size = 20, font_color = 'k')

ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.grid()
ax.set_aspect('equal', adjustable='box')
# plt.savefig('成对距离矩阵_无向图_阈值.svg')
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 亲近度矩阵：高斯核函数

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# 12个坐标点
points = np.array([[1,6],[4,6],[1,5],[6,0],
                   [3,8],[8,3],[4,1],[3,5],
                   [9,2],[5,9],[4,9],[8,4]])

# 自定义高斯核函数
def gaussian_kernel(distance, sigma=1.0):
    return np.exp(- (distance ** 2) / (2 * sigma ** 2))

# 计算成对距离矩阵
D = np.linalg.norm(points[:, np.newaxis, :] - points, axis=2)

K = gaussian_kernel(D,3)
# 参数sigma设为3


# 可视化亲近度矩阵
plt.figure(figsize=(8,8))
sns.heatmap(K, square = True,
            cmap = 'viridis', vmin = 0, vmax = 1,
            # annot=True, fmt=".3f",
            xticklabels = [], yticklabels = [])
# plt.savefig('亲近度矩阵_heatmap.svg')

np.fill_diagonal(K, 0)
# 将对角线元素置0，不画自环

# 用邻接矩阵创建无向图
G = nx.Graph(K, nodetype=int)

# 添加节点和边
for i in range(12):
    G.add_node(i, pos=(points[i, 0], points[i, 1]))

# 取出节点位置
pos = nx.get_node_attributes(G, 'pos')

# 增加节点属性
node_labels = {i: chr(ord('a') + i) for i in range(len(G.nodes))}
edge_weights = [G[i][j]['weight'] for i, j in G.edges]
edge_labels = {(i, j): f'{K[i, j]:.2f}' for i, j in G.edges}


# 可视化图
fig, ax = plt.subplots(figsize = (16, 16))
nx.draw_networkx(G, pos,
                 with_labels = True,
                 labels = node_labels,
                 node_size = 800, font_size = 20,
                 edge_vmin = 0, edge_vmax = 1,
                 node_color = 'grey',
                 font_color = 'black',
                 edge_color = edge_weights,
                 edge_cmap = plt.cm.viridis,
                 width = 1, alpha = 0.7)
ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.grid()
ax.set_aspect('equal', adjustable='box')
# plt.savefig('亲近度矩阵_无向图.svg')
plt.show()


# 可视化图
fig, ax = plt.subplots(figsize = (16,16))
nx.draw_networkx(G, pos,
                 with_labels=True,
                 labels=node_labels,
                 node_size = 800, font_size = 20,
                 edge_vmin = 0, edge_vmax = 1,
                 node_color='grey',
                 font_color='black',
                 edge_color=edge_weights,
                 edge_cmap=plt.cm.viridis,
                 width=1, alpha=0.7)

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='k', font_size=20)

ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.grid()
ax.set_aspect('equal', adjustable='box')
plt.show()



#################### 设定高斯核阈值
threshold = 0.4
K_threshold = np.copy(K)
# 副本，非视图
K_threshold[K_threshold < threshold] = 0
# 低于阈值置零，改为小于号

# 可视化成对距离矩阵
plt.figure(figsize=(8,8))
sns.heatmap(K_threshold, square = True,
            cmap = 'viridis', vmin = 0, vmax = 1,
            # annot=True, fmt=".3f",
            xticklabels = [], yticklabels = [])
# plt.savefig('亲近度矩阵_heatmap, 设置阈值.svg')

# 创建无向图
G_threshold = nx.Graph(K_threshold, nodetype=int)
# 用邻接矩阵创建无向图

# 添加节点和边
for i in range(12):
    G_threshold.add_node(i, pos=(points[i, 0], points[i, 1]))

# 取出节点位置
pos = nx.get_node_attributes(G_threshold, 'pos')

# 增加节点属性
node_labels = {i: chr(ord('a') + i) for i in range(len(G_threshold.nodes))}
edge_weights = [G_threshold[i][j]['weight'] for i, j in G_threshold.edges]
edge_labels = {(i, j): f'{K_threshold[i, j]:.2f}' for i, j in G_threshold.edges}


# 可视化图
fig, ax = plt.subplots(figsize = (6,6))
nx.draw_networkx(G_threshold, pos,
                 with_labels=True,
                 labels=node_labels,
                 node_size = 800, font_size = 20,
                 edge_vmin = 0, edge_vmax = 1,
                 node_color='grey',
                 font_color='black',
                 edge_color=edge_weights,
                 edge_cmap=plt.cm.viridis,
                 width=1, alpha=0.7)

ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.grid()
ax.set_aspect('equal', adjustable='box')
# plt.savefig('亲近度矩阵_无向图, 设置阈值.svg')
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 鸢尾花数据到无向图

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel

# 自定义高斯核函数
def gaussian_kernel(distance, sigma=1.0):
    return np.exp(- (distance ** 2) / (2 * sigma ** 2))

# 加载鸢尾花数据集
iris = load_iris()
data = iris.data[:, :2]
# 只使用前两个特征
target = iris.target

# 计算高斯核矩阵
sigma = 0.5  # 高斯核的参数
# K = rbf_kernel(data, data)

# 计算成对距离矩阵
D = np.linalg.norm(data[:, np.newaxis, :] - data, axis=2)
K = gaussian_kernel(D, sigma)

# 计算欧氏距离矩阵
# distances = euclidean_distances(data)
# 用成对距离矩阵也可以构造无向图

sns.heatmap(K, square = True, cmap = 'RdYlBu_r', xticklabels = [], yticklabels = [])
# 对角线置0，避免自环
np.fill_diagonal(K, 0)

# 创建无向图
G = nx.Graph()
threshold = 0.8

# 添加节点
for i in range(len(data)):
    G.add_node(i, label=f"{iris.target_names[target[i]]} ({i+1})")

# 添加边，满足欧氏距离小于阈值的条件
# for i in range(len(data)):
#     for j in range(i + 1, len(data)):
#         if K[i, j] > threshold:
#             G.add_edge(i, j)

# 找到满足条件的节点对，并添加边
# 避免使用 for 循环
indices = np.where(K > threshold)
edges = list(zip(indices[0], indices[1]))

G.add_edges_from(edges)


# 使用鸢尾花数据的真实位置绘制图形
pos = {i: (data[i, 0], data[i, 1]) for i in range(len(data))}
labels = nx.get_edge_attributes(G, 'weight')

fig, ax = plt.subplots(figsize = (6,6))
nx.draw_networkx(G, pos, with_labels=False, node_size = 18)
# node的颜色代表鸢尾花类别
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

ax.set_xlim(4,8)
ax.set_ylim(1,5)
ax.grid()
ax.set_aspect('equal', adjustable='box')
# plt.savefig('鸢尾花_高斯核_无向图, 真实位置.svg')
plt.show()


# 不使用真实数据位置

fig, ax = plt.subplots(figsize = (6,6))
nx.draw_networkx(G, with_labels=False, node_size = 18)
# plt.savefig('鸢尾花_高斯核_无向图, 随机位置.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 基于相关性系数矩阵创建的无向图

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from networkx.algorithms.community.centrality import girvan_newman

df = pd.read_pickle('stock_levels_df_2020.pkl')

df.head(5)

# 计算日收益率
returns_df = df['Adj Close'].pct_change()


# 整列、整行都为NaN的删除
returns_df.dropna(axis = 1,how='all', inplace = True)
returns_df.dropna(axis = 0,how='all', inplace = True)
returns_df


# 计算相关性系数矩阵
corr = returns_df.corr()

# 可视化相关性系数矩阵
sns.heatmap(corr)
plt.savefig('相关性系数矩阵.png')

# 将相关性系数矩阵转换为邻接矩阵
A = corr.copy()

# 设定阈值
threshold = 0.8

# 低于阈值，置0
A[A < threshold] = 0

# 超过阈值，置1
A[A >= threshold] = 1
A

A = A - np.identity(len(A))
# 将对角线元素置0，不画自环
A

# 可视化邻接矩阵
sns.heatmap(A)
# plt.savefig('邻接矩阵.png')


# 创建图
G = nx.from_numpy_array(A.to_numpy())

# 修改节点名称
G = nx.relabel_nodes(G, dict(enumerate(A.columns)))

# 可视化图
pos = nx.spring_layout(G, seed = 8)

plt.figure(figsize = (6,6))
nx.draw_networkx(G,pos = pos, with_labels = False, node_size = 8)
# plt.savefig('图.png')


# 最大连通分量
Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
pos_Gcc = {k: pos[k] for k in list(Gcc.nodes())}
# 取出子图节点坐标

plt.figure(figsize = (6,6))
nx.draw_networkx(Gcc, pos_Gcc, with_labels = False, node_size=8)
# plt.savefig('最大连通分量子图.svg')

# 划分社区
communities = girvan_newman(G)

node_groups = []
for com in next(communities):
    node_groups.append(list(com))

print(len(node_groups))
# 按子列表长度排列
node_groups.sort(key=len, reverse = True)

plt.figure(figsize = (6,6))
nx.draw_networkx(G,pos = pos, node_color = '0.8', with_labels = False, node_size = 8)
# 排名前4个社区
list_colors = ['r', 'b', 'orange', 'pink']
for idx,color_i in enumerate(list_colors):
    nx.draw_networkx(G,pos = pos, nodelist = node_groups[idx], node_color = color_i, with_labels = False, node_size = 8)

# plt.savefig('社区.png')




































































































































































































































































































