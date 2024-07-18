

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html


# ◄ networkx.circular_layout() 节点圆周布局
# ◄ networkx.DiGraph() 创建有向图的类，用于表示节点和有向边的关系以进行图论分析
# ◄ networkx.draw_networkx() 用于绘制图的节点和边，可根据指定的布局将图可视化呈现在平面上
# ◄ networkx.get_edge_attributes() 获取图中边的特定属性的字典
# ◄ networkx.get_node_attributes() 获取图中节点的特定属性的字典
# ◄ networkx.Graph() 创建无向图的类，用于表示节点和边的关系以进行图论分析
# ◄ networkx.relabel_nodes() 对节点重命名
# ◄ networkx.to_numpy_matrix() 用于将图表示转换为 NumPy 矩阵，方便在数值计算和线性代数操作中使用
# ◄ numpy.linalg.norm() 计算范数

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 将无向图转换为邻接矩阵

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
# 创建无向图的实例
undirected_G = nx.Graph()

# 添加多个顶点
undirected_G.add_nodes_from(['a', 'b', 'c', 'd'])

# 增加一组边
undirected_G.add_edges_from([('a','b'),
                             ('b','c'),
                             ('b','d'),
                             ('c','d'),
                             ('c','a')])

plt.figure(figsize = (6,6))
nx.draw_networkx(undirected_G, node_size = 880)


# 邻接矩阵
# adjacency_matrix = nx.to_numpy_matrix(undirected_G)
adjacency_matrix = nx.adjacency_matrix(undirected_G).todense()
adjacency_matrix
# matrix([[0, 1, 1, 0],
#         [1, 0, 1, 1],
#         [1, 1, 0, 1],
#         [0, 1, 1, 0]])


# 空手道俱乐部图的邻接矩阵
G_karate = nx.karate_club_graph()
# 空手道俱乐部图
pos = nx.spring_layout(G_karate,seed=2)

plt.figure(figsize = (6,6))
nx.draw_networkx(G_karate, pos = pos)
# plt.savefig('空手道俱乐部图.svg')

A_karate = nx.adjacency_matrix(G_karate).todense()
# 邻接矩阵

sns.heatmap(A_karate,cmap = 'RdYlBu_r', square = True, xticklabels = [], yticklabels = [])
# plt.savefig('A邻接矩阵.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 将邻接矩阵转换为无向图

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 定义邻接矩阵
adjacency_matrix = np.array([[0, 1, 1, 0],
                             [1, 0, 1, 1],
                             [1, 1, 0, 1],
                             [0, 1, 1, 0]])
# 用邻接矩阵创建无向图
G = nx.Graph(adjacency_matrix, nodetype=int)

# 创建字典，可视化时用作节点标签
node_labels = {i: chr(ord('a') + i) for i in range(len(G.nodes))}
node_labels
# {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(G, with_labels=True, labels=node_labels)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 补图的邻接矩阵

import networkx as nx
import matplotlib.pyplot as plt
import random
import seaborn as sns

def visualize(G, fig_title):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (12,6))
    pos = nx.circular_layout(G)
    # 左子图
    nx.draw_networkx(G, ax = axs[0], pos = pos, with_labels = False, node_size = 100)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].axis('off')

    # 邻接矩阵
    A = nx.adjacency_matrix(G).todense()
    # 右子图
    sns.heatmap(A, cmap = 'Blues', ax = axs[1], annot = True, fmt = '.0f',
                xticklabels = list(G.nodes), yticklabels = list(G.nodes),
                linecolor = 'k', square = True, linewidths = 0.2, cbar = False)
    # plt.savefig(fig_title + '.svg')

# 创建完全图
G_complete = nx.complete_graph(9)
print(len(G_complete.edges))
visualize(G_complete,'完全图')

# 随机删除一半边
random.seed(8)
# 副本，非视图
G = G_complete.copy(as_view=False)
edges_removed = random.sample(list(G.edges), 18)
G.remove_edges_from(edges_removed)
visualize(G,'图G')

# 补图
G_complement = nx.complement(G)
visualize(G_complement,'图G补图')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 特殊图的邻接矩阵

import networkx as nx
import matplotlib.pyplot as plt
import random
import seaborn as sns

def visualize(G, pos, fig_title):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (12,6))
    # 左子图
    nx.draw_networkx(G, ax = axs[0], pos = pos, with_labels = False, node_size = 128)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].axis('off')

    # 邻接矩阵
    C = nx.adjacency_matrix(G).todense()

    # 右子图
    sns.heatmap(C, cmap = 'Blues', ax = axs[1], annot = False, xticklabels = [], yticklabels = [],
                linecolor = 'k', square = True, linewidths = 0.2, cbar = True)
    # plt.savefig(fig_title + '.svg')

# 完全图
G_complete = nx.complete_graph(12)
pos = nx.circular_layout(G_complete)
visualize(G_complete, pos, '完全图')


# 完全二分图
G_complete_bipartite = nx.complete_bipartite_graph(5,7)
left = nx.bipartite.sets(G_complete_bipartite)[0]
pos = nx.bipartite_layout(G_complete_bipartite, left)
visualize(G_complete_bipartite, pos, '完全二分图')

# 正四面体图
tetrahedral_graph = nx.tetrahedral_graph()
pos = nx.spring_layout(tetrahedral_graph)
visualize(tetrahedral_graph, pos, '正四面体图')

# 正六面体图
cubical_graph = nx.cubical_graph()
pos = nx.spring_layout(cubical_graph)
visualize(cubical_graph, pos, '正六面体图')

# 正八面体图
octahedral_graph = nx.octahedral_graph()
pos = nx.spring_layout(octahedral_graph)
visualize(octahedral_graph, pos, '正八面体图')


# 正十二面体图¶
dodecahedral_graph = nx.dodecahedral_graph()
pos = nx.spring_layout(dodecahedral_graph)
visualize(dodecahedral_graph, pos, '正十二面体图')


# 正二十面体图
icosahedral_graph = nx.icosahedral_graph()
pos = nx.spring_layout(icosahedral_graph)
visualize(icosahedral_graph, pos, '正二十面体图')


# 平衡树
balanced_tree = nx.balanced_tree(3,3)
pos = nx.spring_layout(balanced_tree)
visualize(balanced_tree, pos, '平衡树')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 将有向图转换为邻接矩阵
import matplotlib.pyplot as plt
import networkx as nx
# 创建有向图的实例
directed_G = nx.DiGraph()

# 添加多个顶点
directed_G.add_nodes_from(['a', 'b', 'c', 'd'])

# 增加一组有向边
directed_G.add_edges_from([('b','a'),
                           ('c','b'),
                           ('b','d'),
                           ('d','c'),
                           ('a','c')])


plt.figure(figsize = (6,6))
nx.draw_networkx(directed_G, node_size = 180)

adjacency_matrix = nx.adjacency_matrix(directed_G).todense()
adjacency_matrix
nx.to_numpy_matrix(directed_G)
# matrix([[0., 0., 1., 0.],
#         [1., 0., 0., 1.],
#         [0., 1., 0., 0.],
#         [0., 0., 1., 0.]])

directed_G.in_degree()
directed_G.out_degree()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 传球问题

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# # 创建有向图的实例
directed_G = nx.DiGraph()

# # 添加多个顶点
directed_G.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f'])

directed_G.add_edges_from([('a','b'),
                            ('a','c'),
                            ('a','d'),
                            ('a','e'),
                            ('a','f')])

directed_G.add_edges_from([('b','a'),
                            ('b','c'),
                            ('b','d'),
                            ('b','e'),
                            ('b','f')])

directed_G.add_edges_from([('c','a'),
                            ('c','b'),
                            ('c','d'),
                            ('c','e'),
                            ('c','f')])

directed_G.add_edges_from([('d','a'),
                            ('d','b'),
                            ('d','c'),
                            ('d','e'),
                            ('d','f')])

directed_G.add_edges_from([('e','a'),
                            ('e','b'),
                            ('e','c'),
                            ('e','d'),
                            ('e','f')])

directed_G.add_edges_from([('f','a'),
                            ('f','b'),
                            ('f','c'),
                            ('f','d'),
                            ('f','e')])

G = nx.complete_graph(6, nx.DiGraph())
G.number_of_nodes()
# 6
G.number_of_edges()
# 30

mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
node_color = ['purple', 'blue', 'green', 'orange', 'red', 'pink']
G = nx.relabel_nodes(G, mapping)
pos = nx.circular_layout(G)
plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos = pos, connectionstyle='arc3, rad = 0.1', node_color = node_color, node_size = 880, font_size  =20)
# plt.savefig('6-node directed G, complete.svg')


A = nx.adjacency_matrix(G).todense()
# matrix([[0, 1, 1, 1, 1, 1],
#         [1, 0, 1, 1, 1, 1],
#         [1, 1, 0, 1, 1, 1],
#         [1, 1, 1, 0, 1, 1],
#         [1, 1, 1, 1, 0, 1],
#         [1, 1, 1, 1, 1, 0]])

# 球在A手里
x0 = np.array([[1,0,0,0,0,0]]).T
# 第1次传球
x1 = A @ x0
x1
# matrix([[0],
#         [1],
#         [1],
#         [1],
#         [1],
#         [1]])


x1.sum() # 5
A.sum() # 30

# 第2次传球
x2 = A @ x1
x2
# matrix([[5],
#         [4],
#         [4],
#         [4],
#         [4],
#         [4]])
A @ A

(A @ A).sum()

x2.sum()

# 第3次传球
x3 = A @ x2
x3

x3.sum()
A@A@A

# 第4次传球
x4 = A @ x3
x4
x4.sum()
A@A@A@A


# 组合数求法
m = 6
n = 4
import math
total_num = 0
for idx in range(1,math.floor(n/2) + 1):
    num_idx = (m - 1)**idx*(m - 2)**(n - 2*idx) * math.comb(n - 1 - idx, idx - 1)
    total_num = total_num + num_idx
total_num


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 邻接矩阵乘法


import matplotlib.pyplot as plt
import networkx as nx
# 创建有向图的实例
directed_G = nx.DiGraph()

# 添加多个顶点
directed_G.add_nodes_from(['a', 'b', 'c', 'd'])
# 增加一组有向边
directed_G.add_edges_from([('a','b'),('b','a'),
                           ('c','b'),('b','c'),
                           ('b','d'),('d','c'),
                           ('a','c')])

plt.figure(figsize = (6,6))
nx.draw_networkx(directed_G,
                 pos = nx.spring_layout(directed_G,seed = 8),
                 node_size = 180)
# plt.savefig('有向图.svg')

A = nx.adjacency_matrix(directed_G).todense()
A
A @ A
A @ A @ A @ A
A @ A.T
A.T @ A


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 特征向量中心性

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# 空手道俱乐部
G_karate = nx.karate_club_graph()
# 空手道俱乐部图
pos_karate = nx.spring_layout(G_karate,seed=2)

# 特征向量中心性
centrality_karate = nx.eigenvector_centrality(G_karate)

# 取出特征向量中心性具体值
list_c_karate = np.array(list(centrality_karate.values()))
max(list_c_karate)
min(list_c_karate)

# 可视化
plt.figure(figsize = (6,6))
# nx.draw_networkx(G_karate, pos=pos_karate, edge_color = '0.18', with_labels=False, cmap = 'RdYlBu_r',  alpha = 0.68, width=0.15)
nx.draw_networkx(G_karate, pos_karate, cmap = 'RdYlBu_r', with_labels = False, node_color = list_c_karate)
# plt.savefig('空手道俱乐部，特征向量中心性.svg')

# 更复杂的图
seed = 89
# 创建图
G = nx.gnp_random_graph(100, 0.5, seed=seed)
pos = nx.spring_layout(G, seed = 88)

# 特征向量中心性
centrality = nx.eigenvector_centrality(G)

# 取出特征向量中心性具体值
list_c = np.array(list(centrality.values()))

# 图 27. 特征向量中心性的分布
fig, ax = plt.subplots(figsize=(8, 8))
plt.hist(list_c, ec = 'k')
plt.ylabel('Count')
plt.xlabel('Eigenvector centrality')
# plt.savefig('特征向量中心性分布.svg')

max_c = max(list_c)
min_c = min(list_c)

# 以特征向量中心性大小设定节点大小
node_size = 100 * (list_c - min_c)/(max_c - min_c)

# 图 26. 根据特征向量中心性大小渲染节点
fig, ax = plt.subplots(figsize=(8, 8))
nx.draw_networkx(G, pos = pos, node_size = node_size, edge_color = '0.18', with_labels = False, cmap = 'RdYlBu_r', node_color = node_size, alpha = 0.68, width = 0.15)
plt.axis("off")
# plt.savefig('特征向量中心性.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%












