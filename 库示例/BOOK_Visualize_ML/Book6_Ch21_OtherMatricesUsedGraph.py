

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 将无向图转换为关联矩阵

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
# 创建无向图的实例
G = nx.Graph()

# 添加多个顶点
G.add_nodes_from(['a', 'b', 'c', 'd'])

# 增加一组边
G.add_edges_from([('a','b'), ('b','c'), ('b','d'), ('c','d'), ('c','a')])

pos = nx.spring_layout(G, seed=2)
plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos, node_size = 880)

# 邻接矩阵
A = nx.adjacency_matrix(G).todense()
plt.figure(figsize = (6,6))
sns.heatmap(A, cmap = 'Blues',
            annot = True, fmt = '.0f',
            xticklabels = list(G.nodes),
            yticklabels = list(G.nodes),
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('邻接矩阵.svg')

# 关联矩阵
G.nodes()
# NodeView(('a', 'b', 'c', 'd'))
G.edges()
# EdgeView([('a', 'b'), ('a', 'c'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

# 关联矩阵
C = nx.incidence_matrix(G).todense()
plt.figure(figsize = (6,6))
sns.heatmap(C, cmap = 'Blues', annot = True, fmt = '.0f',
            yticklabels = list(G.nodes), xticklabels = list(G.edges),
            linecolor = 'k', square = True, linewidths = 0.2)
# plt.savefig('关联矩阵.svg')

G_karate = nx.karate_club_graph()
# 空手道俱乐部图
pos_karate = nx.spring_layout(G_karate,seed=2)

plt.figure(figsize = (6,6))
nx.draw_networkx(G_karate, pos = pos_karate)
# plt.savefig('空手道俱乐部图.svg')

C_karate = nx.incidence_matrix(G_karate).todense()
plt.figure(figsize = (8,6))
sns.heatmap(C_karate, cmap = 'Blues',
            annot = False,
            yticklabels = [],
            xticklabels = [],
            cbar = False,
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('关联矩阵，空手道俱乐部.svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 无向图转换为线图

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

# 创建无向图的实例
undirected_G = nx.Graph()

# 添加多个顶点
undirected_G.add_nodes_from(['a', 'b', 'c', 'd'])

# 增加一组边
undirected_G.add_edges_from([('a','b'), ('b','c'), ('b','d'), ('c','d'), ('c','a')])

plt.figure(figsize = (6,6))
nx.draw_networkx(undirected_G, node_size = 880, font_size = 20)

# 获取无向图边的序列，用于关联矩阵列排序
sequence_edges_G = list(undirected_G.edges())
sequence_edges_G

# 线图
# 转换成线图
L_G = nx.line_graph(undirected_G)

# 图 6. 线图的邻接矩阵
# 可视化线图
plt.figure(figsize = (6,6))
nx.draw_networkx(L_G, pos = nx.spring_layout(L_G), node_size = 880, font_size = 10)
# plt.savefig('线图.svg')

# 矩阵关系
L_G.nodes()
# NodeView((('b', 'd'), ('c', 'd'), ('b', 'c'), ('a', 'b'), ('a', 'c')))
# 需要调整列顺序，列置换
nx.adjacency_matrix(L_G).todense()


# 图 6. 线图的邻接矩阵
# 线图的链接矩阵，调整列顺序
A_LG = nx.adjacency_matrix(L_G, nodelist = sequence_edges_G).todense()
plt.figure(figsize = (6,6))
sns.heatmap(A_LG, cmap = 'Blues',
            annot = True, fmt = '.0f',
            yticklabels = list(sequence_edges_G),
            xticklabels = list(sequence_edges_G),
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('线图的邻接矩阵.svg')

# 图的关联矩阵
C = nx.incidence_matrix(undirected_G).todense()
CTC_2I = C.T @ C - 2 * np.identity(5)
print(CTC_2I - A_LG)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 将有向图转换为关联矩阵

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
# 创建有向图的实例
G = nx.DiGraph()

# 添加多个顶点
G.add_nodes_from(['a', 'b', 'c', 'd'])

# 增加一组有向边
G.add_edges_from([('b','a'),('c','b'), ('b','d'),('d','c'), ('a','c')])

plt.figure(figsize = (6,6))
nx.draw_networkx(G, node_size = 180)

plt.figure(figsize = (6,6))
A = nx.adjacency_matrix(G).todense()
sns.heatmap(A, cmap = 'Blues', annot = True, fmt = '.0f',
            xticklabels = list(G.nodes), yticklabels = list(G.nodes),
            linecolor = 'k', square = True, linewidths = 0.2)
# plt.savefig('邻接矩阵，有向图.svg')

G.nodes()
# NodeView(('a', 'b', 'c', 'd'))
G.edges()
# OutEdgeView([('a', 'c'), ('b', 'a'), ('b', 'd'), ('c', 'b'), ('d', 'c')])
nx.incidence_matrix(G).todense()
# 不考虑方向，等同于与无向图

# 关联矩阵，考虑方向
C = nx.incidence_matrix(G, oriented = True).todense()
# array([[-1.,  1.,  0.,  0.,  0.],
#        [ 0., -1., -1.,  1.,  0.],
#        [ 1.,  0.,  0., -1.,  1.],
#        [ 0.,  0.,  1.,  0., -1.]])

np.abs(C).sum(axis = 1) # array([2., 3., 3., 2.])
dict(G.degree())
# {'a': 2, 'b': 3, 'c': 3, 'd': 2}

# 有向图的入度
C.sum(axis = 1, where = (C == 1)) # array([1., 1., 2., 1.])
# 有向图的入度
G.in_degree()
# InDegreeView({'a': 1, 'b': 1, 'c': 2, 'd': 1})

# 有向图的出度
C.sum(axis = 1, where = (C == -1))
# 有向图的出度
G.out_degree()
# 图 9. 从有向图到关联矩阵热图
plt.figure(figsize = (6,6))
sns.heatmap(C, cmap = 'Blues', annot = True, fmt = '.0f',
            yticklabels = list(G.nodes), xticklabels = list(G.edges),
            linecolor = 'k', square = True, linewidths = 0.2)
# plt.savefig('关联矩阵，有向图.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 特殊图的关联矩阵

import networkx as nx
import matplotlib.pyplot as plt
import random
import seaborn as sns

def visualize(G, pos, fig_title):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (12, 6))

    # 左子图
    nx.draw_networkx(G, ax = axs[0], pos = pos, with_labels = False, node_size = 28)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].axis('off')

    # 关联矩阵
    A = nx.incidence_matrix(G).todense()

    # 右子图
    sns.heatmap(A, cmap = 'Blues', ax = axs[1], annot = False, xticklabels = [], yticklabels = [], linecolor = 'k', square = True, linewidths = 0.2, cbar = False)
    # plt.savefig(fig_title + '.svg')

# 表 3 中图和关联矩阵热图，
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

# 正十二面体图
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



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 无向图度矩阵

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

# 创建无向图的实例
G = nx.Graph()

# 添加多个顶点
G.add_nodes_from(['a', 'b', 'c', 'd'])

# 增加一组边
G.add_edges_from([('a','b'), ('b','c'), ('b','d'),('c','d'), ('c','a')])

# 度矩阵
A = nx.adjacency_matrix(G).todense()
A
# matrix([[0, 1, 1, 0],
#         [1, 0, 1, 1],
#         [1, 1, 0, 1],
#         [0, 1, 1, 0]])

D = A.sum(axis = 0)
# matrix([[2, 3, 3, 2]])


dict(G.degree()).values()
# dict_values([2, 3, 3, 2])

D = np.diag(np.array(D)[0])
D
# 图 11. 无向图到度矩阵
plt.figure(figsize = (6,6))
sns.heatmap(D, cmap = 'Blues',
            annot = True, fmt = '.0f',
            xticklabels = list(G.nodes),
            yticklabels = list(G.nodes),
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('度矩阵.svg')

# 图 12. 空手道俱乐部人员关系图，以及对应度矩阵热图
G_karate = nx.karate_club_graph()
# 空手道俱乐部图
pos_karate = nx.spring_layout(G_karate,seed=2)

dict_degrees = dict(G_karate.degree())

D_karate = np.diag(list(dict_degrees.values()))
plt.figure(figsize = (6,6))
sns.heatmap(D_karate, cmap = 'Blues',
            annot = False,
            xticklabels = [],
            yticklabels = [],
            cbar = False,
            linecolor = 'k', square = True,
            linewidths = 0.1)
# plt.savefig('度矩阵，空手道俱乐部.svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  有向图度矩阵

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
# 创建有向图的实例
G = nx.DiGraph()

# 添加多个顶点
G.add_nodes_from(['a', 'b', 'c', 'd'])

# 增加一组有向边
G.add_edges_from([('b','a'),('c','b'),
                  ('b','d'),('d','c'),
                  ('a','c')])

# 入度矩阵
D_in = G.in_degree()
D_in

D_in = np.diag(list(dict(D_in).values()))
D_in
# array([[1, 0, 0, 0],
#        [0, 1, 0, 0],
#        [0, 0, 2, 0],
#        [0, 0, 0, 1]])
plt.figure(figsize = (6,6))
sns.heatmap(D_in, cmap = 'Blues',
            annot = True, fmt = '.0f',
            xticklabels = list(G.nodes),
            yticklabels = list(G.nodes),
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('入度矩阵.svg')


# 出度矩阵
D_out = G.out_degree()
D_out
D_out = np.diag(list(dict(D_out).values()))
D_out
# array([[1, 0, 0, 0],
#        [0, 2, 0, 0],
#        [0, 0, 1, 0],
#        [0, 0, 0, 1]])

# 图 13. 从有向图到邻接矩阵热图
plt.figure(figsize = (6,6))
sns.heatmap(D_out, cmap = 'Blues',
            annot = True, fmt = '.0f',
            xticklabels = list(G.nodes),
            yticklabels = list(G.nodes),
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('出度矩阵.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  无向图的拉普拉斯矩阵

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

# 创建无向图的实例
G = nx.Graph()

# 添加多个顶点
G.add_nodes_from(['a', 'b', 'c', 'd'])

# 增加一组边
G.add_edges_from([('a','b'),('b','c'),
                  ('b','d'),('c','d'),
                  ('c','a')])

# 拉普拉斯矩阵¶
L = nx.laplacian_matrix(G).toarray()
# array([[ 2, -1, -1,  0],
#        [-1,  3, -1, -1],
#        [-1, -1,  3, -1],
#        [ 0, -1, -1,  2]])
plt.figure(figsize = (6,6))
sns.heatmap(L, cmap = 'Blues',
            annot = True, fmt = '.0f',
            xticklabels = list(G.nodes),
            yticklabels = list(G.nodes),
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('拉普拉斯矩阵.svg')


# 验证拉普拉斯矩阵
A = nx.adjacency_matrix(G).todense()
A
# matrix([[0, 1, 1, 0],
#         [1, 0, 1, 1],
#         [1, 1, 0, 1],
#         [0, 1, 1, 0]])
D = A.sum(axis = 0)
D
# matrix([[2, 3, 3, 2]])
D = np.diag(np.array(D)[0])
D
# array([[2, 0, 0, 0],
#        [0, 3, 0, 0],
#        [0, 0, 3, 0],
#        [0, 0, 0, 2]])

D - A ## == L
# matrix([[ 2, -1, -1,  0],
#         [-1,  3, -1, -1],
#         [-1, -1,  3, -1],
#         [ 0, -1, -1,  2]])





# 归一化 (对称) 拉普拉斯矩阵
L_N = nx.normalized_laplacian_matrix(G).todense()
L_N
# matrix([[ 1.        , -0.40824829, -0.40824829,  0.        ],
#         [-0.40824829,  1.        , -0.33333333, -0.40824829],
#         [-0.40824829, -0.33333333,  1.        , -0.40824829],
#         [ 0.        , -0.40824829, -0.40824829,  1.        ]])
plt.figure(figsize = (6,6))
sns.heatmap(L_N, cmap = 'Blues',
            annot = True, fmt = '.3f',
            xticklabels = list(G.nodes),
            yticklabels = list(G.nodes),
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('归一化拉普拉斯矩阵.svg')

# 验证归一化拉普拉斯矩阵¶
D_sqrt_inv = np.diag(np.array(1/np.sqrt(A.sum(axis = 0)))[0])  #np.diag(1/np.sqrt(A.sum(axis = 0)))
D_sqrt_inv
# array([[0.70710678, 0.        , 0.        , 0.        ],
#        [0.        , 0.57735027, 0.        , 0.        ],
#        [0.        , 0.        , 0.57735027, 0.        ],
#        [0.        , 0.        , 0.        , 0.70710678]])
D_sqrt_inv @ L @ D_sqrt_inv # == L_N
# array([[ 1.        , -0.40824829, -0.40824829,  0.        ],
#        [-0.40824829,  1.        , -0.33333333, -0.40824829],
#        [-0.40824829, -0.33333333,  1.        , -0.40824829],
#        [ 0.        , -0.40824829, -0.40824829,  1.        ]])

# 有向图的拉普拉斯矩阵，请大家参考：
# https://networkx.org/documentation/stable/reference/generated/networkx.linalg.laplacianmatrix.directed_laplacian_matrix.html


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  特殊图的拉普拉斯矩阵
import networkx as nx
import matplotlib.pyplot as plt
import random
import seaborn as sns


def visualize(G, pos, fig_title):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (12,6))

    # 左子图
    nx.draw_networkx(G, ax = axs[0], pos = pos, with_labels = False, node_size = 28)
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].axis('off')

    # 归一化拉普拉斯矩阵矩阵
    L_N = nx.normalized_laplacian_matrix(G).todense()

    # 右子图
    sns.heatmap(L_N, cmap = 'Blues',
                ax = axs[1],
                annot = False,
                xticklabels = [],
                yticklabels = [],
                linecolor = 'k', square = True,
                linewidths = 0.2, cbar = False)
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

# 正十二面体图
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


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 拉普拉斯矩阵谱分析

import matplotlib.pyplot as plt
import networkx as nx
import numpy.linalg


n = 1000  # 1000 nodes
m = 5000  # 5000 edges
# 创建图
G = nx.gnm_random_graph(n, m, seed=8)

# 拉普拉斯矩阵谱分解
eig_values_L = nx.laplacian_spectrum(G) #  (1000,)

# 验证
# L = nx.laplacian_matrix(G)
# numpy.linalg.eigvals(L.toarray())
print("Largest eigenvalue:", max(eig_values_L))
print("Smallest eigenvalue:", min(eig_values_L))

# 图 17. 拉普拉斯矩阵谱分析结果
fig, ax = plt.subplots(figsize = (6,3))
ax.hist(eig_values_L, bins=100, ec = 'k', range = [0,25])
ax.set_ylabel("Count")
ax.set_xlabel("Eigenvalues of Laplacian matrix")
ax.set_xlim(0,25)
ax.set_ylim(0,30)
# plt.savefig('拉普拉斯矩阵谱.svg')


# 归一化拉普拉斯矩阵谱分解
eig_values_L_N = nx.normalized_laplacian_spectrum(G)

# L_N = nx.normalized_laplacian_matrix(G)
# numpy.linalg.eigvals(L_N.toarray())

print("Largest eigenvalue:", max(eig_values_L_N))
print("Smallest eigenvalue:", min(eig_values_L_N))

# 图 17. 拉普拉斯矩阵谱分析结果
fig, ax = plt.subplots(figsize = (6,3))
ax.hist(eig_values_L_N, bins=100, ec = 'k', range = [0,2])
ax.set_ylabel("Count")
ax.set_xlabel("Eigenvalues of normalized Laplacian matrix")
ax.set_xlim(0,2)
ax.set_ylim(0,30)
# plt.savefig('归一化拉普拉斯矩阵谱.svg')




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  谱分解拉普拉斯矩阵完成聚类
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

# 图 18. 空手道俱乐部人员关系图，以及对应拉普拉斯矩阵热图
G = nx.karate_club_graph()
pos = nx.spring_layout(G,seed=2)

plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos = pos)
# plt.savefig('空手道俱乐部图.svg')

# 拉普拉斯矩阵
L = nx.laplacian_matrix(G).todense()

# 特征值分解
lambdas, V = np.linalg.eig(L)

# 按特征值有小到大排列
lambdas_sorted = np.sort(lambdas)
V_sorted = V[:, lambdas.argsort()] # (34, 34)
# 图 18. 空手道俱乐部人员关系图，以及对应拉普拉斯矩阵热图
plt.figure(figsize = (6,6))
sns.heatmap(L, cmap = 'RdYlBu_r', square = True, xticklabels = [], yticklabels = [])
# plt.savefig('L热图.svg')

# 图 19. 拉普拉斯矩阵谱分解结果
plt.figure(figsize = (6,6))
sns.heatmap(V_sorted,cmap = 'RdYlBu_r', square = True, xticklabels = [], yticklabels = [])
# plt.savefig('V热图.svg')

plt.figure(figsize = (6,6))
sns.heatmap(np.diag(lambdas_sorted), cmap = 'RdYlBu_r',square = True, xticklabels = [], yticklabels = [])
# plt.savefig('lambda热图.svg')

plt.figure(figsize = (6,6))
plt.plot(range(0, 34), lambdas_sorted)
plt.ylabel('Ascending eigenvalue')
plt.xlabel('Rank')
plt.xlim(0,34)
plt.ylim(0,50)
# plt.savefig('特征值线图.svg')


plt.figure(figsize = (6,6))
plt.scatter(np.array(V_sorted[:,0]).flatten(), np.array(V_sorted[:,1]).flatten() )
plt.xlabel('First eigenvector')
plt.ylabel('Second eigenvector')
# plt.savefig('特征向量散点图.svg')


# 聚类标签
colors = [ "r" for i in range(0,34)]
for i in range(0,34):
    if (V_sorted[i,1] < 0):
        colors[i] = "b"

plt.figure(figsize = (6,6))
nx.draw_networkx(G,pos,
                 # with_labels = False,
                 node_color=colors)
# plt.savefig('图节点聚类.svg')






















































