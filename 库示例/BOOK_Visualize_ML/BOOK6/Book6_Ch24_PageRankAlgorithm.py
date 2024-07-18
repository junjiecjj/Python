

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PageRank算法

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

# 创建有向图的实例
directed_G = nx.DiGraph()
# 添加多个顶点
directed_G.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f'])

# 添加几组有向边
directed_G.add_edges_from([('a','b'),('a','c'),('a','d'), ('a','e'),('a','f')])
directed_G.add_edges_from([('b','d'),('b','e')])
directed_G.add_edges_from([('c','a'),('c','d'),('c','e')])
directed_G.add_edges_from([('d','b'),('d','e')])
directed_G.add_edges_from([('e','a')])
directed_G.add_edges_from([('f','b'),('f','c'),('f','e')])

pos = nx.circular_layout(directed_G)
node_color = ['purple', 'blue', 'green', 'orange', 'red', 'pink']

# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(directed_G, pos = pos, node_color = node_color, node_size = 880, font_size = 20)
# plt.savefig('网页之间关系的有向图.svg')

# 邻接矩阵
A = nx.adjacency_matrix(directed_G).todense()
A

list(directed_G.nodes)
plt.figure(figsize = (6,6))
sns.heatmap(A, cmap = 'Blues',
            annot = True, fmt = '.0f',
            xticklabels = list(directed_G.nodes),
            yticklabels = list(directed_G.nodes),
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('邻接矩阵.svg')

# 转移矩阵
deg_out = np.diag(np.array(A.sum(axis = 1)).flatten())
# 节点出度
plt.figure(figsize = (6,6))
sns.heatmap(deg_out, cmap = 'Blues',
            annot = True, fmt = '.0f',
            xticklabels = list(directed_G.nodes),
            yticklabels = list(directed_G.nodes),
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('节点出度.svg')

# 邻接矩阵的行归一化
T_T = A /  np.array(A.sum(axis = 1))
# 转置获得转移矩阵
T = T_T.T

# T.sum(axis = 0)
plt.figure(figsize = (6,6))
sns.heatmap(T, cmap = 'Blues',
            annot = True, fmt = '.3f',
            xticklabels = list(directed_G.nodes),
            yticklabels = list(directed_G.nodes),
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('转移矩阵.svg')

# 幂迭代
# 自定义函数幂迭代
def power_iteration(T_, num_iterations: int, L2_norm = False):
    # 初始状态
    r_k = np.ones((len(T_), 1))/len(T_)
    r_k_iter = r_k
    for _ in range(num_iterations):
        # 矩阵乘法 T @ r
        r_k1 = T_ @ r_k
        if L2_norm:
            # 计算L2范数
            r_k1_norm = np.linalg.norm(r_k1)
            # L2范数单位化
            r_k = r_k1 / r_k1_norm
        else:
            # 归一化
            r_k = r_k1 / r_k1.sum()
        # 记录迭代过程结果
        r_k_iter = np.column_stack((r_k_iter, r_k))
    return r_k, r_k_iter

# 调用自行函数完成幂迭代
r_k, r_k_iter = power_iteration(T, 20)

# 可视化幂迭代过程
fig, ax = plt.subplots( figsize = (6,6) )
for i,node_i in zip(range(len(node_color)),list(directed_G.nodes)):
    ax.plot(np.array(r_k_iter[i,:]).flatten(), color = node_color[i], label = node_i)
ax.set_xlim(0,20)
ax.set_ylim(0,0.4)
ax.set_xlabel('Iteration')
ax.set_ylabel('PageRank')
ax.legend(loc = 'upper right')
# plt.savefig('幂迭代.svg')
r_k
# matrix([[0.28662424],
#         [0.1571125 ],
#         [0.0764331 ],
#         [0.16135882],
#         [0.2611465 ],
#         [0.05732483]])

# 特征值分解
eigenValues, eigenVectors = np.linalg.eig(T)

idx = eigenValues.argsort()[::-1]
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]

eigenValues

v0 = eigenVectors[:,0]
v0 = v0.real
# -v0/np.linalg.norm(v0)
v0/v0.sum()
# matrix([[0.2866242 ],
#         [0.15711253],
#         [0.07643312],
#         [0.16135881],
#         [0.2611465 ],
#         [0.05732484]])

############################### 修改有向图
directed_G_2 = directed_G.copy()
directed_G_2.remove_edge('e','a')

# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(directed_G_2,
                 pos = pos,
                 node_color = node_color,
                 node_size = 880)


A_2 = nx.adjacency_matrix(directed_G_2).todense()
plt.figure(figsize = (6,6))
sns.heatmap(A_2, cmap = 'Blues',
            annot = True, fmt = '.0f',
            xticklabels = list(directed_G.nodes),
            yticklabels = list(directed_G.nodes),
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('邻接矩阵，删除ea.svg')


def A_2_T(A):
    T_T = A /  np.array(A.sum(axis = 1))
    # 转置获得转移矩阵
    T = T_T.T
    T[np.isnan(T)] = 0
    return T

T_2 = A_2_T(A_2)
plt.figure(figsize = (6,6))
sns.heatmap(T_2, cmap = 'Blues',
            annot = True, fmt = '.3f',
            xticklabels = list(directed_G.nodes),
            yticklabels = list(directed_G.nodes),
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('转移矩阵，删除ea.svg')


# 幂迭代修正
# 幂迭代，修正
def power_iteration_adjust(T_, num_iterations: int, d = 0.85, tol=1e-6, L2_norm = False):
    n = len(T_)
    # 初始状态
    r_k = np.ones((len(T_),1))/n
    r_k_iter = r_k

    # 幂迭代过程
    for _ in range(num_iterations):
        # 核心迭代计算式
        r_k1 = d * T_ @ r_k + (1-d)/n

        # 检测是否收敛
        if np.linalg.norm(r_k - r_k1, 1) < tol:
            break

        if L2_norm:
            # 计算L2范数
            r_k1_norm = np.linalg.norm(r_k1)

            # L2范数单位化
            r_k = r_k1 / r_k1_norm
        else:
            # 归一化
            r_k = r_k1 / r_k1.sum()
        # 记录迭代过程结果
        r_k_iter = np.column_stack((r_k_iter,r_k))
    return r_k,r_k_iter

r_k_adj, r_k_iter_adj = power_iteration_adjust(T_2, 20, 0.85)
r_k_adj


# 可视化幂迭代过程
fig, ax = plt.subplots(figsize = (6,6))
for i,node_i in zip(range(len(node_color)),list(directed_G.nodes)):
    ax.plot(np.array(r_k_iter_adj[i,:]).flatten() , color = node_color[i], label = node_i)
ax.set_xlim(0,20)
ax.set_ylim(0,0.4)
ax.set_xlabel('Iteration')
ax.set_ylabel('PageRank')
ax.legend(loc = 'upper right')
# plt.savefig('幂迭代，修正.svg')

















#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



































































































































































































































































































































