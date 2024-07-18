

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 无向图


import matplotlib.pyplot as plt
import networkx as nx


undirected_G = nx.Graph()
# 创建无向图的实例

# 顶点
undirected_G.add_node('a')
# 添加单一顶点
undirected_G.add_nodes_from(['b', 'c', 'd'])
# 添加多个顶点


# 边
undirected_G.add_edge('a', 'b')
# 添加一条边
undirected_G.add_edges_from([('b','c'),
                             ('b','d'),
                             ('c','d'),
                             ('c','a')])
# 增加一组边


# 可视化
random_pos = nx.random_layout(undirected_G, seed=188)
# 设定随机种子，保证每次绘图结果一致

pos = nx.spring_layout(undirected_G, pos=random_pos)
# 使用弹簧布局算法来排列图中的节点
# 使得节点之间的连接看起来更均匀自然
plt.figure(figsize = (6,6))
nx.draw_networkx(undirected_G, pos = pos, node_size = 180)
# plt.savefig('G_4顶点_5边.svg')

plt.figure(figsize = (6,6))
nx.draw_networkx(undirected_G, node_size = 180)
# plt.savefig('G_4顶点_5边,位置不固定.svg')

# 属性
# 图的阶
undirected_G.order()
# 4

# 图的节点数
undirected_G.number_of_nodes()
# 4

# 列出图的节点
undirected_G.nodes
# NodeView(('a', 'b', 'c', 'd'))

for node_i in undirected_G.nodes:
    print(node_i, list(undirected_G.neighbors(node_i)))
# a ['b', 'c']
# b ['a', 'c', 'd']
# c ['b', 'd', 'a']
# d ['b', 'c']

# 图的大小
undirected_G.size()
# 5

# 列出图的边
undirected_G.edges
# EdgeView([('a', 'b'), ('a', 'c'), ('b', 'c'), ('b', 'd'), ('c', 'd')])

for edge_i in undirected_G.edges:
    print(edge_i)
# ('a', 'b')
# ('a', 'c')
# ('b', 'c')
# ('b', 'd')
# ('c', 'd')

for u,v in undirected_G.edges:
    print(u,v)
# a b
# a c
# b c
# b d
# c d

# 图的边数
undirected_G.number_of_edges()
# 5
# 判断是否存在ab边
undirected_G.has_edge('a', 'b')
# True
# 判断是否存在ad边
undirected_G.has_edge('a', 'd')
# False

# 图的度
undirected_G.degree()
# DegreeView({'a': 2, 'b': 3, 'c': 3, 'd': 2})

dict(undirected_G.degree())
# {'a': 2, 'b': 3, 'c': 3, 'd': 2}

# 图的度
undirected_G.degree('a')
# 2

# 邻居
list(undirected_G.neighbors('a'))
type(undirected_G.neighbors('a'))


# 删除
# undirected_G.remove_node('a')
# undirected_G.remove_nodes_from(['b','a'])
# undirected_G.remove_edge('b','a')

# undirected_G.remove_edges_from([('b','a'),('b','c')])

# 自环
undirected_G.add_edge('a', 'a')
# 添加一条自环

plt.figure(figsize = (6,6))
nx.draw_networkx(undirected_G, pos = pos, node_size = 180)
# plt.savefig('G_4顶点_5边_a自环.svg')

# 图的大小
undirected_G.size()

# 列出图的边
undirected_G.edges

# 节点a的度
undirected_G.degree('a')


# 邻居
list(undirected_G.neighbors('a'))




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 同构

import networkx as nx
import matplotlib.pyplot as plt

# 第一幅图
G = nx.cubical_graph()
# 立方体图

plt.figure(figsize = (6,6))

nx.draw_networkx(G, pos = nx.spring_layout(G, seed = 8), with_labels=True, node_color="c")
# plt.savefig('图G.svg')


# 第二幅图
H = nx.Graph()
H.add_edges_from([('a','b'),('b','c'),
                  ('c','d'),('e','f'),
                  ('f','g'),('g','h'),
                  ('b','h'),('c','g'),
                  ('d','a'),('e','h'),
                  ('e','a'),('d','f')])

plt.figure(figsize = (6,6))

nx.draw_networkx(H, pos = nx.circular_layout(H), with_labels=True, node_color="orange")
# plt.savefig('图H.svg')

# 判断是否同构
nx.is_isomorphic(G,H)


# 节点对应关系
nx.vf2pp_isomorphism(G,H, node_label="label")



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 多图¶

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


Multi_G = nx.MultiGraph()
# 多图对象shli

Multi_G.add_nodes_from(['a', 'b', 'c', 'd'])
# 添加多个顶点

Multi_G.add_edges_from([('a','b'), # 平行边
                        ('a','b'),
                        ('a','c'), # 平行边
                        ('a','c'),
                        ('a','d'),
                        ('b','d'),
                        ('c','d')])
# 添加多条边
print(Multi_G.edges)

# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(Multi_G, with_labels=True)

# 获得邻接矩阵
adjacency_matrix = nx.to_numpy_matrix(Multi_G)

print(adjacency_matrix)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 子图
import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()
# 创建无向图的实例

G.add_nodes_from(['a', 'b', 'c', 'd'])
# 添加多个顶点

G.add_edges_from([('a','b'),
                  ('b','c'),
                  ('b','d'),
                  ('c','d'),
                  ('c','a')])
# 增加一组边

# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(G, with_labels = True)

# 基于节点子集的子图
Sub_G_nodes = G.subgraph(['a','b','c'])

# 基于节点子集的子图
set(G.nodes) - set(Sub_G_nodes.nodes)

# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(Sub_G_nodes, with_labels = True)


# 基于边子集的子图
Sub_G_edges = G.edge_subgraph([('a','b'),
                               ('b','c'),
                               ('c','d')])

set(G.edges) - set(Sub_G_edges.edges)

# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(Sub_G_edges, with_labels = True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  加权无向图
import matplotlib.pyplot as plt
import networkx as nx

# 创建无向图的实例
weighted_G = nx.Graph()

# 添加多个顶点
weighted_G.add_nodes_from(['a', 'b', 'c', 'd'])

# 增加一组边，并赋予权重
weighted_G.add_edges_from([('a','b', {'weight':10}),
                           ('b','c', {'weight':20}),
                           ('b','d', {'weight':30}),
                           ('c','d', {'weight':40}),
                           ('c','a', {'weight':50})])
# 取出节点a的邻居
weighted_G['a']

# 取出ab边的权重，结果为字典
weighted_G['a']['b']

# 取出ab边的权重，结果为数值
weighted_G['a']['b']['weight']

edge_weights = [weighted_G[i][j]['weight'] for i, j in weighted_G.edges]
# [10, 50, 20, 30, 40]

# 所有边的权重
edge_labels = nx.get_edge_attributes(weighted_G, "weight")

# 所有边的标签
plt.figure(figsize = (6,6))
pos = nx.spring_layout(weighted_G)
nx.draw_networkx(weighted_G, pos = pos, with_labels = True, node_size = 280, edge_color=edge_weights, edge_cmap = plt.cm.RdYlBu, edge_vmin = 10, edge_vmax = 50)

nx.draw_networkx_edge_labels(weighted_G, pos = pos, edge_labels=edge_labels, font_color='k')


# plt.savefig('加权无向图.svg')
nx.to_numpy_matrix(weighted_G)




























#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







































































































































































































































































