#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:50:51 2023

@author: jack
"""

import networkx as nx
import matplotlib.pyplot as plt

# 1、生成了标号为0到9的十个点
import matplotlib.pyplot as plt
import networkx as nx
H = nx.path_graph(10)
G=nx.Graph()
G.add_nodes_from(H)
nx.draw(G, with_labels=True)
plt.show()

# 2、添加各节点之间的边
G = nx.Graph()
#导入所有边，每条边分别用tuple表示
G.add_edges_from([(1,2),(1,3),(2,4),(2,5),(3,6),(4,8),(5,8),(3,7)])
nx.draw(G, with_labels=True, edge_color='b', node_color='g', node_size=1000)
plt.show()
#plt.savefig('./generated_image.png') 如果你想保存图片，去除这句的注释



# 4、画个五角星

import networkx as nx
import matplotlib.pyplot as plt
#画图！
G = nx.Graph()
G.add_node(1)
G.add_nodes_from([2,3,4,5])
for i in range(5):
    for j in range(i):
        if (abs(i-j) not in (1,4)):
            G.add_edge(i+1, j+1)
nx.draw(G,
        with_labels=True, #这个选项让节点有名称
        edge_color='b', # b stands for blue!
        pos=nx.circular_layout(G), # 这个是选项选择点的排列方式，具体可以用 help(nx.drawing.layout) 查看
     # 主要有spring_layout  (default), random_layout, circle_layout, shell_layout
     # 这里是环形排布，还有随机排列等其他方式
        node_color='r', # r = red
        node_size=1000, # 节点大小
        width=3, # 边的宽度
       )
plt.show()



# 5、加入权重

import random
G = nx.gnp_random_graph(10,0.3)
for u,v,d in G.edges(data=True):
    d['weight'] = random.random()

edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())

pos = nx.spring_layout(G)
nx.draw(G, pos, node_color='b', edgelist=edges, edge_color=weights, width=10.0, edge_cmap=plt.cm.Blues)
# plt.savefig('edges.png')
plt.show()


# 6、有向图

import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()

G.add_edge('a', 'b', weight=0.6)
G.add_edge('a', 'c', weight=0.2)
G.add_edge('c', 'd', weight=0.1)
G.add_edge('c', 'e', weight=0.7)
G.add_edge('c', 'f', weight=0.9)
G.add_edge('a', 'd', weight=0.3)

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]

pos = nx.spring_layout(G)  # positions for all nodes

# nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
nx.draw_networkx_edges(G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color='b', style='dashed')

# labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

plt.axis('off')
plt.show()

# 7、多层感知机
import matplotlib.pyplot as plt
import networkx as nx
left, right, bottom, top, layer_sizes = .1, .9, .1, .9, [4, 7, 7, 2]
# 网络离上下左右的距离
# layter_sizes可以自己调整
import random
G = nx.Graph()
v_spacing = (top - bottom)/float(max(layer_sizes))
h_spacing = (right - left)/float(len(layer_sizes) - 1)
node_count = 0
for i, v in enumerate(layer_sizes):
    layer_top = v_spacing*(v-1)/2. + (top + bottom)/2.
    for j in range(v):
        G.add_node(node_count, pos=(left + i*h_spacing, layer_top - j*v_spacing))
        node_count += 1
# 这上面的数字调整我想了好半天，汗
for x, (left_nodes, right_nodes) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
    for i in range(left_nodes):
        for j in range(right_nodes):
            G.add_edge(i+sum(layer_sizes[:x]), j+sum(layer_sizes[:x+1]))
# 慢慢研究吧
pos=nx.get_node_attributes(G,'pos')
# 把每个节点中的位置pos信息导出来
nx.draw(G, pos,
        node_color=range(node_count),
        with_labels=True,
        node_size=200,
        edge_color=[random.random() for i in range(len(G.edges))],
        width=3,
        cmap=plt.cm.Dark2, # matplotlib的调色板，可以搜搜，很多颜色呢
        edge_cmap=plt.cm.Blues
       )
plt.show()

##==============================================================================================
##                  python 图论算法包 network 简单使用
##==============================================================================================


import networkx as nx

G = nx.Graph()
G.add_edge('A', 'B', weight=4)
G.add_edge('B', 'D', weight=2)
G.add_edge('A', 'C', weight=3)
G.add_edge('C', 'D', weight=5)
G.add_edge('A', 'D', weight=6)
G.add_edge('C', 'F', weight=7)
G.add_edge('A', 'G', weight=1)
G.add_edge('H', 'B', weight=2)
pos = nx.spring_layout(G)

# 生成邻接矩阵
mat = nx.to_numpy_matrix(G)
print(mat)


# 计算两点间的最短路
# dijkstra_path
path=nx.dijkstra_path(G, source='H', target='F')
print('节点H到F的路径：', path)
distance=nx.dijkstra_path_length(G, source='H', target='F')
print('节点H到F的最短距离为：', distance)

# 一点到所有点的最短路
p=nx.shortest_path(G,source='H') # target not specified
d=nx.shortest_path_length(G,source='H')
for node in G.nodes():
    print("H 到",node,"的最短路径为:",p[node])
    print("H 到",node,"的最短距离为:",d[node])

# 所有点到一点的最短距离
p=nx.shortest_path(G,target='H') # target not specified
d=nx.shortest_path_length(G,target='H')
for node in G.nodes():
    print(node,"到 H 的最短路径为:",p[node])
    print(node,"到 H 的最短距离为:",d[node])
# 任意两点间的最短距离
p=nx.shortest_path_length(G)
p=dict(p)
d=nx.shortest_path_length(G)
d=dict(d)
# for node1 in G.nodes():
#     for node2 in G.nodes():
#         print(node1,"到",node2,"的最短距离为:",d[node1][node2])

# 最小生成树
T=nx.minimum_spanning_tree(G) # 边有权重
# print(sorted(T.edges(data=True)))

mst=nx.minimum_spanning_edges(G,data=False) # a generator of MST edges
edgelist=list(mst) # make a list of the edges
print(sorted(edgelist))

# 使用A *算法的最短路径和路径长度
# p=nx.astar_path(G, source='H', target='F')
# print('节点H到F的路径：', path)
# print('节点H到F的路径：', p)
# print('节点H到F的路径：', p1)
# d=nx.astar_path_length(G, source='H', target='F')
# print('节点H到F的距离为：', distance)







































































































































































































































