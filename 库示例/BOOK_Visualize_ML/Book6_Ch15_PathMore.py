

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html








#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 节点间路径，无向图

import networkx as nx
import matplotlib.pyplot as plt


G = nx.complete_graph(5)
# 完全图



# 可视化图
plt.figure(figsize = (6,6))
pos = nx.circular_layout(G)
nx.draw_networkx(G,
                 pos = pos,
                 with_labels = True,
                 node_size = 188)
# plt.savefig('完全图.svg')

# 节点0、3之间所有路径

all_paths_nodes = nx.all_simple_paths(G, source=0, target=3)
# 节点0、3之间所有路径上的节点

all_paths_edges = nx.all_simple_edge_paths(G, source=0, target=3)
# 节点0、3之间所有路径上的边

fig, axes = plt.subplots(4, 4,
                         figsize = (8,8))

axes = axes.flatten()

for nodes_i, edges_i, ax_i in zip(all_paths_nodes,
                                  all_paths_edges,
                                  axes):

    nx.draw_networkx(G,
                     ax = ax_i,
                     pos = pos,
                     with_labels = True,
                     node_size = 88)

    nx.draw_networkx_nodes(G,
                           ax = ax_i,
                           nodelist = nodes_i,
                           pos = pos,
                           node_size = 88,
                           node_color = 'r')

    nx.draw_networkx_edges(G,
                           pos = pos,
                           ax = ax_i,
                           edgelist = edges_i,
                           edge_color = 'r')

    ax_i.set_title(' → '.join(str(node) for node in nodes_i))
    ax_i.axis('off')

# plt.savefig('节点0、3之间所有路径.svg')


# 最短路径
print(nx.shortest_path(G, source=0, target=3))



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 节点间路径，有向图


import matplotlib.pyplot as plt
import networkx as nx
import random

# 创建一个包含5个节点的无向完全图
G_undirected = nx.complete_graph(5)


# 创建一个新的有向图
G_directed = nx.DiGraph()



# 为每对节点随机选择方向
random.seed(8)
for u, v in G_undirected.edges():
    if random.choice([True, False]):
        G_directed.add_edge(u, v)
    else:
        G_directed.add_edge(v, u)

# 可视化图
plt.figure(figsize = (6,6))
pos = nx.circular_layout(G_directed)
nx.draw_networkx(G_directed,
                 pos = pos,
                 with_labels = True,
                 node_size = 188)
# plt.savefig('完全图，设定边方向.svg')


# 节点0为始点、3为终点之间所有路径

all_paths_nodes = nx.all_simple_paths(G_directed, source=0, target=3)
# 节点0为始点、3为终点所有路径上的节点

all_paths_edges = nx.all_simple_edge_paths(G_directed, source=0, target=3)
# 节点0为始点、3为终点所有路径上的边

fig, axes = plt.subplots(1, 2,
                         figsize = (8,4))

axes = axes.flatten()

for nodes_i, edges_i, ax_i in zip(all_paths_nodes,
                                  all_paths_edges,
                                  axes):

    nx.draw_networkx(G_directed,
                     ax = ax_i,
                     pos = pos,
                     with_labels = True,
                     node_size = 88)

    nx.draw_networkx_nodes(G_directed,
                           ax = ax_i,
                           nodelist = nodes_i,
                           pos = pos,
                           node_size = 88,
                           node_color = 'r')

    nx.draw_networkx_edges(G_directed,
                           pos = pos,
                           ax = ax_i,
                           edgelist = edges_i,
                           edge_color = 'r')

    ax_i.set_title(' → '.join(str(node) for node in nodes_i))
    ax_i.axis('off')

# plt.savefig('节点0为始点、3为终点的所有路径，有向图.svg')


# 节点3为始点、0为终点之间所有路径

all_paths_nodes = nx.all_simple_paths(G_directed, source=3, target=0)
# 节点3为始点、0为终点所有路径上的节点

all_paths_edges = nx.all_simple_edge_paths(G_directed, source=3, target=0)
# 节点3为始点、0为终点所有路径上的边

fig, axes = plt.subplots(1, 2,
                         figsize = (8,4))

axes = axes.flatten()

for nodes_i, edges_i, ax_i in zip(all_paths_nodes,
                                  all_paths_edges,
                                  axes):

    nx.draw_networkx(G_directed,
                     ax = ax_i,
                     pos = pos,
                     with_labels = True,
                     node_size = 88)

    nx.draw_networkx_nodes(G_directed,
                           ax = ax_i,
                           nodelist = nodes_i,
                           pos = pos,
                           node_size = 88,
                           node_color = 'r')

    nx.draw_networkx_edges(G_directed,
                           pos = pos,
                           ax = ax_i,
                           edgelist = edges_i,
                           edge_color = 'r')

    ax_i.set_title(' → '.join(str(node) for node in nodes_i))
    ax_i.axis('off')

# plt.savefig('节点3为始点、0为终点的所有路径，有向图.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 环，有向图

import matplotlib.pyplot as plt
import networkx as nx
import random


# 创建一个包含5个节点的无向完全图
G_undirected = nx.complete_graph(5)


# 创建一个新的有向图
G_directed = nx.DiGraph()



# 为每对节点随机选择方向
random.seed(8)
for u, v in G_undirected.edges():
    if random.choice([True, False]):
        G_directed.add_edge(u, v)
    else:
        G_directed.add_edge(v, u)


# 可视化图
plt.figure(figsize = (6,6))
pos = nx.circular_layout(G_directed)
nx.draw_networkx(G_directed,
                 pos = pos,
                 with_labels = True,
                 node_size = 188)
# plt.savefig('完全图，设定边方向.svg')


# 发现 cycle

cycle = nx.find_cycle(G_directed, orientation="original")
print(cycle)


# 可视化图
plt.figure(figsize = (6,6))
pos = nx.circular_layout(G_directed)
nx.draw_networkx(G_directed,
                 pos = pos,
                 with_labels = True,
                 node_size = 188)
nx.draw_networkx_edges(G_directed, pos = pos,
                       edgelist=cycle, edge_color="r", width=2)
# plt.savefig('有向图中的cycle.svg')



list_cycles = list(nx.simple_cycles(G_directed))
# 找到有向图中所有环
list_cycles

len(list_cycles)


# 自定义函数将节点序列转化为边序列 (闭环)

def nodes_2_edges(node_list):

    # 使用列表生成式创建边的列表
    list_edges = [(node_list[i], node_list[i+1])
                  for i in range(len(node_list)-1)]

    # 加上一个额外的边从最后一个节点回到第一个节点，形成闭环
    closing_edge = [(node_list[-1], node_list[0])]
    list_edges = list_edges + closing_edge
    return  list_edges

# 也可以用函数
# nx.utils.pairwise(, cyclic = True)




# 可视化有向图中3个环
fig, axes = plt.subplots(1, 3, figsize = (9,3))

axes = axes.flatten()

for nodes_i, ax_i in zip(list_cycles, axes):

    edges_i = nodes_2_edges(nodes_i)

    nx.draw_networkx(G_directed,
                     ax = ax_i,
                     pos = pos,
                     with_labels = True,
                     node_size = 88)

    nx.draw_networkx_nodes(G_directed,
                           ax = ax_i,
                           nodelist = nodes_i,
                           pos = pos,
                           node_size = 88,
                           node_color = 'r')

    nx.draw_networkx_edges(G_directed,
                           pos = pos,
                           ax = ax_i,
                           edgelist = edges_i,
                           edge_color = 'r')

    ax_i.set_title(' → '.join(str(node) for node in nodes_i))
    ax_i.axis('off')

# plt.savefig('有向图中所有cycles.svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 最短路径问题，无向图

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 创建图
G = nx.Graph()
G.add_nodes_from(["A", "B", "C", "D", "E", "F", "G", "H"])
G.add_edge("A", "B", weight=4)
G.add_edge("A", "H", weight=8)
G.add_edge("B", "C", weight=8)
G.add_edge("B", "H", weight=11)
G.add_edge("C", "D", weight=7)
G.add_edge("C", "F", weight=4)
G.add_edge("C", "I", weight=2)
G.add_edge("D", "E", weight=9)
G.add_edge("D", "F", weight=14)
G.add_edge("E", "F", weight=10)
G.add_edge("F", "G", weight=2)
G.add_edge("G", "H", weight=1)
G.add_edge("G", "I", weight=6)
G.add_edge("H", "I", weight=7)

pos = nx.spring_layout(G, seed = 8)
edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}


plt.figure(figsize = (6,6))
nx.draw_networkx(G,
                 pos = pos,
                 node_color = '#0058FF',
                 node_size = 180)
nx.draw_networkx_edge_labels(G,
                             pos = pos,
                             edge_labels=edge_labels)
# plt.savefig('无向图.svg')

# 指定两节点最短距离

# 确定是否存在路径

nx.has_path(G,
            source = "A", # 始点
            target = "E", # 终点
           )

# 找到A、E节点之间的最短距离，考虑权重
path_A_2_E = nx.shortest_path(G,
                              source = "A", # 始点
                              target = "E", # 终点
                              weight="weight")
print(path_A_2_E)


# 最短距离值
nx.shortest_path_length(G,
                        source = "A", # 始点
                        target = "E", # 终点
                        weight="weight")
# 最短路径的边，列表的每个元素为元组
# 元组有两个元素，代表边的两个节点
path_edges = list(zip(path_A_2_E, path_A_2_E[1:]))

# 最短路径所在边：红色
# 其他边：黑色
edge_colors = ["#FF5800"
               if edge in path_edges
                       or tuple(reversed(edge)) in path_edges
               else "black" for edge in G.edges()]

# 可视化图
plt.figure(figsize = (6,6))
# 路径上的节点
nx.draw_networkx_nodes(G, pos,
                       nodelist = path_A_2_E,
                       node_color = '#FF5800',
                       node_size = 180)

# 路径之外其他节点
not_path_A_2_E = set(G.nodes()) - set(path_A_2_E)
nx.draw_networkx_nodes(G, pos,
                       nodelist = not_path_A_2_E,
                       node_color = '#0058FF',
                       node_size = 180)

nx.draw_networkx_edges(G, pos,
                       edge_color=edge_colors)

nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edge_labels(
    G, pos, edge_labels=edge_labels)

# plt.savefig('A、E最短距离.svg')

# 起点为A的所有最短路径
path_from_A = nx.shortest_path(G,
                               source = "A", # 起点
                               weight="weight")
print(path_from_A)

# 最短距离值
nx.shortest_path_length(G,
                        source = "A", # 始点
                        weight="weight")

# 终点为E的所有最短路径
path_2_E = nx.shortest_path(G,
                            target = "E", # 终点
                            weight="weight")
# nx.single_source_shortest_path()
print(path_2_E)


# 最短距离值
nx.shortest_path_length(G,
                        target = "E", # 终点
                        weight="weight")
# nx.single_source_shortest_path_length()


# 图中任意两点所有最短路径
path_all = nx.shortest_path(G,
                            weight="weight")


path_all['A']['E']

# Shortest_path_matrix
# 最短距离值
distances_all = dict(nx.shortest_path_length(G,
                        weight="weight"))

# 将成对最短距离整理为矩阵
Shortest_D_matrix = np.array([[v[j] for j in list(distances_all.keys())]
          for k, v in distances_all.items()])

sns.heatmap(Shortest_D_matrix, cmap = 'Blues',
            annot = True, fmt = '.0f',
            xticklabels = list(G.nodes),
            yticklabels = list(G.nodes),
            linecolor = 'k', square = True,
            cbar = False,
            linewidths = 0.2)
plt.savefig('Shortest_D_matrix.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 最短路径问题，有向图

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



# 创建图
G = nx.DiGraph()
G.add_nodes_from(["A", "B", "C", "D", "E", "F", "G", "H"])

# 有向边
G.add_edge("A", "B", weight=4)
G.add_edge("H", "A", weight=8)
G.add_edge("B", "C", weight=8)
G.add_edge("H", "B", weight=11)
G.add_edge("C", "D", weight=7)
G.add_edge("F", "C", weight=4)
G.add_edge("C", "I", weight=2)
G.add_edge("E", "D", weight=9)
G.add_edge("D", "F", weight=14)
G.add_edge("E", "F", weight=10) # 颠倒顺序
G.add_edge("G", "F", weight=2)
G.add_edge("G", "H", weight=1)
G.add_edge("I", "G", weight=6)
G.add_edge("H", "I", weight=7)

pos = nx.spring_layout(G, seed = 28)
edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}



plt.figure(figsize = (6,6))
nx.draw_networkx(G,
                 pos = pos,
                 node_color = '#0058FF',
                 node_size = 180)
nx.draw_networkx_edge_labels(G,
                             pos = pos,
                             edge_labels=edge_labels)
# plt.savefig('有向图.svg')


# 指定两节点最短距离
# 确定是否存在路径

nx.has_path(G,
            source = "A", # 始点
            target = "E", # 终点
           )



nx.has_path(G,
            source = "E", # 始点
            target = "A", # 终点
            )


# 找到E、A节点之间的最短距离，考虑权重
path_E_2_A = nx.shortest_path(G,
                              source = "E", # 始点
                              target = "A", # 终点
                              weight="weight")
print(path_E_2_A)


# 最短距离值
nx.shortest_path_length(G,
                        source = "E", # 始点
                        target = "A", # 终点
                        weight="weight")


# 最短路径的边，列表的每个元素为元组
# 元组有两个元素，代表边的两个节点
path_edges = list(zip(path_E_2_A, path_E_2_A[1:]))

# 最短路径所在边：红色
# 其他边：黑色
edge_colors = ["#FF5800"
               if edge in path_edges
                       or tuple(reversed(edge)) in path_edges
               else "black" for edge in G.edges()]


# 可视化图
plt.figure(figsize = (6,6))
# 路径上的节点
nx.draw_networkx_nodes(G, pos,
                       nodelist = path_E_2_A,
                       node_color = '#FF5800',
                       node_size = 180)

# 路径之外其他节点
not_path_E_2_A = set(G.nodes()) - set(path_E_2_A)
nx.draw_networkx_nodes(G, pos,
                       nodelist = not_path_E_2_A,
                       node_color = '#0058FF',
                       node_size = 180)

nx.draw_networkx_edges(G, pos,
                       edge_color=edge_colors)

nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edge_labels(
    G, pos, edge_labels=edge_labels)

# plt.savefig('E、A最短距离，有向图.svg')


# 起点为A的所有最短路径
path_from_A = nx.shortest_path(G,
                               source = "A", # 起点
                               weight="weight")
print(path_from_A)

# 最短距离值
nx.shortest_path_length(G,
                        source = "A", # 始点
                        weight="weight")





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 终点为E的所有最短路径

path_2_E = nx.shortest_path(G,
                            target = "E", # 终点
                            weight="weight")
# nx.single_source_shortest_path()
print(path_2_E)

# 最短距离值
nx.shortest_path_length(G,
                        target = "E", # 终点
                        weight="weight")
# nx.single_source_shortest_path_length()




# 图中任意两点所有最短路径
path_all = nx.shortest_path(G,
                            weight="weight")

path_all['E']['A']



# 最短距离值
distances_all = dict(nx.shortest_path_length(G,
                        weight="weight"))


list_nodes = list(G.nodes())
Shortest_D_matrix = np.full((len(G.nodes()),
                             len(G.nodes())), np.nan)


for i,i_node in enumerate(list_nodes):
    for j,j_node in enumerate(list_nodes):
        try:
            d_ij = distances_all[i_node][j_node]
            Shortest_D_matrix[i][j] = d_ij
        except KeyError:
            print(i_node + ' to ' + j_node + ': no path')



sns.heatmap(Shortest_D_matrix, cmap = 'Blues',
            annot = True, fmt = '.0f',
            xticklabels = list(G.nodes),
            yticklabels = list(G.nodes),
            linecolor = 'k', square = True,
            cbar = False,
            linewidths = 0.2)
# plt.savefig('Shortest_D_matrix，有向图.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 推销员问题

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.approximation as nx_app
import math

G = nx.random_geometric_graph(20, radius=0.4, seed=3)
pos = nx.get_node_attributes(G, "pos")

pos[0] = (0.5, 0.5)

H = G.copy()


# 计算节点距离作为权重
for i in range(len(pos)):
    for j in range(i + 1, len(pos)):
        # 斜边
        dist = math.hypot(pos[i][0] - pos[j][0],
                          pos[i][1] - pos[j][1])

        G.add_edge(i, j, weight=dist)


cycle = nx_app.christofides(G, weight="weight")
# Christofides 算法是一种解决度量空间中推销员问题的近似算法
edge_list = list(nx.utils.pairwise(cycle))
print("The route of the traveller is:", cycle)

edge_list

plt.figure(figsize = (8,8))
nx.draw_networkx_edges(H,
                       pos,
                       edge_color="blue",
                       width=0.5)

nx.draw_networkx(G, pos, with_labels=True,
                 edgelist=edge_list,
                 edge_color="red",
                 node_size=200,width=3)

plt.show()





















































































































































































































































































