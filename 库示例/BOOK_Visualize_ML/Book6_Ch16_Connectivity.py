

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html









#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 连通性，空手道俱乐部


import networkx as nx
import matplotlib.pyplot as plt



G = nx.karate_club_graph()
# 空手道俱乐部图
pos = nx.spring_layout(G,seed=2)



plt.figure(figsize = (6,6))

nx.draw_networkx(G, pos)
nx.draw_networkx_nodes(G,pos,
                       nodelist = [11,18],
                       node_color = 'r')
# plt.savefig('空手道俱乐部图.svg')

nx.has_path(G, 11, 18)
# 检查两个节点是否连通

path_nodes = nx.shortest_path(G, 11, 18)
# 最短路径
path_nodes

# 自定义函数将节点序列转化为边序列
def nodes_2_edges(node_list):

    # 使用列表生成式创建边的列表
    list_edges = [(node_list[i], node_list[i+1])
                  for i in range(len(node_list)-1)]

    return  list_edges

path_edges = nodes_2_edges(path_nodes)
# 将节点序列转化为边序列

plt.figure(figsize = (6,6))

nx.draw_networkx(G, pos)
nx.draw_networkx_nodes(G,pos,
                       nodelist = path_nodes,
                       node_color = 'r')
nx.draw_networkx_edges(G,pos,
                       edgelist = path_edges,
                       edge_color = 'r')
# plt.savefig('空手道俱乐部图，节点11、18最短路径.svg')

# 删除一条边
G.remove_edge(11,0)

nx.has_path(G, 11, 8)
# 再次检查两个节点是否连通


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 连通图 vs 非连通图

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt



p = [0.03, 0.05]

fig, axes = plt.subplots(1,2,figsize = (8,4))

for ax_i,p_i in zip(axes.flatten(), p):

    G = nx.gnp_random_graph(100, p_i, seed=8)
    # 参数n控制图中节点的数量
    # 参数p控制节点间连线存在的概率

    # 检查无向图是否连通
    print(nx.is_connected(G))

    pos = nx.spring_layout(G, seed = 8)
    nx.draw_networkx(G, pos,
                     ax = ax_i,
                     with_labels = False,
                     node_size=20)
    # ax_i.axis('off')

# plt.savefig('图的连通.svg')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 连通分量

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


G = nx.gnp_random_graph(100, 0.01, seed=8)
# 创建随机图

plt.figure(figsize = (6,6))
pos = nx.spring_layout(G, seed = 8)
nx.draw_networkx(G, pos,
                 with_labels = False,
                 node_size=20)
# plt.savefig('全图.svg')


# 检查无向图是否连通
nx.is_connected(G)


# 连通分量的数量
nx.number_connected_components(G)



# 连通分量
list_cc = nx.connected_components(G)

# 根据节点数从大到小排列连通分量
list_cc = sorted(list_cc, key=len, reverse=True)
for cc_idx in list_cc:
    print(cc_idx)

# 取出节点数较多的连通分量
fig, axes = plt.subplots(2,3,figsize = (9,6))
axes = axes.flatten()

for idx in range(6):
    Gcc_idx = G.subgraph(list_cc[idx])

    pos_Gcc_idx = {k: pos[k] for k in list(Gcc_idx.nodes())}

    # 可视化连通分量

    nx.draw_networkx(Gcc_idx,
                     pos_Gcc_idx,
                     ax = axes[idx],
                     with_labels = False,
                     node_size=20)

# plt.savefig('前6大连通分量.svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 强连通、弱连通

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


nx.is_strongly_connected(G_directed)


nx.is_weakly_connected(G_directed)


list_strongly_cc = sorted(nx.strongly_connected_components(G_directed), key=len, reverse=True)


list(list_strongly_cc)


# 自定义函数将节点序列转化为边序列 (闭环)

def nodes_2_edges(node_list):
    # 使用列表生成式创建边的列表
    list_edges = [(node_list[i], node_list[i+1]) for i in range(len(node_list)-1)]

    # 加上一个额外的边从最后一个节点回到第一个节点，形成闭环
    closing_edge = [(node_list[-1], node_list[0])]
    list_edges = list_edges + closing_edge
    return  list_edges


list_cc = list(list_strongly_cc[0])

# 可视化图
plt.figure(figsize = (6,6))
edges_strongly_cc = nodes_2_edges(list_cc)

pos_Gcc_idx = {k: pos[k] for k in list_cc}

nx.draw_networkx(G_directed,
                 pos = pos,
                 with_labels = True,
                 node_size = 188)

nx.draw_networkx_nodes(G_directed,
                       nodelist = list_cc,
                       pos = pos_Gcc_idx,
                       node_size = 188,
                       node_color = 'r')

nx.draw_networkx_edges(G_directed,
                       pos = pos_Gcc_idx,
                       edgelist = edges_strongly_cc,
                       edge_color = 'r')

# plt.savefig('强连通分量.svg')

# 可视化有向图中3个环

fig, axes = plt.subplots(1, 3, figsize = (9,3))

axes = axes.flatten()

for nodes_i, ax_i in zip(list_cc, axes):
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

largest = max(list_strongly_cc, key=len)
largest

G_directed_2 = nx.complete_graph(5, create_using = nx.DiGraph())
# 可视化图
plt.figure(figsize = (6,6))
nx.draw_networkx(G_directed_2,
                 pos = pos,
                 with_labels = True,
                 node_size = 188)
# plt.savefig('完全图，设定边方向.svg')
nx.is_strongly_connected(G_directed_2)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 桥、局部桥

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# G = nx.balanced_tree(3, 5)
G = nx.karate_club_graph()
# 创建随机图
pos = nx.spring_layout(G, seed = 2)


plt.figure(figsize = (6,6))

nx.draw_networkx(G, pos,
                 node_size = 18,
                 with_labels = False)
# plt.savefig('全图.svg')

nx.has_bridges(G)

#
bridges = list(nx.bridges(G))
len(bridges)


plt.figure(figsize = (6,6))

nx.draw_networkx(G, pos,
                 with_labels = False)

nx.draw_networkx_nodes(G,
                       pos=pos,
                       nodelist = set(sum(bridges, ())),
                       node_color = 'r')

nx.draw_networkx_edges(
    G, pos, edgelist=bridges, width=1, edge_color="r"
)  # red color for bridges

# plt.axis("off")
# plt.savefig('桥.svg')


#  局部桥
local_bridges = list(nx.local_bridges(G, with_span=False))
len(local_bridges)




plt.figure(figsize = (6,6))

nx.draw_networkx(G, pos,
                 with_labels = False)

nx.draw_networkx_nodes(G,
                       pos=pos,
                       nodelist = set(sum(local_bridges, ())),
                       node_color = 'pink')

nx.draw_networkx_edges(G, pos,
                       edgelist=local_bridges,
                       width=1, edge_color="pink")

# plt.axis("off")
# plt.savefig('局部桥.svg')


plt.figure(figsize = (6,6))

nx.draw_networkx(G, pos,
                 with_labels = False)

nx.draw_networkx_edges(G, pos,
                       edgelist=set(local_bridges) - set(bridges),
                       width=1, edge_color="orange")

nx.draw_networkx_nodes(G,
                       pos=pos,
                       nodelist = set(sum(list(set(local_bridges) - set(bridges)), ())),
                       node_color = 'orange')

# plt.axis("off")
# plt.savefig('局部桥中不是桥的边.svg')




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 最大k边连通分量

import matplotlib.pyplot as plt
import networkx as nx

G = nx.barbell_graph(8, 0)
pos = nx.spring_layout(G, seed = 8)


# 可视化
plt.figure(figsize = (6,3))
nx.draw_networkx(G,
                 pos = pos,
                 node_size = 88,
                 with_labels=False)
# plt.savefig('哑铃.svg')


nx.is_k_edge_connected(G, k=1)


nx.is_k_edge_connected(G, k=2)


list(nx.k_edge_components(G, k=2))



list_k_edge_cc = list(nx.k_edge_components(G, k=2))



fig, axes = plt.subplots(2,1,figsize = (6,6))

for idx in range(2):

    nx.draw_networkx(G,
                     pos = pos,
                     ax = axes[idx],
                     node_size = 88,
                     with_labels=False)
    nx.draw_networkx_nodes(G,
                           pos = pos,
                           ax = axes[idx],
                           node_color = 'r',
                           node_size = 88,
                           nodelist = list_k_edge_cc[idx])

# plt.savefig('最大k边连通组件.svg')




































































































































































































































