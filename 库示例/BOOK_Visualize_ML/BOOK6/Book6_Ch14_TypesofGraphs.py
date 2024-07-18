

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html

# networkx.balanced_tree() 创建平衡树图
# networkx.barbell_graph() 创建哑铃型图
# networkx.binomial_tree() 创建二叉图
# networkx.bipartite.gnmk_random_graph() 创建随机二分图
# networkx.bipartite_layout() 二分图布局
# networkx.circular_layout() 圆周布局
# networkx.complete_bipartite_graph() 创建完全二分图
# networkx.complete_graph() 创建完全图
# networkx.complete_multipartite_graph() 创建完全多向图
# networkx.cycle_graph() 创建循环图
# networkx.ladder_graph() 创建梯子图
# networkx.lollipop_graph() 创建棒棒糖型图
# networkx.multipartite_layout() 多向图布局
# networkx.path_graph() 创建路径图
# networkx.star_graph() 创建星型图
# networkx.wheel_graph() 创建轮型图
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 完全图

import networkx as nx
import matplotlib.pyplot as plt

# for循环，绘制9幅完全图
for num_nodes in range(2,11):
    # 创建完全图对象
    G_i = nx.complete_graph(num_nodes)

    # 环形布局
    pos_i = nx.circular_layout(G_i)

    # 可视化，请大家试着绘制 3 * 3 子图布局
    plt.figure(figsize = (6,6))
    nx.draw_networkx(G_i,
                     pos = pos_i,
                     with_labels = False,
                     node_size = 28)
    # plt.savefig(str(num_nodes) + '_完全图.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 完全图

import matplotlib.pyplot as plt
import networkx as nx

# A rainbow color mapping using matplotlib's tableau colors
node_dist_to_color = {
    1: "tab:red",
    2: "tab:orange",
    3: "tab:olive",
    4: "tab:green",
    5: "tab:blue",
    6: "tab:purple",
}

# Create a complete graph with an odd number of nodes
nnodes = 13
G = nx.complete_graph(nnodes)

# A graph with (2n + 1) nodes requires n colors for the edges
n = (nnodes - 1) // 2
ndist_iter = list(range(1, n + 1))

# Take advantage of circular symmetry in determining node distances
ndist_iter += ndist_iter[::-1]


def cycle(nlist, n):
    return nlist[-n:] + nlist[:-n]


# Rotate nodes around the circle and assign colors for each edge based on
# node distance
nodes = list(G.nodes())
for i, nd in enumerate(ndist_iter):
    for u, v in zip(nodes, cycle(nodes, i + 1)):
        G[u][v]["color"] = node_dist_to_color[nd]

pos = nx.circular_layout(G)
# Create a figure with 1:1 aspect ratio to preserve the circle.
fig, ax = plt.subplots(figsize=(8, 8))
node_opts = {"node_size": 500, "node_color": "w", "edgecolors": "k", "linewidths": 2.0}
nx.draw_networkx_nodes(G, pos, **node_opts)
nx.draw_networkx_labels(G, pos, font_size=14)
# Extract color from edge data
edge_colors = [edgedata["color"] for _, _, edgedata in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, width=2.0, edge_color=edge_colors)

ax.set_axis_off()
fig.tight_layout()
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 补图

import networkx as nx
import matplotlib.pyplot as plt
import random


# 创建完全图
G = nx.complete_graph(9)
print(len(G.edges))

# 随机删除一半边
edges_removed = random.sample(list(G.edges), 18)
G.remove_edges_from(edges_removed)

# 环形布局
pos = nx.circular_layout(G)

plt.figure(figsize = (6,6))
nx.draw_networkx(G,
                 pos = pos,
                 with_labels = False,
                 node_size = 28)
# plt.savefig('图.svg')


# 补图
G_complement = nx.complement(G)
# 补图

# 可视化补图
plt.figure(figsize = (6,6))
nx.draw_networkx(G_complement,
                 pos = pos,
                 with_labels = False,
                 node_size = 28)
# plt.savefig('补图.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 二分图，手动

import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite

G = nx.Graph()

# 增加节点
G.add_nodes_from(['u1','u2','u3','u4'],
                 bipartite=0)

G.add_nodes_from(['v1','v2','v3'],
                 bipartite=1)

# 增加边
G.add_edges_from([('u1', "v3"),
                  ('u1', "v2"),
                  ('u4', "v1"),
                  ('u2', "v1"),
                  ('u2', "v3"),
                  ('u3', "v1")])

# 判断是否是二分图
bipartite.is_bipartite(G)

# 可视化
pos = nx.bipartite_layout(G, ['u1','u2','u3','u4'])
nx.draw_networkx(G, pos = pos, width = 2)
# plt.savefig('二分图，手动.svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 二分图，随机生成
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite

G = nx.bipartite.gnmk_random_graph(6, 8, 16, seed=88)

# 判断是否是二分图
bipartite.is_bipartite(G)

left = nx.bipartite.sets(G)[0]
pos = nx.bipartite_layout(G, left)

left

# 可视化
nx.draw_networkx(G, pos = pos, width = 2)
# plt.savefig('二分图，随机.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 完全多分图

import networkx as nx
import matplotlib.pyplot as plt

# 创建完全多分图
G = nx.complete_multipartite_graph(3, 6, 9)

# 多分图布局
pos = nx.multipartite_layout(G)

# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos = pos, width = 0.25, with_labels = False)
# plt.savefig('完全多分图.svg')


#%%%%%%%%%%%%%% 且对不同分层节点分别着色
import itertools
import matplotlib.pyplot as plt
import networkx as nx

subset_sizes = [3, 5, 5, 3, 2, 4, 4, 3]
subset_color = [
    "gold",
    "violet",
    "violet",
    "violet",
    "violet",
    "limegreen",
    "limegreen",
    "darkorange",
]


def multilayered_graph(*subset_sizes):
    extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
    layers = [range(start, end) for start, end in extents]
    G = nx.Graph()
    for i, layer in enumerate(layers):
        G.add_nodes_from(layer, layer=i)
    for layer1, layer2 in nx.utils.pairwise(layers):
        G.add_edges_from(itertools.product(layer1, layer2))
    return G

G = multilayered_graph(*subset_sizes)
color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
pos = nx.multipartite_layout(G, subset_key="layer")
plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_color=color, with_labels=False)
plt.axis("equal")
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 随机正则图
import networkx as nx
import matplotlib.pyplot as plt

list_specs = [(1,6),(2,6),(3,6),
              (1,8),(2,8),(3,8),
              (1,12),(2,12),(3,12)]
# d, the degree of each node.
# n, the number of nodes.

fig, axes = plt.subplots(3, 3, figsize = (8,8))
axes = axes.flatten()
for (d,n), ax_i in zip(list_specs,axes):
    # 生成正则图
    G = nx.random_regular_graph(d,n)
    # d, the degree of each node.
    # n, the number of nodes.
    nx.draw_networkx(G,
                     ax = ax_i,
                     pos = nx.circular_layout(G),
                     with_labels = False,
                     node_size = 100)
# plt.savefig('正则图.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 平衡树

import matplotlib.pyplot as plt
import networkx as nx

G = nx.balanced_tree(3, 5)
# balanced_tree(r, h, create_using=None)
# r, int
# Branching factor of the tree; each node will have r children.
# h, int
# Height of the tree.

plt.figure(figsize=(8, 8))
nx.draw_networkx(G, node_size=20,
                 alpha=0.5,
                 node_color="blue",
                 with_labels=False)
plt.axis("equal")
# plt.savefig('平衡树.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 二叉树

import matplotlib.pyplot as plt
import networkx as nx

G = nx.binomial_tree(8)
# n, int
# Order of the binomial tree.

plt.figure(figsize=(8, 8))
nx.draw_networkx(G, node_size=20,
                 alpha=0.5,
                 node_color="blue",
                 with_labels=False)
plt.axis("equal")
# plt.savefig('二叉树.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 星图
import networkx as nx
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
fig, axes = plt.subplots(2, 4, figsize = (12,6))
for n,ax_n in zip([6,8,9,12,16,18,19,38],axes.flat):
    # star graph
    S_n = nx.star_graph(n)
    nx.draw_networkx(S_n,
                     pos = nx.spring_layout(S_n),
                     node_size=20,
                     node_color="blue",
                     with_labels=False,
                     ax = ax_n)
    ax_n.axis("off")
# plt.savefig('星图.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 链图

import networkx as nx
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
fig, axes = plt.subplots(2, 4, figsize = (12,6))
for n,ax_n in zip([6,8,9,12,16,18,19,38],axes.flat):
    # star graph
    S_n = nx.path_graph(n)
    nx.draw_networkx(S_n,
                     pos = nx.spring_layout(S_n),
                     node_size=20,
                     node_color="blue",
                     with_labels=False,
                     ax = ax_n)
    ax_n.axis("off")
# plt.savefig('链图.svg')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 柏拉图图
import networkx as nx
import matplotlib.pyplot as plt


def visualize_G(G,fig_name):
    plt.figure(figsize = (6,6))
    nx.draw_networkx(G,
                     pos = nx.spring_layout(G),
                     with_labels = False,
                     node_size = 28)
    # plt.savefig(fig_name + '.svg')


# 正四面体图
tetrahedral_graph = nx.tetrahedral_graph()
print(len(tetrahedral_graph.edges))
print(len(tetrahedral_graph.nodes))
visualize_G(tetrahedral_graph, 'tetrahedral_graph')

# 正六面体图
cubical_graph = nx.cubical_graph()
print(len(cubical_graph.edges))
print(len(cubical_graph.nodes))
visualize_G(cubical_graph, 'cubical_graph')


# 正八面体图
octahedral_graph = nx.octahedral_graph()
print(len(octahedral_graph.edges))
print(len(octahedral_graph.nodes))
visualize_G(octahedral_graph, 'octahedral_graph')

# 正十二面体图
dodecahedral_graph = nx.dodecahedral_graph()
print(len(dodecahedral_graph.edges))
print(len(dodecahedral_graph.nodes))
visualize_G(dodecahedral_graph, 'dodecahedral_graph')

# 正二十面体图
icosahedral_graph = nx.icosahedral_graph()
print(len(icosahedral_graph.edges))
print(len(icosahedral_graph.nodes))
visualize_G(icosahedral_graph, 'icosahedral_graph')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 柏拉图图，平面化

import networkx as nx
import matplotlib.pyplot as plt

def visualize_G(G,fig_name):
    plt.figure(figsize = (6,6))
    nx.draw_networkx(G,
                     pos = nx.planar_layout(G), # 平面化 (planar)
                     with_labels = False,
                     node_size = 28)
    # plt.savefig(fig_name + '.svg')

# 正四面体图
tetrahedral_graph = nx.tetrahedral_graph()
nx.is_planar(tetrahedral_graph)

visualize_G(tetrahedral_graph, 'tetrahedral_graph')


# 正六面体图
cubical_graph = nx.cubical_graph()
nx.is_planar(cubical_graph)


visualize_G(cubical_graph, 'cubical_graph')

# 正八面体图
octahedral_graph = nx.octahedral_graph()
nx.is_planar(octahedral_graph)

visualize_G(octahedral_graph, 'octahedral_graph')


# 正十二面体图
dodecahedral_graph = nx.dodecahedral_graph()
nx.is_planar(dodecahedral_graph)
visualize_G(dodecahedral_graph, 'dodecahedral_graph')

# 正二十面体图
icosahedral_graph = nx.icosahedral_graph()
nx.is_planar(icosahedral_graph)

visualize_G(icosahedral_graph, 'icosahedral_graph')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





























































































































































































































