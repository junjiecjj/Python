

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 使用networkx.draw_networkx()，节点位置

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np



# 图
# 创建无向图
G = nx.random_geometric_graph(200, 0.2, seed=888)

# 位置
# 使用弹簧模型算法布局节点
pos = nx.spring_layout(G, seed = 888)
pos


# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos = pos, node_color = '#3388FF', alpha = 0.5, with_labels = False, node_size = 68)
# plt.savefig('节点布局，弹簧算法布局.svg')


# 自定义节点位置
data_loc = np.random.rand(200,2)
# 随机数发生器生成节点平面坐标

# 创建节点位置坐标字典
pos_loc = {i: (data_loc[i, 0], data_loc[i, 1]) for i in range(len(data_loc))}


# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos = pos_loc, node_color = '#3388FF', alpha = 0.5, with_labels = False, node_size = 68)
# plt.savefig('节点布局，随机数.svg')




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 使用networkx.draw_networkx()，节点位置


import matplotlib.pyplot as plt
import networkx as nx


m = 3
n = 9
G = nx.complete_bipartite_graph(m,n)
# 创建完全二分图


pos = {}
# m = 3 位于上层；n = 9 位于下层
pos.update((i, (i - m/2, 1)) for i in range(m))
pos.update((i, (i - m - n/2, 0)) for i in range(m, m + n))


pos

fig, ax = plt.subplots(figsize = (6,6))
nx.draw_networkx(G, with_labels=True, pos=pos, node_size=300, width=0.4)
# plt.savefig('m = 3 位于上层；n = 9 位于下层.svg')

# m = 3 位于下层；n = 9 位于上层
pos = {}
pos.update((i, (i - m/2, 0)) for i in range(m))
pos.update((i, (i - m - n/2, 1)) for i in range(m, m + n))


fig, ax = plt.subplots(figsize = (6,6))
nx.draw_networkx(G, with_labels=True, pos=pos, node_size=300, width=0.4)
# plt.savefig('m = 3 位于下层；n = 9 位于上层.svg')

# m = 3 位于右侧；n = 9 位于左侧
pos = {}
pos.update((i, (1, i - m/2)) for i in range(m))
pos.update((i, (0, i - m - n/2)) for i in range(m, m + n))

pos

fig, ax = plt.subplots(figsize = (6,6))
nx.draw_networkx(G, with_labels=True, pos=pos, node_size=300, width=0.4)
# plt.savefig('m = 3 位于右侧；n = 9 位于左侧.svg')

# m = 3 位于左侧；n = 9 位于右侧
pos = {}
pos.update((i, (0, i - m/2)) for i in range(m))
pos.update((i, (1, i - m - n/2)) for i in range(m, m + n))

fig, ax = plt.subplots(figsize = (6,6))
nx.draw_networkx(G, with_labels=True, pos=pos, node_size=300, width=0.4)
# plt.savefig('m = 3 位于左侧；n = 9 位于右侧.svg')




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 不同边权重，不同线型

import networkx as nx
import matplotlib.pyplot as plt


G = nx.path_graph(28)

pos = nx.circular_layout(G)
# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(G, with_labels = False, pos=pos, node_size = 38)
# plt.savefig('圆周布局.svg')


pos = nx.spiral_layout(G)
plt.figure(figsize = (6,6))
nx.draw_networkx(G, with_labels = False, pos=pos, node_size = 38)
# plt.savefig('螺旋布局.svg')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 调整节点位置

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# 创建图
G = nx.path_graph(20)
# 需要调整位置的节点序号
center_node = 5
# 剩余节点子集
edge_nodes = set(G) - {center_node}
# {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
print(edge_nodes)

# 圆周布局 (除了节点5以外)
pos = nx.circular_layout(G.subgraph(edge_nodes))

# 在字典中增加一个键值对，节点5的坐标
pos[center_node] = np.array([0, 0])

# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos, with_labels=True)
# plt.savefig('调整节点位置.svg')



# 圆周布局，所有节点
pos_all = nx.circular_layout(G)

pos_all[5] = np.array([0, 0])

# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos_all, with_labels=True)
# plt.savefig('调整节点位置，第二种.svg')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 使用networkx.draw_networkx()，节点

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# 图
# 创建无向图
G = nx.random_geometric_graph(200, 0.2, seed=888)
# 使用弹簧模型算法布局节点
pos = nx.spring_layout(G, seed = 888)

# 节点
plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos = pos, alpha = 0.5, with_labels = False, node_size = 68)
# plt.savefig('节点大小.svg')

# 节点颜色 + 形状
plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos = pos, node_shape = 's', with_labels = False, node_size = 68, node_color = '#66FF33')
# plt.savefig('节点颜色 + 形状.svg')

# 节点颜色映射
plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos = pos, node_shape = 's', cmap = 'RdYlBu_r', node_color = np.random.rand(200), with_labels = False, node_size = 68)
# plt.savefig('节点颜色映射.svg')

# 节点颜色映射 + 大小
plt.figure(figsize = (6,6))
nx.draw_networkx(G,
                 pos = pos,
                 node_shape = 's',
                 cmap = 'hsv',
                 vmax = 1,
                 vmin = 0,
                 node_color = np.random.rand(200),
                 node_size  = np.random.rand(200)*88,
                 with_labels = False)
# plt.savefig('节点颜色映射 + 大小.svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 绘制十二面体图，图着色问题

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 创建十二面体图
G = nx.dodecahedral_graph()

# 贪心着色算法
graph_color_code = nx.greedy_color(G)
print(graph_color_code)
# {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 2, 7: 0, 8: 2, 9: 0, 10: 1, 11: 0, 12: 1, 13: 2, 14: 1, 15: 0, 16: 2, 17: 1, 18: 2, 19: 3}

# 独特颜色，{0, 1, 2, 3}
unique_colors = set(graph_color_code.values())
print(unique_colors)

# 颜色映射
color_mapping = {0: '#0099FF',
                 1: '#FF6600',
                 2: '#99FF33',
                 3: '#FF99FF'}

# 完成每个节点的颜色映射
node_colors = [color_mapping[graph_color_code[n]] for n in G.nodes()]

# 节点位置布置
pos = nx.spring_layout(G, seed=14)

# 可视化
fig, ax = plt.subplots(figsize = (6,6))
nx.draw_networkx(
    G,
    pos,
    with_labels=True,
    node_size=500,
    node_color = node_colors,
    edge_color="grey",
    font_size=12,
    font_color="#333333",
    width=2,
)
# plt.savefig('十二面体图，着色问题.svg')




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 使用networkx.draw_networkx()，边
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# 图
G_star = nx.star_graph(88)
pos_star = nx.spring_layout(G_star, seed=88)

# 边
plt.figure(figsize = (6,6))
nx.draw_networkx(G_star,
        pos = pos_star,
        with_labels = False,
        node_size = 38,
        node_color = '0.5',
        edge_color = '#0070C0',
        width = 0.2)
# plt.savefig('边粗细 + 颜色.svg')


colors = range(88)
options = {
    "node_color": "0.5",
    "node_size": 38,
    "edge_color": colors,
    "width": 0.25,
    "edge_cmap": plt.cm.hsv,
    "with_labels": False,
}
plt.figure(figsize = (6,6))
nx.draw_networkx(G_star, pos_star, **options)
# plt.savefig('边颜色映射.svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 鸢尾花样本数据到图

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import euclidean_distances


# 加载鸢尾花数据集
iris = load_iris()
data = iris.data[:, :2]

# 计算欧氏距离矩阵
D = euclidean_distances(data)
# 用成对距离矩阵可以构造无向图

# 绘制成对欧氏距离热图
sns.heatmap(D, square = True, cbar = False,
            xticklabels=[], yticklabels=[],
            cmap = 'RdYlBu_r')
# plt.savefig('成对欧氏距离热图.svg')


# 创建无向图
G = nx.Graph(D, nodetype=int)

# 提取边的权重，即欧氏距离值
edge_weights = [G[i][j]['weight'] for i, j in G.edges]

# 使用鸢尾花数据的真实位置绘制图形
pos = {i: (data[i, 0], data[i, 1]) for i in range(len(data))}

# 绘制无向图，所有边
fig, ax = plt.subplots(figsize = (6,6))
nx.draw_networkx(G,
                 pos,
                 node_color = '0.28',
                 edge_color = edge_weights,
                 edge_cmap = plt.cm.RdYlBu_r,
                 linewidths = 0.2,
                 with_labels=False,
                 node_size = 18)

ax.set_xlim(4,8)
ax.set_ylim(1,5)
ax.grid()
ax.set_aspect('equal', adjustable='box')
# plt.savefig('鸢尾花_欧氏距离矩阵_无向图, 真实位置.svg')

#>>>>>>>>>>  绘制部分边 <<<<<<<<<<<<<<
# 欧氏距离的直方图
fig, ax = plt.subplots(figsize = (6,4))
plt.hist(edge_weights, bins = 20, ec = 'k', density = True)
plt.xlabel('Euclidean distance (cm)')
plt.ylabel('Density')
# plt.savefig('欧氏距离直方图.svg')

len(edge_weights)
# 150 * 149/2

# 选择需要保留的边
edge_kept = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

# 绘制无向图，剔除欧氏距离大于0.5的边
fig, ax = plt.subplots(figsize = (6,6))
nx.draw_networkx(G, pos,
                 edgelist = edge_kept, # 通过设定 edgelist 保留特定边。
                 node_color = '0.28',
                 edge_color = '#3388FF',
                 linewidths = 0.2,
                 with_labels = False,
                 node_size = 18)
ax.set_xlim(4,8)
ax.set_ylim(1,5)
ax.grid()
ax.set_aspect('equal', adjustable='box')
# plt.savefig('鸢尾花_欧氏距离矩阵_无向图，保留部分边.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 不同边权重，不同线

import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()
# 添加边
G.add_edge("a", "b", weight=0.6)
G.add_edge("a", "c", weight=0.2)
G.add_edge("c", "d", weight=0.1)
G.add_edge("c", "e", weight=0.7)
G.add_edge("c", "f", weight=0.9)
G.add_edge("a", "d", weight=0.3)

# 将边分成两组
# 第一组：边权重 > 0.5
elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]

# 第二组：边权重 <= 0.5
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

# 节点布局
pos = nx.spring_layout(G, seed=7)

# 可视化
plt.figure(figsize = (6,6))
# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=700)

# 节点标签
nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

# 绘制第一组边
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1)

# 绘制第二组边
nx.draw_networkx_edges(G, pos, edgelist=esmall, width=1, alpha=0.5, edge_color="b", style="dashed")

# 边标签
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)

# plt.savefig('不同边权重，不同线型.svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 鸢尾花样本数据到图，展示分类标签

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import euclidean_distances



# 加载鸢尾花数据集
iris = load_iris()
data = iris.data[:, :2]
label = iris.target


# 计算欧氏距离矩阵
D = euclidean_distances(data)
# 用成对距离矩阵可以构造无向图


# 创建无向图
G = nx.Graph(D, nodetype=int)

# 提取边的权重，即欧氏距离值
edge_weights = [G[i][j]['weight'] for i, j in G.edges]

# 使用鸢尾花数据的真实位置绘制图形
pos = {i: (data[i, 0], data[i, 1]) for i in range(len(data))}


# 选择需要保留的边
edge_kept = [(u, v)
             for (u, v, d)
             in G.edges(data=True)
             if d["weight"] <= 0.5]



# 节点颜色映射
color_mapping = {0: '#0099FF',
                 1: '#FF6600',
                 2: '#99FF33'}

# 完成每个节点的颜色映射
node_color = [color_mapping[label[n]] for n in G.nodes()]


# 绘制无向图，分别绘制边和节点

fig, ax = plt.subplots(figsize = (6,6))

nx.draw_networkx_edges(G, pos, edgelist=edge_kept, width = 0.2, edge_color='0.58')

nx.draw_networkx_nodes(G, pos, node_size = 18, node_color=node_color)

ax.set_xlim(4,8)
ax.set_ylim(1,5)
ax.grid()
ax.set_aspect('equal', adjustable='box')
# plt.savefig('鸢尾花_欧氏距离矩阵_无向图，展示分类标签.svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 有向图
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx



# 创建图
seed = 13648
G = nx.random_k_out_graph(10, 3, 0.5, seed=seed)
pos = nx.spring_layout(G, seed=seed)

# 节点大小
node_sizes = [3 + 18 * i for i in range(len(G))]



# 边的颜色和透明度
M = G.number_of_edges()
edge_colors = range(2, M + 2)
edge_alphas = [(5 + i) / (M + 4) for i in range(M)]


# 颜色映射
cmap = plt.cm.plasma




# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx_nodes(G, pos,
                       node_size=node_sizes,
                       node_color="indigo")
edges = nx.draw_networkx_edges(
    G,
    pos,
    node_size=node_sizes,
    arrowstyle="->",
    arrowsize=10,
    edge_color=edge_colors,
    edge_cmap=cmap,
    width=2)

# 每条边设置不同透明度
for i in range(M):
    edges[i].set_alpha(edge_alphas[i])

# 增加颜色条
pc = mpl.collections.PatchCollection(edges, cmap=cmap)
pc.set_array(edge_colors)
ax = plt.gca()
ax.set_axis_off()
plt.colorbar(pc, ax=ax)
# plt.savefig('有向图.svg')




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 节点标签

import matplotlib.pyplot as plt
import networkx as nx


G = nx.cubical_graph()
pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes

# nodes
options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}


nx.draw_networkx_nodes(G, pos, nodelist=[0, 1, 2, 3], node_color="tab:red", **options)
nx.draw_networkx_nodes(G, pos, nodelist=[4, 5, 6, 7], node_color="tab:blue", **options)

# edges
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[(0, 1), (1, 2), (2, 3), (3, 0)],
    width=8,
    alpha=0.5,
    edge_color="tab:red",
)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[(4, 5), (5, 6), (6, 7), (7, 4)],
    width=8,
    alpha=0.5,
    edge_color="tab:blue",
)


# some math labels
labels = {}
labels[0] = r"$a$"
labels[1] = r"$b$"
labels[2] = r"$c$"
labels[3] = r"$d$"
labels[4] = r"$\alpha$"
labels[5] = r"$\beta$"
labels[6] = r"$\gamma$"
labels[7] = r"$\delta$"
nx.draw_networkx_labels(G, pos, labels, font_size=22, font_color="whitesmoke")

plt.tight_layout()
plt.axis("off")
# plt.savefig('节点标签.svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 根据度数大小用颜色映射渲染节点

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# 产生随机图
G = nx.random_geometric_graph(400, 0.125)

# 提取节点平面坐标
pos = nx.get_node_attributes(G, "pos")

# 度数大小排序
degree_sequence = sorted((d for n, d in G.degree()),  reverse=True)

# 将结果转为字典
dict_degree = dict(G.degree())

# 自定义函数，过滤dict
def filter_value(dict_, unique):
    newDict = {}
    for (key, value) in dict_.items():
        if value == unique:
            newDict[key] = value
    return newDict


unique_deg = set(degree_sequence)
# 取出节点度数独特值

colors = plt.cm.viridis(np.linspace(0, 1, len(unique_deg)))
# 独特值的颜色映射


# 可视化
plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G, pos, edge_color = '0.8')

# 根据度数大小渲染节点
for deg_i, color_i in zip(unique_deg,colors):

    dict_i = filter_value(dict_degree,deg_i)
    nx.draw_networkx_nodes(G, pos,
                           nodelist = list(dict_i.keys()),
                           node_size = deg_i*8,
                           node_color = color_i)
plt.axis("off")
# plt.savefig('根据度数大小用颜色映射渲染节点.svg')
plt.show()
























































































































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






























































































































































































































































































