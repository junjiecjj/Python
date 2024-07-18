

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 度分析，空手道俱乐部

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

G = nx.karate_club_graph()
# 空手道俱乐部图
pos = nx.spring_layout(G,seed=2)

plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos)
# plt.savefig('全图.svg')

# 度分析
# 度数大小排序
degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
dmax = max(degree_sequence)

# 将结果转为字典
dict_degree = dict(G.degree())

# pd.DataFrame(list(dict_degree.items()), columns=['Key', 'Values'])
########## 每个节点的具体度数
plt.figure(figsize = (6,6))
plt.bar(dict_degree.keys(), dict_degree.values())
plt.xlabel('Node label')
plt.ylabel('Degree')
# plt.savefig('节点度数bar chart.svg')

########## 节点度数排序
# set(degree_sequence)
fig, ax = plt.subplots(figsize = (6,3))
ax.plot(degree_sequence, "b-", marker="o")
ax.set_ylabel("Degree")
ax.set_xlabel("Rank")
ax.set_xlim(0, 34)
ax.set_ylim(0, 17)
# plt.savefig('度数等级图.svg')

########## 节点度数柱状图
fig, ax = plt.subplots(figsize = (6,3))
ax.bar(*np.unique(degree_sequence, return_counts=True))
ax.set_xlabel("Degree")
ax.set_ylabel("Number of Nodes")
# plt.savefig('度数直方图.svg')

# 根据度数渲染节点
# 自定义函数，过滤dict
def filter_value(dict_, unique):
    newDict = {}
    for (key, value) in dict_.items():
        if value == unique:
            newDict[key] = value
    return newDict

# 根据度数大小渲染节点
# 取出节点度数独特值
unique_deg = set(degree_sequence)

# 独特值的颜色映射
colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(unique_deg)))

plt.figure(figsize = (6,6))
# 绘制图的边
nx.draw_networkx_edges(G, pos)
# 分别绘制不同度数节点
for deg_i, color_i in zip(unique_deg,colors):
    dict_i = filter_value(dict_degree,deg_i)
    nx.draw_networkx_nodes(G, pos, nodelist = list(dict_i.keys()), node_color = color_i)
# plt.savefig('根据度数大小渲染节点.svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 度分析

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 创建随机图
G = nx.gnp_random_graph(100, 0.02, seed=8)
plt.figure(figsize = (6,6))
pos = nx.spring_layout(G, seed = 8)
nx.draw_networkx(G, pos, with_labels = False, node_size=20)
# plt.savefig('全图.svg')

# 连通分量
Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
num_nodes = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
num_nodes
# [84, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# 取出子图节点坐标
pos_Gcc = {k: pos[k] for k in list(Gcc.nodes())}


plt.figure(figsize = (6,6))
nx.draw_networkx(Gcc, pos_Gcc, with_labels = False, node_size=20)
# plt.savefig('最大连通分量.svg')

# 度分析
# 度数大小排序
degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
dmax = max(degree_sequence)
# 将结果转为字典
dict_degree = dict(G.degree())

# 度分析，节点度数排序
fig, ax = plt.subplots(figsize = (6,3))
ax.plot(degree_sequence, "b-", marker="o")
ax.set_ylabel("Degree")
ax.set_xlabel("Rank")
ax.set_xlim(0,100)
ax.set_ylim(0,8)
# plt.savefig('度数等级图.svg')

# 节点度数柱状图
fig, ax = plt.subplots(figsize = (6,3))
ax.bar(*np.unique(degree_sequence, return_counts=True))
ax.set_xlabel("Degree")
ax.set_ylabel("Number of Nodes")
# plt.savefig('度数柱状图.svg')

##### 根据度数渲染节点
# 自定义函数，过滤dict
def filter_value(dict_, unique):
    newDict = {}
    for (key, value) in dict_.items():
        if value == unique:
            newDict[key] = value

    return newDict

# 根据度数大小渲染节点
# 取出节点度数独特值
unique_deg = set(degree_sequence)
colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(unique_deg)))
# 独特值的颜色映射

plt.figure(figsize = (6,6))
nx.draw_networkx_edges(G, pos)
# 绘制图的边

# 分别绘制不同度数节点
for deg_i, color_i in zip(unique_deg,colors):
    dict_i = filter_value(dict_degree,deg_i)
    nx.draw_networkx_nodes(G, pos, nodelist = list(dict_i.keys()), node_size=20, node_color = color_i)
# plt.savefig('根据度数大小渲染节点.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 距离度量

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 空手道俱乐部图
G = nx.karate_club_graph()
pos = nx.spring_layout(G,seed=2)

plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos)
# nx.draw_networkx_nodes(G,pos, nodelist = [11,18], node_color = 'r')
# plt.savefig('空手道俱乐部图.svg')

# 图距离
path_nodes = nx.shortest_path(G, 15, 16)
# 节点15、16之间最短路径
path_nodes
# [15, 32, 2, 0, 5, 16]

# 路径节点序列转化为边序列 (不封闭)
path_edges = list(nx.utils.pairwise(path_nodes))

plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos)
nx.draw_networkx_nodes(G, pos, nodelist = path_nodes, node_color = 'r')
nx.draw_networkx_edges(G, pos, edgelist = path_edges, edge_color = 'r')
# plt.savefig('空手道俱乐部图，15、16最短路径.svg')

# 图距离矩阵
# 成对最短距离值 (图距离)
distances_all = dict(nx.shortest_path_length(G))

# 创建图距离矩阵
list_nodes = list(G.nodes())
Shortest_D_matrix = np.full((len(G.nodes()), len(G.nodes())), np.nan) # (34, 34)
for i,i_node in enumerate(list_nodes):
    for j,j_node in enumerate(list_nodes):
        try:
            d_ij = distances_all[i_node][j_node]
            Shortest_D_matrix[i][j] = d_ij
        except KeyError:
            print(i_node + ' to ' + j_node + ': no path')

# 图距离最大值
Shortest_D_matrix.max()
Shortest_D_matrix.min()

#### 图 10. 空手道俱乐部会员关系图，图距离矩阵
# 用热图可视化图距离矩阵
sns.heatmap(Shortest_D_matrix, cmap = 'Blues',
            annot = False,
            xticklabels = list(G.nodes), yticklabels = list(G.nodes),
            linecolor = 'k', square = True, cbar = True, linewidths = 0.2)
# plt.savefig('图距离矩阵，无权图.svg')

# 取出图距离矩阵中所有成对图距离 (不含对角线下三角元素)
# 使用numpy.tril获取下三角矩阵，并排除对角线元素
lower_tri_wo_diag = np.tril(Shortest_D_matrix, k=-1) # (34, 34)

# 获取下三角矩阵（不含对角线）的索引
rows, cols = np.tril_indices(Shortest_D_matrix.shape[0], k=-1)

# 使用索引从原矩阵中取出对应的元素
list_shortest_distances = Shortest_D_matrix[rows, cols] # (561,)

# 使用numpy.unique函数获取独特值及其出现次数
unique_values, counts = np.unique(list_shortest_distances, return_counts=True)

#### 图 9. 空手道俱乐部会员关系图，成对图距离柱状图
plt.bar(unique_values, counts)
plt.xlabel('Graph distance')
plt.ylabel('Count')
# plt.savefig('图距离柱状图.svg')

#### 图 11. 空手道俱乐部会员关系图，各个节点平均图距离
# 节点平均距离
average_path_lengths = [ np.mean(list(spl.values())) for spl in distances_all.values() ]
# 节点平均图距离的均值
average_all = np.mean(average_path_lengths)
fig, ax = plt.subplots(figsize = (6,3))
plt.bar(G.nodes(), average_path_lengths)
plt.axhline(y = average_all, c = 'r')
plt.xlabel('Node label')
plt.ylabel('Average node graph distance')
# plt.savefig('节点平均图距离.svg')

#### 图 12. 空手道俱乐部会员关系图，节点平均图距离直方图
fig, ax = plt.subplots(figsize = (6,3))
plt.hist(average_path_lengths, ec = 'k')
plt.axvline(x = average_all, c = 'r')
plt.xlabel('Average node graph distance')
plt.ylabel('Count')
# plt.savefig('节点图距离直方图.svg')


# dict_ave_graph_d = {index: value for index, value in enumerate(average_path_lengths)}

# 根据节点平均图距离大小渲染节点
# 取出节点离心率独特值
# unique_ave_graph_d = set(average_path_lengths)

# colors = plt.cm.RdYlBu(np.linspace(0, 1, len(unique_ave_graph_d)))
# 独特值的颜色映射
#### 图 13. 空手道俱乐部会员关系图，用节点平均图距离渲染节点
fig, ax = plt.subplots(figsize = (10, 10))
nx.draw_networkx(G, pos, cmap = 'RdYlBu_r', with_labels = False, node_color = average_path_lengths)
# plt.savefig('根据平均图距离大小渲染节点.svg')

#>>>>>>>>>>>>>>>>>> 离心率 <<<<<<<<<<<<<<<<<<<<<
eccentricity = nx.eccentricity(G)
# 计算每个节点离心率
eccentricity_list = list(eccentricity.values())
# eccentricity_list
# 自定义函数，过滤dict
def filter_value(dict_, unique):
    newDict = {}
    for (key, value) in dict_.items():
        if value == unique:
            newDict[key] = value
    return newDict

# 图 16. 空手道俱乐部会员关系图，根据节点离心率渲染节点
# 取出节点离心率独特值
unique_ecc = set(eccentricity_list)
# 独特值的颜色映射
colors = plt.cm.RdYlBu(np.linspace(0, 1, len(unique_ecc)))
plt.figure(figsize = (6,6))
# 绘制图的边
nx.draw_networkx_edges(G, pos)
# 分别绘制不同离心率
for deg_i, color_i in zip(unique_ecc,colors):
    dict_i = filter_value(eccentricity,deg_i)
    nx.draw_networkx_nodes(G, pos, nodelist = list(dict_i.keys()), node_color = color_i)
# plt.savefig('根据离心率大小渲染节点.svg')


# 图 14. 空手道俱乐部会员关系图，各个节点离心率
plt.figure(figsize = (6,6))
plt.bar(G.nodes(),eccentricity_list)
plt.xlabel('Node label')
plt.ylabel('Eccentricity')
# plt.savefig('节点离心率.svg')

# 图 15. 空手道俱乐部会员关系图，离心率柱状图
# 使用numpy.unique函数获取独特值及其出现次数
unique_values, counts = np.unique(eccentricity_list, return_counts=True)
plt.figure(figsize = (6,6))
plt.bar(unique_values, counts)
plt.xlabel('Eccentricity')
plt.ylabel('Frequency')
# plt.savefig('图离心率柱状图.svg')


# 图 17. 空手道俱乐部会员关系图，中心点
# 图距离矩阵所有元素最大值叫做图直径 (graph diameter)
# 图直径
nx.diameter(G)

# 获取图的中心点
list_centers = list(nx.center(G))

plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos)
nx.draw_networkx_nodes(G, pos, nodelist = list_centers, node_color = 'r')
# plt.savefig('空手道俱乐部图，中心点.svg')

# 图 18. 空手道俱乐部会员关系图，边缘点
list_periphery = list(nx.periphery(G))
# 获取图的边缘点

plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos)
nx.draw_networkx_nodes(G, pos, nodelist = list_periphery, node_color = 'r')
# plt.savefig('空手道俱乐部图，边缘点.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 中心性，空手道俱乐部
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

G = nx.karate_club_graph()
# 加载空手道俱乐部数据
pos = nx.spring_layout(G,seed=2)
plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos)
# plt.savefig('空手道俱乐部图.svg')


# 自定义可视化函数
def visualize(x_cent, xlabel):
    fig, axes = plt.subplots(1,2,figsize = (12,6))

    # 中心性度量值直方图
    axes[0].hist(x_cent.values(),bins = 15, ec = 'k')
    # 中心性度量值均值
    mean_cent = np.array(list(x_cent.values())).mean()
    axes[0].axvline(x = mean_cent, c = 'r')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel('Count')

    # 可视化图，用中心性度量值渲染节点颜色
    degree_colors = [x_cent[i] for i in range(0,34)]
    nx.draw_networkx(G, pos, ax = axes[1], with_labels = True, node_color=degree_colors)
    # plt.savefig(xlabel + '.svg')

##### 图 20. 度中心性，空手道俱乐部会员关系图
# 度中心性
degree_cent = nx.degree_centrality(G)
# 计算度中心性
visualize(degree_cent, 'Degree centrality')


# 入度中心性、出度中心性
# degree_cent_in = nx.in_degree_centrality(G)
# visualize(degree_cent_in, 'In Degree centrality')
# degree_cent_out = nx.out_degree_centrality(G)
# visualize(degree_cent_out, 'Out Degree centrality')


##### 图 22. 归一化介数中心性，空手道俱乐部会员关系图
# 介数中心性
nx.betweenness_centrality(G, normalized = False)
# 计算归一化介数中心性
betweenness_cent = nx.betweenness_centrality(G)
visualize(betweenness_cent, 'Betweeness centrality')

##### 图 26. 紧密中心性，空手道俱乐部会员关系图
# 紧密中心性
closeness_cent = nx.closeness_centrality(G)
visualize(closeness_cent, 'Closeness centrality')

##### 图 27. 特征向量中心性，空手道俱乐部会员关系图
# 计算特征向量中心性
eigen_cent = nx.eigenvector_centrality(G)
visualize(eigen_cent, 'Eigenvector centrality')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 社区划分，空手道俱乐部

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman

G = nx.karate_club_graph()

plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos = nx.circular_layout(G), with_labels=True)

# 社区划分
communities = girvan_newman(G)

node_groups = []
for com in next(communities):
    node_groups.append(list(com))
node_groups
# [[0, 1, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21],
 # [2, 8, 9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]]

# 根据社区划分设置节点颜色
color_map = []
for node in G:
    if node in node_groups[0]:
        color_map.append('blue')
    else:
        color_map.append('red')

plt.figure(figsize = (6,6))
pos = nx.spring_layout(G,seed=2)
nx.draw_networkx(G, pos = pos, node_color=color_map, with_labels=False)
# plt.savefig('社区划分，spring_layout.svg')
plt.show()

plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos = nx.circular_layout(G), node_color=color_map, with_labels=False)
# plt.savefig('社区划分，circular_layout.svg')
plt.show()
plt.close('all')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 基因关系

from random import sample
import networkx as nx
import matplotlib.pyplot as plt


G = nx.read_edgelist("WormNet.v3.benchmark.txt")

# remove randomly selected nodes (to make example fast)
num_to_remove = int(len(G) / 1.5)
nodes = sample(list(G.nodes), num_to_remove)
G.remove_nodes_from(nodes)

# remove low-degree nodes
low_degree = [n for n, d in G.degree() if d < 10]
G.remove_nodes_from(low_degree)

# largest connected component
components = nx.connected_components(G)
largest_component = max(components, key=len)
H = G.subgraph(largest_component)

# compute centrality
centrality = nx.betweenness_centrality(H, k=10, endpoints=True)


# compute community structure
lpc = nx.community.label_propagation_communities(H)
community_index = {n: i for i, com in enumerate(lpc) for n in com}

# 图 23. WormNet v.3-GS 数据来测量基因之间的正向功能关联
fig, ax = plt.subplots(figsize=(20, 15))
pos = nx.spring_layout(H, k=0.15, seed=4572321)
node_color = [community_index[n] for n in H]
node_size = [v * 20000 for v in centrality.values()]
nx.draw_networkx(
    H,
    pos=pos,
    with_labels=False,
    node_color=node_color,
    node_size=node_size,
    edge_color="gainsboro",
    alpha=0.4,
)

plt.axis("off")
plt.show()

























































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%































































































































































































































































































