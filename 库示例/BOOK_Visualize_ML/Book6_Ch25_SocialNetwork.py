

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 社交网络分析SNA
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random import randint
import seaborn as sns

# 导入数据
facebook = pd.read_csv(
    "data/facebook_combined.txt.gz",
    compression="gzip",
    sep=" ",
    names=["start_node", "end_node"])

facebook.head(5)
len(facebook)


# 创建无向图
# 为了加快运算速度，仅仅随机取出5000行数据
# G = nx.from_pandas_edgelist(facebook.sample(n=5000), "start_node", "end_node")
# 为了加快运算速度，取出前5000行数据
G = nx.from_pandas_edgelist(facebook.head(5000), "start_node", "end_node")

plot_options = {"node_size": 18,
                "with_labels": False,
                "width": 0.15,
                "edge_color": '0.18',
                "alpha": 0.28}
pos = nx.spring_layout(G, iterations=15, seed=1721)

# 可视化社交关系图
fig, ax = plt.subplots(figsize=(18, 18))
ax.axis("off")
nx.draw_networkx(G, pos=pos, ax=ax, **plot_options)
plt.show()
plt.close('all')

# 无向图节点数
G.number_of_nodes()
# 1724

# 无向图边数
G.number_of_edges()
# 5000

# 无向图连通分量数量
nx.number_connected_components(G)
# 1

# 度分析
# 节点度数排序
degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
len(degree_sequence)
# 1724

# 度数大小排序
dmax = max(degree_sequence)
# 1045

# 将结果转为字典
dict_degree = dict(G.degree())

# 节点度数均值
mean_degree = np.mean([d for _, d in G.degree()])

# 节点度数排序
fig, ax = plt.subplots(figsize = (9,8))
ax.plot(degree_sequence, "b-", marker="o")
ax.grid()
ax.axhline(y = mean_degree, color = 'r')
ax.set_ylabel("Degree")
ax.set_xlabel("Rank")
# ax.set_xscale('log')
plt.show()
plt.close('all')


# 节点度数柱状图
fig, ax = plt.subplots(figsize = (6,5))
ax.bar(*np.unique(degree_sequence, return_counts=True), ec = 'k')
ax.axvline(x = mean_degree, color = 'r')
ax.grid()
ax.set_xlabel("Degree")
ax.set_ylabel("Number of Nodes")
# ax.set_xscale('log')
# plt.savefig('度数柱状图.svg')

# 自定义函数，过滤dict
def filter_value(dict_, unique):
    newDict = {}
    for (key, value) in dict_.items():
        if value == unique:
            newDict[key] = value
    return newDict

################ 根据度数大小渲染节点
# 取出节点度数独特值
unique_deg = sorted(set(degree_sequence))

# 独特值的颜色映射
colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(unique_deg)))
# 图 3. 社交网络图，节点度数
fig, ax = plt.subplots(figsize=(18, 18))
ax.axis("off")
# 绘制图的边
nx.draw_networkx_edges(G, pos, edge_color = 'gray', width = 0.15)
# 分别绘制不同度数节点
for deg_i, color_i in zip(unique_deg, colors):
    dict_i = filter_value(dict_degree, deg_i)
    nx.draw_networkx_nodes(G, pos, nodelist = list(dict_i.keys()), node_size=18, node_color = np.array(color_i)[:3].reshape(1,-1))
# plt.savefig('根据度数大小渲染节点.svg')


# 图 4. 社交网络图，节点度数超过 100 的节点
# 取出度数超过 100
deg_threshold = 100
fig, ax = plt.subplots(figsize=(18, 18))
ax.axis("off")
nx.draw_networkx_edges(G, pos, edge_color = '0.18', width = 0.15)
# 绘制图的边
for deg_i in reversed(unique_deg):
    dict_i = filter_value(dict_degree, deg_i)
    nx.draw_networkx_nodes(G, pos, nodelist = list(dict_i.keys()), node_size=108, node_color = 'red', alpha = 0.28)
    # print(deg_i)
    if deg_i <= deg_threshold:
        break


################### 图距离
shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))

# 节点平均距离
average_path_lengths = [np.mean(list(spl.values())) for spl in shortest_path_lengths.values()]
# 节点平均图距离的均值
average_all = np.mean(average_path_lengths)

# 图 6. 平均图距离直方图
fig, ax = plt.subplots(figsize = (6,5))
plt.hist(average_path_lengths, ec = 'k', bins = 30)
plt.axvline(x = average_all, c = 'r')
plt.xlabel('Average node graph distance')
plt.ylabel('Count')
# plt.savefig('节点图距离直方图.svg')

##################### 图距离分布
diameter = nx.diameter(G) # 图直径
path_lengths = np.zeros(diameter + 1, dtype=int)

for pls in shortest_path_lengths.values():
    pl, cnts = np.unique(list(pls.values()), return_counts=True)
    path_lengths[pl] += cnts
freq_percent = 100 * path_lengths[1:] / path_lengths[1:].sum()
fig, ax = plt.subplots(figsize = (6,5))
ax.bar(np.arange(1, diameter + 1), height=freq_percent)
ax.set_xlabel("Graph distance")
ax.set_ylabel("Frequency (%)")


# 图 7 根据节点平均图距离大小渲染节点
unique_ave_graph_d = set(average_path_lengths)
# 取出节点离心率独特值

# 独特值的颜色映射
# colors = plt.cm.RdYlBu(np.linspace(0, 1, len(unique_ave_graph_d)))
fig, ax = plt.subplots(figsize=(18, 18))
ax.axis("off")
nx.draw_networkx(G, pos, cmap = 'RdYlBu_r', with_labels = False, node_size=18, node_color = average_path_lengths)


# 图 8. 图距离矩阵
Shortest_D_matrix = np.array([[v[j] for j in list(shortest_path_lengths.keys())] for k, v in shortest_path_lengths.items()])
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(Shortest_D_matrix, cmap = 'Blues', annot = False, xticklabels = [], yticklabels = [], ax = ax, cbar_kws={'shrink': 0.38}, square = True, cbar = True)

###################### 离心率
eccentricity = nx.eccentricity(G)
# 计算每个节点离心率
eccentricity_list = list(eccentricity.values())

# 图 10. 根据离心率大小渲染节点
# 取出节点离心率独特值
unique_ecc = set(eccentricity_list)
# 独特值的颜色映射
colors = plt.cm.RdYlBu(np.linspace(0, 1, len(unique_ecc)))
fig, ax = plt.subplots(figsize=(18, 18))
ax.axis("off")
# 绘制图的边
nx.draw_networkx_edges(G, pos, edge_color = '0.18', width = 0.15)
# 分别绘制不同离心率
for deg_i, color_i in zip(unique_ecc,colors):
    dict_i = filter_value(eccentricity,deg_i)
    nx.draw_networkx_nodes(G, pos, node_size=18, nodelist = list(dict_i.keys()), node_color = np.array(color_i).reshape(1,-1))

# 图 9. 离心率柱状图
# 使用numpy.unique函数获取独特值及其出现次数
unique_values, counts = np.unique(eccentricity_list, return_counts=True)
# 绘制柱状图
fig, ax = plt.subplots(figsize = (6,5))
plt.bar(unique_values, counts)
plt.xlabel('Eccentricity')
plt.ylabel('Frequency')
# plt.savefig('图离心率柱状图.svg')


############################## 度中心性
# 图直径
diameter = nx.diameter(G)
diameter
# 6
# 图半径
radius = nx.radius(G)
radius
# 3
# 度中心性
degree_centrality = nx.centrality.degree_centrality(G)

# 图 12.度中心性直方图
fig, ax = plt.subplots(figsize = (8,5))
plt.hist(degree_centrality.values(), bins=25, ec = 'k')
plt.xlabel("Degree Centrality")
plt.ylabel("Count")
plt.yscale('log')
plt.grid(which='major', color='k', linestyle='-')
plt.grid(which='minor', color='0.8', linestyle='-')

# 图 11. 根据度中心性大小渲染节点
node_size = [v * 1000 for v in degree_centrality.values()]
fig, ax = plt.subplots(figsize=(18, 18))
nx.draw_networkx(G, pos=pos, node_size=node_size, edge_color = '0.18', with_labels=False, cmap = 'RdYlBu_r', node_color = node_size, alpha = 0.68, width=0.15)
plt.axis("off")

#############################  介数中心性
betweenness_centrality = nx.centrality.betweenness_centrality(G)
# 图 14 所示为社交网络节点介数中心性直方图。
fig, ax = plt.subplots(figsize = (8,5))
plt.hist(betweenness_centrality.values(), bins=25, ec = 'k')
plt.xlabel("Betweeness Centrality")
plt.ylabel("Count")
plt.yscale('log')
plt.grid(which='major', color='k', linestyle='-')
plt.grid(which='minor', color='0.8', linestyle='-')

# 图 13 所示为根据节点介数中心性渲染节点；
node_size = [v * 1200 for v in betweenness_centrality.values()]
fig, ax = plt.subplots(figsize=(18, 18))
nx.draw_networkx(G, pos=pos,
                 node_size=node_size,
                 edge_color = '0.18',
                 with_labels=False,
                 cmap = 'RdYlBu_r',
                 node_color = node_size,
                 alpha = 0.68,
                 width=0.15)
plt.axis("off")


########################### 紧密中心性
closeness_centrality = nx.centrality.closeness_centrality(G)
# 图 16 所示为社交网络节点紧密中心性直方图。
fig, ax = plt.subplots(figsize = (8,5))
plt.hist(closeness_centrality.values(), bins=25, ec = 'k')
plt.xlabel("Closeness Centrality")
plt.ylabel("Count")
plt.yscale('log')
plt.grid(which='major', color='k', linestyle='-')
plt.grid(which='minor', color='0.8', linestyle='-')

# 图 15 所示为根据节点紧密中心性渲染节点
node_size = [v * 50 for v in closeness_centrality.values()]
fig, ax = plt.subplots(figsize=(18, 18))
nx.draw_networkx(G, pos=pos,
                 node_size=node_size,
                 edge_color = '0.18',
                 with_labels=False,
                 cmap = 'RdYlBu_r',
                 node_color = node_size,
                 alpha = 0.68,
                 width=0.15)
plt.axis("off")

######################### 特征向量中心性
eigenvector_centrality = nx.centrality.eigenvector_centrality(G)
# 图 18 所示为社交网络节点特征向量中心性直方图。
fig, ax = plt.subplots(figsize = (8,5))
plt.hist(eigenvector_centrality.values(), bins=25, ec = 'k')
plt.xlabel("Eigenvector Centrality")
plt.ylabel("Count")
plt.yscale('log')
plt.grid(which='major', color='k', linestyle='-')
plt.grid(which='minor', color='0.8', linestyle='-')

# 图 17 所示为根据节点特征向量中心性渲染节点
node_size = [v * 4000 for v in eigenvector_centrality.values()]
fig, ax = plt.subplots(figsize=(18, 18))
nx.draw_networkx(G, pos=pos, node_size=node_size, edge_color = '0.18', with_labels=False, cmap = 'RdYlBu_r', node_color = node_size, alpha = 0.68, width=0.15)
plt.axis("off")


######################### 桥
nx.has_bridges(G)

bridges = list(nx.bridges(G))
len(bridges)
# 1113

fig, ax = plt.subplots(figsize=(18, 18))
nx.draw_networkx(G, pos=pos, node_size=18, edge_color = '0.68', with_labels=False, width=0.15, node_color = '0.58')
nx.draw_networkx_nodes(G, pos=pos, nodelist = set(np.unique(bridges)), node_color = 'r', node_size = 58)
nx.draw_networkx_edges(G, pos, edgelist=bridges, width=1, edge_color="r")
plt.axis("off")


####################### 局部桥
local_bridges = list(nx.local_bridges(G, with_span=False))
len(local_bridges)
# 1119
fig, ax = plt.subplots(figsize=(18, 18))
nx.draw_networkx(G, pos=pos, node_size=18, edge_color = '0.68', with_labels=False, width=0.15, node_color = '0.38')
nx.draw_networkx_nodes(G, pos=pos, nodelist = set(np.unique(local_bridges)), node_color = 'blue', node_size = 58)
nx.draw_networkx_edges(G, pos, edgelist=local_bridges, width=1, edge_color="blue")
plt.axis("off")

fig, ax = plt.subplots(figsize=(18, 18))
nx.draw_networkx(G, pos=pos, node_size=18, edge_color = '0.68', with_labels=False, width=0.15, node_color = '0.38')
nx.draw_networkx_edges(G, pos, edgelist=set(local_bridges) - set(bridges), width=1, edge_color="blue")
nx.draw_networkx_nodes(G, pos=pos, nodelist = set(np.unique(local_bridges)) - set(np.unique(bridges)), node_color = 'blue', node_size = 58)
plt.axis("off")


################# 社区
print(f"{len(G.nodes())}, {max(G.nodes())}, {min(G.nodes())}") # 1724, 3290, 0
#(1) 初始化标签：将每个节点初始化为一个唯一的标签。
node_mapping = {node: i for i, node in enumerate(G.nodes())}
# 创建一个新图 H
H = nx.relabel_nodes(G, node_mapping)
# 这部分代码是为了保证选取部分节点构造无向图时，代码依然能跑通
pos_H = nx.spring_layout(H, iterations=15, seed=1721)
print(f"{len(H.nodes())}, {max(H.nodes())}, {min(H.nodes())}") # 1724, 1723, 0

#(2) 标签传播：在每一轮中，节点会将其当前标签传播给邻居节点。具体来说，节点选择其邻居中标签数最多的标签，并将自己的标签更新为这个最多的标签。
colors = ["" for x in range(H.number_of_nodes())]
counter = 0
for com in nx.community.label_propagation_communities(H): # 返回每个社区的node的集合
    color = "#%06X" % randint(0, 0xFFFFFF)
    # creates random RGB color
    counter += 1
    for node in list(com):
        colors[node] = color
counter
# 22


fig, ax = plt.subplots(figsize=(18, 18))
plt.axis("off")
nx.draw_networkx(H, pos=pos_H, node_size=18, with_labels=False, width=0.15, node_color=colors)

# 社区划分，异步流体社区算法
colors = ["" for x in range(H.number_of_nodes())]
c = 0
for com in nx.community.asyn_fluidc(H, 8, seed=0):
    c += 1
    color = "#%06X" % randint(0, 0xFFFFFF)
    # creates random RGB color
    for node in list(com):
        colors[node] = color
# c = 8
fig, ax = plt.subplots(figsize=(18, 18))
plt.axis("off")
nx.draw_networkx(H, pos = pos_H, node_size=18, with_labels=False, width=0.15, node_color = colors)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Facebook Network Analysis
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random import randint


facebook = pd.read_csv(
    "data/facebook_combined.txt.gz",
    compression="gzip",
    sep=" ",
    names=["start_node", "end_node"],)
facebook
# 创建无向图
G = nx.from_pandas_edgelist(facebook, "start_node", "end_node")


# 可视化社交关系图
fig, ax = plt.subplots(figsize=(15, 9))
ax.axis("off")
plot_options = {"node_size": 10, "with_labels": False, "width": 0.15}
nx.draw_networkx(G, pos=nx.random_layout(G), ax=ax, **plot_options)

# 可视化社交关系图
pos = nx.spring_layout(G, iterations=15, seed=1721)
fig, ax = plt.subplots(figsize=(15, 9))
ax.axis("off")
nx.draw_networkx(G, pos=pos, ax=ax, **plot_options)



# 无向图节点数
G.number_of_nodes() # 4039

# 无向图边数
G.number_of_edges() # 88234
# 节点度数均值
np.mean([d for _, d in G.degree()])
# 43.69101262688784
##################### 图距离分布
shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))
shortest_path_lengths[0][42]  # Length of shortest path between nodes 0 and 42
# 1
# 图直径
diameter = max(nx.eccentricity(G, sp=shortest_path_lengths).values())
diameter
# 8

# 节点平均距离
average_path_lengths = [np.mean(list(spl.values())) for spl in shortest_path_lengths.values()]
# 节点平均图距离的均值
np.mean(average_path_lengths)
# 3.691592636562027

path_lengths = np.zeros(diameter + 1, dtype=int)
for pls in shortest_path_lengths.values():
    pl, cnts = np.unique(list(pls.values()), return_counts=True)
    path_lengths[pl] += cnts
freq_percent = 100 * path_lengths[1:] / path_lengths[1:].sum()

fig, ax = plt.subplots(figsize=(15, 8))
ax.bar(np.arange(1, diameter + 1), height=freq_percent)
ax.set_title("Distribution of shortest path length in G", fontdict={"size": 35}, loc="center")
ax.set_xlabel("Shortest Path Length", fontdict={"size": 22})
ax.set_ylabel("Frequency (%)", fontdict={"size": 22})



nx.density(G)

# 无向图连通分量数量
nx.number_connected_components(G)
############################## 度中心性
degree_centrality = nx.centrality.degree_centrality(G)  # save results in a variable to use again
(sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True))[:8]
(sorted(G.degree, key=lambda item: item[1], reverse=True))[:8]

# 图 .度中心性直方图
plt.figure(figsize=(15, 8))
plt.hist(degree_centrality.values(), bins=25)
plt.xticks(ticks=[0, 0.025, 0.05, 0.1, 0.15, 0.2])  # set the x axis ticks
plt.title("Degree Centrality Histogram ", fontdict={"size": 35}, loc="center")
plt.xlabel("Degree Centrality", fontdict={"size": 20})
plt.ylabel("Counts", fontdict={"size": 20})
# 图 . 根据度中心性大小渲染节点
node_size = [v * 1000 for v in degree_centrality.values()]  # set up nodes size for a nice graph representation
plt.figure(figsize=(15, 8))
nx.draw_networkx(G, pos=pos, node_size=node_size, with_labels=False, width=0.15)
plt.axis("off")

#############################  介数中心性
betweenness_centrality = nx.centrality.betweenness_centrality(G)  # save results in a variable to use again
(sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True))[:8]

# 图  所示为社交网络节点介数中心性直方图。
plt.figure(figsize=(15, 8))
plt.hist(betweenness_centrality.values(), bins=100)
plt.xticks(ticks=[0, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5])  # set the x axis ticks
plt.title("Betweenness Centrality Histogram ", fontdict={"size": 35}, loc="center")
plt.xlabel("Betweenness Centrality", fontdict={"size": 20})
plt.ylabel("Counts", fontdict={"size": 20})

# 图. 所示为根据节点介数中心性渲染节点；
node_size = [v * 1200 for v in betweenness_centrality.values()]  # set up nodes size for a nice graph representation
plt.figure(figsize=(15, 8))
nx.draw_networkx(G, pos=pos, node_size=node_size, with_labels=False, width=0.15)
plt.axis("off")

########################### 紧密中心性
closeness_centrality = nx.centrality.closeness_centrality(G)  # save results in a variable to use again
(sorted(closeness_centrality.items(), key=lambda item: item[1], reverse=True))[:8]

1 / closeness_centrality[107]
# 图: 所示为社交网络节点紧密中心性直方图。
plt.figure(figsize=(15, 8))
plt.hist(closeness_centrality.values(), bins=60)
plt.title("Closeness Centrality Histogram ", fontdict={"size": 35}, loc="center")
plt.xlabel("Closeness Centrality", fontdict={"size": 20})
plt.ylabel("Counts", fontdict={"size": 20})

# 图:所示为根据节点紧密中心性渲染节点
node_size = [v * 50 for v in closeness_centrality.values()]  # set up nodes size for a nice graph representation
plt.figure(figsize=(15, 8))
nx.draw_networkx(G, pos=pos, node_size=node_size, with_labels=False, width=0.15)
plt.axis("off")

######################### 特征向量中心性
eigenvector_centrality = nx.centrality.eigenvector_centrality(G)  # save results in a variable to use again
(sorted(eigenvector_centrality.items(), key=lambda item: item[1], reverse=True))[:10]


high_eigenvector_centralities = (sorted(eigenvector_centrality.items(), key=lambda item: item[1], reverse=True))[1:10]  # 2nd to 10th nodes with heighest eigenvector centralities
high_eigenvector_nodes = [tuple[0] for tuple in high_eigenvector_centralities]
# set list as [2266, 2206, 2233, 2464, 2142, 2218, 2078, 2123, 1993]
neighbors_1912 = [n for n in G.neighbors(1912)]
# list with all nodes connected to 1912
all(item in neighbors_1912 for item in high_eigenvector_nodes)
# check if items in list high_eigenvector_nodes exist in list neighbors_1912

# 图:所示为社交网络节点紧密中心性直方图。
plt.figure(figsize=(15, 8))
plt.hist(eigenvector_centrality.values(), bins=60)
plt.xticks(ticks=[0, 0.01, 0.02, 0.04, 0.06, 0.08])  # set the x axis ticks
plt.title("Eigenvector Centrality Histogram ", fontdict={"size": 35}, loc="center")
plt.xlabel("Eigenvector Centrality", fontdict={"size": 20})
plt.ylabel("Counts", fontdict={"size": 20})

# 图: 所示为根据节点特征向量中心性渲染节点
node_size = [v * 4000 for v in eigenvector_centrality.values()]  # set up nodes size for a nice graph representation
plt.figure(figsize=(15, 8))
nx.draw_networkx(G, pos=pos, node_size=node_size, with_labels=False, width=0.15)
plt.axis("off")


################################### Clustering Effects
nx.average_clustering(G)

plt.figure(figsize=(15, 8))
plt.hist(nx.clustering(G).values(), bins=50)
plt.title("Clustering Coefficient Histogram ", fontdict={"size": 35}, loc="center")
plt.xlabel("Clustering Coefficient", fontdict={"size": 20})
plt.ylabel("Counts", fontdict={"size": 20})


triangles_per_node = list(nx.triangles(G).values())
sum(triangles_per_node) / 3  # divide by 3 because each triangle is counted once for each node

np.mean(triangles_per_node)
np.median(triangles_per_node)

######################### 桥
nx.has_bridges(G)

bridges = list(nx.bridges(G))
len(bridges)

plt.figure(figsize=(15, 8))
nx.draw_networkx(G, pos=pos, node_size=10, with_labels=False, width=0.15)
nx.draw_networkx_edges(G, pos, edgelist=bridges, width=0.5, edge_color="r")  # red color for bridges
plt.axis("off")
####################### 局部桥
local_bridges = list(nx.local_bridges(G, with_span=False))
len(local_bridges)


plt.figure(figsize=(15, 8))
nx.draw_networkx(G, pos=pos, node_size=10, with_labels=False, width=0.15)
nx.draw_networkx_edges(G, pos, edgelist=local_bridges, width=0.5, edge_color="lawngreen")  # green color for local bridges
plt.axis("off")

#######################  Assortativity
nx.degree_assortativity_coefficient(G)
nx.degree_pearson_correlation_coefficient(G)  # use the potentially faster scipy.stats.pearsonr function.


################# 社区
colors = ["" for x in range(G.number_of_nodes())]  # initialize colors list
counter = 0
for com in nx.community.label_propagation_communities(G):
    color = "#%06X" % randint(0, 0xFFFFFF)  # creates random RGB color
    counter += 1
    for node in list( com ):  # fill colors list with the particular color for the community nodes
        colors[node] = color
counter
# 44

plt.figure(figsize=(15, 9))
plt.axis("off")
nx.draw_networkx(G, pos=pos, node_size=10, with_labels=False, width=0.15, node_color=colors)


colors = ["" for x in range(G.number_of_nodes())]
for com in nx.community.asyn_fluidc(G, 8, seed=0):
    color = "#%06X" % randint(0, 0xFFFFFF)  # creates random RGB color
    for node in list(com):
        colors[node] = color


plt.figure(figsize=(15, 9))
plt.axis("off")
nx.draw_networkx(G, pos=pos, node_size=10, with_labels=False, width=0.15, node_color=colors)































































