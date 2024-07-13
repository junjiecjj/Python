

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  有向图


import matplotlib.pyplot as plt
import networkx as nx

# 创建有向图的实例
directed_G = nx.DiGraph()

# 添加多个顶点
directed_G.add_nodes_from(['a', 'b', 'c', 'd'])

# 增加一组有向边
directed_G.add_edges_from([('b','a'),
                           ('c','b'),
                           ('b','d'),
                           ('d','c'),
                           ('a','c')])

# 设定随机种子，保证每次绘图结果一致
random_pos = nx.random_layout(directed_G, seed=188)

# 使用弹簧布局算法来排列图中的节点
# 使得节点之间的连接看起来更均匀自然
pos = nx.spring_layout(directed_G, pos=random_pos)

plt.figure(figsize = (6,6))
nx.draw_networkx(directed_G, pos = pos, arrowsize = 28, node_size = 180)
# plt.savefig('G_D_4顶点_5边.svg')



# 属性
# 图的阶
directed_G.order()

# 图的节点数
directed_G.number_of_nodes()

# 列出图的节点
directed_G.nodes

# 图的大小
directed_G.size()

# 列出图的边
directed_G.edges

# 图的边数
directed_G.number_of_edges()

# 判断是否存在ab有向边
directed_G.has_edge('a', 'b')


# 判断是否存在ba有向边
directed_G.has_edge('b', 'a')

# 图的度
directed_G.degree()

dict(directed_G.degree())
# 有向图的入度
directed_G.in_degree()

# 有向图的出度
directed_G.out_degree()
directed_G.out_degree()['a']

# 节点a的度
directed_G.degree('a')

# 节点a的入度
directed_G.in_degree('a')

# 节点a的出度
directed_G.out_degree('a')

# 节点b的入度
directed_G.in_degree('b')

# 节点b的出度
directed_G.out_degree('b')

# 节点a所有邻居
list(nx.all_neighbors(directed_G, 'a'))

# 节点a的 (出度) 邻居
list(directed_G.neighbors('a'))

# 节点a的出度邻居
list(directed_G.successors('a'))

# 节点a的入度邻居
list(directed_G.predecessors('a'))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 有向多图

import matplotlib.pyplot as plt
import networkx as nx

# 创建有向图的实例
directed_G = nx.MultiDiGraph()

# 添加多个顶点
directed_G.add_nodes_from(['a', 'b', 'c', 'd'])

# 增加一组有向边
directed_G.add_edges_from([('b','a'),
                           ('a','b'),
                           ('c','b'),
                           ('b','d'),
                           ('d','c'),
                           ('a','c'),
                           ('c','a')])

#>>>>>> 人为设定节点位置 <<<<<<<<
nodePosDict = {'b':[0, 0],
               'c':[1, 0],
               'd':[1, 1],
               'a':[0, 1]}

plt.figure(figsize = (6,6))
nx.draw_networkx(directed_G, pos = nodePosDict, arrowsize = 28, connectionstyle='arc3, rad = 0.1', node_size = 180)
# plt.savefig('G_D_4顶点_7边.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 各种三元组 (triads) 类型

import networkx as nx
import matplotlib.pyplot as plt

# 16种三元组的名称
list_triads = ('003', '012', '102', '021D',
               '021U', '021C', '111D', '111U',
               '030T', '030C', '201', '120D',
               '120U', '120C', '210', '300')

# 可视化
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for triad_i, ax in zip(list_triads, axes.flatten()):
    G = nx.triad_graph(triad_i)
    # 绘制三元组
    nx.draw_networkx( G, ax=ax, with_labels=False, node_size=100, arrowsize=30, width=1.25, pos=nx.planar_layout(G))
    ax.set_xlim(val * 1.2 for val in ax.get_xlim())
    ax.set_ylim(val * 1.2 for val in ax.get_ylim())
    # 增加三元组名称
    ax.text(0,0, triad_i, fontsize=15, font = 'Roboto', fontweight="light", horizontalalignment="center")
fig.tight_layout()
# plt.savefig('16种三元组.svg')
plt.show()

# 第二种：完全自定义方法
triads = {
    "003": [],
    "012": [(1, 2)],
    "102": [(1, 2), (2, 1)],
    "021D": [(3, 1), (3, 2)],
    "021U": [(1, 3), (2, 3)],
    "021C": [(1, 3), (3, 2)],
    "111D": [(1, 2), (2, 1), (3, 1)],
    "111U": [(1, 2), (2, 1), (1, 3)],
    "030T": [(1, 2), (3, 2), (1, 3)],
    "030C": [(1, 3), (3, 2), (2, 1)],
    "201": [(1, 2), (2, 1), (3, 1), (1, 3)],
    "120D": [(1, 2), (2, 1), (3, 1), (3, 2)],
    "120U": [(1, 2), (2, 1), (1, 3), (2, 3)],
    "120C": [(1, 2), (2, 1), (1, 3), (3, 2)],
    "210": [(1, 2), (2, 1), (1, 3), (3, 2), (2, 3)],
    "300": [(1, 2), (2, 1), (2, 3), (3, 2), (1, 3), (3, 1)],
}

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for (title, triad), ax in zip(triads.items(), axes.flatten()):
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3])
    G.add_edges_from(triad)
    nx.draw_networkx( G, ax=ax, with_labels=False, node_size=58, arrowsize=20, width=0.25, pos=nx.planar_layout(G))

    ax.set_xlim(val * 1.2 for val in ax.get_xlim())
    ax.set_ylim(val * 1.2 for val in ax.get_ylim())
    ax.text(0, 0, title, fontsize=15, font = 'Roboto', fontweight="light", horizontalalignment="center")
fig.tight_layout()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 发现三元组
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph([(1, 2), (2, 3),
                (3, 1), (4, 3),
                (4, 1), (1, 4)])

pos = nx.spring_layout(G,seed = 68)

plt.figure(figsize=(8, 8))
nx.draw_networkx(G, pos = pos, with_labels = True)
# plt.savefig('有向图.svg')

# 判断G是否为三元组triad
nx.is_triad(G)

# 寻找G中三元组
fig, axes = plt.subplots(2, 2, figsize = (8,8))
axes = axes.flatten()
for triad_i, ax_i in zip(nx.all_triads(G), axes):
    nx.draw_networkx(G, pos = pos, ax = ax_i, width=0.25, with_labels = False)
    # 绘制三元组子图
    nx.draw_networkx_nodes(G, nodelist = triad_i.nodes, node_color = 'r', ax = ax_i, pos = pos)

    nx.draw_networkx_edges(G, edgelist = triad_i.edges, edge_color = 'r', width=1, ax = ax_i, pos = pos) #  绘制边
    ax_i.set_title(nx.triad_type(triad_i))
# plt.savefig('有向图中4个三元组子图.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 用列表创建有向图

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 创建list
edgelist = [('b', 'a'),
            ('b', 'd'),
            ('d', 'c'),
            ('c', 'b'),
            ('a', 'c')]
# 创建无向图
G = nx.from_edgelist(edgelist, create_using=nx.Graph())
G.nodes()
G.edges()

# 可视化
plt.figure(figsize = (6,6))
pos = nx.spring_layout(G, seed = 88)
nx.draw_networkx(G, pos = pos, node_color = '#0058FF', with_labels = True, node_size = 188)
# plt.savefig('无向图.svg')


# 创建有向图
Di_G = nx.from_edgelist(edgelist, create_using=nx.DiGraph())

# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(Di_G, pos = pos, node_color = '#0058FF', with_labels = True, node_size = 188)
plt.savefig('有向图.svg')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 用数据帧创建有向图
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 创建数据帧
edges_df = pd.DataFrame({
    'source': ['b', 'b', 'd', 'c', 'a'],
    'target': ['a', 'd', 'c', 'b', 'c'],
    'edge_key': ['ba', 'bd', 'dc', 'cb', 'ac'],
    'weight': [1, 2, 3, 4, 5]})

# edges_df.to_csv('edges_df.csv')
edges_df

#### 创建无向图
G = nx.from_pandas_edgelist( edges_df, source = "source", target = "target", edge_key="edge_key", edge_attr=["weight"], create_using=nx.Graph())

G_edge_labels = nx.get_edge_attributes(G, "weight")
G_edge_labels
# {('b', 'a'): 1, ('b', 'd'): 2, ('b', 'c'): 4, ('a', 'c'): 5, ('d', 'c'): 3}

# 可视化
plt.figure(figsize = (6,6))
pos = nx.spring_layout(G, seed = 28)
nx.draw_networkx(G, pos = pos, node_color = '#0058FF', with_labels = True, node_size = 188)
nx.draw_networkx_edge_labels(G, pos, G_edge_labels)
# plt.savefig('无向图.svg')

#### 创建有向图
Di_G = nx.from_pandas_edgelist( edges_df, source = "source", target = "target", edge_key="edge_key", edge_attr=["weight"], create_using=nx.DiGraph())
# 边权重
Di_G_edge_labels = nx.get_edge_attributes(Di_G, "weight")
# {('b', 'a'): 1, ('b', 'd'): 2, ('a', 'c'): 5, ('d', 'c'): 3, ('c', 'b'): 4}
# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(Di_G, pos = pos, node_color = '#0058FF', with_labels = True, node_size = 188)
nx.draw_networkx_edge_labels(Di_G, pos, Di_G_edge_labels) #  添加边标签
# plt.savefig('有向图.svg')


#### 创建有向图
Di_G = nx.from_pandas_edgelist( edges_df, source = "source", target = "target", edge_key="edge_key", edge_attr=["weight"], create_using=nx.DiGraph())
# 边权重
Di_G_edge_labels = nx.get_edge_attributes(Di_G, "weight")
# {('b', 'a'): 1, ('b', 'd'): 2, ('a', 'c'): 5, ('d', 'c'): 3, ('c', 'b'): 4}
# 可视化
plt.figure(figsize = (6,6))
nx.draw(Di_G, pos = pos, node_color = '#0058FF', with_labels = True, node_size = 188)
nx.draw_networkx_edge_labels(Di_G, pos, Di_G_edge_labels)
# plt.savefig('有向图.svg')





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 用NumPy数组 (邻接矩阵) 创建图，无权

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


matrix_G = np.array([[0, 1, 1, 0],
                     [1, 0, 1, 1],
                     [1, 1, 0, 1],
                     [0, 1, 1, 0]])
# 定义无向图邻接矩阵
G = nx.from_numpy_array(matrix_G, create_using=nx.Graph)
G.nodes()
G.edges()

# 修改节点标签
mapping = {0: "a", 1: "b", 2: "c", 3: "d"}
G = nx.relabel_nodes(G, mapping)
G.nodes()
G.edges()

# 可视化
plt.figure(figsize = (6,6))
pos = nx.spring_layout(G, seed = 8)
nx.draw_networkx(G, pos = pos, node_color = '#0058FF', with_labels = True, node_size = 188)
# plt.savefig('无向图.svg')

# 定义有向图邻接矩阵
matrix_Di_G = np.array([[0, 0, 1, 0],
                        [1, 0, 0, 1],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0]])
Di_G = nx.from_numpy_array(matrix_Di_G, create_using=nx.DiGraph)
# 修改节点标签
Di_G = nx.relabel_nodes(Di_G, mapping)

# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(Di_G, pos = pos, node_color = '#0058FF', with_labels = True, node_size = 188)
# plt.savefig('有向图.svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 用NumPy数组 (邻接矩阵) 创建图，有权

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

##### 定义无向图邻接矩阵
matrix_G = np.array([[0, 1, 5, 0],
                     [1, 0, 4, 2],
                     [5, 4, 0, 3],
                     [0, 2, 3, 0]])
G = nx.from_numpy_array(matrix_G, create_using=nx.Graph)
G.nodes()
G.edges()

# 修改节点标签
mapping = {0: "a", 1: "b", 2: "c", 3: "d"}
G = nx.relabel_nodes(G, mapping) # 用 networkx.relabel_nodes() 修改无向图节点标签。

G.nodes()
G.edges()

G_edge_labels = nx.get_edge_attributes(G, "weight")
G_edge_labels

# 可视化
plt.figure(figsize = (6,6))
pos = nx.spring_layout(G, seed = 28)
nx.draw_networkx(G, pos = pos, node_color = '#0058FF', with_labels = True, node_size = 188)
nx.draw_networkx_edge_labels(G, pos, G_edge_labels)
# plt.savefig('无向图.svg')

##### 定义有向图邻接矩阵
matrix_Di_G = np.array([[0, 0, 5, 0],
                        [1, 0, 0, 2],
                        [0, 4, 0, 0],
                        [0, 0, 3, 0]])
Di_G = nx.from_numpy_array(matrix_Di_G, create_using=nx.DiGraph)
# 修改节点标签
Di_G = nx.relabel_nodes(Di_G, mapping)

# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(Di_G, pos = pos, node_color = '#0058FF', with_labels = True, node_size = 188)
nx.draw_networkx_edge_labels(G, pos, G_edge_labels)
# plt.savefig('有向图.svg')



























































































































































































































































