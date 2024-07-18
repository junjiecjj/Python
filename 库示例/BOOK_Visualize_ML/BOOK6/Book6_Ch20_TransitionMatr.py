

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html








#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 再看邻接矩阵

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
# 创建无向图的实例
G = nx.Graph()

# 添加多个顶点
G.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f'])

# 增加一组边
G.add_edges_from([('a','b'),('a','f'),
                  ('a','c'),('a','d'),
                  ('d','e'),('c','e'),
                  ('d','c'),('b','c'),
                  ('b','e'),('b','f'),
                  ('e','f'),('c','f')])

# 城市具体位置
pos = {'a':(50.6463,6.5692),
       'b':(70.3019,41.0777),
       'c':(35.6363,36.0673),
       'd':(6.8205,16.8171),
       'e':(26.1044,68.8277),
       'f':(59.1871,69.6398)}


plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos = pos, node_size = 880)
# plt.savefig('城市位置.svg')


A = nx.adjacency_matrix(G).todense()
# 无向图的邻接矩阵

sns.heatmap(A,square = True,
            cmap = 'Blues',
            annot = True, fmt = '.0f',
            xticklabels = list(G.nodes),
            yticklabels = list(G.nodes),
            linecolor = 'k', linewidths = 0.2)
# plt.savefig('无向图的邻接矩阵A.svg')
A @ A

sns.heatmap(A @ A,square = True,
            cmap = 'Blues',
            annot = True, fmt = '.0f',
            xticklabels = list(G.nodes),
            yticklabels = list(G.nodes),
            linecolor = 'k', linewidths = 0.2)
# plt.savefig('A @ A.svg')

A @ A @ A
sns.heatmap(A @ A @ A,square = True, cmap = 'Blues', annot = True)


A + A @ A + A @ A @ A
sns.heatmap(A + A @ A + A @ A @ A,square = True, cmap = 'Blues', annot = True)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 鸡兔互变有向图

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

G = nx.DiGraph()
# 创建有向图的实例

G.add_nodes_from(['Chicken', 'Rabbit'])
# 添加多个顶点

edges_with_weights = [('Chicken','Chicken', 0.7),
                      ('Chicken','Rabbit', 0.3),
                      ('Rabbit','Rabbit', 0.8),
                      ('Rabbit','Chicken', 0.2)]

for u, v, w in edges_with_weights:
    G.add_edge(u, v, weight=w)

# 位置
pos = {'Chicken':(-0.2,0), 'Rabbit': (0.2,0)}


# 可视化
plt.figure(figsize = (16, 16))
nx.draw_networkx(G, pos = pos, node_size = 180, font_size = 20, )
plt.ylim(-0.5,0.5)
plt.xlim(-0.5,0.5)

# 邻接矩阵
A = nx.adjacency_matrix(G).todense()
A

sns.heatmap(A, cmap = 'Blues',
            annot = True, fmt = '.1f',
            xticklabels = list(G.nodes),
            yticklabels = list(G.nodes),
            vmin = 0, vmax = 1,
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('有向图邻接矩阵.svg')

T = A.T
sns.heatmap(T, cmap = 'Blues',
            annot = True, fmt = '.1f',
            xticklabels = list(G.nodes),
            yticklabels = list(G.nodes),
            vmin = 0, vmax = 1,
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('转移矩阵.svg')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 有向图邻接矩阵

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
# 创建有向图的实例
G = nx.DiGraph()

# 添加多个顶点
G.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f'])

G.add_edges_from([('a','c'),('a','f'),
                  ('b','a'),('b','e'),
                  ('c','b'),('c','d'),
                  ('d','a'),('d','e'),
                  ('e','f'),('e','c'),
                  ('f','c'),('f','b')])

# 城市具体位置
pos = {'a':(50.6463,6.5692),
       'b':(70.3019,41.0777),
       'c':(35.6363,36.0673),
       'd':(6.8205,16.8171),
       'e':(26.1044,68.8277),
       'f':(59.1871,69.6398)}

# 可视化
plt.figure(figsize = (6,6))
nx.draw_networkx(G, pos = pos, node_size = 1000, font_size = 20)

# 邻接矩阵
A = nx.adjacency_matrix(G).todense()
A
plt.figure(figsize = (6,6))
sns.heatmap(A, cmap = 'Blues',
            annot = True, fmt = '.0f',
            xticklabels = list(G.nodes),
            yticklabels = list(G.nodes),
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('有向图邻接矩阵.svg')

A @ A
A @ A @ A
A @ A @ A @ A

# 每行元素之和为1
T_T = A /  A.sum(axis=1)
T_T
plt.figure(figsize = (6,6))
sns.heatmap(T_T, cmap = 'Blues', annot = True, fmt = '.1f',
            xticklabels = list(G.nodes),
            yticklabels = list(G.nodes),
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('转移矩阵，每行元素之和为1.svg')


# 每列元素之和为1
A_T = A.T
T = A_T / A_T.sum(axis=0)
plt.figure(figsize = (6,6))
sns.heatmap(T, cmap = 'Blues', annot = True, fmt = '.1f',
            xticklabels = list(G.nodes),
            yticklabels = list(G.nodes),
            linecolor = 'k', square = True,
            linewidths = 0.2)
# plt.savefig('转移矩阵，每列元素之和为1.svg')

T @ T
T @ T @ T
T @ T @ T @ T

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 三种天气状况

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

T = np.matrix([[0.7, 0.45, 0.55],
               [0.25, 0.3,  0.3],
               [0.05, 0.25, 0.15 ]])


A = T.T
# 使用邻接矩阵创建有向图
G = nx.from_numpy_array(A, create_using=nx.DiGraph)
# 修改节点名称
G = nx.relabel_nodes(G, {0: 'Sun', 1: 'Cloud', 2: 'Rain'})
list(G.edges())

plt.figure(figsize = (10, 10))
pos = nx.spring_layout(G,seed=8)
nx.draw_networkx(G, pos = pos, node_size = 880, font_size = 20)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,  font_size = 20)

# 稳态
sstate = np.linalg.eig(T)[1][:,0]
sstate = sstate/sstate.sum()
print(sstate)
# [[0.61538462]
#  [0.26923077]
#   [0.11538462]]

# initial states
initial_x_3 = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
num_iterations = 10

# 图 29. 从晴天经过转移矩阵变换得到的稳态
# 图 30. 从阴天经过转移矩阵变换得到的稳态
# 图 31. 从雨天经过转移矩阵变换得到的稳态
for i in np.arange(0,3):
    initial_x = initial_x_3[:,i][:, None]
    x_i = np.zeros_like(initial_x)
    x_i = initial_x
    X =   initial_x.T

    # matrix power through iterations
    for x in np.arange(0, num_iterations):
        x_i = T@x_i
        X = np.concatenate([X, x_i.T],axis = 0) # (11, 3)

    fig, ax = plt.subplots()
    itr = np.arange(0, num_iterations+1);
    plt.plot(itr,X[:, 0], marker = 'x', color = (1,0,0))
    plt.plot(itr,X[:, 1], marker = 'x', color = (0,0.6,1))
    plt.plot(itr,X[:, 2], marker = 'x', color = (0,0.36,0.635))

    # decorations
    ax.grid(linestyle = '--', linewidth = 0.25, color = [0.5,0.5,0.5])
    ax.set_xlim(0, num_iterations)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Iteration, i')
    ax.set_ylabel('State')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 三种天气状况，随机行走

import numpy as np
import matplotlib.pyplot as plt

def simulate_markov_chain(T, initial_state, num_steps):
    states = [initial_state]
    # 初始状态
    current_state = initial_state
    for _ in range(num_steps):
        # 根据当前状态和转移矩阵决定下一状态
        next_state = np.random.choice(np.arange(len(T)), p=T[current_state])
        states.append(next_state)
        current_state = next_state

    states = np.array(states)
    return states

# 定义转移矩阵
T = np.array([[0.70, 0.45, 0.55],
              [0.25, 0.30, 0.30],
              [0.05, 0.25, 0.15]])

# 模拟马尔科夫链
initial_state = 0  # 假设初始状态为 0
num_steps = 300    # 假设模拟步数
states = simulate_markov_chain(T.T, initial_state, num_steps)

indices = np.arange(num_steps + 1)
labels = ["Sun", "Cloud", "Rain"]

# 图 32. 马尔科夫链随机行走
plt.figure(figsize=(8, 3))
plt.plot(indices, states, c = '0.8')

for state_i, label_i in zip([0, 1, 2], labels):
    plt.scatter(indices[states == state_i], states[states == state_i], label=label_i)
# plt.legend()
plt.xlabel('Iteration')
plt.ylabel('State')
plt.yticks([0, 1, 2], labels)
plt.grid(True)
# plt.savefig('马尔科夫链随机行走.svg')

# 计算0, 1, 2的累积出现频率
cumulative_counts = np.cumsum(np.eye(3)[states], axis=0)
cumulative_probs = cumulative_counts / np.arange(1, len(states) + 1).reshape(-1, 1)
# 图 33. 累积概率变化
plt.figure(figsize=(8, 3))
plt.plot(cumulative_probs, label = labels)
plt.xlabel("Iteration")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
# plt.savefig('累积概率曲线.svg')
plt.show()


states_list, counts_list = np.unique(states, return_counts=True)
prob_list = counts_list/num_steps
prob_list
# 图 34. 马尔科夫链随机行走最终概率结果
plt.figure(figsize=(5, 3))
plt.bar(states_list, prob_list)
plt.xticks(states_list)
plt.xticks([0, 1, 2], labels)
# plt.savefig('最终结果.svg')
























#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
































































































































































































































































































