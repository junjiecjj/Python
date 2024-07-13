

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 最小生成树


import networkx as nx
import matplotlib.pyplot as plt


# 构造图
G = nx.Graph()

# 创建边
G.add_edges_from([(0, 1, {"weight": 4}),
                  (0, 7, {"weight": 8}),
                  (1, 7, {"weight": 11}),
                  (1, 2, {"weight": 8}),
                  (2, 8, {"weight": 2}),
                  (2, 5, {"weight": 4}),
                  (2, 3, {"weight": 7}),
                  (3, 4, {"weight": 9}),
                  (3, 5, {"weight": 14}),
                  (4, 5, {"weight": 10}),
                  (5, 6, {"weight": 2}),
                  (6, 8, {"weight": 6}),
                  (7, 8, {"weight": 7})])


# Visualize the graph and the minimum spanning tree
pos = nx.spring_layout(G)

# 可视化
nx.draw_networkx_nodes(G,
                       pos,
                       node_color="0.8",
                       node_size=500)

edge_labels = nx.get_edge_attributes(G,
                                     "weight")

nx.draw_networkx_edges(G,
                       pos,
                       edge_color=edge_labels.values(),
                       edge_cmap=plt.cm.RdYlBu_r,
                       width = 3)

nx.draw_networkx_labels(G,
                        pos,
                        font_size=12,
                        font_family="sans-serif")

edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G,
                             pos,
                             edge_labels=edge_labels)

plt.axis("off")
# plt.savefig('图.svg')

# 找到最小生成树
T = nx.minimum_spanning_tree(G)



# Visualize the graph and the minimum spanning tree
pos = nx.spring_layout(G)

# 可视化
nx.draw_networkx_nodes(G,
                       pos,
                       node_color="0.8",
                       node_size=500)

edge_labels = nx.get_edge_attributes(G,
                                     "weight")

nx.draw_networkx_edges(G,
                       pos,
                       edge_color=edge_labels.values(),
                       edge_cmap=plt.cm.RdYlBu_r,
                       width = 3)

nx.draw_networkx_labels(G,
                        pos,
                        font_size=12,
                        font_family="sans-serif")

edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G,
                             pos,
                             edge_labels=edge_labels)

nx.draw_networkx_edges(T,
                       pos,
                       style = '--',
                       edge_color="black",
                       width=1)
plt.axis("off")
# plt.savefig('最小生成树.svg')




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 树形图

# initializations
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats



# load historical price levels for 12 stocks
list_tickers = ['TSLA','WMT','MCD','USB',
                'YUM','NFLX','JPM','PFE',
                'F','GM','COST','JNJ']

stock_levels_df = yf.download(list_tickers, start='2020-01-01', end='2020-5-31')
stock_levels_df.to_pickle('stock_levels_df.pkl')
stock_levels_df.to_csv("stock_levels_df.csv")


stock_levels_df.round(2).head()

# calculate daily returns
daily_returns_df = stock_levels_df['Adj Close'].pct_change()




#%% Lineplot of stock prices
sns.set_style("whitegrid")
sns.set_theme(font = 'Times New Roman')



# normalize the initial stock price levels to 1
normalized_stock_levels = stock_levels_df['Adj Close']/stock_levels_df['Adj Close'].iloc[0]

g = sns.relplot(data=normalized_stock_levels,dashes = False,
                kind="line") # , palette="coolwarm"
g.set_xlabels('Date')
g.set_ylabels('Adjusted closing price')
g.set_xticklabels(rotation=45)



#%% Heatmap of correlation matrix
fig, ax = plt.subplots()
# Compute the correlation matrix
corr_P = daily_returns_df.corr()

sns.heatmap(corr_P, cmap="coolwarm",
            square=True, linewidths=.05,
            annot=True)



#%% Cluster map based on correlation
g = sns.clustermap(corr_P, cmap="coolwarm",
                   annot=True)
g.ax_row_dendrogram.remove()





















#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





































































































































































































































































































































