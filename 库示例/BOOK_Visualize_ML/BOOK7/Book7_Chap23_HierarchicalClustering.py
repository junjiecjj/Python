




# 不同簇间距离

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap




p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
# p["ytick.minor.visible"] = True
# p["xtick.minor.visible"] = True
# p["axes.grid"] = True
# p["grid.color"] = "0.5"
# p["grid.linewidth"] = 0.5


# Create color maps
rgb = [[255, 238, 255],  # red
       [219, 238, 244],  # blue
       [228, 228, 228]]  # black
rgb = np.array(rgb)/255.

cmap_light = ListedColormap(rgb)
iris = load_iris()
X = iris.data[:, :2]

clustering_algorithms = (
    ('Single linkage', 'single'),
    ('Average linkage', 'average'),
    ('Complete linkage', 'complete'),
    ('Ward linkage', 'ward'),
)
for name, method in clustering_algorithms:

    # 绘制树形图
    fig, ax = plt.subplots()

    plt.title(name)
    dend = dendrogram(linkage(X,
                              method = method))

    # 层次聚类
    cluster = AgglomerativeClustering(n_clusters=3,
                                      metric='euclidean',
                                      linkage=method)

    # 完成聚类预测
    Z = cluster.fit_predict(X)

    # 可视化聚类结果
    fig, ax = plt.subplots()
    plt.title(name)

    # 可视化散点图
    plt.scatter(x=X[:, 0], y=X[:, 1], c=Z, alpha=1.0,
                    linewidth = 1, edgecolor=[1,1,1])

    ax.set_xticks(np.arange(4, 8.5, 0.5))
    ax.set_yticks(np.arange(1.5, 5, 0.5))
    ax.set_xlim(4, 8)
    ax.set_ylim(1.5, 4.5)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    ax.set_aspect('equal')
    plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 亲近度层次度量

from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets



p = plt.rcParams
p["font.sans-serif"] = ["Roboto"]
p["font.weight"] = "light"
p["ytick.minor.visible"] = True
p["xtick.minor.visible"] = True
p["axes.grid"] = True
p["grid.color"] = "0.5"
p["grid.linewidth"] = 0.5


# load iris data
iris = datasets.load_iris()
X = iris.data[:, [0,1]]

# generate pairwise RBF affinity matrix
rbf_X = rbf_kernel(X)


# heatmap of RBF affinity matrix
fig, ax = plt.subplots()

sns.heatmap(rbf_X, cmap="RdYlBu_r",
            square=True)

# lower triangle for heatmap of RBF affinity matrix
fig, ax = plt.subplots()

mask = np.triu(np.ones_like(rbf_X, dtype=bool))

sns.heatmap(rbf_X, cmap="RdYlBu_r",
            mask = mask,
            square=True)



# fig, ax = plt.subplots()

g = sns.clustermap(rbf_X, cmap="RdYlBu_r")
g.ax_row_dendrogram.remove()






















































































































































































































































































































































































































































































































































































































































