


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% K均值聚类

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans

# Create color maps
rgb = [[255, 238, 255],  # red
       [219, 238, 244],  # blue
       [228, 228, 228]]  # black
rgb = np.array(rgb)/255.

cmap_light = ListedColormap(rgb)


# 导入鸢尾花数据
iris = datasets.load_iris()

# Only use the first two features: sepal length, sepal width
# 取出鸢尾花前两个特征
X_train = iris.data[:, :2]


# Vector of labels
y_train = iris.target

# 创建KMeans对象
kmeans = KMeans(n_clusters=3, n_init = 'auto')


# 使用KMeans算法训练数据
kmeans.fit(X_train)

# 生成网格数据
plot_step = 0.02
xx, yy = np.meshgrid(np.linspace(4, 8, int(4/plot_step + 1)),
                     np.linspace(1.5, 4.5, int(3/plot_step + 1)))
# 使用KMeans模型对网格中的点进行预测
# 并将预测结果整形成与网格相同形状的矩阵
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
fig, ax = plt.subplots()

# plot regions
plt.contourf(xx, yy, Z, cmap=cmap_light)

# plot sample data
plt.scatter(x=X_train[:, 0], y=X_train[:, 1], color=np.array([0, 68, 138])/255., alpha=1.0,
                linewidth = 1, edgecolor=[1,1,1])

# plot decision boundaries
plt.contour(xx, yy, Z, levels=[0,1,2], colors=np.array([0, 68, 138])/255.)

# plot centroids
centroids = kmeans.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=100, linewidths=1.5,
            color="k")

ax.set_xticks(np.arange(4, 8.5, 0.5))
ax.set_yticks(np.arange(1.5, 5, 0.5))
ax.set_xlim(4, 8)
ax.set_ylim(1.5, 4.5)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_aspect('equal')
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 肘部法则
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans

# import the iris data
iris = datasets.load_iris()

# Only use the first two features: sepal length, sepal width
X_train = iris.data[:, :2]

# Vector of labels
y_train = iris.target

distortions = []
for i in range(1, 11):

    # GK-Means
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0)
    # train the parameters
    km.fit(X_train)
    distortions.append(km.inertia_)

fig, ax = plt.subplots()
plt.plot(range(1, 11), distortions, marker='x')
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
ax.set_xticks(range(1, 11))
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 轮廓图

from sklearn import datasets
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import ListedColormap



# Create color maps
rgb = [[255, 238, 255],  # red
       [219, 238, 244],  # blue
       [228, 228, 228]]  # black
rgb = np.array(rgb)/255.

cmap_light = ListedColormap(rgb)


# import the iris data
iris = datasets.load_iris()

# Only use the first two features: sepal length, sepal width
X = iris.data[:, :2]

plot_step = 0.02
xx, yy = np.meshgrid(np.arange(4, 8+plot_step, plot_step),
                     np.arange(1.5, 4.5+plot_step, plot_step))

range_n_clusters = [3, 4, 5]




for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, ax= plt.subplots()

    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = kmeans.fit_predict(X)

    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')

    visualizer.fit(X)
    # Fit the data to the visualizer
    visualizer.show()
    # Finalize and render the figure

    # Labeling the clusters
    centers = kmeans.cluster_centers_
    # Generate mesh

    # predict clusters
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)


    fig, ax = plt.subplots()

    plt.plot(centers[0,:],centers[1,:],'x')
    # plot regions
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # plot sample data
    plt.scatter(x=X[:, 0], y=X[:, 1], color=np.array([0, 68, 138])/255., alpha=1.0,
                    linewidth = 1, edgecolor=[1,1,1])

    # plot decision boundaries
    levels = np.unique(Z).tolist();
    plt.contour(xx, yy, Z, levels=levels,colors='r')

    # plot centroids
    centroids = kmeans.cluster_centers_

    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=100, linewidths=1.5,
                color="r")
    ax.set_xticks(np.arange(4, 8.5, 0.5))
    ax.set_yticks(np.arange(1.5, 5, 0.5))
    ax.set_xlim(4, 8)
    ax.set_ylim(1.5, 4.5)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    ax.set_aspect('equal')
    plt.show()





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 沃罗诺伊图


import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d






# generate data/speed values
points = np.random.uniform(size=[30, 2])
points = np.append(points, [[999,999], [-999,999], [999,-999], [-999,-999]], axis = 0)
speed = np.random.uniform(low=0.0, high=5.0, size=50)





# generate Voronoi tessellation
vor = Voronoi(points)


# find min/max values for normalization
minima = min(speed)
maxima = max(speed)


# normalize chosen colormap
norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)


fig, ax = plt.subplots()

# plot Voronoi diagram, and fill finite regions with color mapped from speed value
voronoi_plot_2d(vor, show_points=True, show_vertices=False, s=1, ax = ax)
for r in range(len(vor.point_region)):
    region = vor.regions[vor.point_region[r]]
    if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon), color=mapper.to_rgba(speed[r]), alpha=0.5)
ax.set_aspect('equal', 'box')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()










































































































































































































































































































































































































































































































































































































