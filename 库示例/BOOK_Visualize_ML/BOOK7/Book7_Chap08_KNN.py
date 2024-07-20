







#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  kNN分类

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


# import the iris data
iris = datasets.load_iris()

# Only use the first two features: sepal length, sepal width
X = iris.data[:, :2]

# Vector of labels
y = iris.target

# generate mesh
h = .02  # step size in the mesh
x1_min, x1_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
x2_min, x2_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),np.arange(x2_min, x2_max, h))

# Create color maps
rgb = [[255, 238, 255],  # red
       [219, 238, 244],  # blue
       [228, 228, 228]]  # black
rgb = np.array(rgb)/255.
cmap_light = ListedColormap(rgb)

cmap_bold = [[255, 51, 0], [0, 153, 255],[138,138,138]]
cmap_bold = np.array(cmap_bold)/255.

# 近邻数量
k_neighbors = 4

# kNN分类器
clf = neighbors.KNeighborsClassifier(k_neighbors, weights = 'uniform') # distance

# 拟合数据
clf.fit(X, y)

# 查询点
q = np.c_[xx1.ravel(), xx2.ravel()];

# 预测
y_predict = clf.predict(q)

# 规整形状
y_predict = y_predict.reshape(xx1.shape)


# visualization
fig, ax = plt.subplots()

# plot decision regions
plt.contourf(xx1, xx2, y_predict, cmap=cmap_light)

# plot decision boundaries
plt.contour(xx1, xx2, y_predict, levels=[0,1,2], colors=np.array([0, 68, 138])/255.)

# Plot data points
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y], palette=cmap_bold, alpha=1.0, linewidth = 1, edgecolor=[1,1,1])

# Figure decorations
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())
plt.title("k-NN classifier (k = %i, weights = 'uniform')" % (k_neighbors))
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
plt.tight_layout()
plt.axis('equal')
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5 最近质心分类

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.neighbors import NearestCentroid


# import the iris data
iris = datasets.load_iris()

# Only use the first two features: sepal length, sepal width
X = iris.data[:, :2]
y = iris.target

# generate mesh
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

# Create color maps
rgb = [[255, 238, 255],  # red
       [219, 238, 244],  # blue
       [228, 228, 228]]  # black
rgb = np.array(rgb)/255.
cmap_light = ListedColormap(rgb)

cmap_bold = [[255, 51, 0], [0, 153, 255],[138,138,138]]
cmap_bold = np.array(cmap_bold)/255.

# shrinkage: It "shrinks" each of the class centroids toward
# the overall centroid for all classes
# by an amount we call the shrink threshold
for shrinkage in [None, 2, 5, 8]:
    # Create an instance of Neighbours Classifier and fit the data.
    clf = NearestCentroid(metric='euclidean', shrink_threshold=shrinkage)

    # kNN classification, weight = uniform
    clf.fit(X, y)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    # visualization
    fig, ax = plt.subplots()

    # plot decision regions
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # plot decision boundaries
    plt.contour(xx, yy, Z, levels=[0,1,2], colors=['k'])

    # Plot data points
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y], palette=cmap_bold, alpha=1.0, linewidth = 1, edgecolor=[1,1,1])

    # Plot the centroid of each class
    centroids = clf.centroids_;

    sns.scatterplot(x=centroids[:, 0], y=centroids[:, 1], color=".2", marker="+", s=100)

    # calculate the centroid of all sample data
    data_center_x1 = np.mean(X[:, 0]);
    data_center_x2 = np.mean(X[:, 1]);

    # plot the centroid of all sample data
    plt.plot(data_center_x1,data_center_x2, marker="x", color = 'r', markersize=12)

    # Figure decorations
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("NCC, shrink threshold = %r)" % shrinkage)
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    plt.tight_layout()
    plt.axis('equal')
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5 kNN回归

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import StratifiedShuffleSplit

# import the iris data
iris = datasets.load_iris()

# Only use the first two features: sepal length, sepal width
X = iris.data[:, 0]  # sepal length is the input
X = X.reshape(-1, 1) # reshape it to a column vector
y = iris.data[:, 1]  # sepal width is the output
label = iris.target

# Fit regression model
n_neighbors = 8

# Create color maps
cmap_bold = [[255, 51, 0], [0, 153, 255],[138,138,138]]
cmap_bold = np.array(cmap_bold)/255.

for i, weights in enumerate(['uniform', 'distance']):
    fig, ax = plt.subplots()
    for i, class_i in enumerate([0, 1, 2]):
        # split the sample data based on the labels
        idx = (label == class_i)

        # split data based on class
        X_train=X[idx,:]
        y_train=y[idx]

        # kNN regressor
        knn_regress = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)

        # Query points, Q
        Q = np.linspace(X_train.min()-0.2, X_train.max()+0.2, 100)[:, np.newaxis] # (100, 1)

        # Fit the data
        y_predict = knn_regress.fit(X_train, y_train)

        # Predict based on the regression model
        y_predict = knn_regress.predict(Q) # (100,)

        # Visualizations
        plt.scatter(X_train, y_train, color=cmap_bold[i,:], label=iris.target_names[i])
        plt.plot(Q, y_predict, color = cmap_bold[i,:]*0.75)
        plt.axis('tight')
        plt.title("kNN regressor, k = %i, weights = '%s')" % (n_neighbors,weights))

    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
    plt.tight_layout()
    plt.axis('equal')
    plt.legend()
    plt.show()


















































































































































































































































































































































