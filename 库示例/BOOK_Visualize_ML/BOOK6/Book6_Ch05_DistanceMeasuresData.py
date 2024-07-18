

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 欧氏距离

from scipy.spatial import distance
import numpy as np


x_i = (0, 0, 0) # data point
q   = (4, 8, 6) # query point

# calculate Euclidean distance
dst_1 = distance.euclidean(x_i, q)

dst_2 = np.linalg.norm(np.array(x_i) - np.array(q))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 成对欧氏距离

from sklearn.metrics.pairwise import euclidean_distances

# Sample data points
X = [[-5, 0], [4, 3], [3, -4]]

# Query point
q = [[0, 0]]


# pairwise distances between rows of X and q
dst_pairwise_X_q = euclidean_distances(X, q)
print('Pairwise distances between X and q')
print(dst_pairwise_X_q)


# pairwise distances between rows of X and itself
dst_pairwise_X_X = euclidean_distances(X, X)
print('Pairwise distances between X and X')
print(dst_pairwise_X_X)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 标准欧氏距离
from scipy.spatial import distance
import numpy as np

# Variance-covariance matrix
SIGMA = np.array([[2, 1], [1, 2]])

q   = [0, 0];       # query point
x_1 = [-3.5, -4];   # data point 1
x_2 = [2.75, -1.5]; # data point 1

# Calculate standardized Euclidean distances

d_1 = distance.seuclidean(q, x_1, np.diag(SIGMA)) # np.sqrt((3.5/1.414)**2 + (4/1.414)**2)
d_2 = distance.seuclidean(q, x_2, np.diag(SIGMA))

# Note1: V is an 1-D array of component variances

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 马氏距离
from scipy.spatial import distance
import numpy as np
from numpy.linalg import inv

# Variance-covariance matrix
SIGMA = np.array([[2, 1], [1, 2]])

q   = [0, 0];       # query point
x_1 = [-3.5, -4];   # data point 1
x_2 = [2.75, -1.5]; # data point 1

# Calculate Mahal distances
d_1 = distance.mahalanobis(q, x_1, inv(SIGMA))
d_2 = distance.mahalanobis(q, x_2, inv(SIGMA))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 城市街区距离
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances

# Sample data points
X = [[-5, 0], [4, 3], [3, -4]]

# Query point
q = [0, 0]

# Compute the City Block (Manhattan) distance.
d_1 = distance.cityblock(q, X[0])
d_2 = distance.cityblock(q, X[1])
d_3 = distance.cityblock(q, X[2])

# pairwise distances between rows of X and q
dst_pairwise_X_q = pairwise_distances(X, [q], metric='cityblock')
print('Pairwise City Block distances between X and q')
print(dst_pairwise_X_q)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 切比雪夫距离
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances

# Sample data points
X = [[-5, 0], [4, 3], [3, -4]]

# Query point
q = [0, 0]

# Compute the Chebyshev distance.
d_1 = distance.chebyshev(q, X[0])
d_2 = distance.chebyshev(q, X[1])
d_3 = distance.chebyshev(q, X[2])

# pairwise distances between rows of X and q
dst_pairwise_X_q = pairwise_distances(X, [q], metric='chebyshev')
print('Pairwise Chebyshev distances between X and q')
print(dst_pairwise_X_q)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 闵氏距离 (Minkowski distance) 类似 Lp范数，对应定义如下：
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances

# Sample data points
X = [[-5, 0], [4, 3], [3, -4]]

# Query point
q = [0, 0]

# Compute the Chebyshev distance.
d_1 = distance.minkowski(q, X[0], p = 2)
d_2 = distance.minkowski(q, X[1], p = 2)
d_3 = distance.minkowski(q, X[2], p = 2)

# pairwise distances between rows of X and q
dst_pairwise_X_q = pairwise_distances(X, [q], metric='minkowski',p = 2)
print('Pairwise Chebyshev distances between X and q')
print(dst_pairwise_X_q)




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 余弦相似性
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

x = np.array([[8, 2]])
q = np.array([[7, 9]])

k_x_q = cosine_similarity(x,q)
print(k_x_q)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 余弦距离
import numpy as np
from scipy.spatial import distance

x = np.array([8, 2])
q = np.array([7, 9])

d_x_q = distance.cosine(x,q)
print(d_x_q)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 相关性距离
import numpy as np
from scipy.spatial import distance

x = np.array([8, 2])
q = np.array([7, 9])

d_x_q = distance.correlation(x,q)
print(d_x_q)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 成对欧氏距离

from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# define sample data
X = np.array([[1,6], [4,6], [1,5], [6,0], [3,8], [8,3], [4,1], [3,5], [9, 2], [5, 9], [4, 9], [8, 4]])
# define labels
labels = ['a','b','c','d','e','f','g','h','i','j','k','l']

fig, ax = plt.subplots()
# plot scatter of sample data
plt.scatter(x = X[:, 0], y = X[:, 1], color = np.array([0, 68, 138])/255., alpha = 1.0, linewidth = 1, edgecolor = [1,1,1])

for i, (x,y) in enumerate(zip(X[:, 0], X[:, 1])):
    # add labels to the sample data
    label = labels[i] + f"({x},{y})"
    plt.annotate(label, xy = X[i], xytext = (-10, 10), textcoords="offset points", ha='center')

ax.set_xticks(np.arange(0, 11, 1))
ax.set_yticks(np.arange(0, 11, 1))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
plt.xlabel('x_1')
plt.ylabel('x_2')
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_aspect('equal')
plt.show()



Pairwise_d = euclidean_distances(X)
fig, ax = plt.subplots()
h = sns.heatmap(Pairwise_d, cmap="coolwarm", square=True, linewidths=.05, annot=True, xticklabels = labels, yticklabels = labels)

# lower triangle for heatmap of RBF affinity matrix
fig, ax = plt.subplots()
mask = np.triu(np.ones_like(Pairwise_d, dtype=bool))
h = sns.heatmap(Pairwise_d, cmap="coolwarm", mask = mask, square=True, linewidths=.05, annot=True, xticklabels = labels, yticklabels = labels)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




























































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


























#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%











































































































































































































































































