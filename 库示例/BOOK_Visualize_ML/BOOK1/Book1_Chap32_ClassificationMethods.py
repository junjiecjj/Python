





import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

# matplotlib.use('Agg')
# matplotlib.rcParams['text.usetex'] = True
filepath2 = '/home/jack/snap/'
font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)

fontpath = "/usr/share/fonts/truetype/windows/"

fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"

fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% kNN分类

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

# 导入并整理数据
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# df_iris = sns.load_dataset("iris")
# df_iris['size'] = 60


# 生成网格化数据
x1_array = np.linspace(4,8,101)
x2_array = np.linspace(1,5,101)
xx1, xx2 = np.meshgrid(x1_array,x2_array)


# 创建色谱
rgb = [[255, 238, 255],
       [219, 238, 244],
       [228, 228, 228]]
rgb = np.array(rgb)/255.
cmap_light = ListedColormap(rgb)
cmap_bold = [[255, 51, 0],
             [0, 153, 255],
             [138,138,138]]
cmap_bold = np.array(cmap_bold)/255.0

k_neighbors = 4 # 定义kNN近邻数量k
# 创建kNN分类器对象
kNN = neighbors.KNeighborsClassifier(k_neighbors)
kNN.fit(X, y) # 用训练数据训练kNN

q = np.c_[xx1.ravel(), xx2.ravel()]  # (10201, 2)
# 用kNN对一系列查询点进行预测
y_predict = kNN.predict(q)
y_predict = y_predict.reshape(xx1.shape) #  (101, 101)



# 可视化
fig, ax = plt.subplots(figsize = (10,10))
plt.contourf(xx1, xx2, y_predict, cmap=cmap_light)
plt.contour(xx1, xx2, y_predict, levels=[0,1,2], colors=np.array([0, 68, 138])/255.)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y], ax = ax, palette=dict(setosa=cmap_bold[0,:], versicolor=cmap_bold[1,:], virginica=cmap_bold[2,:]), alpha=1.0, linewidth = 1, edgecolor=[1,1,1],  legend="full")
plt.xlim(4, 8); plt.ylim(1, 5)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_aspect('equal', adjustable='box')
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 高斯朴素贝叶斯

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.naive_bayes import GaussianNB

# 导入并整理数据
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# 生成网格化数据
x1_array = np.linspace(4,8,101)
x2_array = np.linspace(1,5,101)
xx1, xx2 = np.meshgrid(x1_array, x2_array)


# 创建色谱
rgb = [[255, 238, 255],
       [219, 238, 244],
       [228, 228, 228]]
rgb = np.array(rgb)/255.
cmap_light = ListedColormap(rgb)
cmap_bold = [[255, 51, 0],
             [0, 153, 255],
             [138,138,138]]
cmap_bold = np.array(cmap_bold)/255.


# 创建高斯朴素贝叶斯分类器对象
gnb = GaussianNB()
gnb.fit(X, y)
# 用高斯朴素贝叶斯分类器对一系列查询点进行预测
q = np.c_[xx1.ravel(), xx2.ravel()]
y_predict = gnb.predict(q)
y_predict = y_predict.reshape(xx1.shape)


# 可视化
fig, ax = plt.subplots(figsize = (10,10))
plt.contourf(xx1, xx2, y_predict, cmap=cmap_light)
plt.contour(xx1, xx2, y_predict, levels=[0,1,2], colors=np.array([0, 68, 138])/255.)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y], ax = ax, palette=dict(setosa=cmap_bold[0,:], versicolor=cmap_bold[1,:], virginica=cmap_bold[2,:]), alpha=1.0, linewidth = 1, edgecolor=[1,1,1])
plt.xlim(4, 8); plt.ylim(1, 5)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_aspect('equal', adjustable='box')
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 支持向量机，线性核

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import svm

# 导入并整理数据
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target



# 生成网格化数据
x1_array = np.linspace(4,8,101)
x2_array = np.linspace(1,5,101)
xx1, xx2 = np.meshgrid(x1_array,x2_array)


# 创建色谱
rgb = [[255, 238, 255],
       [219, 238, 244],
       [228, 228, 228]]
rgb = np.array(rgb)/255.
cmap_light = ListedColormap(rgb)
cmap_bold = [[255, 51, 0],
             [0, 153, 255],
             [138,138,138]]
cmap_bold = np.array(cmap_bold)/255.


q = np.c_[xx1.ravel(), xx2.ravel()]

# 创建支持向量机 (线性核) 分类器对象
SVM = svm.SVC(kernel='linear')
# 用训练数据训练kNN
SVM.fit(X, y)
# 用支持向量机 (线性核) 分类器对一系列查询点进行预测
y_predict = SVM.predict(q)
y_predict = y_predict.reshape(xx1.shape)

# 可视化
fig, ax = plt.subplots(figsize = (10,10))
plt.contourf(xx1, xx2, y_predict, cmap=cmap_light)
plt.contour(xx1, xx2, y_predict, levels=[0,1,2], colors=np.array([0, 68, 138])/255.)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y], ax = ax, palette=dict(setosa=cmap_bold[0,:], versicolor=cmap_bold[1,:], virginica=cmap_bold[2,:]), alpha=1.0, linewidth = 1, edgecolor=[1,1,1])
plt.xlim(4, 8); plt.ylim(1, 5)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_aspect('equal', adjustable='box')
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 支持向量机，高斯核

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import svm

# 导入并整理数据
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target


# 生成网格化数据
x1_array = np.linspace(4,8,101)
x2_array = np.linspace(1,5,101)
xx1, xx2 = np.meshgrid(x1_array,x2_array)

# 创建色谱
rgb = [[255, 238, 255],
       [219, 238, 244],
       [228, 228, 228]]
rgb = np.array(rgb)/255.
cmap_light = ListedColormap(rgb)
cmap_bold = [[255, 51, 0],
             [0, 153, 255],
             [138,138,138]]
cmap_bold = np.array(cmap_bold)/255.

q = np.c_[xx1.ravel(), xx2.ravel()]

# 创建支持向量机 (高斯核) 分类器对象
SVM = svm.SVC(kernel='rbf', gamma= 'auto')
# 用训练数据训练kNN
SVM.fit(X, y)
# 用支持向量机 (高斯核) 分类器对一系列查询点进行预测
y_predict = SVM.predict(q)
y_predict = y_predict.reshape(xx1.shape)

# 可视化
fig, ax = plt.subplots(figsize = (10,10))
plt.contourf(xx1, xx2, y_predict, cmap=cmap_light)
plt.contour(xx1, xx2, y_predict, levels=[0,1,2], colors=np.array([0, 68, 138])/255.)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=iris.target_names[y], ax = ax, palette=dict(setosa=cmap_bold[0,:], versicolor=cmap_bold[1,:], virginica=cmap_bold[2,:]), alpha=1.0, linewidth = 1, edgecolor=[1,1,1])
plt.xlim(4, 8); plt.ylim(1, 5)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])
ax.set_aspect('equal', adjustable='box')
font2 = {'family': 'Times New Roman', 'style': 'normal', 'size': 22}
plt.legend(loc='best', borderaxespad=0, edgecolor='black', prop=font2,)




































































































































































































































































































