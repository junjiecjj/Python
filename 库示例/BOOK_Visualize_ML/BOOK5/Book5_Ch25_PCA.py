#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 15:19:05 2024

@author: jack
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

mean = [1, 2]
# center of data
cov = [[1, 1], [1, 1.5]]
# covariance matrix

X = np.random.multivariate_normal(mean, cov, 500)

# 图 20. 原始二维数据 X
fig, ax = plt.subplots()
plt.scatter(X[:, 0], X[:, 1], alpha = 0.5, marker = '.')

plt.axvline(x=mean[0], color='r', linestyle='--')
plt.axhline(y=mean[1], color='r', linestyle='--')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('scaled')

ax.set_xlim([-3,5])
ax.set_ylim([-2,6])

X_df = pd.DataFrame(X, columns=['x_1', 'x_2'])
# 图 20. 原始二维数据 X
sns.jointplot(data=X_df,x = 'x_1', y = 'x_2', kind = 'kde', fill = True, xlim = (-3,5), ylim = (-2,6))
ax.set_aspect('equal')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()



#%% PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
# 图 21. 主成分数据分布
fig, ax = plt.subplots()
plt.scatter(X[:, 0], X[:, 1], alpha = 0.5, marker = '.')
plt.axvline(x=mean[0], color='r', linestyle='--')
plt.axhline(y=mean[1], color='r', linestyle='--')

# plot first principal component, PC1
PC1_x = pca.components_[0,0]
PC1_y = pca.components_[0,1]
ax.quiver(mean[0], mean[1], PC1_x, PC1_y, angles='xy', scale_units='xy',scale=1/3, edgecolor='none', facecolor= 'b')

# plot second principal component, PC2
PC2_x = pca.components_[1,0]
PC2_y = pca.components_[1,1]
ax.quiver(mean[0], mean[1], PC2_x,PC2_y, angles='xy', scale_units='xy',scale=1/3, edgecolor='none', facecolor= 'r')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('scaled')
ax.set_xlim([-3,5])
ax.set_ylim([-2,6])
plt.show()

# 图 21. 主成分数据分布
# convert X to Z
Z = pca.transform(X)
Z_df = pd.DataFrame(Z, columns=['z_1', 'z_2'])

fig, ax = plt.subplots()

sns.kdeplot(data=Z_df)
sns.rugplot(data=Z_df)
plt.show()

#### 图 22. 数据 Y 在 [v1, v2] 中散点图
fig, ax = plt.subplots()
plt.scatter(Z[:, 0], Z[:, 1], alpha = 0.5, marker = '.')
plt.axvline(x=0, color='r', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('$z_1$')
plt.ylabel('$z_2$')
plt.axis('scaled')
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
plt.show()

#### 图 22. 数据 Y 在 [v1, v2] 中散点图
# fig, ax = plt.subplots()
sns.jointplot(data=Z_df, x = 'z_1', y = 'z_2', kind = 'kde', fill = True, xlim = (-5,5), ylim = (-5,5), ax = ax)
ax.set_aspect('equal')
plt.xlabel('$z_1$')
plt.ylabel('$z_2$')
# plt.show()
#%% dimension reduction

pca_PC1 = PCA(n_components=1)
pca_PC1.fit(X)

z1 = pca_PC1.transform(X)
x1_proj = pca_PC1.inverse_transform(z1)
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], alpha = 0.5, marker = '.')
# plot first principal component, PC1

PC1_x = pca_PC1.components_[0,0]
PC1_y = pca_PC1.components_[0,1]

ax.quiver(mean[0], mean[1], PC1_x, PC1_y, angles='xy', scale_units='xy', scale=1/3, edgecolor='none', facecolor= 'b')
plt.scatter(x1_proj[:, 0], x1_proj[:, 1], alpha=0.5, c = 'k', marker = 'x')

plt.axvline(x=mean[0], color='r', linestyle='--')
plt.axhline(y=mean[1], color='r', linestyle='--')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('scaled')

ax.set_xlim([-3,5])
ax.set_ylim([-2,6])

plt.show()


#%% 3D特征值分解投影
# 方式一
plt.subplot(projection='3d')

# # 方式二
# # get current axes
# plt.figure(figsize=(14, 10))
# axes = plt.gca(projection='3d')

x = np.linspace(0, 10, 20)
y = np.linspace(2, 8, 20)
z = 2*x + 5*y + 3
plt.plot(x, y, z)

# https://www.cnblogs.com/shanger/p/13201139.html
plt.subplot(projection='3d')
# 三维平面, 要求X,Y都是二维的
x = np.linspace(0, 10, 20)
y = np.linspace(2, 8, 20)
X,Y = np.meshgrid(x, y)
Z = 2*X + 5*Y + 3

plt.figure(figsize=(14, 10))
axes = plt.gca(projection='3d')
axes.plot_surface(X, Y, Z, color = 'b', alpha = 0.2)

axes.plot(x, y, z, c='k', lw=2,)


#%% https://blog.csdn.net/soaryy/article/details/82884691
# https://blog.csdn.net/red_stone1/article/details/70260070
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
import matplotlib
from matplotlib.font_manager import FontProperties
from pylab import tick_params
import copy
from matplotlib.pyplot import MultipleLocator

font = FontProperties(fname="/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", size=14)
fontpath = "/usr/share/fonts/truetype/windows/"
fontpath1 = "/usr/share/fonts/truetype/msttcorefonts/"
fontpath2 = "/usr/share/fonts/truetype/NerdFonts/"

x = np.linspace(2, 10, 20)
y = np.linspace(0, 10, 20)
XX, YY = np.meshgrid(x, y)
Z = XX + YY - 26

# Load the iris data
iris_sns = sns.load_dataset("iris")
# A copy from Seaborn
iris = load_iris()
# A copy from Sklearn

X = iris.data
label = iris.target
feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$', 'Petal length, $X_3$','Petal width, $X_4$']
# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)
X = X_df.iloc[:, [0,1,2]].to_numpy()


w = np.array([1, 1, -1]).reshape(-1,1)
b = -26
Xq = X.T - (w.T@X.T + b) * w/(w.T@w)
Xq = Xq.T

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(projection='3d')

rainbow = plt.get_cmap("rainbow")
ax.scatter(X[:,0], X[:,1], X[:,2],  s = 15,  c = label, cmap=rainbow)
ax.scatter(Xq[:,0], Xq[:,1], Xq[:,2],  s = 15,  c = label, cmap=rainbow)

ax.plot_surface(XX, YY, Z, color = 'b', alpha = 0.2)

ax.set_proj_type('ortho')
# ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) # 3D坐标区的背景设置为白色
# ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
# ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
font3 = FontProperties(fname=fontpath1+"Times_New_Roman.ttf", size = 22)
ax.set_xlabel('$\it{x_1}$', fontproperties=font3)
ax.set_ylabel('$\it{x_2}$', fontproperties=font3)
ax.set_zlabel('$\it{f}$($\it{x_1}$,$\it{x_2}$)', fontproperties=font3)

# ax.set_xlim(X[:,0].min()-4, X[:,0].max()+4)
# ax.set_ylim(X[:,1].min()-4, X[:,1].max()+4)
# ax.set_zlim(X[:,2].min()-4, X[:,2].max()+4)

ax.view_init(azim=-160, elev=30)
ax.grid(False)
plt.show()





























































































































































