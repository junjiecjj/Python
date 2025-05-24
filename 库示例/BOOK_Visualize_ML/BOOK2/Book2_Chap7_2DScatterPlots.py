





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 使用matplotlib.pyplot.scatter()绘制平面散点图

# 导入包
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
# 导入色谱

import os

# 规则网格，低颗粒度
num_points = 21
XX,YY = np.meshgrid(np.linspace(-2, 2, num_points),np.linspace(-2, 2, num_points))
# 指定特定函数
color_function = XX * np.exp(-(XX**2 + YY**2))

#>>>>>>>>>>>>>>>>>>>>>>>>  使用 红黄蓝_翻转 色谱
fig, ax = plt.subplots(figsize=(5,5))
# plt.scatter(XX, YY, c = color_function, s = 20, cmap='RdYlBu_r')
# 或者
plt.scatter(XX, YY, c = color_function, s = 12, cmap=cm.RdYlBu_r)

ax.set_xlim(-2.05, 2.05)
ax.set_ylim(-2.05, 2.05)

ax.set_xticks(np.linspace(-2, 2, 6))
ax.set_yticks(np.linspace(-2, 2, 6))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.yaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$', rotation = 0)

plt.grid(color = (0.8, 0.8, 0.8))
ax.set_axisbelow(True)

#>>>>>>>>>>>>>>>>>>>>>>>>  规则网格，高颗粒度
num_points = 51
XX,YY = np.meshgrid(np.linspace(-2, 2, num_points),np.linspace(-2, 2, num_points))
color_function = XX * np.exp(-(XX**2 + YY**2))
# 指定特定函数

fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(XX, YY, c = color_function, s = 20, cmap='RdYlBu_r') # 使用 红黄蓝_翻转 色谱

ax.set_xlim(-2.05, 2.05)
ax.set_ylim(-2.05, 2.05)

ax.set_xticks(np.linspace(-2, 2, 6))
ax.set_yticks(np.linspace(-2, 2, 6))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.yaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$', rotation = 0)

plt.grid(color = (0.8, 0.8, 0.8))
ax.set_axisbelow(True)

#>>>>>>>>>>>>>>>>>>>>>>>>  同时调整颜色和大小
num_points = 21
XX,YY = np.meshgrid(np.linspace(-2, 2, num_points),np.linspace(-2, 2, num_points))

random_color = np.random.random(XX.shape)
# numpy.random.random() 返回 [0.0, 1.0) 区间连续均匀随机数


fig, ax = plt.subplots(figsize=(5,5))
plt.scatter(XX, YY, c = random_color, s = random_color*50, cmap=cm.RdYlBu_r) # 使用 红黄蓝_翻转 色谱

ax.set_xlim(-2.05,2.05)
ax.set_ylim(-2.05,2.05)

ax.set_xticks(np.linspace(-2,2,6))
ax.set_yticks(np.linspace(-2,2,6))

ax.yaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$', rotation = 0)

plt.grid(color = (0.8,0.8,0.8))
ax.set_axisbelow(True)

#>>>>>>>>>>>>>>>>>>>>>>>>  不规则的散点
num = 200
x = np.random.rand(num) * 4 - 2
y = np.random.rand(num) * 4 - 2
colors = np.random.rand(num) # (200,)
area = (30 * np.random.rand(num))**2  # (200,)

fig, ax = plt.subplots(figsize=(5,5))

plt.scatter(x, y, s=area, c=colors, alpha=0.5, cmap = 'RdYlBu_r')

ax.set_xlim(-2.05,2.05)
ax.set_ylim(-2.05,2.05)

ax.set_xticks(np.linspace(-2,2,6))
ax.set_yticks(np.linspace(-2,2,6))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.yaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$', rotation = 0)

plt.grid(color = (0.8,0.8,0.8))
ax.set_axisbelow(True)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 使用seaborn.heatmap()绘制平面散点图

# 导入包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.datasets import load_iris

# 从seaborn中导入鸢尾花样本数据
# iris_sns = sns.load_dataset("/home/jack/seaborn-data/iris")

iris_sns = pd.read_csv("/home/jack/seaborn-data/iris.csv")
iris_sns.head()
# 打印数据帧前5行

# 打印数据帧列名
iris_sns.columns

iris_sns.species.unique()

# 鸢尾花数据统计特征
iris_sns.describe()

#>>>>>>>>>>>>>>>>>>>>>>>>  2. 绘制样本数据散点图，不加标签
fig, ax = plt.subplots()
# 利用 seaborn.scatterplot() 绘制散点图
# x对应横轴特征，鸢尾花数据帧列名 "sepal_length"
# y对应纵轴特征，鸢尾花数据帧列名 "sepal_width"
ax = sns.scatterplot(data = iris_sns, x = "sepal_length", y = "sepal_width")

# 设置横纵轴标签
ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')


ax.set_xticks(np.arange(4, 8 + 1, step = 1))
ax.set_yticks(np.arange(1, 5 + 1, step = 1))
# 设置横纵轴刻度

ax.axis('scaled')
# 设定横纵轴尺度1:1

ax.grid(linestyle = '--', linewidth = 0.25, color = [0.7,0.7,0.7])
# 增加刻度网格，颜色为浅灰 (0.8,0.8,0.8)

ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)
# 设置横纵轴取值范围

#>>>>>>>>>>>>>>>>>>>>>>>> 3. 调转横、纵轴特征
fig, ax = plt.subplots()
ax = sns.scatterplot(data = iris_sns, x = "sepal_width", y = "sepal_length")
# 横轴，花萼宽度
# 纵轴，花萼长度
ax.set_xlabel('Sepal width (cm)')
ax.set_ylabel('Sepal length (cm)')

ax.set_xticks(np.arange(1, 5 + 1, step = 1))
ax.set_yticks(np.arange(4, 8 + 1, step = 1))

ax.axis('scaled')
ax.grid(linestyle = '--', linewidth = 0.25, color = [0.7,0.7,0.7])
ax.set_xbound(lower = 1, upper = 5)
ax.set_ybound(lower = 4, upper = 8)

#>>>>>>>>>>>>>>>>>>>>>>>>  4. 绘制样本数据散点图，增加鸢尾花分类标签
fig, ax = plt.subplots()
# hue 用不同色调表达鸢尾花的类别
ax = sns.scatterplot(data = iris_sns, x="sepal_length", y="sepal_width", hue = "species")

ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
ax.set_xticks(np.arange(4, 8 + 1, step = 1))
ax.set_yticks(np.arange(1, 5 + 1, step = 1))
ax.axis('scaled')
ax.grid(linestyle = '--', linewidth = 0.25, color = [0.7, 0.7, 0.7])
ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)

#>>>>>>>>>>>>>>>>>>>>>>>>  5. 利用色调hue可视化第三特征 (花瓣长度)
fig, ax = plt.subplots()
# hue 用不同色调表达花萼长度 (连续数值）
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width", hue = "petal_length", palette = 'RdYlBu_r')

ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
ax.set_xticks(np.arange(4, 8 + 1, step=1))
ax.set_yticks(np.arange(1, 5 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)

#>>>>>>>>>>>>>>>>>>>>>>>>  6. 利用散点大小可视化第四特征 (花瓣宽度)
fig, ax = plt.subplots()
# size 用散点大小表达花瓣宽度
ax = sns.scatterplot(data = iris_sns, x = "sepal_length", y = "sepal_width", hue = "petal_length", size = "petal_width", palette = 'RdYlBu_r')

ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
ax.set_xticks(np.arange(4, 8 + 1, step=1))
ax.set_yticks(np.arange(1, 5 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)
ax.legend(loc='lower left', bbox_to_anchor=(1.1, 0), ncol=1)
# 将图例置于散点图之外，放置方式为左下
# 图例定位点为 (1.1, 0)
# 其他放置方式：'best', 'upper right', 'upper left', 'lower left',
# 'lower right', 'right', 'center left', 'center right',
# 'lower center', 'upper center', 'center'

#>>>>>>>>>>>>>>>>>>>>>>>>  7. 利用色调hue可视化第三特征 (花瓣长度)，分类标签
fig, ax = plt.subplots()
# size 用散点大小表达花瓣宽度
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width", hue = "species", size = "petal_width")

ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
ax.set_xticks(np.arange(4, 8 + 1, step=1))
ax.set_yticks(np.arange(1, 5 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)
ax.legend(loc='lower left', bbox_to_anchor=(1.1, 0), ncol=1)

#>>>>>>>>>>>>>>>>>>>>>>>>  8. 用不同的标记符号marker
markers = {"setosa": "s", "versicolor": "X", "virginica": "."}
fig, ax = plt.subplots(figsize = (10, 10))
ax = sns.scatterplot(data = iris_sns, x = "sepal_length", y = "sepal_width", hue = "petal_length", style = 'species', markers = markers, size = "petal_width", palette = 'RdYlBu_r')

ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
ax.set_xticks(np.arange(4, 8 + 1, step=1))
ax.set_yticks(np.arange(1, 5 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)
ax.legend(loc='lower left', bbox_to_anchor=(1.1, 0), ncol=1)

#>>>>>>>>>>>>>>>>>>>>>>>> 9. 可视化紧密程度 (分布概率密度）
from sklearn.neighbors import KernelDensity
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(iris_sns[['sepal_length', 'sepal_width']])

# 输出概率密度的对数值
log_PDF = kde.score_samples(iris_sns[['sepal_length', 'sepal_width']]) # (150,)

fig, ax = plt.subplots()
# 利用 seaborn.scatterplot() 绘制散点图
# x对应横轴特征，鸢尾花数据帧列名 "sepal_length"
# y对应纵轴特征，鸢尾花数据帧列名 "sepal_width"
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width", c = log_PDF, cmap = 'RdYlBu_r')

# 设置横纵轴标签
ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')

# 设置横纵轴刻度
ax.set_xticks(np.arange(4, 8 + 1, step=1))
ax.set_yticks(np.arange(1, 5 + 1, step=1))

# 设定横纵轴尺度1:1
ax.axis('scaled')

# 增加刻度网格，颜色为浅灰 (0.8,0.8,0.8)
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])

ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)

norm = plt.Normalize(log_PDF.min(), log_PDF.max())
sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
sm.set_array([])
# ax.figure.colorbar(sm)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 散点包络线
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import seaborn as sns

def encircle(x,y, ax=None, **kw):
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)
    return

# 从seaborn中导入鸢尾花样本数据
iris_sns = sns.load_dataset("iris")
# 打印数据帧前5行
iris_sns.head()

fig, ax = plt.subplots()
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width", hue = "species")
# hue 用不同色调表达鸢尾花的类别
encircle(iris_sns.loc[iris_sns.species == 'setosa','sepal_length'].to_numpy(),
         iris_sns.loc[iris_sns.species == 'setosa','sepal_width'].to_numpy(),
         ec="blue", fc="blue", alpha=0.2)

encircle(iris_sns.loc[iris_sns.species == 'versicolor','sepal_length'].to_numpy(),
         iris_sns.loc[iris_sns.species == 'versicolor','sepal_width'].to_numpy(),
         ec="orange", fc="orange", alpha=0.2)

encircle(iris_sns.loc[iris_sns.species == 'virginica','sepal_length'].to_numpy(),
         iris_sns.loc[iris_sns.species == 'virginica','sepal_width'].to_numpy(),
         ec="green", fc="green", alpha=0.2)

ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
ax.set_xticks(np.arange(4, 8 + 1, step=1))
ax.set_yticks(np.arange(1, 5 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  68-95-99.7法则

import numpy as np
import matplotlib.pyplot as plt

import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# 产生随机数
# 一元，服从标准正态分布
num =   2000  # 随机数数量
mu =    0    # 期望值
sigma = 1    # 标准差
X =  np.random.normal(mu, sigma, num)

# 可视化随机数
fig, ax = plt.subplots()
plt.scatter(np.arange(num), X, marker = '.')
plt.axhline(y = 0,  color = 'r')
plt.axhline(y = -2, color = 'r')
plt.axhline(y = 2,  color = 'r')
plt.xlim(0, num)
plt.ylim(-4, 4)

############# 区分内外
mask_outside = ((X > -2) & (X < 2))

fig, ax = plt.subplots()

# ±2 内的点；不满足的置 NaN
X_inside = X.copy()
X_inside[~mask_outside] = np.nan

# ±2 外的点；满足的置 NaN
X_outside = X.copy()
X_outside[mask_outside] = np.nan

circ = plt.Circle((0, 0), radius=1, edgecolor='k', linewidth = 4, facecolor='red')
ax.add_patch(circ)

colors = np.array(['#377eb8', '#ff7f00'])
plt.scatter(np.arange(num),  X_inside,  color = colors[0], marker = '.')
plt.scatter(np.arange(num),  X_outside, color = colors[1], marker = 'x')

plt.axhline(y = 0,  color = 'r')
plt.axhline(y = -2, color = 'r')
plt.axhline(y = 2,  color = 'r')
plt.xlim(0, num)
plt.ylim(-4, 4)

print('Number of points outside = ' + str(mask_outside.sum()))
print('Percentage of points outside = ' + str(mask_outside.sum()/num*100) + '%')
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 估算圆周率
import numpy as np
import matplotlib.pyplot as plt

import os
# 如果文件夹不存在，创建文件夹
if not os.path.isdir("Figures"):
    os.makedirs("Figures")

# 二元，服从连续均匀分布
num = 2000
# 2000 个散点
X = np.random.uniform(low=-1, high=1, size=(num,2))

fig, ax = plt.subplots()
plt.scatter(X[:,0], X[:,1], marker = '.')

circ = plt.Circle((0, 0), radius=1, edgecolor='red', linewidth = 2, facecolor='none')
ax.add_patch(circ)

ax.set_aspect('equal', adjustable='box')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
ax.set_xticks((-1,0,1))
ax.set_yticks((-1,0,1))

#>>>>>>>>>>>>>>>>>>>>>>>>  使用面具Mask，区分单位圆内外散点

# 生成面具
mask_inside = (X[:,0]**2 + X[:,1]**2 <= 1)
mask_inside
X_inside  = X[mask_inside,:]  # 单位圆内 (包括圆上) 的点
X_outside = X[~mask_inside,:] # 单位圆外部的点


fig, ax = plt.subplots()

circ = plt.Circle((0, 0), radius=1, edgecolor='red', linewidth = 2, facecolor='none')
ax.add_patch(circ)

colors = np.array(['#377eb8', '#ff7f00'])
plt.scatter(X_inside[:,0],  X_inside[:,1],  color = colors[0], marker = '.')
plt.scatter(X_outside[:,0], X_outside[:,1], color = colors[1], marker = 'x')

ax.set_aspect('equal', adjustable='box')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
ax.set_xticks((-1,0,1))
ax.set_yticks((-1,0,1))

print('Number of points inside = ' + str(mask_inside.sum()))
print('Percentage of points inside = ' + str(mask_inside.sum()/num*100) + '%')


















































