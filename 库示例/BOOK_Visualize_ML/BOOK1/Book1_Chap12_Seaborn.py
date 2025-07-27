



import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# import os
# # 如果文件夹不存在，创建文件夹
# if not os.path.isdir("Figures"):
#     os.makedirs("Figures")


# 导入鸢尾花数据
iris_sns = sns.load_dataset("iris")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 一元 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% 直方图
# 绘制花萼长度样本数据直方图
fig, ax = plt.subplots(figsize = (8, 6))

sns.histplot(data=iris_sns, x="sepal_length", binwidth=0.2, ax = ax)
# 纵轴三个选择：频率、概率、概率密度
ax.axvline(x = iris_sns.sepal_length.mean(), color = 'r', ls = '--')
# 增加均值位置竖直参考线


# 绘制花萼长度样本数据直方图，考虑鸢尾花分类
fig, ax = plt.subplots(figsize = (8,6))
# 纵轴为概率密度
sns.histplot(data = iris_sns, x = "sepal_length", hue = 'species', binwidth = 0.2, ax = ax, element = "step", stat = 'density')
sns.histplot( iris_sns, x = "sepal_length", y = "species", hue = "species", legend = True)
sns.barplot(data=iris_sns, x="species", y="sepal_length",  )


#%% 核密度估计KDE
# 绘制花萼长度样本数据，高斯核密度估计
fig, ax = plt.subplots(figsize = (8,6))

sns.kdeplot(data=iris_sns, x="sepal_length", bw_adjust=0.3, fill = True)
sns.rugplot(data=iris_sns, x="sepal_length")

# 绘制花萼长度样本数据，高斯核密度估计，考虑鸢尾花类别
fig, ax = plt.subplots(figsize = (8,6))

sns.kdeplot(data=iris_sns, x="sepal_length", hue = 'species', bw_adjust=0.5, fill = True)
sns.rugplot(data=iris_sns, x="sepal_length", hue = 'species')


# 绘制花萼长度样本数据，高斯核密度估计，考虑鸢尾花类别，堆叠
fig, ax = plt.subplots(figsize = (8,6))
sns.kdeplot(data=iris_sns, x="sepal_length", hue="species", multiple="stack", bw_adjust=0.5)


# 绘制后验概率 (成员值)
fig, ax = plt.subplots(figsize = (8,6))
sns.kdeplot(data=iris_sns, x="sepal_length", hue="species", bw_adjust=0.5, multiple = 'fill')

# 第二种方法
fig, ax = plt.subplots(figsize = (8,6))
sns.displot(data=iris_sns, x="sepal_length", hue="species", kind="kde", bw_adjust=0.3, multiple="fill")



#%% 分散图
# 绘制鸢尾花花萼长度分散点图
fig, ax = plt.subplots(figsize = (8,6))
sns.stripplot(data=iris_sns, x="sepal_length", y="species", hue="petal_length", palette="RdYlBu_r", ax = ax)


sns.stripplot(
    data=iris_sns, x="sepal_length", y="species", hue="petal_length",
    jitter=False, s=20, marker="D", linewidth=1, alpha=.1, palette="RdYlBu_r",
)



#%% 小提琴图

# 绘制花萼长度样本数据，小提琴图
fig, ax = plt.subplots(figsize = (8,2))
sns.violinplot(data=iris_sns, x="sepal_length", ax = ax)

# 绘制花萼长度样本数据，小提琴图，考虑分类
fig, ax = plt.subplots(figsize = (8,4))
sns.violinplot(data=iris_sns, x="sepal_length", y="species", ax = ax)

sns.violinplot(data=iris_sns, x="sepal_length", y="species", inner = 'stick')



#%% 蜂群图
# 绘制花萼长度样本数据，蜂群图
fig, ax = plt.subplots(figsize = (8,4))

sns.swarmplot(data=iris_sns, x="sepal_length", ax = ax)

# 绘制花萼长度样本数据，蜂群图，考虑分类
fig, ax = plt.subplots(figsize = (8,4))
sns.swarmplot(data=iris_sns, x="sepal_length", y = 'species', hue = 'species', ax = ax)

# 蜂群图 + 小提琴图，考虑鸢尾花分类
sns.catplot(data=iris_sns, x="sepal_length", y="species", kind="violin", color=".9", inner=None)

sns.swarmplot(data=iris_sns, x="sepal_length", y="species", size=3)

####################### 箱型图
# 绘制鸢尾花花萼长度箱型图
fig, ax = plt.subplots(figsize = (8,2))
sns.boxplot(data=iris_sns, x="sepal_length", ax = ax)

# 绘制鸢尾花花萼长度箱型图，考虑鸢尾花分类
fig, ax = plt.subplots(figsize = (8,3))
sns.boxplot(data=iris_sns, x="sepal_length", y = 'species', ax = ax)

sns.boxenplot(data=iris_sns, x="sepal_length", y="species", scale="linear")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 二元 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% 散点图 + 毛毯图
# 鸢尾花散点图 + 毛毯图
fig, ax = plt.subplots(figsize = (4,4))

sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width")
sns.rugplot(data=iris_sns, x="sepal_length", y="sepal_width")


fig, ax = plt.subplots(figsize = (4,4))
sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width", hue = 'species')
sns.rugplot(data=iris_sns, x="sepal_length", y="sepal_width", hue = 'species')

# fig.savefig('Figures\二元，scatterplot + rugplot + hue.svg', format='svg')



g = sns.JointGrid(data=iris_sns, x="sepal_length", y="sepal_width")
g.plot_joint(sns.scatterplot)
g.plot_marginals(sns.boxplot)



g = sns.JointGrid(data=iris_sns, x="sepal_length", y="sepal_width")
g.plot_joint(sns.scatterplot)
g.plot_marginals(sns.kdeplot)



#%% 频率/概率热图
# 鸢尾花二元频率直方热图

sns.displot(data=iris_sns, x="sepal_length", y="sepal_width", binwidth=(0.2, 0.2), cbar=True)



#%% 二元概率密度估计KDE
# 联合分布概率密度等高线
sns.displot(data=iris_sns, x="sepal_length",  y="sepal_width", kind="kde")


# 联合分布概率密度等高线，考虑分布
sns.kdeplot(data=iris_sns, x="sepal_length", y="sepal_width", hue = 'species')



#%% 线性回归
# 可视化线性回归关系
sns.lmplot(data=iris_sns, x="sepal_length", y="sepal_width")

# 可视化线性回归关系，考虑鸢尾花分类
sns.lmplot(data=iris_sns, x="sepal_length", y="sepal_width", hue = 'species')



#%% 散点图 + 边缘直方图
sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width")


# 散点图 + 边缘KDE
# 联合分布、边缘分布
sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind = 'kde', fill = True)

sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", hue = 'species')

# 联合分布、边缘分布，考虑鸢尾花分类
sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", hue = 'species', kind="kde")

sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind="reg")

sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind="hist")

sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind="hex")

g = sns.JointGrid(data=iris_sns, x="sepal_length", y="sepal_width")
g.plot(sns.scatterplot, sns.histplot)
g.refline(x=iris_sns['sepal_length'].mean(), y=iris_sns['sepal_width'].mean())





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 多元 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 分散图
iris_melt = pd.melt(iris_sns, "species", var_name="measurement")


iris_melt = pd.melt(iris_sns, "species", var_name="measurement")
# 数据从宽格式 (wide format) 转换为长格式 (long format)

# 绘制多特征分散图
sns.stripplot(
    data=iris_melt, x="value", y="measurement", hue="species",
    dodge=True, alpha=.25, zorder=1,  )

plt.grid()


# 小提琴图
# 绘制多特征小提琴图
sns.violinplot(
    data=iris_melt, x="value", y="measurement", hue="species",
    dodge=True, alpha=.25, zorder=1, legend=True)

plt.grid()
# 热图
sns.heatmap(iris_sns.iloc[:,:-1], cmap = 'RdYlBu_r', vmin = 0, vmax = 8)

# 聚类热图
sns.clustermap(iris_sns.iloc[:,:-1], cmap = 'RdYlBu_r', vmin = 0, vmax = 8)

# 成对特征散点图
sns.pairplot(iris_sns)

# 绘制成对特征散点图
sns.pairplot(iris_sns, hue = 'species')

g = sns.pairplot(iris_sns, hue = 'species')
g.map_lower(sns.kdeplot, levels=8, color=".2")

# 平行坐标图
from pandas.plotting import parallel_coordinates
# 可视化函数来自pandas
parallel_coordinates(iris_sns, 'species', colormap=plt.get_cmap("Set2"))
plt.show()

# 安德鲁斯曲线
from pandas.plotting import andrews_curves
andrews_curves(iris_sns, 'species', colormap=plt.get_cmap("Set2"))

# Radviz雷达图
from pandas.plotting import radviz
radviz(iris_sns, 'species', colormap=plt.get_cmap("Set2"))

# 协方差矩阵
SIGMA = iris_sns.iloc[:,:-1].cov()
fig, axs = plt.subplots()
h = sns.heatmap(SIGMA,cmap='RdYlBu_r', linewidths=.05, annot = True)
h.set_aspect("equal")



f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)
features = list(iris_sns.columns)[:-1]

g1 = sns.heatmap(iris_sns.loc[iris_sns.species == 'setosa', features].cov(),
                 cmap="RdYlBu_r",fmt='.2f',
                 annot=True,cbar=False,ax=ax1,square=True,
                 vmax = 0.4, vmin = 0)
ax1.set_title('Setosa')

g2 = sns.heatmap(iris_sns.loc[iris_sns.species == 'versicolor', features].cov(),
                 cmap="RdYlBu_r",fmt='.2f',
                 annot=True,cbar=False,ax=ax2,square=True,
                 vmax = 0.4, vmin = 0)
ax2.set_title('Versicolor')

g3 = sns.heatmap(iris_sns.loc[iris_sns.species == 'virginica', features].cov(),
                 cmap="RdYlBu_r",fmt='.2f',
                 annot=True,cbar=False,ax=ax3,square=True,
                 vmax = 0.4, vmin = 0)
ax3.set_title('Virginica')



# 相关性系数矩阵
RHO = iris_sns.iloc[:,:-1].corr()

fig, axs = plt.subplots()

h = sns.heatmap(RHO,cmap='RdYlBu_r', linewidths=.05, annot = True)
h.set_aspect("equal")




f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)
features = list(iris_sns.columns)[:-1]

g1 = sns.heatmap(iris_sns.loc[iris_sns.species == 'setosa', features].corr(),
                 cmap="RdYlBu_r",fmt='.2f',
                 annot=True,cbar=False,ax=ax1,square=True,
                 vmax = 1, vmin = 0)
ax1.set_title('Setosa')

g2 = sns.heatmap(iris_sns.loc[iris_sns.species == 'versicolor', features].corr(),
                 cmap="RdYlBu_r",fmt='.2f',
                 annot=True,cbar=False,ax=ax2,square=True,
                 vmax = 1, vmin = 0)
ax2.set_title('Versicolor')

g3 = sns.heatmap(iris_sns.loc[iris_sns.species == 'virginica', features].corr(),
                 cmap="RdYlBu_r",fmt='.2f',
                 annot=True,cbar=False,ax=ax3,square=True,
                 vmax = 1, vmin = 0)
ax3.set_title('Virginica')





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Seaborn散点图 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 导入包
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris



# 1. 导入并简单分析数据


# 从seaborn中导入鸢尾花样本数据
iris_sns = sns.load_dataset("iris")
# 打印数据帧前5行
iris_sns.head()

# 打印数据帧列名
iris_sns.columns


# 鸢尾花数据统计特征
iris_sns.describe()

# 绘制样本数据散点图，不加标签
fig, ax = plt.subplots()

ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width")

# 利用 seaborn.scatterplot() 绘制散点图
# x对应横轴特征，鸢尾花数据帧列名 "sepal_length"
# y对应纵轴特征，鸢尾花数据帧列名 "sepal_width"

ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
# 设置横纵轴标签

ax.set_xticks(np.arange(4, 8 + 1, step=1))
ax.set_yticks(np.arange(1, 5 + 1, step=1))
# 设置横纵轴刻度

ax.axis('scaled')
# 设定横纵轴尺度1:1

ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
# 增加刻度网格，颜色为浅灰 (0.8,0.8,0.8)

ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)
# 设置横纵轴取值范围



# 调转横、纵轴特征
fig, ax = plt.subplots()
ax = sns.scatterplot(data=iris_sns, x="sepal_width", y="sepal_length")
# 横轴，花萼宽度
# 纵轴，花萼长度

ax.set_xlabel('Sepal width (cm)')
ax.set_ylabel('Sepal length (cm)')

ax.set_xticks(np.arange(1, 5 + 1, step=1))
ax.set_yticks(np.arange(4, 8 + 1, step=1))

ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 1, upper = 5)
ax.set_ybound(lower = 4, upper = 8)


# 绘制样本数据散点图，增加鸢尾花分类标签
fig, ax = plt.subplots()
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width", hue = "species")

# hue 用不同色调表达鸢尾花的类别
ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
ax.set_xticks(np.arange(4, 8 + 1, step=1))
ax.set_yticks(np.arange(1, 5 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)
plt.show()

# 利用色调hue可视化第三特征 (花瓣长度)
fig, ax = plt.subplots()
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width", hue = "petal_length", palette = 'RdYlBu_r')
# hue 用不同色调表达花萼长度 (连续数值）
ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
ax.set_xticks(np.arange(4, 8 + 1, step=1))
ax.set_yticks(np.arange(1, 5 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)
plt.show()



# 利用散点大小size可视化第四特征 (花瓣宽度)
fig, ax = plt.subplots()
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width", hue = "petal_length", size = "petal_width", palette = 'RdYlBu_r')
# size 用散点大小表达花瓣宽度
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
plt.show()




fig, ax = plt.subplots()
ax = sns.scatterplot(data=iris_sns, x="sepal_length", y="sepal_width", hue = "species", size = "petal_width")
# size 用散点大小表达花瓣宽度
ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
ax.set_xticks(np.arange(4, 8 + 1, step=1))
ax.set_yticks(np.arange(1, 5 + 1, step=1))
ax.axis('scaled')
ax.grid(linestyle='--', linewidth=0.25, color=[0.7,0.7,0.7])
ax.set_xbound(lower = 4, upper = 8)
ax.set_ybound(lower = 1, upper = 5)
ax.legend(loc='lower left', bbox_to_anchor=(1.1, 0), ncol=1)
# 将图例置于散点图之外，位置为左下
#
# 其他位置：'best', 'upper right', 'upper left', 'lower left',
# 'lower right', 'right', 'center left', 'center right',
# 'lower center', 'upper center', 'center'
plt.show()































































































































































































































































































































