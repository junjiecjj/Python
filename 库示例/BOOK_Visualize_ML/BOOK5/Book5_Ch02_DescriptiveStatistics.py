




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

# Load the iris data
iris_sns = sns.load_dataset("iris")
# A copy from Seaborn
iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target

feature_names = ['Sepal length, $X_1$', 'Sepal width, $X_2$', 'Petal length, $X_3$', 'Petal width, $X_4$']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

#%% Heatmap of X
plt.close('all')

# Visualize the heatmap of X
fig, ax = plt.subplots()
ax = sns.heatmap(X_df, cmap='RdYlBu_r', xticklabels=list(X_df.columns), cbar_kws={"orientation": "vertical"}, vmin=-1, vmax=9)
plt.title('X')

#%% Histograms, 图 4. 鸢尾花四个特征数据的直方图，纵轴为频数
fig, axes = plt.subplots(2,2, figsize = (12, 8))

sns.histplot(data = X_df, x = feature_names[0], binwidth = 0.1, ax = axes[0][0])
axes[0][0].set_xlim([0, 8]); axes[0][0].set_ylim([0, 40])

sns.histplot(data = X_df, x = feature_names[1], binwidth = 0.1, ax = axes[0][1])
axes[0][1].set_xlim([0, 8]); axes[0][1].set_ylim([0, 40])

sns.histplot(data = X_df, x = feature_names[2], binwidth = 0.1, ax = axes[1][0])
axes[1][0].set_xlim([0, 8]); axes[1][0].set_ylim([0, 40])

sns.histplot(data = X_df, x = feature_names[3], binwidth = 0.1, ax = axes[1][1])
axes[1][1].set_xlim([0, 8]); axes[1][1].set_ylim([0, 40])

plt.tight_layout()
plt.show()
plt.close('all')

#%% draw multiple histograms on the same plot, 图 5. 直方图，比较频数和概率密度
fig, ax = plt.subplots()
sns.histplot(data=X_df, palette = "rocket_r", binwidth = 0.2)
plt.show()

fig, ax = plt.subplots()
sns.histplot(data=X_df, palette = "viridis", binwidth = 0.2, stat="density", common_norm=False)
plt.show()

fig, ax = plt.subplots()
sns.histplot(data=X_df, palette = "viridis", binwidth = 0.2, stat="density", common_norm=True)
plt.show()

plt.close('all')

#%% cumulative, 图 6. 累积频数图，累积概率图
fig, ax = plt.subplots()
sns.histplot(data=X_df, palette = "viridis",fill = False, binwidth = 0.2,element="step", cumulative=True, common_norm=False)
plt.show()

fig, ax = plt.subplots()
sns.histplot(data=X_df, palette = "viridis",fill = False, binwidth = 0.2,element="step",stat="density", cumulative=True, common_norm=False)
plt.show()
plt.close('all')
#%% variations of histograms, 图 7. 比较多边形图和和概率密度估计曲线
fig, ax = plt.subplots()
sns.histplot(data=X_df, palette = "viridis",fill = False, binwidth = 0.2,element="poly", stat="density", common_norm=False)
plt.show()

fig, ax = plt.subplots()
sns.histplot(data=X_df, palette = "viridis", binwidth = 0.2, element="step", kde = True, stat="density", common_norm=False)
plt.show()
plt.close('all')

#%% KDE
plt.tight_layout()

fig, ax = plt.subplots()
sns.kdeplot(data=X_df, fill=True, common_norm=False, alpha=.3, linewidth=1, palette = "viridis")
plt.show()
plt.close('all')

#%% bivariate, 图 10. 二维数据直方图热图，二维 KDE 概率密度曲面等高线

fig, ax = plt.subplots()
sns.histplot(iris_sns, x="sepal_length", y="sepal_width", bins = 20)
sns.displot(iris_sns, x="sepal_length", y="sepal_width", kind="kde", rug=True)
plt.show()
plt.close('all')

#%% variations of joint plots
# 图 9. 二维数据散点图及扩展
# fig, ax = plt.subplots()
sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", marginal_kws=dict(bins=20, fill=True))
sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind = 'reg', marginal_kws=dict(bins=20, fill=True))
plt.show()
plt.close('all')

# 图 10. 二维数据直方图热图，二维 KDE 概率密度曲面等高线
# fig, ax = plt.subplots()
sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind = 'hex', bins = 20, marginal_kws=dict(bins=20, fill=True))
sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind = 'kde')
plt.show()
plt.close('all')

# 图 11. 直方图热图和概率密度曲面等高线拓展
# fig, ax = plt.subplots()
sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind = 'hist', bins = 20, marginal_kws=dict(bins=20, fill=True))
g = sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind = 'kde', fill = True)
g.plot_joint(sns.kdeplot, color="r",lw = 3, zorder=0, levels=20)
plt.show()
plt.close('all')
#%% Categorical data
#%% 图 14. 直方图，考虑鸢尾花分类标签
for i in [0,1,2,3]:
    fig, ax = plt.subplots()
    sns.histplot(data=iris_sns, x=iris_sns.columns[i], hue="species", binwidth = 0.2, element="step")
    ax.set_xlim([0,8])

#%% classes, bivariate, 图 16. 二维数据散点图，KDE 概率密度曲面等高线，考虑鸢尾花分类标签
sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", hue="species")
sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind = 'kde', hue="species")
sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", kind = 'kde', fill = True, hue="species")

#%% Regression by classes
sns.lmplot(data = iris_sns, x="sepal_length", y="sepal_width", hue="species")
sns.lmplot(data = iris_sns, x="sepal_length", y="sepal_width", hue="species", col="species")



#%% penguins
penguins = sns.load_dataset("penguins")
sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")

sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species")

sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species", kind="kde")

sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", kind="reg")

sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", kind="hist")

sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", kind="hex")

sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", marker="+", s=100, marginal_kws=dict(bins=25, fill=False),)

sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", height=5, ratio=2, marginal_ticks=True)

g = sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")
g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)



#%% multivariate pairwise
# without class labels
fig, ax = plt.subplots()
g = sns.pairplot(iris_sns)
g.map_upper(sns.scatterplot, color = 'b')
g.map_lower(sns.kdeplot, levels=8, fill=True, cmap="Blues_d")
g.map_diag(sns.histplot, kde=False, color = 'b')


# 成对特征散点图
sns.pairplot(iris_sns)

# 绘制成对特征散点图
sns.pairplot(iris_sns, hue = 'species')

g = sns.pairplot(iris_sns, hue = 'species')
g.map_lower(sns.kdeplot, levels=4, color=".2")

plt.show()
plt.close('all')


#%% pairwise
# with class labels, 图 17. 鸢尾花数据成对特征分析图，考虑鸢尾花分类标签
g = sns.pairplot(iris_sns, hue="species", plot_kws={"s": 6}, palette = "viridis")
g.map_lower(sns.kdeplot)

#%% parallel coordinates, 图 18. 鸢尾花数据的平行坐标图
fig, ax = plt.subplots()
# Make the plot
pd.plotting.parallel_coordinates(iris_sns, 'species', colormap=plt.get_cmap("Set2"))
plt.show()

#%% Joy plot, 图 15. 鸢尾花山数据山脊图，特征分类
import joypy
# you might have to install joypy
joypy.joyplot(iris_sns, ylim='own')

joypy.joyplot(iris_sns, column=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],  by="species", ylim='own')

joypy.joyplot(iris_sns, by="species", column="sepal_width", hist=True, bins=40, overlap=0,grid=True)

#%% add mean values to the histograms, 图 19. 鸢尾花四个特征数据均值在直方图位置
fig, axes = plt.subplots(2,2)

sns.histplot(data=X_df, x = feature_names[0], binwidth = 0.2, ax = axes[0][0])
axes[0][0].set_xlim([0,8]); axes[0][0].set_ylim([0,40])
axes[0][0].vlines(x = X_df.mean()[feature_names[0]], ymin = 0, ymax = 40, color = 'r')

sns.histplot(data=X_df, x = feature_names[1], binwidth = 0.2, ax = axes[0][1])
axes[0][1].set_xlim([0,8]); axes[0][1].set_ylim([0,40])
axes[0][1].vlines(x = X_df.mean()[feature_names[1]], ymin = 0, ymax = 40, color = 'r')

sns.histplot(data=X_df, x = feature_names[2], binwidth = 0.2, ax = axes[1][0])
axes[1][0].set_xlim([0,8]); axes[1][0].set_ylim([0,40])
axes[1][0].vlines(x = X_df.mean()[feature_names[2]], ymin = 0, ymax = 40, color = 'r')

sns.histplot(data=X_df, x = feature_names[3], binwidth = 0.2, ax = axes[1][1])
axes[1][1].set_xlim([0,8]); axes[1][1].set_ylim([0,40])
axes[1][1].vlines(x = X_df.mean()[feature_names[3]], ymin = 0, ymax = 40, color = 'r')

plt.tight_layout()

#%% centroid added to jointplot, 图 20. 均值在散点图的位置
scatter_ax = sns.jointplot(data = iris_sns, x="sepal_length", y="sepal_width", marginal_kws=dict(bins=20, fill=True))

scatter_ax.ax_joint.axvline(x = X_df.mean()[feature_names[0]], color = 'r')
scatter_ax.ax_joint.axhline(y = X_df.mean()[feature_names[1]], color = 'r')

scatter_ax.ax_joint.plot(X_df.mean()[feature_names[0]], X_df.mean()[feature_names[1]], marker = 'x', markersize = '12', color = 'r')
scatter_ax.ax_joint.set_xlim(4,8)
scatter_ax.ax_joint.set_ylim(2,4.5)

#%% centroid added to jointplot, with classes, 图 21. 均值在散点图的位置，考虑类别标签
scatter_ax = sns.jointplot(data=iris_sns, x="sepal_length", y="sepal_width", hue="species")
for label,color in zip(['setosa','versicolor','virginica'], ['b','r','g']):
    mu_x1_class = iris_sns.loc[iris_sns['species'] == label, 'sepal_length'].mean()
    mu_x2_class = iris_sns.loc[iris_sns['species'] == label, 'sepal_width'].mean()

    scatter_ax.ax_joint.axvline(x=mu_x1_class, color = color)
    scatter_ax.ax_joint.axhline(y=mu_x2_class, color = color)
    scatter_ax.ax_joint.plot(mu_x1_class, mu_x2_class, marker = 'x', markersize = '12', color = color)


#%% add mean values and std bands to the histograms, 图 23. 鸢尾花四个特征数据均值、标准差所在位置在直方图位置
num = 0
fig, axes = plt.subplots(2,2)

for i in [0,1]:
    for j in [0,1]:
        sns.histplot(data=X_df, x = feature_names[num], binwidth = 0.2, ax = axes[i][j])
        axes[i][j].set_xlim([0,8]); axes[0][0].set_ylim([0,40])

        mu  = X_df[feature_names[num]].mean()
        std = X_df[feature_names[num]].std()

        axes[i][j].axvline(x=mu, color = 'r')
        axes[i][j].axvline(x=mu - std, color = 'r')
        axes[i][j].axvline(x=mu + std, color = 'r')
        axes[i][j].axvline(x=mu - 2*std, color = 'r')
        axes[i][j].axvline(x=mu + 2*std, color = 'r')
        num = num + 1

#%% print the summary of iris data

print(iris_sns.describe(percentiles = [0.01, 0.25, 0.5, 0.75, 0.99]))

#%% 4-quantiles, quartiles
# visualize locations of three quartiles, 图 24. 鸢尾花数据直方图，以及 25%、50%和 75%百分位

num = 0
fig, axes = plt.subplots(2,2)
for i in [0,1]:
    for j in [0,1]:
        sns.histplot(data=X_df, x = feature_names[num], binwidth = 0.2, ax = axes[i][j])
        axes[i][j].set_xlim([0,8]); axes[0][0].set_ylim([0,40])

        q75, q50, q25 = np.percentile(X_df[feature_names[num]], [75,50,25])
        axes[i][j].axvline(x=q75, color = 'r')
        axes[i][j].axvline(x=q50, color = 'r')
        axes[i][j].axvline(x=q25, color = 'r')

        num = num + 1

#%% 100-quantiles, percentile
# visualize two tails (1%, 99%), 图 25. 鸢尾花数据直方图，以及 1%和 99%百分位

num = 0
fig, axes = plt.subplots(2,2)

for i in [0,1]:
    for j in [0,1]:
        sns.histplot(data=X_df, x = feature_names[num], binwidth = 0.2, ax = axes[i][j])
        axes[i][j].set_xlim([0,8]); axes[0][0].set_ylim([0,40])

        q1, q50, q99 = np.percentile(X_df[feature_names[num]], [1,50,99])
        axes[i][j].axvline(x=q1, color = 'r')
        axes[i][j].axvline(x=q50, color = 'r')
        axes[i][j].axvline(x=q99, color = 'r')
        num = num + 1

#%% box plot of data, 图 28. 鸢尾花数据箱型图
fig, ax = plt.subplots()
sns.boxplot(data=X_df, palette="Set3")
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])


#%% violin plot of data
# 图 29. 鸢尾花数据小提琴图
fig, ax = plt.subplots()
sns.violinplot(data=X_df, palette="Set3", bw=.2, cut=1, linewidth=0.25, inner="points", orient="v")
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

# 图 30. 分布散点图 (stripplot)
fig, ax = plt.subplots()
sns.swarmplot(data=X_df, palette="Set3", linewidth=0.25, orient="v")
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

#%% combine boxplot and swarmplot, 图 31. 鸢尾花箱型图，叠加分布散点图 swarmplot
fig, ax = plt.subplots()

sns.boxplot(data=X_df, orient="h")
sns.swarmplot(data=X_df, linewidth=0.25, orient="h", color=".2")

#%% boxplot by labels, 图 32. 鸢尾花箱型图，考虑分类标签
iris_long = iris_sns.melt(id_vars=['species'])
fig, ax = plt.subplots()
sns.boxplot(data=iris_long, x="value", y="variable", orient="h", hue = 'species', palette="Set3")
ax.grid(linestyle='--', linewidth=0.25, color=[0.5,0.5,0.5])

#%% Heatmap of covariance matrix, 图 37. 协方差矩阵、相关性系数矩阵热图

SIGMA = X_df.cov()

fig, axs = plt.subplots()

h = sns.heatmap(SIGMA,cmap='RdYlBu_r', linewidths=.05,annot=True)
h.set_aspect("equal")
h.set_title('Covariance matrix')

RHO = X_df.corr()
fig, axs = plt.subplots()

h = sns.heatmap(RHO,cmap='RdYlBu_r', linewidths=.05,annot=True)
h.set_aspect("equal")
h.set_title('Correlation matrix')

#%% skewness and kurtosis

print(X_df.skew())
print(X_df.kurt())

#%% compare covariance matrices
f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)

g1 = sns.heatmap(X_df[y==0].cov(),cmap="RdYlBu_r", annot=True,cbar=False,ax=ax1,square=True, vmax = 0.4, vmin = 0)
ax1.set_title('Y = 0, setosa')

g2 = sns.heatmap(X_df[y==1].cov(),cmap="RdYlBu_r", annot=True,cbar=False,ax=ax2,square=True, vmax = 0.4, vmin = 0)
ax2.set_title('Y = 1, versicolor')

g3 = sns.heatmap(X_df[y==2].cov(),cmap="RdYlBu_r", annot=True,cbar=False,ax=ax3,square=True, vmax = 0.4, vmin = 0)
ax3.set_title('Y = 2, virginica')

#%% compare correlation matrices
f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)

g1 = sns.heatmap(X_df[y==0].corr(),cmap="RdYlBu_r", annot=True, cbar=False,ax=ax1, square=True, vmax = 1, vmin = 0.15)
ax1.set_title('Y = 0, setosa')

g2 = sns.heatmap(X_df[y==1].corr(),cmap="RdYlBu_r", annot=True,cbar=False,ax=ax2,square=True, vmax = 1, vmin = 0.15)
ax2.set_title('Y = 1, versicolor')

g3 = sns.heatmap(X_df[y==2].corr(),cmap="RdYlBu_r", annot=True,cbar=False,ax=ax3,square=True, vmax = 1, vmin = 0.15)
ax3.set_title('Y = 2, virginica')





































































































































































































































































































