

# https://www.osgeo.cn/networkx/auto_examples/index.html

# https://networkx.org/documentation/latest/auto_examples/index.html







#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 离群值


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from scipy import stats


# Load the iris data
# iris_sns = sns.load_dataset("iris")
# A copy from Seaborn
iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target

feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$', 'Petal length, $X_3$','Petal width, $X_4$']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)



# 直方图
# visualize two tails (1%, 99%)

num = 0
fig, axes = plt.subplots(2,2)

for i in [0,1]:
    for j in [0,1]:
        sns.histplot(data=X_df, x = feature_names[num], binwidth = 0.2, ax = axes[i][j])
        axes[i][j].set_xlim([0,8]);
        axes[i][j].set_ylim([0,40])

        q1, q50, q99 = np.percentile(X_df[feature_names[num]], [1,50,99])
        axes[i][j].axvline(x=q1, color = 'r')
        axes[i][j].axvline(x=q50, color = 'r')
        axes[i][j].axvline(x=q99, color = 'r')

        num = num + 1

# visualize locations of three quartiles
num = 0
fig, axes = plt.subplots(2,2)
for i in [0,1]:
    for j in [0,1]:
        sns.histplot(data=X_df, x = feature_names[num], binwidth = 0.2, ax = axes[i][j])
        axes[i][j].set_xlim([0,8]);
        axes[i][j].set_ylim([0,40])

        q75, q50, q25 = np.percentile(X_df[feature_names[num]], [75,50,25])
        axes[i][j].axvline(x=q75, color = 'r')
        axes[i][j].axvline(x=q50, color = 'r')
        axes[i][j].axvline(x=q25, color = 'r')

        num = num + 1

#%%  概率密度
#%% KDE +rug plot
num = 0
fig, axes = plt.subplots(2,2)
for i in [0,1]:
    for j in [0,1]:
        sns.kdeplot(data=X_df, x = feature_names[num], ax = axes[i][j], fill = True)
        sns.rugplot(data=X_df, x = feature_names[num], ax = axes[i][j], color = 'k', height=.05)

        q1, q50, q99 = np.percentile(X_df[feature_names[num]], [1,50,99])
        axes[i][j].axvline(x=q1, color = 'r')
        axes[i][j].axvline(x=q50, color = 'r')
        axes[i][j].axvline(x=q99, color = 'r')

        num = num + 1


#%% 平面散点图
#%% scatter plot
from sklearn import   datasets

# # 导入并整理数据
# iris = datasets.load_iris()
# # y = iris.target


cmap_bold = [[255, 51, 0],
             [0, 153, 255],
             [138,138,138]]
cmap_bold = np.array(cmap_bold)/255.0

for i in [1,2,3]:
    fig, axes = plt.subplots()
    sns.scatterplot(data=X_df, x=feature_names[0], y=feature_names[i], hue=iris.target_names[y], ax = axes, palette=dict(setosa=cmap_bold[0,:], versicolor=cmap_bold[1,:], virginica=cmap_bold[2,:]), )
    sns.rugplot(data=X_df, x=feature_names[0], y=feature_names[i], height=.05, hue=iris.target_names[y], ax = axes,)



#%%  成对特征
#%% pairplot
g = sns.pairplot(X_df)
g.map_upper(sns.scatterplot, color = 'b')
g.map_lower(sns.kdeplot, levels=8, fill=True, cmap="Blues_r")
g.map_diag(sns.histplot, kde=False, color = 'b')



#%%QQ图
#%% QQ plot
import pylab
num = 0;
for i in [0,1]:
    for j in [0,1]:
        fig, axes = plt.subplots(1,2)
        sns.histplot(data=X_df, x = feature_names[num], binwidth = 0.2, ax = axes[0])
        axes[0].set_xlim([0,8]); axes[0].set_ylim([0,40])
        values = X_df[feature_names[num]]
        stats.probplot(values, dist="norm", plot=pylab)

        plt.xlabel('Normal distribution')
        plt.ylabel('Empirical distribution')
        plt.title(feature_names[num])
        num = num + 1


#%% 箱型图
#%% box plot of data

fig, ax = plt.subplots()
sns.boxplot(data=X_df, palette="Set3", orient="h")

print(X_df.describe())

X_df.quantile(q=[0.25, 0.5, 0.75], axis=0, numeric_only=True, interpolation='midpoint')




#%%  箱型图 + 蜂群图
#%% combine boxplot and swarmplot
fig, ax = plt.subplots()
sns.boxplot(data=X_df, orient="h", palette="Set3")
sns.swarmplot(data=X_df, linewidth=0.25, orient="h", color=".5")


#%% z分数
#%% z score
from scipy import stats
df_zscore = (X_df - X_df.mean())/X_df.std()
# z_score = stats.zscore(X_df)
num = 0
fig, axes = plt.subplots(2,2)
for i in [0,1]:
    for j in [0,1]:
        sns.histplot(data=df_zscore, x = feature_names[num], binwidth = 0.2, ax = axes[i][j])
        axes[i][j].set_xlim([-4,4]); axes[i][j].set_ylim([0,40])

        axes[i][j].axvline(x=3, color = 'r')
        axes[i][j].axvline(x=2, color = 'r')
        axes[i][j].axvline(x=-3, color = 'r')
        axes[i][j].axvline(x=-2, color = 'r')
        axes[i][j].axvline(x=0, color = 'r')
        num = num + 1



#%% 马氏距离
#%% Mahal distance

from sklearn.covariance import EllipticEnvelope

clf = EllipticEnvelope(contamination=0.05)

xx, yy = np.meshgrid(np.linspace(3, 9, 50), np.linspace(1, 5, 50))

clf.fit(X_df.values[:,:2])
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()

ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='r')

ax.scatter(X_df.values[:, 0], X_df.values[:, 1], color='b')

ax.set_xlim((3,9))
ax.set_ylim((0,6))

ax.set_ylabel(feature_names[0]);
ax.set_xlabel(feature_names[1]);
ax.set_aspect('equal', adjustable='box')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 离群值，多元
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.neighbors import KernelDensity

iris = load_iris()
# A copy from Sklearn

X = iris.data
y = iris.target

feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$', 'Petal length, $X_3$','Petal width, $X_4$']

# Convert X array to dataframe
X_df = pd.DataFrame(X, columns=feature_names)

xx, yy = np.meshgrid(np.linspace(4, 8, 50), np.linspace(1, 5, 50))


kde = KernelDensity(bandwidth=0.3)
kde.fit(X_df.values[:,:2])
pred = kde.score_samples(X_df.values[:,:2])
pred_1_0 = (pred > np.percentile(pred, 10)).astype(int) # 基于对数概率密度值，将前 10%的点标记为异常值 (0)，其余标记为正常值 (1)。
X_df['pred_1_0'] = pred_1_0

dec = kde.score_samples(np.c_[xx.ravel(), yy.ravel()]) # 使用训练好的核密度估计模型计算在特征空间中每个点的对数概率密度值。
dens = kde.score_samples(X_df.values[:,:2]) # 使用训练好的核密度估计模型计算数据集中每个点的对数概率密度值。

plt.figure(figsize=(8,8))
plt.scatter(X_df.values[:,0], X_df.values[:,1], c = dens, cmap='RdYlBu')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])


plt.figure(figsize=(8,8))
colors = {1:'tab:blue', 0:'tab:red'}
plt.contourf(xx, yy, dec.reshape(xx.shape), alpha=.5, levels = 20) # 绘制核密度估计的轮廓线，并使用颜色填充轮廓线之间的区域。alpha 设置透明度，levels设置轮廓线的数量。
plt.scatter(X_df.values[:,0], X_df.values[:,1], c = X_df['pred_1_0'].map(colors)) # 绘制散点图，异常值用红色表示，正常值用蓝表示。
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])



#%% SVM
from sklearn.svm import OneClassSVM

xx, yy = np.meshgrid(np.linspace(4, 8, 200), np.linspace(1, 5, 200))

# scaler = StandardScaler()
X_train = X_df.values[:,:2]

oneclass = OneClassSVM(nu=.1).fit(X_train) # 训练，根据训练样本和上面两个参数探测边界。（注意是无监督）

pred = oneclass.predict(X_train).astype(np.int) # 返回预测值，+1就是正常样本，-1就是异常样本。

X_df['pred_1_minus_1'] = pred

dec = oneclass.decision_function(np.c_[xx.ravel(), yy.ravel()]) # 返回各样本点到超平面的函数距离（signed distance），正的维正常样本，负的为异常样本。

plt.figure(figsize=(8,8))

colors = {1:'tab:blue', -1:'tab:red'}
plt.contourf(xx, yy, dec.reshape(xx.shape), alpha=.5)
plt.contour(xx, yy, dec.reshape(xx.shape), levels=[0], linewidths=2, colors="black")

plt.scatter(X_df.values[:,0], X_df.values[:,1], c=X_df['pred_1_minus_1'].map(colors))

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])



#%% IsolationForest

from sklearn.ensemble import IsolationForest
iso = IsolationForest().fit(X_train)

pred = iso.predict(X_train).astype(np.int)

X_df['pred_1_minus_1'] = pred

dec = iso.decision_function(np.c_[xx.ravel(), yy.ravel()])

plt.figure(figsize=(8,8))

colors = {1:'tab:blue', -1:'tab:red'}
plt.contourf(xx, yy, dec.reshape(xx.shape), alpha=.5)
plt.contour(xx, yy, dec.reshape(xx.shape), levels=[0], linewidths=2, colors="black")

plt.scatter(X_df.values[:,0], X_df.values[:,1], c=X_df['pred_1_minus_1'].map(colors))

plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])



















































































































































































































































































