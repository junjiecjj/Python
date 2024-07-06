

#%% 平面散点图 + 椭圆


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.graphics.plot_grids import scatter_ellipse

# 导入鸢尾花数据
data_raw = sns.load_dataset('iris')
labels = ['Sepal length','Sepal width', 'Petal length','Petal width']

fig = plt.figure(figsize=(8,8))
scatter_ellipse(data_raw.iloc[:,:-1],  varnames=labels, fig=fig)


for s_idx in data_raw.species.unique():
    data= data_raw.loc[data_raw.species == s_idx].iloc[:,:-1]
    fig = plt.figure(figsize=(8,8))
    scatter_ellipse(data, varnames=labels, fig=fig)




#%% 一元OLS线性回归
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# 生成随机数据
num = 50
np.random.seed(0)
x_data = np.random.uniform(0,10,num)
y_data = 0.5 * x_data + 1 + np.random.normal(0, 1, num)
data = np.column_stack([x_data, y_data])


# 添加常数列
X = sm.add_constant(x_data)

# 创建一元OLS线性回归模型
model = sm.OLS(y_data, X)

# 拟合模型
results = model.fit()

# 打印回归结果
print(results.summary())


# 预测
x_array = np.linspace(0,10,101)
predicted = results.params[1] * x_array + results.params[0]

fig, ax = plt.subplots()
ax.scatter(x_data, y_data)
ax.scatter(x_data, results.fittedvalues, color = 'k', marker = 'x')
ax.plot(x_array, predicted, color = 'r')

data_ = np.column_stack([x_data,results.fittedvalues])
ax.plot(([i for (i,j) in data_], [i for (i,j) in data]), ([j for (i,j) in data_], [j for (i,j) in data]), c=[0.6,0.6,0.6], alpha = 0.5)

ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0,10); ax.set_ylim(-2,8)
# fig.savefig('一元线性回归.svg', format='svg')




#%% 主成分分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
# pip install pandas_datareader
import seaborn as sns
import statsmodels.multivariate.pca as pca

# 下载数据
df = pdr.data.DataReader(['DGS6MO','DGS1',
                          'DGS2','DGS5',
                          'DGS7','DGS10',
                          'DGS20','DGS30'],
                          data_source='fred',
                          start='01-01-2022',
                          end='12-31-2022')
df = df.dropna()

# 修改数据帧列标签
df = df.rename(columns={'DGS6MO': '0.5 yr',
                        'DGS1': '1 yr',
                        'DGS2': '2 yr',
                        'DGS5': '5 yr',
                        'DGS7': '7 yr',
                        'DGS10': '10 yr',
                        'DGS20': '20 yr',
                        'DGS30': '30 yr'})

# 绘制利率走势
fig, ax = plt.subplots(figsize = (6,3))
sns.lineplot(df,markers=False,dashes=False,
             palette = "husl",ax = ax)
ax.legend(loc='lower right',ncol=3)

# 计算日收益率
X_df = df.pct_change()
X_df = X_df.dropna()

# 可视化收益率
fig, ax = plt.subplots(figsize = (6,3))
sns.lineplot(X_df,markers=False,
             dashes=False,palette = "husl",ax = ax)
ax.legend(loc='upper right',ncol=3)



# 成对特征散点图
sns.pairplot(X_df, corner=True, diag_kind="kde")



# 相关性系数矩阵
C = X_df.corr()
fig, ax = plt.subplots()
sns.heatmap(C, ax = ax,  annot=True, cmap = 'RdYlBu_r', square = True)



# 主成分分析
pca_model = pca.PCA(X_df, standardize=True)

variance_V = pca_model.eigenvals
# 计算主成分的方差解释比例

explained_var_ratio = variance_V / variance_V.sum()

PC_range = np.arange(len(variance_V)) + 1

labels = ['$PC_' + str(index) + '$' for index in PC_range]


# 陡坡图
fig, ax1 = plt.subplots(figsize = (6,3))

ax1.plot(PC_range, variance_V, 'b', marker = 'x')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Eigen value $\lambda$ (PC variance)', color='b')
ax1.set_ylim(0,1600); ax1.set_xticks(PC_range)

ax2 = ax1.twinx()
ax2.plot(PC_range, np.cumsum(explained_var_ratio)*100, 'r', marker = 'x')
ax2.set_ylabel('Cumulative ratio of explained variance (%)', color='r')
ax2.set_ylim(20,100)
ax2.set_xlim(PC_range.min() - 0.1,PC_range.max() + 0.1)


# PCA载荷
loadings= pca_model.loadings[['comp_0','comp_1','comp_2']]

fig, ax = plt.subplots(figsize = (6,4))
sns.lineplot(data=loadings, markers=True, dashes=False, palette = "husl")
plt.axhline(y=0, color='r', linestyle='-')

# 用前3主成分获得还原数据
X_df_ = pca_model.project(3)


# 比较原始数据和还原数据
# 线图
fig, axes = plt.subplots(4,2,figsize=(4,8))
axes = axes.flatten()

for col_idx, ax_idx in zip(list(X_df_.columns),axes):
    sns.lineplot(X_df_[col_idx],ax = ax_idx)
    sns.lineplot(X_df[col_idx],ax = ax_idx)
    sns.lineplot(X_df[col_idx] - X_df_[col_idx],
                 c = 'k', ax = ax_idx)
    ax_idx.set_xticks([]); ax_idx.set_yticks([])
    ax_idx.axhline(y = 0, c = 'k')



# 散点图
fig, axes = plt.subplots(4,2,figsize=(4,8))
axes = axes.flatten()

for col_idx, ax_idx in zip(list(X_df_.columns),axes):
    sns.scatterplot(x = X_df_[col_idx], y = X_df[col_idx], ax = ax_idx)
    ax_idx.plot([-0.3, 0.3],[-0.3, 0.3],c = 'r')
    ax_idx.set_aspect('equal', adjustable='box')
    ax_idx.set_xticks([]); ax_idx.set_yticks([])
    ax_idx.set_xlim(-0.3, 0.3); ax_idx.set_ylim(-0.3, 0.3)







#%% 一元概率密度估计

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris


# 从Scikit-Learn库加载鸢尾花数据
iris = load_iris()
y = iris.target
X_df = pd.DataFrame(iris.data)
X1_df = X_df.iloc[:,0]

# 自定义可视化函数
def visualize(x1,pdf,color):
    fig, ax = plt.subplots(figsize = (8,3))
    ax.fill_between(x1, pdf, facecolor = color,alpha = 0.2)
    ax.plot(x1, pdf,color = color)
    ax.set_ylim([0,1.4])
    ax.set_xlim([4,8])
    ax.set_ylabel('PDF')
    ax.set_xlabel('Sepal length, $x_1$')

# 不考虑标签
KDE = sm.nonparametric.KDEUnivariate(X1_df)
KDE.fit(bw = 0.1)

x1 = np.linspace(4, 8, 101)
f_x1 = KDE.evaluate(x1)
visualize(x1, f_x1, '#00448A')


# 考虑鸢尾花标签，用KDE描述样本数据花萼长度分布
colors = ['#FF3300','#0099FF','#8A8A8A']
x1 = np.linspace(4,8,161)

for idx in range(3):
    KDE_C_i = sm.nonparametric.KDEUnivariate(X1_df[y==idx])
    KDE_C_i.fit(bw=0.1)
    f_x1_given_C_i = KDE_C_i.evaluate(x1)

    visualize(x1,f_x1_given_C_i,colors[idx])




#%% 二元概率密度估计
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.datasets import load_iris
import scipy.stats as st

# 定义可视化函数
def plot_surface(xx1, xx2, surface, x1_s, x2_s,  z_height, color, title_txt):
    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_wireframe(xx1, xx2, surface, cstride = 8, rstride = 8, color = [0.7,0.7,0.7], linewidth = 0.25)
    ax.scatter(x1_s, x2_s, x2_s*0, c = color)
    ax.contour(xx1, xx2, surface, 20, cmap = 'RdYlBu_r')

    ax.set_proj_type('ortho')
    ax.set_xlabel('Sepal length, $x_1$')
    ax.set_ylabel('Sepal width, $x_2$')
    ax.set_zlabel('PDF')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlim(x1.min(), x1.max())
    # ax.set_ylim(x2.min(), x2.max())
    ax.set_zlim([0,z_height])
    ax.view_init(azim=-120, elev=30)
    ax.set_title(title_txt)
    ax.grid(False)

    ax = fig.add_subplot(1, 2, 2)
    ax.contourf(xx1, xx2, surface, 12, cmap='RdYlBu_r')
    ax.contour(xx1, xx2, surface, 12, colors='w')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(x1.min(), x1.max())
    # ax.set_ylim(x2.min(), x2.max())
    ax.set_xlabel('Sepal length, $x_1$')
    ax.set_ylabel('Sepal width, $x_2$')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title_txt)


# 导入鸢尾花数据
iris = load_iris()
X_1_to_4 = iris.data
y = iris.target
feature_names = ['Sepal length, $X_1$','Sepal width, $X_2$', 'Petal length, $X_3$','Petal width, $X_4$']

X_df = pd.DataFrame(X_1_to_4)
X1_2_df = X_df.iloc[:,[0,1]]

x1 = np.linspace(4,8,161); x2 = np.linspace(1,5,161)
xx1, xx2 = np.meshgrid(x1,x2)
positions = np.vstack([xx1.ravel(), xx2.ravel()])
colors = ['#FF3300','#0099FF','#8A8A8A']

KDE = st.gaussian_kde(X1_2_df.values.T)
f_x1_x2 = np.reshape(KDE(positions).T, xx1.shape)

x1_s = X1_2_df.iloc[:,0]
x2_s = X1_2_df.iloc[:,1]

# 可视化证据因子
z_height = 0.5
title_txt = '$f_{X1, X2}(x_1, x_2)$, evidence'
plot_surface(xx1, xx2, f_x1_x2, x1_s, x2_s, z_height, '#00448A', title_txt)


# 考虑不同鸢尾花分类
for idx in range(3):
    KDE_idx = st.gaussian_kde(X1_2_df[y==idx].values.T)
    f_x1_x2_given_C_i = np.reshape(KDE_idx(positions).T, xx1.shape)

    x1_s_C_i = X1_2_df.iloc[:,0][y==idx]
    x2_s_C_i = X1_2_df.iloc[:,1][y==idx]

    z_height = 1
    title_txt = 'Likelihood'
    plot_surface(xx1, xx2, f_x1_x2_given_C_i, x1_s_C_i, x2_s_C_i, z_height, colors[idx], title_txt)






























































































































































































































































































































































































