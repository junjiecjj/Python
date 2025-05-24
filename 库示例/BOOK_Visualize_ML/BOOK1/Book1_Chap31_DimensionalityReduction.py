










#  sklearn.preprocessing.StandardScaler() 用于对数据进行标准化处理
#  sklearn.decomposition.PCA() 执行主成分分析 PCA 以减少数据维度
#  sklearn.covariance.EmpiricalCovariance() 计算基于样本的经验协方差矩阵
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 从几何角度理解PCA



# 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
# pip install pandas_datareader
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance

# 下载数据
df = pdr.data.DataReader(['DGS6MO','DGS1'], data_source='fred', start='01-01-2022', end='12-31-2022')
# df.to_csv('IR_data.csv')
# 如果不能下载数据，请用pandas.read_csv() 读入配套数据
df = df.dropna()

# 修改列标签
df = df.rename(columns={'DGS6MO': 'X1', 'DGS1': 'X2'})
df.head()


X_df = df.pct_change();
# 计算日收益率
X_df = X_df.dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)

np.cov(X_scaled.T)

x1_array = np.linspace(-6,6,601)
x2_array = np.linspace(-6,6,601)
xx1, xx2 = np.meshgrid(x1_array, x2_array)
xx12 = np.c_[xx1.ravel(), xx2.ravel()] # (361201, 2)

# 计算曼哈顿距离
COV = EmpiricalCovariance().fit(X_scaled)
mahal_sq_Xc = COV.mahalanobis(xx12)
mahal_sq_dd = mahal_sq_Xc.reshape(xx1.shape)
mahal_dd = np.sqrt(mahal_sq_dd)

###################### 图 8. 标准化数据的散点图
fig, ax = plt.subplots()
# 绘制马氏距离填充等高线
plt.contourf(xx1, xx2, mahal_dd, cmap='Blues_r', levels=np.linspace(0,6,13))
# 绘制样本数据 (标准化) 散点图
plt.scatter(X_scaled[:,0], X_scaled[:,1], s = 38, edgecolor = 'w', alpha = 0.5, marker = '.', color = 'k')
# 绘制样本数据质心
plt.plot(X_scaled[:,0].mean(), X_scaled[:,1].mean(), marker = 'x', color = 'k', markersize = 18)


ax.axvline(x = 0, c = 'k')
ax.axhline(y = 0, c = 'k')
ax.grid('off')
ax.set_aspect('equal', adjustable='box')
ax.set_xbound(lower = -6, upper = 6)
ax.set_ybound(lower = -6, upper = 6)

#####################
fig, ax = plt.subplots()
plt.contour(xx1, xx2, mahal_dd, colors='k', levels=np.linspace(0,6,13))

ax.axvline(x = 0, c = 'k')
ax.axhline(y = 0, c = 'k')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')
ax.set_aspect('equal', adjustable='box')
ax.set_xbound(lower = -6, upper = 6)
ax.set_ybound(lower = -6, upper = 6)


from sklearn.decomposition import PCA

# 主成分分析
n_components = 2 # 主成分数量
pca = PCA(n_components = 2)

# 拟合PCA模型
pca.fit(X_scaled)

# 获取loadings（主成分方向向量）
loadings = pca.components_.T
V = loadings
print(np.round(loadings.T @ loadings))
# [[1. 0.]
# [0. 1.]]
v1 = V[:,[0]] # 第一主成分方向
v2 = V[:,[1]] # 第二主成分方向


pca.explained_variance_
pca.explained_variance_ratio_


# 自定义绘制向量函数
def draw_vector(vector,RBG):
    array = np.array([[0, 0, vector[0], vector[1]]], dtype=object)
    X, Y, U, V = zip(*array)
    plt.quiver(X, Y, U, V,angles='xy', scale_units='xy',scale=1,color = RBG, zorder = 1e5)

# 图 9. 主成分方向
fig, ax = plt.subplots()
plt.contourf(xx1, xx2, mahal_dd, cmap='Blues_r', levels=np.linspace(0,6,13))
plt.scatter(X_scaled[:,0],X_scaled[:,1], s = 38, edgecolor = 'w', alpha = 0.5, marker = '.', color = 'k')
plt.plot(X_scaled[:,0].mean(),X_scaled[:,1].mean(), marker = 'x', color = 'k', markersize = 18)
# 可视化两个主成分方向
draw_vector(v1,'r')
draw_vector(v2,'r')

# 绘制两条参考线
ax.plot(x1_array,x1_array*v1[1]/v1[0], 'r', lw = 0.25, ls = 'dashed')
ax.plot(x1_array,x1_array*v2[1]/v2[0], 'r', lw = 0.25, ls = 'dashed')

ax.axvline(x = 0, c = 'k'); ax.axhline(y = 0, c = 'k')
ax.grid('off')
ax.set_aspect('equal', adjustable='box')
ax.set_xbound(lower = -6, upper = 6); ax.set_ybound(lower = -6, upper = 6)


# 完成投影
proj1 = v1@v1.T
z1_2D = X_scaled@proj1
fig, ax = plt.subplots()
plt.contourf(xx1, xx2, mahal_dd, cmap='Blues_r', levels=np.linspace(0,6,13))
plt.scatter(X_scaled[:,0],X_scaled[:,1], s = 38, edgecolor = 'w', alpha = 0.5, marker = '.', color = 'k')
plt.plot(X_scaled[:,0].mean(),X_scaled[:,1].mean(), marker = 'x', color = 'k', markersize = 18)

plt.scatter(z1_2D[:,0],z1_2D[:,1], s = 38, edgecolor = 'w', alpha = 0.5, marker = '.', color = 'r')
plt.plot(([i for (i,j) in z1_2D], [i for (i,j) in X_scaled]), ([j for (i,j) in z1_2D], [j for (i,j) in X_scaled]), c='k', lw = 0.25)

draw_vector(v1,'r')
ax.plot(x1_array,x1_array*v1[1]/v1[0], 'r', lw = 0.25)

ax.axvline(x = 0, c = 'k')
ax.axhline(y = 0, c = 'k')
ax.grid('off')
ax.set_aspect('equal', adjustable='box')
ax.set_xbound(lower = -6, upper = 6)
ax.set_ybound(lower = -6, upper = 6)

# 完成投影
proj2 = v2@v2.T
z2_2D = X_scaled@proj2
fig, ax = plt.subplots()
plt.contourf(xx1, xx2, mahal_dd, cmap='Blues_r', levels=np.linspace(0,6,13))
plt.scatter(X_scaled[:,0],X_scaled[:,1], s = 38, edgecolor = 'w', alpha = 0.5, marker = '.', color = 'k')
plt.plot(X_scaled[:,0].mean(),X_scaled[:,1].mean(), marker = 'x', color = 'k', markersize = 18)
plt.scatter(z2_2D[:,0],z2_2D[:,1], s = 38, edgecolor = 'w', alpha = 0.5, marker = '.', color = 'r')
plt.plot(([i for (i,j) in z2_2D], [i for (i,j) in X_scaled]), ([j for (i,j) in z2_2D], [j for (i,j) in X_scaled]), c='k', lw = 0.25)

draw_vector(v2,'r')
ax.plot(x1_array,x1_array*v2[1]/v2[0], 'r', lw = 0.25)

ax.axvline(x = 0, c = 'k')
ax.axhline(y = 0, c = 'k')
ax.grid('off')
ax.set_aspect('equal', adjustable='box')
ax.set_xbound(lower = -6, upper = 6)
ax.set_ybound(lower = -6, upper = 6)



Z = X_scaled @ V
COV = EmpiricalCovariance().fit(Z)

mahal_sq_Xc = COV.mahalanobis(xx12)
mahal_sq_dd = mahal_sq_Xc.reshape(xx1.shape)
mahal_dd = np.sqrt(mahal_sq_dd)

fig, ax = plt.subplots()
plt.contourf(xx1, xx2, mahal_dd, cmap='Blues_r', levels=np.linspace(0,6,13))
plt.scatter(Z[:,0],Z[:,1], s = 38, edgecolor = 'w', alpha = 0.5, marker = '.', color = 'k')
plt.plot(X_scaled[:,0].mean(),X_scaled[:,1].mean(), marker = 'x', color = 'k', markersize = 18)

ax.axvline(x = 0, c = 'k')
ax.axhline(y = 0, c = 'k')
ax.grid('off')
ax.set_aspect('equal', adjustable='box')
ax.set_xbound(lower = -6, upper = 6)
ax.set_ybound(lower = -6, upper = 6)




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

































#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




















































































































































































































































































































































































































































































































































































































































































































































































































