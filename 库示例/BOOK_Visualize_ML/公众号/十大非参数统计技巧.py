#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:46:28 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=Mzk0MjUxMzg3OQ==&mid=2247490040&idx=1&sn=0e0ff9ebaeee172a2bace64a6f01880a&chksm=c35e77efea5bc1b75e14c75a674e44a96765a55af8b73c4590f7bea08546a147aa41fcab80f1&mpshare=1&scene=1&srcid=0911oFA7sDGElm8HwlDVsgIQ&sharer_shareinfo=3154df298015c72dc0b3594716a52671&sharer_shareinfo_first=3154df298015c72dc0b3594716a52671&exportkey=n_ChQIAhIQ8frl0ETX%2BADmuTxi3mDU4xKfAgIE97dBBAEAAAAAACngMze3DhEAAAAOpnltbLcz9gKNyK89dVj0mMTniIOXelLMXJjOFpasVcWgJLqdvUrCgTtKf2aZ2RL2AvYYw5uwRlspQi0yiql7vaH840nXj8wIsBpblb3BhDfUD0kX7G9btDgjCTMELtf617DqeSQpq9tklImc9NWls8s89XnES2IALql1L9Fk6bNoW1GL%2F1PAjXJ0Pj8RUbW1vd5kmeImA2nim3y0TJb4Xa09i6JGS%2BxCgk%2FPOkKdDiG4mExNghBDU5aWNyBp5uGo%2BqNNdbLmpRiuNcRE4K4k1gKnPXNVZ7r8WbF8aWF7SG57pmFyg0lYBcg%2B78ICl%2Fvrnm%2B7S4XuNR8ThLkLd6ezmHrXaKJZAQzR&acctmode=0&pass_ticket=uqP8At7Dlo%2BEv%2F%2BUjmT%2BOnwv3XLeGLPBmY1d8N753c%2F3yn%2BkSeykeDSz4xfZirno&wx_header=0#rd


"""

#%%>>>>>>>>>>>>>>>>>>>>>>> 1. K-近邻

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 加载数据集
iris = load_iris()
X, y = iris.data[:, :2], iris.target  # 仅使用前两个特征，方便可视化

# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 寻找最佳 K 值
k_range = range(1, 31)
accuracies = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# 绘制 K 值选择与分类准确率的关系图
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.plot(k_range, accuracies, marker='o', color='magenta', linestyle='-', linewidth=2, markersize=8)
plt.title('Accuracy vs. K Value', fontsize=16)
plt.xlabel('Number of Neighbors K', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(True)

# 绘制决策边界
knn = KNeighborsClassifier(n_neighbors=5)  # 选取最佳 K 值
knn.fit(X_train, y_train)

# 定义网格范围
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测网格点上的分类
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', s=50, cmap=plt.cm.Paired, edgecolor='k')
plt.title('Decision Boundary with KNN (K=5)', fontsize=16)
plt.xlabel('Feature 1 (Standardized)', fontsize=14)
plt.ylabel('Feature 2 (Standardized)', fontsize=14)
plt.grid(True)

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 2. 核密度估计
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# 生成一维数据
np.random.seed(42)
data = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)])

# 定义带宽范围
bandwidths = [0.1, 0.5, 1.0]

# 绘制原始数据的直方图
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.3, color='gray', label='Histogram')

# 计算并绘制不同带宽下的KDE
X_plot = np.linspace(-3, 8, 1000)[:, np.newaxis]

for bandwidth in bandwidths:
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data[:, np.newaxis])
    log_dens = kde.score_samples(X_plot)
    plt.plot(X_plot[:, 0], np.exp(log_dens), label=f'Bandwidth = {bandwidth}', lw=2)

plt.title('Kernel Density Estimation with Different Bandwidths', fontsize=16)
plt.xlabel('Data Value', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 3. 非参数回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 生成示例数据
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.3, X.shape[0])

# 多项式回归
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
y_poly_pred = model.predict(X_poly)

# 绘制结果
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color='blue', s=30, marker='o', label='Data Points')
plt.plot(X, y_poly_pred, color='red', lw=2, label='Polynomial Regression')
plt.title('Nonparametric Regression: Polynomial Fit', fontsize=16)
plt.xlabel('X', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 4. 决策树
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 构建决策树
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y)

# 绘制决策树结构
plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title('Decision Tree Structure', fontsize=16)
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 5. 随机森林
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve

# 加载数据集
data = load_breast_cancer()
X, y = data.data, data.target

# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 构建随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 特征重要性分析
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = data.feature_names

# 绘制特征重要性
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.title("Feature Importances", fontsize=16)
plt.bar(range(X.shape[1]), importances[indices], color="orange", align="center")
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Importance', fontsize=14)

# 学习曲线
train_sizes, train_scores, test_scores = learning_curve(rf, X_train, y_train, cv=5, n_jobs=-1)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.subplot(1, 2, 2)
plt.plot(train_sizes, train_mean, 'o-', color="red", label="Training Score")
plt.plot(train_sizes, test_mean, 'o-', color="blue", label="Cross-Validation Score")
plt.title('Learning Curve', fontsize=16)
plt.xlabel('Training Set Size', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 6. 支持向量机
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# 加载数据集
digits = datasets.load_digits()
X = digits.data
y = digits.target

# 使用PCA降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 训练SVM模型
svc = SVC(kernel='rbf', C=1.0, gamma=0.01)
svc.fit(X_pca, y)

# 绘制决策边界
plt.figure(figsize=(12, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.rainbow, edgecolor='k', s=50)
plt.title('Support Vector Machine Decision Boundary', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)

# 支持向量可视化
sv = svc.support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], facecolors='none', s=100, edgecolor='k', label='Support Vectors')

plt.legend()
plt.grid(True)
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 7. 最近邻图

import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering

# 生成瑞士卷数据
X, _ = make_swiss_roll(n_samples=10000, noise=0.8)

# 构建K-近邻图
kng = kneighbors_graph(X, n_neighbors=10, mode='connectivity', include_self=False)

# 谱聚类
sc = SpectralClustering(n_clusters=6, affinity='precomputed', random_state=42)
labels = sc.fit_predict(kng)

# 可视化
plt.figure(figsize=(12, 6))
plt.scatter(X[:, 0], X[:, 2], c=labels, cmap='viridis', edgecolor='k', s=20)
plt.title('Spectral Clustering on Nearest Neighbor Graph', fontsize=16)
plt.xlabel('X axis', fontsize=14)
plt.ylabel('Z axis', fontsize=14)
plt.grid(True)
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 8. 核主成分分析

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA

# 生成瑞士卷数据
X, color = make_swiss_roll(n_samples=10000, noise=0.8)

# 使用RBF核进行KPCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.04)
X_kpca = kpca.fit_transform(X)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=color, cmap=plt.cm.Spectral, s=20)
plt.title('Kernel PCA on Swiss Roll Data', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)
plt.colorbar()
plt.grid(True)
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 9. 自适应平滑法


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 生成心电图模拟数据
np.random.seed(42)
t = np.linspace(0, 10, 500)
signal = np.sin(2 * np.pi * t) + np.random.normal(0, 0.5, t.shape)

# 自适应平滑方法：Savitzky-Golay滤波器
window_length = 51
poly_order = 3
smoothed_signal = savgol_filter(signal, window_length, poly_order)

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(t, signal, color='gray', label='Original Signal')
plt.plot(t, smoothed_signal, color='red', lw=2, label='Smoothed Signal')
plt.title('Adaptive Smoothing of ECG Signal', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()






#%%>>>>>>>>>>>>>>>>>>>>>>> 10. 分位数回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 加载数据集
data = fetch_california_housing()
X = data.data[:, 0]  # 使用第一个特征
y = data.target

# 数据转换为DataFrame格式
import pandas as pd
df = pd.DataFrame({'MedInc': X, 'MedHouseVal': y})

# 分位数回归
quantiles = [0.1, 0.5, 0.9]
models = {}
predictions = pd.DataFrame({'MedInc': np.linspace(df.MedInc.min(), df.MedInc.max(), 100)})

plt.figure(figsize=(10, 6))
plt.scatter(df.MedInc, df.MedHouseVal, alpha=0.3, color='blue', label='Data')

for q in quantiles:
    model = smf.quantreg('MedHouseVal ~ MedInc', df).fit(q=q)
    models[q] = model
    predictions[f'quantile_{q}'] = model.predict(predictions['MedInc'])
    plt.plot(predictions['MedInc'], predictions[f'quantile_{q}'], label=f'Quantile {q*100}%', lw=2)

plt.title('Quantile Regression on California Housing Data', fontsize=16)
plt.xlabel('Median Income', fontsize=14)
plt.ylabel('Median House Value', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()








