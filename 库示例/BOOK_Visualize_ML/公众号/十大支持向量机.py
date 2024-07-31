#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:52:30 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247485183&idx=1&sn=818e402135ba3b0b80ecb65cbf868285&chksm=c0e5dc39f792552f5182e72b09f283c138922e0904f68ffd929f73b2e337f26bda7717bd2ba1&mpshare=1&scene=1&srcid=0729wRRDKHcfDwF0JcFUOhi5&sharer_shareinfo=1def42763ac1a6dafaa4ce39b3617b21&sharer_shareinfo_first=1def42763ac1a6dafaa4ce39b3617b21&exportkey=n_ChQIAhIQxnP61GzvWBEbhj052IHsQhKfAgIE97dBBAEAAAAAAGqFBdkbgX8AAAAOpnltbLcz9gKNyK89dVj0JUhdMRIq4uGxSqu2cuhX5FuK5Oyd5AShRXR1oYpdfo1Ts52sBygKP1ButXhWtzj2pzj7lbB9dSBshJyD8vC%2BESPwMj8VzvEXZrfna5zoHvEh2sFaFNJv3G8az09cCWanqqX0v9%2Bb19JoJiYV1dWVewrkCym8iQhSWpUEvgUF3gYEpuYsfTA04avYKM7GpeAqIvBQ5QHx3yeIpQdwY0xpMhPlHBbSSatbZLenYRlOIpRbIiTLL6nFENGmM1J%2FVuueXLLfnVsn%2B7kvHCW3GRHkfg0pFFefRBZ0KkDnYj%2B6gVGs4wqoPn7BnBrZA%2BVue9tjPSKGRDe7gwvw&acctmode=0&pass_ticket=UfGu8jrlbq3LcOqA6dEkVn%2BA2egnBzksDMnoWvkPQsfUojm0AJzEvCP6GPw8BiGc&wx_header=0#rd


https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247484101&idx=1&sn=336b6ad938c0fd704824ad535d4cbd8d&chksm=c0e5d803f7925115ce75287cbf683d6a136c7e5f63de04db159b4125dce9e348e4c1ab2d2bbf&cur_album_id=3445855686331105280&scene=190#rd

"""

#%% 1. 标准支持向量机（Standard SVM）
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 生成虚拟数据集
X, y = datasets.make_moons(n_samples=800, noise=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练SVM模型
svm = SVC(kernel='rbf', C=1.0, gamma=0.5)
svm.fit(X_train, y_train)

# 绘制决策边界和支持向量
def plot_decision_boundary(model, X, y):
    h = .02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o', s=20)
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], facecolors='none', edgecolors='r', s=100, linewidth=1.5)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary and Support Vectors')
    plt.show()

# 绘制分类结果图
def plot_classification_results(X_train, y_train, X_test, y_test, model):
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', marker='o', s=20)
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c = model.predict(X_test), edgecolor='k', marker='o', s=20)
    plt.title('Test Data Classification')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.show()

# 绘制决策边界和支持向量
plot_decision_boundary(svm, X, y)

# 绘制分类结果图
plot_classification_results(X_train, y_train, X_test, y_test, svm)



#%% 2. 支持向量回归（Support Vector Regression, SVR）
import numpy as np
import matplotlib.pyplot as plt
# SVR用于回归任务，旨在找到一个与数据点最接近的回归线，同时允许有一定的误差。
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 生成虚拟数据集
np.random.seed(42)
X = np.sort(5 * np.random.rand(500, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X) # (500, 1)
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel() # (500,)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练SVR模型
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train, y_train)

# 预测
y_train_pred = svr.predict(X_train)
y_test_pred = svr.predict(X_test)

# 绘制回归结果
def plot_regression_results(X, y, y_pred, title):
    plt.scatter(X, y, color='blue', label='Actual')
    plt.scatter(X, y_pred, color='red', label='Predicted')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title(title)
    plt.legend()
    plt.show()

# 绘制残差图
def plot_residuals(y, y_pred, title):
    residuals = y - y_pred
    plt.scatter(y_pred, residuals, color='purple')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.show()

# 绘制训练集回归结果
plot_regression_results(X_train, y_train, y_train_pred, 'SVR Regression Results (Training Set)')

# 绘制测试集回归结果
plot_regression_results(X_test, y_test, y_test_pred, 'SVR Regression Results (Test Set)')

# 绘制训练集残差图
plot_residuals(y_train, y_train_pred, 'Residuals Analysis (Training Set)')

# 绘制测试集残差图
plot_residuals(y_test, y_test_pred, 'Residuals Analysis (Test Set)')



#%% 3. 线性支持向量机（Linear SVM）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

# 生成虚拟数据集
X, y = make_classification( n_samples=500, n_features=3, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# 使用线性支持向量机进行训练
clf = SVC(kernel='linear')
clf.fit(X, y)

# 绘制决策边界图（2D）
def plot_2d_decision_boundary(X, y, clf):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=30, edgecolors='k')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建网格以评估模型
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200), np.linspace(ylim[0], ylim[1], 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和平面
    ax.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], alpha=0.3, colors=['blue', 'red'])
    ax.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

    plt.title('2D Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# 绘制3D决策面图
def plot_3d_decision_surface(X, y, clf):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='coolwarm', s=30, edgecolors='k')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    # 创建网格以评估模型
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 30), np.linspace(ylim[0], ylim[1], 30))
    zz = (-clf.intercept_[0] - clf.coef_[0][0] * xx - clf.coef_[0][1] * yy) / clf.coef_[0][2]

    # 绘制决策面
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='blue')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.title('3D Decision Surface')
    plt.show()

# 绘制图形
plot_2d_decision_boundary(X, y, clf)
plot_3d_decision_surface(X, y, clf)




#%% 4. 核支持向量机（Kernel SVM）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# 生成虚拟数据集
X, y = make_circles(n_samples=500, factor=0.3, noise=0.1, random_state=42)

# 使用RBF核的支持向量机进行训练
clf = SVC(kernel='rbf', C=1, gamma=2)
clf.fit(X, y)

# 绘制决策边界图（2D）
def plot_2d_decision_boundary(X, y, clf):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=30, edgecolors='k')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建网格以评估模型
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200), np.linspace(ylim[0], ylim[1], 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和平面
    ax.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], alpha=0.3, colors=['blue', 'red'])
    ax.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

    plt.title('2D Decision Boundary with RBF Kernel')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# 绘制特征空间投影图
def plot_feature_space_projection(X, y, clf):
    # 使用PCA进行特征空间投影
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', s=30, edgecolors='k')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建网格以评估模型
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200), np.linspace(ylim[0], ylim[1], 200))
    Z = clf.decision_function(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和平面
    ax.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], alpha=0.3, colors=['blue', 'red'])
    ax.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

    plt.title('Feature Space Projection with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

# 绘制图形
plot_2d_decision_boundary(X, y, clf)
plot_feature_space_projection(X, y, clf)



#%% 5. 序列最小优化（Sequential Minimal Optimization, SMO）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# 生成虚拟数据集
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# 使用RBF核的支持向量机进行训练
clf = SVC(kernel='rbf', C=1.0)
clf.fit(X, y)

# 绘制决策边界图（2D）
def plot_2d_decision_boundary(X, y, clf):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=30, edgecolors='k')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建网格以评估模型
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200), np.linspace(ylim[0], ylim[1], 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和平面
    ax.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], alpha=0.3, colors=['blue', 'red'])
    ax.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

    plt.title('2D Decision Boundary with SMO-SVM (RBF Kernel)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# 绘制支持向量分布图
def plot_support_vectors(X, y, clf):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=30, edgecolors='k')
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], facecolors='none', edgecolors='k', s=100, linewidths=1.5)

    plt.title('Support Vectors Distribution')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# 绘制图形
plot_2d_decision_boundary(X, y, clf)
plot_support_vectors(X, y, clf)


#%% 6. 拉格朗日对偶问题（Lagrangian Dual Problem）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 生成非线性可分数据集
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用RBF核的支持向量机进行分类
clf = SVC(kernel='rbf', C=1, gamma='scale')
clf.fit(X_scaled, y)

# 绘制原始数据集的散点图
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(X_scaled[y == 0][:, 0], X_scaled[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], color='blue', label='Class 1')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# 绘制分类决策边界及支持向量
ax = plt.subplot(2, 2, 2)
plt.scatter(X_scaled[y == 0][:, 0], X_scaled[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X_scaled[y == 1][:, 0], X_scaled[y == 1][:, 1], color='blue', label='Class 1')

# 绘制支持向量
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label='Support Vectors')

# 绘制决策边界
xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 500), np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
contour = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k')

plt.title('Decision Boundary and Support Vectors')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# 使用PCA进行特征空间变换后的散点图
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.subplot(2, 2, 3)
plt.scatter(X_pca[y == 0][:, 0], X_pca[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X_pca[y == 1][:, 0], X_pca[y == 1][:, 1], color='blue', label='Class 1')
plt.title('PCA Transformed Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# 决策边界在PCA变换后的空间
ax = plt.subplot(2, 2, 4)
plt.scatter(X_pca[y == 0][:, 0], X_pca[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X_pca[y == 1][:, 0], X_pca[y == 1][:, 1], color='blue', label='Class 1')

Z_pca = clf.decision_function(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z_pca = Z_pca.reshape(xx.shape)
contour = plt.contour(xx, yy, Z_pca, levels=[0], linewidths=2, colors='k')

plt.title('Decision Boundary in PCA Transformed Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

plt.tight_layout()
plt.show()






#%% 7. 支持向量聚类（Support Vector Clustering, SVC）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

# 生成虚拟数据集
X, _ = make_blobs(n_samples=1000, centers=3, cluster_std=0.60, random_state=0)

# 使用One-Class SVM进行支持向量聚类
svc = OneClassSVM(kernel='rbf', nu=0.1, gamma=0.1)
svc.fit(X)

# 绘制支持向量聚类边界图
def plot_svc_decision_boundary(X, svc):
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], s=30, edgecolors='k')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500), np.linspace(ylim[0], ylim[1], 500))
    Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k')

    plt.title('Support Vector Clustering Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# 绘制PCA投影图
def plot_pca_projection(X):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=30, edgecolors='k')
    plt.title('PCA Projection of Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

# 绘制图形
plot_svc_decision_boundary(X, svc)
plot_pca_projection(X)



#%% 8. 多类支持向量机（Multiclass SVM）

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 生成虚拟数据集
X, y = datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, n_classes=3, random_state=42)

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练多类SVM模型
svm = SVC(kernel='rbf', decision_function_shape='ovo')
svm.fit(X_train, y_train)

# 创建网格以绘制决策边界
h = .02  # 网格步长
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 预测整个网格
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 图1：决策边界和分类结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', marker='d')
plt.title('Decision Boundary and Classification Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(['Train', 'Test'], loc='upper right')

# 图2：特征空间中的数据分布和支持向量
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
support_vectors = svm.support_vectors_
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', edgecolors='r')
plt.title('Data Distribution and Support Vectors in Feature Space')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(['Data points', 'Support Vectors'], loc='upper right')

plt.tight_layout()
plt.show()




#%% 9. 软间隔支持向量机（Soft Margin SVM）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import  roc_curve, auc

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, flip_y=0.1, class_sep=1.0, random_state=42)

# 创建支持向量机模型
model = SVC(kernel='linear', C = 1.0, probability=True)
model.fit(X, y)

# 预测
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# 绘制分类结果和决策边界
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 子图1：数据点及其分类结果
ax1 = axes[0]
sc = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=30, edgecolors='k')
ax1.set_title('Data Points and Classification Result')

# 绘制决策边界
xlim = ax1.get_xlim()
ylim = ax1.get_ylim()
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax1.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k')

# 子图2：混淆矩阵
ax2 = axes[1]
# plot_confusion_matrix(model, X, y, ax=ax2, cmap='coolwarm')
# ax2.set_title('Confusion Matrix')

plt.tight_layout()
plt.show()

# 绘制ROC曲线
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()






#%% 10. 半监督支持向量机（Semi-Supervised Support Vector Machine, S3VM）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成虚拟数据集
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
# (1000, 2), (1000,)
# 确保有一部分数据有标签且包含两个类别
n_labeled_points = 50
y_unlabeled = np.copy(y)
y_unlabeled[n_labeled_points:] = -1  # 将部分标签设为未标记

# 分割数据集，部分有标签，部分无标签
X_train, X_test, y_train, y_test = train_test_split(X, y_unlabeled, test_size=0.3, random_state=42)
y_train_true = y[y_unlabeled == -1]  # 仅用于计算准确度，不用于训练

# 定义半监督学习模型
base_svc = SVC(kernel='rbf', gamma=0.5, probability=True)
self_training_model = SelfTrainingClassifier(base_svc)

# 训练模型
self_training_model.fit(X_train, y_train)

# 预测测试集
y_pred = self_training_model.predict(X_test)

# 计算准确度
accuracy = accuracy_score(y_test[y_test != -1], y_pred[y_test != -1])
print(f'Accuracy: {accuracy:.2f}')

# 绘制数据分布和分类决策边界
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary with Semi-Supervised SVM')

# 绘制训练集数据分布
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plot_decision_boundary(self_training_model, X_train, y_train)
plt.title("Training Data with Decision Boundary")

# 绘制测试集数据分布
plt.subplot(1, 2, 2)
plot_decision_boundary(self_training_model, X_test, y_test)
plt.title("Test Data with Decision Boundary")
plt.show()




#%%















