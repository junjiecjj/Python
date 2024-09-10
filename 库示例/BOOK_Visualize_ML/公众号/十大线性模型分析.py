#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:31:28 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247486596&idx=1&sn=56fca8ae947281d7aa180f0b0dc9ffa1&chksm=c16c5d290d469b5cc935e269f5f1c0b16ed2ef28cb23e0dc8cbf716d91b0e8559279a581407b&mpshare=1&scene=1&srcid=09109vBkxGraIpjaV0H0axFN&sharer_shareinfo=ad298834acb7dcf4b7b1194d09ffc36a&sharer_shareinfo_first=ad298834acb7dcf4b7b1194d09ffc36a&exportkey=n_ChQIAhIQGin9xQeaYkJPG77u47nvBxKfAgIE97dBBAEAAAAAANqbL%2BopjskAAAAOpnltbLcz9gKNyK89dVj0KB4omnIDXEi2atIB%2F1C0Wo%2FharnFoD2Ztf8l09QubOQlnDFH0cfHvzZlHhUZ9CzACsqB%2Bwv%2B6TmBxVZTGzYH%2FnkNIr0D4cwab%2F9HsYUseeWHpJ%2FJ6VJdGVk%2B1gPbOD3IQa9W8Ifema%2F0I4vm%2BdgHudAslRMW%2FTb8W6MH1rnSAIolE5lqmMWyAdM446UQnlLWq2D8N8lwCdMoH9TQGSTwquq%2FMijl%2Boko5XZbLlbW72SCPMtMYxeVTxZuHNFE%2BqvQzj12IboAU5i28q2BbIcur6XbM4ycEg2wG%2Fpob32kZJRzU4spwKm7i9O8MKaOHmY9mncIIVHMb2iV&acctmode=0&pass_ticket=e6WpifvLhNjmr4rbsH%2BbRAcDfuzeQTaukj7C2qIAbX1hL9BgGcJzPL2DZA8fpyrV&wx_header=0#rd


"""


#%%>>>>>>>>>>>>>>>>>>>>>>> 1. 线性回归
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 设置随机种子保证结果可重复
np.random.seed(42)

# 生成虚拟数据集
# 假设我们有一个线性关系 y = 3x + 5, 加入一些随机噪声
X = 2 * np.random.rand(100, 1)
y = 3 * X + 5 + np.random.randn(100, 1)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型并拟合数据
model = LinearRegression()
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算残差
residuals = y_test - y_pred

# 创建画布，设置大小
plt.figure(figsize=(15, 10))

# 子图1：散点图和回归线
plt.subplot(2, 2, 1)
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.title('Scatter plot with Regression Line', fontsize=14)
plt.xlabel('X values', fontsize=12)
plt.ylabel('y values', fontsize=12)
plt.legend()

# 子图2：残差图
plt.subplot(2, 2, 2)
plt.scatter(y_pred, residuals, color='green', edgecolor='black')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot', fontsize=14)
plt.xlabel('Predicted values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)

# 子图3：残差直方图
plt.subplot(2, 2, 3)
sns.histplot(residuals, kde=True, color='purple', edgecolor='black')
plt.title('Histogram of Residuals', fontsize=14)
plt.xlabel('Residuals', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# 子图4：拟合值和实际值对比
plt.subplot(2, 2, 4)
plt.plot(y_test, y_pred, 'o', color='orange', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='Ideal Fit')
plt.title('Predicted vs Actual values', fontsize=14)
plt.xlabel('Actual values', fontsize=12)
plt.ylabel('Predicted values', fontsize=12)
plt.legend()

# 调整子图布局
plt.tight_layout()

# 显示图形
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 2. 岭回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 生成虚拟数据集
np.random.seed(42)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)

# 为了引入多重共线性，生成高度相关的特征
X[:, 1] = X[:, 0] + np.random.normal(0, 0.1, size=n_samples)
X[:, 2] = X[:, 0] - np.random.normal(0, 0.1, size=n_samples)

# 生成目标变量
coef = np.array([1.5, -2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
y = np.dot(X, coef) + np.random.normal(0, 1, size=n_samples)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 正则化强度列表
alphas = np.logspace(-6, 6, 200)

# 存储系数
coefs = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)

# 转换为numpy数组
coefs = np.array(coefs)

# 训练岭回归模型并预测
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# 线性回归模型用于对比
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 绘制图像
plt.figure(figsize=(14, 6))

# 左图：岭回归系数路径
plt.subplot(1, 2, 1)
plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Ridge Coefficients as a function of Regularization')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend([f'Feature {i+1}' for i in range(n_features)], loc='upper right', bbox_to_anchor=(1.25, 1.05), fontsize=8)
plt.grid(True, linestyle='--', alpha=0.7)

# 右图：预测值 vs 实际值
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_ridge, color='blue', label='Ridge Regression', alpha=0.6)
plt.scatter(y_test, y_pred_lr, color='red', marker='x', label='Linear Regression', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 3. Lasso回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 生成虚拟数据
np.random.seed(42)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)
true_coef = np.zeros(n_features)
true_coef[:4] = [3, -2, 1.5, -4]  # 前四个特征有非零权重
y = X.dot(true_coef) + np.random.normal(0, 0.5, size=n_samples)

# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Lasso回归模型
alphas = np.logspace(-4, 0, 100)
lasso_coefs = []
mse_list = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)
    lasso_coefs.append(lasso.coef_)

lasso_coefs = np.array(lasso_coefs)

# 可视化
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1. 数据点及拟合曲线
axs[0, 0].scatter(y_test, y_pred, color='dodgerblue', label='Predicted vs True', edgecolor='k')
axs[0, 0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, label='Ideal')
axs[0, 0].set_title('True vs Predicted Values')
axs[0, 0].set_xlabel('True Values')
axs[0, 0].set_ylabel('Predicted Values')
axs[0, 0].legend()

# 2. Lasso系数随alpha变化的图
for i in range(n_features):
    axs[0, 1].plot(alphas, lasso_coefs[:, i], lw=2, label=f'Feature {i+1}')
axs[0, 1].set_xscale('log')
axs[0, 1].set_title('Lasso Coefficients vs Alpha')
axs[0, 1].set_xlabel('Alpha')
axs[0, 1].set_ylabel('Coefficient Value')
axs[0, 1].legend()

# 3. Alpha值与MSE之间的关系
axs[1, 0].plot(alphas, mse_list, color='purple', lw=3)
axs[1, 0].set_xscale('log')
axs[1, 0].set_title('Mean Squared Error vs Alpha')
axs[1, 0].set_xlabel('Alpha')
axs[1, 0].set_ylabel('Mean Squared Error')

# 4. 残差图 (Residual plot)
axs[1, 1].scatter(y_pred, y_test - y_pred, color='green', edgecolor='k')
axs[1, 1].axhline(0, color='red', lw=2)
axs[1, 1].set_title('Residual Plot')
axs[1, 1].set_xlabel('Predicted Values')
axs[1, 1].set_ylabel('Residuals')

# 调整布局
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 4. 弹性网回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 设置随机种子
np.random.seed(42)

# 生成虚拟数据集
n_samples, n_features = 100, 200
X = np.random.randn(n_samples, n_features)

# 设置真实的系数，只有前10个特征有信号，其他为噪声
true_coefficients = np.zeros(n_features)
true_coefficients[:10] = np.random.randn(10)

# 生成目标变量
y = np.dot(X, true_coefficients) + np.random.normal(size=n_samples)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 训练弹性网回归模型
alpha = 0.1
l1_ratio = 0.5
elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
elastic_net.fit(X_train, y_train)

# 预测
y_pred_train = elastic_net.predict(X_train)
y_pred_test = elastic_net.predict(X_test)

# 模型评估
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

# 获取系数
coefficients = elastic_net.coef_

# 图形绘制
fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

# 1. 系数路径图
axs[0, 0].stem(range(n_features), coefficients, linefmt='r-', markerfmt='ro', basefmt='b-')
axs[0, 0].set_title('Elastic Net Coefficients')
axs[0, 0].set_xlabel('Feature Index')
axs[0, 0].set_ylabel('Coefficient Value')
axs[0, 0].grid(True)

# 2. 训练集 vs 测试集预测结果散点图
axs[0, 1].scatter(y_train, y_pred_train, color='blue', label='Train', alpha=0.6)
axs[0, 1].scatter(y_test, y_pred_test, color='green', label='Test', alpha=0.6)
axs[0, 1].plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
axs[0, 1].set_title('Predicted vs Actual')
axs[0, 1].set_xlabel('Actual Value')
axs[0, 1].set_ylabel('Predicted Value')
axs[0, 1].legend()
axs[0, 1].grid(True)

# 3. 残差图
residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test
axs[1, 0].scatter(y_pred_train, residuals_train, color='blue', label='Train Residuals', alpha=0.6)
axs[1, 0].scatter(y_pred_test, residuals_test, color='green', label='Test Residuals', alpha=0.6)
axs[1, 0].hlines(0, min(y_pred_train), max(y_pred_train), color='red', linestyle='--')
axs[1, 0].set_title('Residuals vs Predicted')
axs[1, 0].set_xlabel('Predicted Value')
axs[1, 0].set_ylabel('Residuals')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 4. 均方误差柱状图
axs[1, 1].bar(['Train MSE', 'Test MSE'], [mse_train, mse_test], color=['blue', 'green'])
axs[1, 1].set_title('Mean Squared Error')
axs[1, 1].set_ylabel('MSE')
axs[1, 1].grid(True)

plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 5. 逻辑回归

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 生成一个二分类虚拟数据集
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1,
                           flip_y=0.1, class_sep=1.5, random_state=42)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 获取模型的预测概率
y_prob = model.predict_proba(X_test)[:, 1]

# 计算测试集上的损失
loss = log_loss(y_test, y_prob)

# 创建网格用于绘制决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

# 预测网格上每个点的分类概率
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 创建一个空的图像窗口，准备绘制子图
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# 子图1：绘制数据点的分布及模型的决策边界
ax[0].contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
scatter = ax[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, edgecolor='k', cmap=plt.cm.coolwarm)
ax[0].set_title('Data Distribution and Decision Boundary', fontsize=14)
ax[0].set_xlabel('Feature 1', fontsize=12)
ax[0].set_ylabel('Feature 2', fontsize=12)
legend1 = ax[0].legend(*scatter.legend_elements(), loc="upper right", title="Classes")
ax[0].add_artist(legend1)

# 子图2：绘制损失函数的收敛过程（这里用一个简化的示例展示损失变化）
iterations = np.arange(1, 101)
losses = [log_loss(y_train, model.predict_proba(X_train)[:, 1]) for _ in iterations]

ax[1].plot(iterations, losses, color='red', marker='o', linestyle='-', linewidth=2, markersize=5)
ax[1].set_title('Log Loss Convergence', fontsize=14)
ax[1].set_xlabel('Iterations', fontsize=12)
ax[1].set_ylabel('Log Loss', fontsize=12)
ax[1].grid(True)

# 调整整体布局
plt.tight_layout()

# 展示图像
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 7. 线性判别分析
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: 生成虚拟数据集
X, y = make_classification(n_samples=300, n_features=6, n_informative=4, n_classes=3,
                           n_clusters_per_class=1, random_state=42)

# Step 2: 数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: 使用LDA降维
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Step 5: 绘图 - 原始数据和LDA降维后的数据
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=100)

# 设置颜色
colors = ['red', 'blue', 'green']
labels = ['Class 0', 'Class 1', 'Class 2']

# 绘制原始数据分布
for i, color, label in zip(np.unique(y_train), colors, labels):
    ax1.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1],
                color=color, label=label, alpha=0.7, edgecolors='k')

ax1.set_title('Original Data (First two features)', fontsize=14)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend(loc='best')
ax1.grid(True)

# 绘制LDA降维后的数据分布
for i, color, label in zip(np.unique(y_train), colors, labels):
    ax2.scatter(X_train_lda[y_train == i, 0], X_train_lda[y_train == i, 1],
                color=color, label=label, alpha=0.7, edgecolors='k')

ax2.set_title('LDA Transformed Data (2D projection)', fontsize=14)
ax2.set_xlabel('LDA Component 1')
ax2.set_ylabel('LDA Component 2')
ax2.legend(loc='best')
ax2.grid(True)

# 总标题
plt.suptitle('LDA Analysis: Original Data vs LDA Transformed Data', fontsize=16)
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 8. 二次判别分析
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# 生成虚拟数据集
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, class_sep=1.5, random_state=42)

# 数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用QDA模型进行训练
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# 绘制图形
plt.figure(figsize=(12, 10))

# 绘制决策边界和数据点
ax1 = plt.subplot(2, 2, 1)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

Z = qda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax1.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm', s=50)
ax1.set_title('Decision Boundary with Data Points')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

# 添加图例
legend1 = ax1.legend(*scatter.legend_elements(), title="Classes")
ax1.add_artist(legend1)

# 绘制分类结果
ax2 = plt.subplot(2, 2, 2)
y_pred = qda.predict(X_test)
ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k', s=50)
ax2.set_title('Classification Results')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')

# 绘制等高线图
ax3 = plt.subplot(2, 2, 3)
Z_proba = qda.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z_proba = Z_proba.reshape(xx.shape)
contour = ax3.contourf(xx, yy, Z_proba, 20, cmap='coolwarm', alpha=0.75)
plt.colorbar(contour, ax=ax3, orientation='vertical')
ax3.set_title('Probability Contour Plot')
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')

# 绘制训练集数据点分布
ax4 = plt.subplot(2, 2, 4)
scatter2 = ax4.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=50)
ax4.set_title('Training Data Distribution')
ax4.set_xlabel('Feature 1')
ax4.set_ylabel('Feature 2')

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 9. 支持向量机
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# 创建虚拟数据集
X, y = make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=1.2)

# 创建一个线性SVM分类器
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, y)

# 获取分隔超平面的系数
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1)
yy = a * xx - (clf.intercept_[0]) / w[1]

# 计算支持向量的边界线
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin

# 创建图形
plt.figure(figsize=(12, 8))

# 第一个图：原始数据及其分类结果
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 第二个图：SVM决策边界及支持向量
plt.subplot(2, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k')
plt.plot(xx, yy, 'k-', label="Decision Boundary")
plt.plot(xx, yy_down, 'k--', label="Margin")
plt.plot(xx, yy_up, 'k--')
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k', label="Support Vectors")
plt.title('SVM Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# 第三个图：分类决策函数的值
plt.subplot(2, 2, 3)
Z = clf.decision_function(X)
plt.scatter(X[:, 0], X[:, 1], c=Z, cmap='seismic', s=50, edgecolors='k')
plt.colorbar(label='Decision Function Value')
plt.title('Decision Function')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 第四个图：SVM支持向量特征分布
plt.subplot(2, 2, 4)
plt.scatter(X[:, 0], X[:, 1], c='lightgray', s=50, edgecolors='k', label="Other Points")
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], c='blue', s=100, edgecolors='k', label="Support Vectors")
plt.title('Support Vectors Highlighted')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# 展示图形
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 10. 感知机
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap

# 生成虚拟数据集
X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=2.5)

# 定义感知机模型
model = Perceptron(max_iter=1000, tol=1e-3)
model.fit(X, y)

# 获取决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 定义颜色
colors = ListedColormap(['#FF0000', '#0000FF'])
plt.figure(figsize=(12, 8))

# 绘制决策边界和分类结果
plt.contourf(xx, yy, Z, alpha=0.8, cmap=colors)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=colors)

# 标记支持向量
plt.scatter(X[:, 0], X[:, 1], s=100, edgecolors='k', facecolors='none')

# 设置图像属性
plt.title('Perceptron Classification with Decision Boundary', fontsize=16)
plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# 显示图像
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>>







