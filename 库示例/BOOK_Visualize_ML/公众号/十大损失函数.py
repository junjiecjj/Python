#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:23:29 2024

@author: jack
"""

# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247485639&idx=1&sn=b64cfda25bcc596d7a74a22e1d79eeb1&chksm=c0e5d201f7925b17e848c8b38f6d8e3914fe3c1ceef22c2fd0d4150d1c9bb4bc21d76de233b8&cur_album_id=3445855686331105280&scene=190#rd



#%%>>>>>>>>>>>>>>  1. 均方误差 (Mean Squared Error, MSE)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 案例 1：线性回归
# 数据生成
X1 = np.linspace(0, 10, 1000)
y1 = 3 * X1 + 5 + np.random.normal(0, 2, 1000)
X1 = X1.reshape(-1, 1)

# 线性回归模型
model1 = LinearRegression()
model1.fit(X1, y1)
y1_pred = model1.predict(X1)

# 数据分析图形
# 图 1：预测值 vs 实际值的散点图及拟合直线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X1, y1, color='blue', label='Actual values')
plt.plot(X1, y1_pred, color='red', label='Predicted values')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Actual vs Predicted values (Linear Regression)')
plt.legend()

# 图 2：残差图及残差的直方图
residuals1 = y1 - y1_pred
plt.subplot(1, 2, 2)
plt.scatter(X1, residuals1, color='green', label='Residuals')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.title('Residuals plot (Linear Regression)')
plt.legend()

plt.tight_layout()
plt.show()

# 案例 2：决策树回归
# 数据生成
X2 = np.linspace(0, 10, 1000)
y2 = np.sin(X2) + np.random.normal(0, 0.1, 1000)
X2 = X2.reshape(-1, 1)

# 决策树回归模型
model2 = DecisionTreeRegressor(max_depth=3)
model2.fit(X2, y2)
y2_pred = model2.predict(X2)

# 数据分析图形
# 图 1：实际值 vs 预测值的曲线图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(X2, y2, label='Actual values')
plt.plot(X2, y2_pred, label='Predicted values', linestyle='--')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Actual vs Predicted values (Decision Tree Regression)')
plt.legend()

# 图 2：残差图及残差的密度图
residuals2 = y2 - y2_pred
plt.subplot(1, 2, 2)
plt.scatter(X2, residuals2, color='purple', label='Residuals')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.title('Residuals plot (Decision Tree Regression)')
plt.legend()

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>  2. 均方根误差 (Root Mean Squared Error, RMSE)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 数据生成
X1 = np.random.rand(100) * 10
X2 = np.random.rand(100) * 10
y = np.sin(X1) * np.cos(X2) + np.random.normal(0, 0.1, 100)
X = np.column_stack((X1, X2))

# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 随机森林回归模型
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 计算RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse:.4f}')

# 数据分析图形
# 图 1：特征重要性图
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(X.shape[1]), importances[indices], color='r', align='center')
plt.xticks(range(X.shape[1]), [f'Feature {i}' for i in indices])
plt.title('Feature Importances (Random Forest)')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# 图 2：实际值、预测值与特征的三维散点图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='blue', label='Actual values')
ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, color='red', label='Predicted values', alpha=0.6)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.set_title('3D Scatter Plot (Random Forest)')
ax.legend()
plt.show()


#%%>>>>>>>>>>>>>> 3. 平均绝对误差 (Mean Absolute Error, MAE)
# 案例 1：线性回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 生成数据集
np.random.seed(0)
X = np.random.rand(1000, 1) * 10
y = 2.5 * X + np.random.randn(1000, 1) * 2

# 数据集可视化
plt.scatter(X, y, color='blue')
plt.title('Generated Data')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# 线性回归模型
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 计算MAE
mae = mean_absolute_error(y, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# 绘制回归线和残差图
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title('Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# 残差分析
residuals = y - y_pred
plt.scatter(X, residuals, color='green')
plt.hlines(0, X.min(), X.max(), colors='red')
plt.title('Residual Plot')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.show()


# 案例 2：随机森林回归
from sklearn.ensemble import RandomForestRegressor

# 生成数据集
np.random.seed(1)
X = np.linspace(0, 10, 1000).reshape(-1, 1)
y = np.sin(X) + np.random.randn(1000, 1) * 0.2

# 数据集可视化
plt.scatter(X, y, color='purple')
plt.title('Generated Non-linear Data')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# 随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y.ravel())
y_pred = model.predict(X)

# 计算MAE
mae = mean_absolute_error(y, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# 绘制预测结果和残差图
plt.scatter(X, y, color='purple', label='Actual Data')
plt.plot(X, y_pred, color='orange', label='Random Forest Prediction')
plt.title('Random Forest Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# 残差分析
residuals = y - y_pred.reshape(-1, 1)
plt.scatter(X, residuals, color='brown')
plt.hlines(0, X.min(), X.max(), colors='red')
plt.title('Residual Plot')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.show()




#%%>>>>>>>>>>>>>>  4. 交叉熵损失 (Cross-Entropy Loss)
# 案例 1：逻辑回归用于二分类
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, classification_report

# 生成二分类数据集
np.random.seed(0)
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, class_sep=1.5, random_state=0)

# 数据集可视化
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')
plt.title('Generated Classification Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 逻辑回归模型
model = LogisticRegression()
model.fit(X, y)
y_pred_prob = model.predict_proba(X)
y_pred = model.predict(X)

# 计算交叉熵损失
loss = log_loss(y, y_pred_prob)
print(f'Cross-Entropy Loss: {loss}')

# 分类报告
print(classification_report(y, y_pred))

# 决策边界可视化
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()



# 案例 2：神经网络用于多分类
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, classification_report

# 生成多分类数据集
np.random.seed(0)
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=0, cluster_std=1.5)

# 数据集可视化
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='green', label='Class 1')
plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], color='red', label='Class 2')
plt.title('Generated Multi-class Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 神经网络模型
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=0)
model.fit(X, y)
y_pred_prob = model.predict_proba(X)
y_pred = model.predict(X)

# 计算交叉熵损失
loss = log_loss(y, y_pred_prob)
print(f'Cross-Entropy Loss: {loss}')

# 分类报告
print(classification_report(y, y_pred))

# 决策边界可视化
xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 200),
                     np.linspace(X[:, 1].min(), X[:, 1].max(), 200))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='green', label='Class 1')
plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], color='red', label='Class 2')
plt.title('Neural Network Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()


#%%>>>>>>>>>>>>>>  5. 二元交叉熵损失 (Binary Cross-Entropy Loss)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import seaborn as sns

# 创建虚拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 绘制ROC曲线
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# 案例2：神经网络

# 使用一个简单的神经网络来解决一个二元分类问题，并使用二元交叉熵损失来评估模型的表现。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import precision_recall_curve, auc

# 创建虚拟数据集
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建神经网络模型
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# 预测
y_pred_proba = model.predict(X_test).ravel()

# 计算Precision-Recall曲线
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

# 绘制Precision-Recall曲线
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()

# 绘制训练和验证损失
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()



#%%>>>>>>>>>>>>>>  6. 平方合页损失 (Squared Hinge Loss)

# 案例1：线性支持向量机
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
import seaborn as sns

# 创建虚拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练线性SVM模型
model = SVC(kernel='linear', C=1.0, probability=True)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
plot_confusion_matrix(model, X_test, y_test, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# 绘制ROC曲线
plt.figure(figsize=(10, 6))
plot_roc_curve(model, X_test, y_test)
plt.title('ROC Curve')
plt.show()

# 案例2：核支持向量机（RBF核）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_precision_recall_curve, classification_report
import seaborn as sns
import pandas as pd

# 创建虚拟数据集
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练带有RBF核的SVM模型
model = SVC(kernel='rbf', C=1.0, probability=True)  # Remove 'loss' parameter
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 绘制Precision-Recall曲线
plt.figure(figsize=(10, 6))
plot_precision_recall_curve(model, X_test, y_test)
plt.title('Precision-Recall Curve')
plt.show()

# 绘制分类报告
report = classification_report(y_test, y_pred, output_dict=True)
plt.figure(figsize=(10, 6))
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='Blues')
plt.title('Classification Report')
plt.show()



#%%>>>>>>>>>>>>>>  7. 对数似然损失 (Log-Likelihood Loss)
# 案例1：高斯朴素贝叶斯（Gaussian Naive Bayes）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import seaborn as sns

# 创建虚拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练高斯朴素贝叶斯模型
model = GaussianNB()
model.fit(X_train, y_train)

# 预测
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 绘制ROC曲线
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()



# 案例2：泊松回归（Poisson Regression）

# 使用泊松回归模型来解决一个计数数据的回归问题，并使用对数似然损失来评估模型的表现。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# 创建虚拟计数数据集
np.random.seed(42)
X = np.random.rand(1000, 1) * 10  # 特征
y = np.random.poisson(lam=(0.5 * X).flatten())  # 泊松分布的目标变量

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练泊松回归模型
model = PoissonRegressor()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算均方误差和R^2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 绘制实际值与预测值对比图
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', alpha=0.6, label='Predicted')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()

# 绘制残差图
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.show()

# 打印MSE和R^2
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')


#%%>>>>>>>>>>>>>>  8. Huber 损失 (Huber Loss)
# 案例1：线性回归

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 创建虚拟数据集
X, y = make_regression(n_samples=1000, n_features=1, noise=4.0, random_state=42)
y += 20 * (np.random.rand(*y.shape) - 0.5)  # 添加一些离群点

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练Huber回归模型
model = HuberRegressor()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算均方误差和平均绝对误差
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# 绘制实际值与预测值对比图
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', alpha=0.6, label='Predicted')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()

# 绘制残差图
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(X_test, residuals, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Feature')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

# 打印MSE和MAE
print(f'Mean Squared Error: {mse:.2f}')
print(f'Mean Absolute Error: {mae:.2f}')

# 案例2：梯度提升回归树

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# 创建虚拟数据集
X, y = make_friedman1(n_samples=1000, noise=1.0, random_state=42)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练梯度提升回归树模型
model = GradientBoostingRegressor(loss='huber', n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算R^2
r2 = r2_score(y_test, y_pred)

# 绘制特征重要性
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(range(X.shape[1]), feature_importance[sorted_idx])
plt.yticks(range(X.shape[1]), [f'Feature {i}' for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Gradient Boosting Regressor')
plt.show()

# 绘制实际值与预测值对比图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='green', alpha=0.6)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.show()

# 打印R^2
print(f'R^2 Score: {r2:.2f}')


#%%>>>>>>>>>>>>>>  9. KL 散度 (Kullback-Leibler Divergence, KL Divergence)
# 案例一：高斯混合模型 (GMM) 对聚类的KL散度分析

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, entropy

# 生成两个不同的高斯混合模型数据集
def generate_gmm_data(means, covariances, weights, n_samples):
    data = []
    for mean, cov, weight in zip(means, covariances, weights):
        data.append(np.random.multivariate_normal(mean, cov, int(n_samples * weight)))
    return np.vstack(data)

# 定义两个GMM的参数
means1 = [np.array([0, 0]), np.array([3, 3])]
covariances1 = [np.eye(2), np.eye(2)]
weights1 = [0.5, 0.5]

means2 = [np.array([1, 1]), np.array([4, 4])]
covariances2 = [np.eye(2), np.eye(2)]
weights2 = [0.5, 0.5]

# 生成数据
n_samples = 1000
data1 = generate_gmm_data(means1, covariances1, weights1, n_samples)
data2 = generate_gmm_data(means2, covariances2, weights2, n_samples)

# 计算KL散度
def calculate_kl_divergence(data1, data2):
    p = multivariate_normal(mean=np.mean(data1, axis=0), cov=np.cov(data1, rowvar=False))
    q = multivariate_normal(mean=np.mean(data2, axis=0), cov=np.cov(data2, rowvar=False))
    kl_div = entropy(p.pdf(data1), q.pdf(data1))
    return kl_div

kl_divergence = calculate_kl_divergence(data1, data2)
print(f"KL Divergence: {kl_divergence}")

# 数据可视化
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(data1[:, 0], data1[:, 1], alpha=0.5, label='GMM 1')
plt.title('Data from GMM 1')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(data2[:, 0], data2[:, 1], alpha=0.5, label='GMM 2', color='orange')
plt.title('Data from GMM 2')
plt.legend()

plt.show()


# 案例二：隐马尔可夫模型 (HMM) 对序列数据的KL散度分析

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from scipy.special import rel_entr

# 生成两个不同的HMM数据集
def generate_hmm_data(startprob, transmat, means, covars, n_samples):
    model = hmm.GaussianHMM(n_components=len(startprob))
    model.startprob_ = np.array(startprob)
    model.transmat_ = np.array(transmat)
    model.means_ = np.array(means)
    model.covars_ = np.array(covars)

    X, _ = model.sample(n_samples)
    return X

# 定义两个HMM的参数
startprob1 = [0.6, 0.4]
transmat1 = [[0.7, 0.3], [0.4, 0.6]]
means1 = [[0.0], [3.0]]
covars1 = [[0.5], [0.5]]

startprob2 = [0.5, 0.5]
transmat2 = [[0.6, 0.4], [0.3, 0.7]]
means2 = [[1.0], [4.0]]
covars2 = [[0.5], [0.5]]

# 生成数据
n_samples = 1000
data1 = generate_hmm_data(startprob1, transmat1, means1, covars1, n_samples)
data2 = generate_hmm_data(startprob2, transmat2, means2, covars2, n_samples)

# 计算KL散度
def calculate_kl_divergence_hmm(data1, data2, means1, means2, covars1, covars2):
    p = multivariate_normal(mean=means1, cov=covars1)
    q = multivariate_normal(mean=means2, cov=covars2)
    kl_div = np.sum(rel_entr(p.pdf(data1), q.pdf(data1)))
    return kl_div

kl_divergence = calculate_kl_divergence_hmm(data1, data2, means1, means2, covars1, covars2)
print(f"KL Divergence: {kl_divergence}")

# 数据可视化
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(data1[:100], label='HMM 1')
plt.title('Data from HMM 1')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(data2[:100], label='HMM 2', color='orange')
plt.title('Data from HMM 2')
plt.legend()

plt.show()



#%%>>>>>>>>>>>>>>  10. 余弦相似性损失 (Cosine Similarity Loss)
# 案例一：自然语言处理中的余弦相似性损失
# 代码中，生成一些虚拟的文本数据，并使用TF-IDF向量化方法，然后计算余弦相似性损失。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 生成一些虚拟的文本数据
documents = [
    "I love machine learning and natural language processing",
    "Machine learning is great for data analysis",
    "Natural language processing helps in understanding human language",
    "Data analysis and machine learning are closely related fields",
    "Human language understanding is a part of natural language processing"
]

# 使用TF-IDF向量化文本数据
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# 计算余弦相似性矩阵
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 打印余弦相似性矩阵
print("Cosine Similarity Matrix:")
print(cosine_sim_matrix)

# 可视化余弦相似性矩阵
plt.figure(figsize=(8, 6))
plt.imshow(cosine_sim_matrix, interpolation='nearest', cmap='coolwarm')
plt.title('Cosine Similarity Matrix')
plt.colorbar()
plt.show()

# 可视化TF-IDF特征
feature_names = vectorizer.get_feature_names_out()
tfidf_array = tfidf_matrix.toarray()

plt.figure(figsize=(10, 8))
for i in range(len(documents)):
    plt.plot(tfidf_array[i], label=f'Doc {i+1}')
plt.title('TF-IDF Features')
plt.xlabel('Feature Index')
plt.ylabel('TF-IDF Value')
plt.legend()
plt.show()

# 案例二：推荐系统中的余弦相似性损失

# 代码中，我们生成一些虚拟的用户-物品评分数据，并计算物品之间的余弦相似性来推荐相似物品。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# 生成一些虚拟的用户-物品评分数据
np.random.seed(0)
user_item_matrix = np.random.rand(10, 5)  # 10个用户对5个物品的评分

# 计算物品之间的余弦相似性矩阵
item_cosine_sim_matrix = cosine_similarity(user_item_matrix.T)

# 打印物品余弦相似性矩阵
print("Item Cosine Similarity Matrix:")
print(item_cosine_sim_matrix)

# 可视化物品余弦相似性矩阵
plt.figure(figsize=(8, 6))
plt.imshow(item_cosine_sim_matrix, interpolation='nearest', cmap='coolwarm')
plt.title('Item Cosine Similarity Matrix')
plt.colorbar()
plt.show()

# 可视化用户-物品评分矩阵
plt.figure(figsize=(10, 8))
plt.imshow(user_item_matrix, interpolation='nearest', cmap='viridis')
plt.title('User-Item Rating Matrix')
plt.xlabel('Item Index')
plt.ylabel('User Index')
plt.colorbar()
plt.show()


























































