#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:22:24 2024

@author: jack
"""
# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247485608&idx=1&sn=88c025c0e9cbebf4eeb03009c4a6819a&chksm=c0e5d26ef7925b78d8285e0683d75700f87b8b3c1c214903a95685454348da58836f292021e1&mpshare=1&scene=1&srcid=0808ybLiyiuWTMsZKMsCFWCs&sharer_shareinfo=b3bd60eddf7fa885e8a6685065e10ad4&sharer_shareinfo_first=b3bd60eddf7fa885e8a6685065e10ad4&exportkey=n_ChQIAhIQEXx6wc8FH2DZjXvFVzk%2FbhKfAgIE97dBBAEAAAAAALJlJ0vFAu8AAAAOpnltbLcz9gKNyK89dVj0wKl22zIzI3aa0mt1bBfCd%2FH2b19UF56amQIOYXVtmELEhRQ0%2BxzQ8E27sezFEq%2BhNd%2Fc8%2FFc5lIu2IxFThts%2F5%2BTu5Rw1Z3r%2B5glrQvDaQd7ntTEBBk3JVxGhOsRnGBnhkvYmJX8rbDPmJ0ynsO0wZ%2FuqZHvTcisvKnFKITjOVJC6vRAE5vuhltZkAHqIhSdzm77y3BjxYW5%2BjomfEOz5258aTUCqbfFrts7P42rHRWS7equaP6zl2DnUOVK1O3fgg4SLCF17N3dFkg3HMwDxa29F97t8bCBYSOcZHcVSn4Am9BRgfzwuFZ5fPGUJET1nKNOotSbbrJr&acctmode=0&pass_ticket=UcZ4acKBEMml0gGtaC8bsXztEJtb0f%2B%2FUGiouj9yJWiGUCgsRnVjzknN93RkAu9Z&wx_header=0#rd



# 1. 交叉验证 (Cross-Validation)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer
import time

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# 初始化模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 设置交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 存储每一折的分数和训练时间
scores = []
fit_times = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    start_time = time.time()
    model.fit(X_train, y_train)
    fit_times.append(time.time() - start_time)

    score = model.score(X_test, y_test)
    scores.append(score)

# 绘制图形
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 1. 交叉验证每一折的评分分布图（箱线图）
sns.boxplot(data=scores, ax=axs[0])
axs[0].set_title('Cross-Validation Score Distribution')
axs[0].set_ylabel('Score')

# 2. 模型训练时间和评分的关系图（散点图）
axs[1].scatter(fit_times, scores)
axs[1].set_title('Training Time vs. Score')
axs[1].set_xlabel('Training Time (seconds)')
axs[1].set_ylabel('Score')

plt.tight_layout()
plt.show()


# 2. 混淆矩阵 (Confusion Matrix)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve

# 生成虚拟数据
np.random.seed(42)
y_true = np.random.choice([0, 1], size=1000, p=[0.9, 0.1])  # 假设实际标签的分布
y_pred_proba = np.random.rand(1000)  # 模型预测的概率
threshold = 0.3
y_pred = (y_pred_proba > threshold).astype(int)  # 设定阈值来生成预测标签

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)
cm_display = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 计算ROC曲线和AUC
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = roc_auc_score(y_true, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 计算Precision-Recall曲线
precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

plt.figure()
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.show()



# 3. 准确率 (Accuracy)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
                           n_clusters_per_class=2, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# 可视化部分特征的分布
features = [0, 1, 2, 3, 4]
fig, axes = plt.subplots(len(features), 2, figsize=(15, 20))
for i, feature in enumerate(features):
    axes[i, 0].hist(X_train[:, feature], bins=30, color='blue', alpha=0.7, label='Train')
    axes[i, 0].hist(X_test[:, feature], bins=30, color='green', alpha=0.7, label='Test')
    axes[i, 0].set_title(f'Feature {feature} Distribution')
    axes[i, 0].legend()

    axes[i, 1].scatter(X_train[:, feature], y_train, color='blue', alpha=0.5, label='Train')
    axes[i, 1].scatter(X_test[:, feature], y_test, color='green', alpha=0.5, label='Test')
    axes[i, 1].set_title(f'Feature {feature} vs Target')
    axes[i, 1].legend()

plt.tight_layout()
plt.show()


# 4. 精确率 (Precision)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 生成虚拟数据集
X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用支持向量机训练模型
model = SVC(kernel='rbf', C=1, gamma=0.1)
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 评估模型准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.2f}")

# 绘制数据分布图和决策边界图
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.show()

# 绘制训练集数据和决策边界
plot_decision_boundary(model, X_train, y_train)

# 绘制测试集数据和决策边界
plot_decision_boundary(model, X_test, y_test)



# 5. 召回率 (Recall)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, confusion_matrix, ConfusionMatrixDisplay

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10,
                           n_clusters_per_class=1, weights=[0.7, 0.3], flip_y=0, random_state=42)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算召回率
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.2f}")

# 生成混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Diseased', 'Diseased'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# 绘制召回率分析图
plt.figure(figsize=(8, 5))
bars = plt.bar(['Recall'], [recall], color=['blue'])
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Recall Score')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
plt.show()


# 6. F1 分数 (F1 Score)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义模型
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC()
}

# 训练模型并评估F1分数
f1_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores[name] = f1

# 可视化F1分数
plt.figure(figsize=(10, 6))
plt.bar(f1_scores.keys(), f1_scores.values(), color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.title('F1 Score Comparison of Different Models')
plt.ylim(0, 1)
for i, v in enumerate(f1_scores.values()):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
plt.show()


# 7. ROC 曲线 (Receiver Operating Characteristic Curve)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=2, random_state=42)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化分类器
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svc_clf = SVC(kernel='linear', probability=True, C=1.0, gamma='scale', random_state=42)

# 训练分类器
rf_clf.fit(X_train, y_train)
svc_clf.fit(X_train, y_train)

# 预测概率
rf_probs = rf_clf.predict_proba(X_test)[:, 1]
svc_probs = svc_clf.predict_proba(X_test)[:, 1]

# 计算 ROC 曲线
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_svc, tpr_svc, _ = roc_curve(y_test, svc_probs)

# 计算 AUC
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_svc = auc(fpr_svc, tpr_svc)

# 绘制 ROC 曲线
plt.figure(figsize=(10, 7))
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot(fpr_svc, tpr_svc, color='green', lw=2, label=f'SVC (AUC = {roc_auc_svc:.2f})')

# 绘制对角线
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# 8. AUC (Area Under the ROC Curve)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# 生成虚拟数据集
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 预测概率
y_probs = clf.predict_proba(X_test)[:, 1]

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()




# 9. 均方误差 (Mean Squared Error, MSE)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 创建虚拟数据集
np.random.seed(0)
X = 6 * np.random.rand(1000, 1) - 3
y = 0.5 * X**3 - X**2 + 2 * X + np.random.randn(1000, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 多项式特征转换
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# 拟合多项式回归模型
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train_poly, y_train)

# 预测
y_train_pred = poly_reg_model.predict(X_train_poly)
y_test_pred = poly_reg_model.predict(X_test_poly)

# 计算MSE
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f'Training MSE: {train_mse}')
print(f'Testing MSE: {test_mse}')

# 绘制数据及回归曲线
plt.figure(figsize=(14, 6))

# 绘制训练数据及回归曲线
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(np.sort(X_train, axis=0), poly_reg_model.predict(np.sort(X_train_poly, axis=0)), color='red', linewidth=2, label='Model')
plt.title(f'Training Data\nMSE: {train_mse:.2f}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# 绘制测试数据及回归曲线
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='green', label='Testing data')
plt.plot(np.sort(X_test, axis=0), poly_reg_model.predict(np.sort(X_test_poly, axis=0)), color='red', linewidth=2, label='Model')
plt.title(f'Testing Data\nMSE: {test_mse:.2f}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()



# 10. 均方根误差 (Root Mean Squared Error, RMSE)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 生成虚拟数据
np.random.seed(42)
X = np.linspace(0, 10, 1000).reshape(-1, 1)  # 100个样本的特征数据
y = 3 * X.squeeze() + 7 + np.random.normal(0, 1, X.shape[0])  # 线性关系加上噪声

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 进行预测
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 计算RMSE
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# 绘制图形
plt.figure(figsize=(12, 6))

# 训练数据与回归线
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_train, y_pred_train, color='red', linewidth=2, label='Regression Line')
plt.title('Training Data and Regression Line')
plt.xlabel('Feature Value')
plt.ylabel('Target Value')
plt.legend()

# 测试数据与回归线
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X_test, y_pred_test, color='orange', linewidth=2, label='Regression Line')
plt.title('Test Data and Regression Line')
plt.xlabel('Feature Value')
plt.ylabel('Target Value')
plt.legend()

plt.suptitle(f'Training Data RMSE: {rmse_train:.2f} | Test Data RMSE: {rmse_test:.2f}', fontsize=14)
plt.tight_layout()
plt.show()















































































































































































































































































































































































