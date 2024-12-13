#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:17:44 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247488659&idx=1&sn=b74e3f8042a21840de330c9a20b32294&chksm=c121f92e20f02b5bb8297a89e5f2ec4344d2e987098b9eba80a31b27d0681c702e87aec01a70&mpshare=1&scene=1&srcid=1210xc8G1zyJQEt5eilTDDMB&sharer_shareinfo=6f3f622fe2c084e273810edf2430c69c&sharer_shareinfo_first=6f3f622fe2c084e273810edf2430c69c&exportkey=n_ChQIAhIQ1GOLmtJ4CAFmKYWsrw8V8hKfAgIE97dBBAEAAAAAADOVI5Ka0KEAAAAOpnltbLcz9gKNyK89dVj0xJ5TTMG9e%2FekSgd9Y1RTvhJx4ZfpsdEblliOXZLGREtMZvUg1fWtf%2BZFptlHAfYRsuY%2FC5jTbrl1gfwDKLtXRKKY9ZD%2Fza7keLPWARXlTdWEZP28EtSmykInzpDbonv7BSRYAP%2Bn2D8AFVDil5POIxq9FC74Kf0gaBCisA4Z7Qdu3EP4ZXy6PPmyyfw7L3tp57x4MjSZnlITKd5aT5PVcJTMrNPXu7Jk%2FhWXkZegw1CdizToTPYcFi%2FwmtvqOdH5wrMonRUD%2Fj85Ic0lW25SSy4CWt63KQnpfkMaepMxAnDLeqqKK8idVlerWNZJFPhW7liLh1rw93Ie&acctmode=0&pass_ticket=NdPJrOakmExKi8nsEp%2F6jFt0i8b3VyPbRs%2F4nrsuwOb1kFLpm%2FqnJ3%2ForvgUn2uD&wx_header=0#rd

"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

# 1. 加载数据集
digits = datasets.load_digits()
X, y = digits.data, digits.target
target_names = digits.target_names

# 2. 数据降维 (PCA)
pca = PCA(n_components=3)  # 降到3维用于可视化
X_pca = pca.fit_transform(X)

# 3. 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# 4. 训练SVM分类器
svm = SVC(kernel='rbf', C=10, gamma=0.01)  # 使用RBF核
svm.fit(X_train, y_train)

# 5. 预测结果
y_pred = svm.predict(X_test)

# 6. 分类报告和混淆矩阵
print(classification_report(y_test, y_pred))

# 图1：二维PCA可视化分类结果
plt.figure(figsize=(10, 6))
colors = plt.cm.get_cmap('tab10', 10)  # 为每个类别设置不同的颜色
for i in range(10):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                label=f'Digit {i}', alpha=0.7, color=colors(i))
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D PCA Visualization of Digits')
plt.legend(loc='best')
plt.show()

# 图2：三维PCA降维效果
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(10):
    ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], X_pca[y == i, 2],
               label=f'Digit {i}', alpha=0.7, color=colors(i))
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.set_title('3D PCA Visualization of Digits')
plt.legend(loc='best')
plt.show()

# 图3：决策边界（二维投影）
h = .02  # 网格步长
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())])  # 使用2D面生成边界
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(colors(range(10))))
for i in range(10):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                label=f'Digit {i}', edgecolor='k', s=30)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Decision Boundary of SVM (2D Projection)')
plt.legend(loc='best')
plt.show()

# 图4：混淆矩阵热图
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap='coolwarm')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')

# 在每个格子中显示数字
thresh = cm.max() / 2
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, f'{cm[i, j]}',
             horizontalalignment='center',
             color='white' if cm[i, j] > thresh else 'black')
plt.show()






















