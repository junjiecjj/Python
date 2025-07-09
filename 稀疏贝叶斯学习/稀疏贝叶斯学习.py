#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 22:46:24 2025

@author: jack

https://zhuanlan.zhihu.com/p/1893622601374468027

https://blog.csdn.net/qq_44648285/article/details/143313531

https://blog.csdn.net/weixin_40735720/article/details/148583124

https://blog.csdn.net/qq_45471796/article/details/130487580

https://github.com/al5250/sparse-bayes-learn
"""



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn_rvm import EMRVC

# 1. 生成模拟的二分类数据
X, y = make_classification(n_samples=1000, n_features=50, n_informative=10, n_redundant=10, n_classes=2, random_state=42)

# 2. 数据预处理：标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. 初始化并训练 RVM 分类器（线性核）
rvm = EMRVC(kernel='linear', verbose=False)
rvm.fit(X_train, y_train)

# 5. 预测并评估
y_pred = rvm.predict(X_test)
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))



















































































































