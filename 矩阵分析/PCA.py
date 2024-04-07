#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:01:17 2024

@author: jack
"""

from sklearn.datasets import load_iris
import numpy as np


iris = load_iris()

print(iris.data.shape) # 150,4
X = iris.data
# 中心化
X = X - X.mean(axis=0)
# 计算协方差矩阵
XXT = np.matrix(X.T) * np.matrix(X) / (len(X)-1)

# 求特征值和特征向量
eigVals, eigVects = np.linalg.eig(np.mat(XXT))
print("特征值: ", eigVals)
print("特征向量: ", eigVects)








# 工具包自动计算

from sklearn.decomposition import PCA

pca = PCA(n_components=2, whiten='True',svd_solver='full')
iris = load_iris()
X = iris.data
pca.fit(X)
print(pca.explained_variance_)
# [4.22824171 0.24267075]
print(pca.components_.T)
# [[ 0.36138659  0.65658877]
#  [-0.08452251  0.73016143]
#  [ 0.85667061 -0.17337266]
#  [ 0.3582892  -0.07548102]]

# 数据变换
X1 = pca.transform(iris.data.T)
print(X1.shape) # 150,2























