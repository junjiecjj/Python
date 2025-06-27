#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 21:01:46 2025

@author: jack
"""

# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247490923&idx=1&sn=a1c6b2209ff2a125256bc33e6ae1ce2e&chksm=c1034c61eb9ff5ab610eee731ec093f9922b2398cc3d73e179a5710444fb30fa4c09c09de561&mpshare=1&scene=1&srcid=0624xdLt1rVdwwihHU0aFI7E&sharer_shareinfo=c816bbb4e070dc92bd739352f48183a4&sharer_shareinfo_first=c816bbb4e070dc92bd739352f48183a4&exportkey=n_ChQIAhIQ6IVovkvqeXBcgk6%2BgJs%2B5xKfAgIE97dBBAEAAAAAAFRjJPWiVGkAAAAOpnltbLcz9gKNyK89dVj0zV7x8ElnEImu3HlQy8kExFFob%2Fe8I%2BnElR%2B7k%2FUUtaeFV3NzHsEaR256TfF3sFY2oOhJ2AZNb%2F%2FYERHV6VRAPjgca2T19mptfNI3un4NjP5si%2BChXgmn0zafATl6qdSvdzktnh0QsaX8Hm%2BGeQTLZfXMWM2ZSctThFqbl2L1GxMIBHVZqDe%2FUGWxlfmjLjnxj2lwKFGW9A8dgv3qcDdhOdnuKyN1IJqLfW0y9q5sHKbJ0SOpyTb4m%2BnJ3GM%2BFXzP4Aps8Iw3lI0BATRb2KAt9XmwnX4NDnrYWIDI0gQA3By4MxfAg6wOwDq7%2B2%2FCwtwpvjAwcn1dpiIv&acctmode=0&pass_ticket=0XSKrYWOJsJySp8T%2BcI4EPLpi9t4N3ztXBYdv%2FZBdlGlff9AsZI%2Fb4%2BTU9WPrE4v&wx_header=0#rd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import PolynomialFeatures

# 用于构建权重矩阵
def get_weights(query_point, X, tau):
    m = X.shape[0]
    weights = np.exp(-np.sum((X - query_point)**2, axis=1) / (2 * tau**2))
    return np.diag(weights)

# 局部加权回归主函数
def locally_weighted_regression(X, y, tau, query_points):
    m, n = X.shape
    y_preds = []

    for q in query_points:
        W = get_weights(q, X, tau)
        XTWX = X.T @ W @ X
        if np.linalg.det(XTWX) == 0:
            theta = np.linalg.pinv(XTWX) @ X.T @ W @ y
        else:
            theta = np.linalg.inv(XTWX) @ X.T @ W @ y
        y_pred = q @ theta
        y_preds.append(y_pred)
    return np.array(y_preds)

# 构造数据
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = np.sin(X) + 0.3 * np.random.randn(100)

# 多项式特征
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X.reshape(-1, 1))

# 查询点用于可视化预测
query_points = np.linspace(0, 10, 300).reshape(-1, 1)
query_poly = poly.transform(query_points)

# 设置带宽
tau = 0.5
y_preds = locally_weighted_regression(X_poly, y, tau, query_poly)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Training Data', alpha=0.6)
plt.plot(query_points, y_preds, color='blue', linewidth=2.5, label='LWR Prediction')
plt.title('Locally Weighted Regression on Noisy Sine Wave', fontsize=16)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
