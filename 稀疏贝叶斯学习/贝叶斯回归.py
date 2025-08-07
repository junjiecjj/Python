#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 22:43:40 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247491460&idx=1&sn=b985283d2fc75d402f1892373fdc62f7&chksm=c1f59f11bc7f1cf3e7587c08565b1ea9afb3b9ae6e491f97e63c8871c78a24da6faca1a97102&mpshare=1&scene=1&srcid=0802JVJVhk1CXRe5PAez63Li&sharer_shareinfo=803e0d8c42c097a022c43c75613fbb44&sharer_shareinfo_first=bf978cc73c96cffa2c82f49a5aac4ce9&exportkey=n_ChQIAhIQTYFgdR3U9mTD8QP8AojaqRKfAgIE97dBBAEAAAAAAMBjCQE9TWAAAAAOpnltbLcz9gKNyK89dVj0LmGyGfzT3RcuBmx99v%2BIHXPbGlChRN2HLINi2cp64B7zx1HUG627Xs%2FDLl5WGJB%2F%2B1xIv8BYgOc43kBoThTyevHmz9s9m6SpBaNUav9RZ%2BcZCKB8b9ywAboS1u9vmN%2F3b9cvOxIc4wuE5ANqn1VezeU4ASg473SmcNHjVPnRnQx1gaS9bpHvhf9n4wlk%2FJ6ac9e1qX8Tj8qbMqUeKIT76GRg3gWiPg8JWur8nXLE13G6AoBiBap7tWffXZ5d3Ox69WrB3Wixtiw0iqRUrhsmM0eUFolaKsV5suBdlE7Q9nfilwmbrfUABar9%2FaIR%2BZ5dNofMzu%2BLqeUR&acctmode=0&pass_ticket=uhCQX5tm%2BkivpBtedMWaEOhP3a4agKeTnP3m8FtnIq5MkF9qqZkFDWdStaRrxidM&wx_header=0#rd


"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

# 1. 读取数据
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]
# boston = load_boston()
# X, y = boston.data, boston.target

# 2. 数据标准化 & 拆分训练/测试集
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. 训练贝叶斯回归模型
model = BayesianRidge(tol=1e-6, alpha_1=1e-6, lambda_1=1e-6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. 计算误差指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse:.3f}, R^2: {r2:.3f}')

# 5. 预测 vs 真实值 可视化
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, hue=np.abs(y_test - y_pred), palette='coolwarm', s=100)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--k', label='Perfect Prediction')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted House Prices')
plt.legend()
plt.show()

# 6. 误差分布
plt.figure(figsize=(10, 6))
sns.histplot(y_test - y_pred, kde=True, bins=30, color='purple')
plt.axvline(0, color='red', linestyle='--')
plt.xlabel('Prediction Error')
plt.title('Distribution of Prediction Errors')
plt.show()

# 7. 置信区间可视化
coefs_mean = model.coef_
coefs_std = np.sqrt(1 / model.lambda_)

plt.figure(figsize=(12, 6))
plt.errorbar(range(len(coefs_mean)), coefs_mean, yerr=coefs_std, fmt='o', color='darkblue', ecolor='red', capsize=5)
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Bayesian Regression Coefficients with Uncertainty')
plt.show()
