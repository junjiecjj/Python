#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:04:46 2024

@author: jack
https://mp.weixin.qq.com/s?__biz=Mzk0NDM4OTYyOQ==&mid=2247486631&idx=1&sn=4d93f84010ee50f22043f009e2a09524&chksm=c24954a9cd2f8399ff65b4197a3f19a42f31e5a0e9c0236687e24cb0e4fb31bfd77a5ebd94f7&mpshare=1&scene=1&srcid=0921yUliz8fuVYnTY2aE4aC2&sharer_shareinfo=aac94b55c5ccb01cab8f3dd9ca8fc8f8&sharer_shareinfo_first=aac94b55c5ccb01cab8f3dd9ca8fc8f8&exportkey=n_ChQIAhIQUBq2KbYwO%2BmRa36tWZ6aWBKfAgIE97dBBAEAAAAAALxHGJnTsc0AAAAOpnltbLcz9gKNyK89dVj01d5MYsboFfWjSJu1061AT%2FqvYxTtEC5MAaVz49HqWi5XrfKLAGsdME59XRei5uBmM8hHCVHjhwhUCB46RapEi9sTZXBKNssls6asko3YB%2BfUfK24z4wE0JcWyPgejccRDbttYa7gIMo2lhxg7O0AlwuYwGn744JnFJNuyRXeGtVGBvhfS8V1Cw5CqHHDAFncBliror95xQBIbgM0mDOuVUBaqxgQIdcF1U7Np9sD4aK6it3Lcf2LCC%2BP%2Fyd7uMSgEsDObqswlej1GIMD4C3%2Bavixz7viYsZRX5Ii47XNNMymP%2BKZKO%2BbpJPyIOskYmQlX8yFlesHM68G&acctmode=0&pass_ticket=NC8hGxSsI05SU5e8xlPM6YWFD%2BzrBWwIa3mRk0VujMaWP8SWzzLddZUZgal1V3EJ&wx_header=0#rd


"""

# 数据生成

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# 为了增加复杂度，生成多个特征，并使用 XGBoost 来进行回归建模和可视化
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
from sklearn.datasets import make_regression
import xgboost as xgb

# 生成多特征模拟数据集
n_samples = 200
n_features = 5
X_simulated, y_simulated = make_regression(n_samples=n_samples, n_features=n_features, noise=0.1, random_state=42)


# 模型训练
# 将模拟数据分为训练集和测试集
X_train_simulated, X_test_simulated, y_train_simulated, y_test_simulated = train_test_split(X_simulated, y_simulated, test_size=0.25, random_state=42)

# 定义XGBoost模型
model_xgb_simulated = xgb.XGBRegressor(
    learning_rate=0.02,
    booster='gbtree',
    objective='reg:squarederror',
    max_leaves=127,
    verbosity=0,
    seed=42,
    colsample_bytree=0.6,
    subsample=0.7
)

# 训练XGBoost模型
model_xgb_simulated.fit(X_train_simulated, y_train_simulated)


# 模型预测及置信区间计算

# 在测试集上进行预测
y_pred_simulated = model_xgb_simulated.predict(X_test_simulated)

# 计算残差和标准差
residuals_simulated = y_test_simulated - y_pred_simulated
sigma_simulated = np.std(residuals_simulated)

# 计算三种置信区间
conf_intervals_simulated = {
    'mu ± sigma': (y_pred_simulated - sigma_simulated, y_pred_simulated + sigma_simulated),
    'mu ± 2sigma': (y_pred_simulated - 2*sigma_simulated, y_pred_simulated + 2*sigma_simulated),
    'mu ± 3sigma': (y_pred_simulated - 3*sigma_simulated, y_pred_simulated + 3*sigma_simulated),
}


# 可视化预测结果及置信区间

# 绘制可视化图像
test_indices_simulated = np.arange(len(y_test_simulated))

plt.figure(figsize=(12, 8),  )

# 绘制测试数据
plt.scatter(test_indices_simulated, y_test_simulated, color='red', label='Test data', zorder=5)

# 绘制预测均值
plt.plot(test_indices_simulated, y_pred_simulated, color='blue', label='prediction, $\\mu$', zorder=4)

# 绘制三种置信区间
plt.fill_between(test_indices_simulated, conf_intervals_simulated['mu ± sigma'][0], conf_intervals_simulated['mu ± sigma'][1],
                 color='navy', alpha=0.6, label='$\\mu \\pm \\sigma$')
plt.fill_between(test_indices_simulated, conf_intervals_simulated['mu ± 2sigma'][0], conf_intervals_simulated['mu ± 2sigma'][1],
                 color='deepskyblue', alpha=0.5, label='$\\mu \\pm 2\\sigma$')
plt.fill_between(test_indices_simulated, conf_intervals_simulated['mu ± 3sigma'][0], conf_intervals_simulated['mu ± 3sigma'][1],
                 color='lightblue', alpha=0.3, label='$\\mu \\pm 3\\sigma$')

# 设置图表标题和标签
plt.title('Test Data with Prediction and Confidence Intervals (XGBoost Model)')
plt.xlabel('Test Sample')
plt.ylabel('Simulated Output')
plt.legend()
# plt.savefig("Test Data with Prediction and Confidence Intervals (XGBoost Model).pdf",bbox_inches='tight')
# 显示图像
plt.show()











