#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 22:44:27 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzAwNTkyNTUxMA==&mid=2247494110&idx=1&sn=efef33a7cba658d038e5ff7cec3e78a2&chksm=9a12d836e115e163813f996068f7e8f169ae0e4bc152ce60b92168f3d457c87dab6fbe78a0b7&mpshare=1&scene=1&srcid=0804KtPZxeU1u4VwweluzBJ0&sharer_shareinfo=efdc8bb9b2d9563fb2869e00c05ae228&sharer_shareinfo_first=e5f1abcdf1318287ec92fbcc81fdd69b&exportkey=n_ChQIAhIQuSLiyxEepP0Cep7wbu9U%2BxKfAgIE97dBBAEAAAAAAKmgKIUb27IAAAAOpnltbLcz9gKNyK89dVj0YN7oqoxE0xqaFP2j1IupHYgwsNXFSzJu0MN%2F2%2FrK8gpTeCtFU0H0ZFRgiws99St9gx5tRdps%2B7Tkb0P9VrgpU0ZMrszkDTZSk%2FyOE5eX%2F8ln7DIeL32bnie6gmfA9Mk8VIjZQHGWP4JgJ2zlwGSYI6ayANweraERk9ZHyz9zbwFlpNvvKTVkLKOThJ%2FoTv1iCNsPLn%2F1jPDi%2B9jjooJE%2BBCjLSI8xoRuDZFr5xLpMosUr4XVOzkYR4kUfs%2BkG%2BM%2B75xueJPxr8cmqAxCxS5PH9zEWWzGx4YMEH2IMZrQY63%2BHHU72wUIfRMw8Sn89QauU0%2BYl9VQrsy5&acctmode=0&pass_ticket=TM6%2BdBB3rVHQa%2B%2FtO8uvQCnhf9j94JiX6DQDfg2QV3ZPIOXQarylz8OKYwVEgp5I&wx_header=0#rd


"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

np.random.seed(42)

# 构造数据
X = np.sort(np.random.rand(200, 1))
y = np.sin(1.5 * np.pi * X).ravel() + np.random.normal(0, 0.2, X.shape[0])

# 拆分训练与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 构建模型
tree = DecisionTreeRegressor(max_depth=4)
forest = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)

# 拟合模型
tree.fit(X_train, y_train)
forest.fit(X_train, y_train)

# 预测
x_plot = np.linspace(0, 1, 500).reshape(-1, 1)
y_tree_pred = tree.predict(x_plot)
y_forest_pred = forest.predict(x_plot)

# 数据分析可视化
plt.figure(figsize=(18, 12))
sns.set_style("whitegrid")
colors = sns.color_palette("bright")

# 1. 真实函数与训练数据分布图
plt.subplot(2, 2, 1)
plt.title("Ground Truth vs. Noisy Training Data", fontsize=14)
plt.plot(x_plot, np.sin(1.5 * np.pi * x_plot), color=colors[0], label="True Function", linewidth=2)
plt.scatter(X_train, y_train, color=colors[1], alpha=0.6, label="Training Data")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# 2. 回归树拟合图
plt.subplot(2, 2, 2)
plt.title("Decision Tree Prediction", fontsize=14)
plt.plot(x_plot, np.sin(1.5 * np.pi * x_plot), color=colors[0], linestyle='--', label="True Function")
plt.plot(x_plot, y_tree_pred, color=colors[2], label="Tree Prediction", linewidth=2)
plt.scatter(X_test, y_test, color=colors[3], label="Test Data", alpha=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# 3. 随机森林拟合图
plt.subplot(2, 2, 3)
plt.title("Random Forest Prediction", fontsize=14)
plt.plot(x_plot, np.sin(1.5 * np.pi * x_plot), color=colors[0], linestyle='--', label="True Function")
plt.plot(x_plot, y_forest_pred, color=colors[4], label="Forest Prediction", linewidth=2)
plt.scatter(X_test, y_test, color=colors[3], label="Test Data", alpha=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# 4. 残差分布图（Residual Plot）
tree_residuals = y_test - tree.predict(X_test)
forest_residuals = y_test - forest.predict(X_test)

plt.subplot(2, 2, 4)
plt.title("Residual Comparison", fontsize=14)
sns.histplot(tree_residuals, color=colors[2], label="Tree Residuals", kde=True, stat="density", bins=20, alpha=0.6)
sns.histplot(forest_residuals, color=colors[4], label="Forest Residuals", kde=True, stat="density", bins=20, alpha=0.6)
plt.xlabel("Residual")
plt.ylabel("Density")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


tree_mse = mean_squared_error(y_test, tree.predict(X_test))
forest_mse = mean_squared_error(y_test, forest.predict(X_test))

tree_r2 = r2_score(y_test, tree.predict(X_test))
forest_r2 = r2_score(y_test, forest.predict(X_test))

print("决策树 MSE:", tree_mse)
print("随机森林 MSE:", forest_mse)
print("决策树 R²:", tree_r2)
print("随机森林 R²:", forest_r2)

























































































