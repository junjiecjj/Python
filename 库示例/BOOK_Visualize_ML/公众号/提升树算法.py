#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:09:57 2024

@author: jack
"""

# https://mp.weixin.qq.com/s?__biz=Mzk0MjUxMzg3OQ==&mid=2247489296&idx=1&sn=39ab1eb166f53c8bee56e7bbfb28026e&chksm=c2c35ca8f5b4d5be161f08525f0fc8f8e6f7cac00f07af1e060db6a535030755f233a706808d&mpshare=1&scene=1&srcid=0808n43JX1qjdAE514bis9Fm&sharer_shareinfo=be5a95db851eb834ee25600cd145129f&sharer_shareinfo_first=be5a95db851eb834ee25600cd145129f&exportkey=n_ChQIAhIQd%2BOApBux63H6rFrmr%2FD%2BTBKfAgIE97dBBAEAAAAAAARFKOZk29cAAAAOpnltbLcz9gKNyK89dVj0qHO%2F7B0lZ6u9Hf7BKV5U2kMzByXiWmMXH1jDdiEnNTjmb8rqQycthLIl%2Bg7%2FdSV50tUeAgbc3%2BlVkRjHoFdahYMjaMZvZQ5jz0QjLC81lOAXboWG2ln1LF%2BqKWuXlGQZXL1YYyPCpslsUfeaRQQNRSSP0kjG%2BFDNdQSFBzpDYHUvOrGG8hm7JE2TQpDu%2FTvChqwQkLEfjVpGlaAHDTj%2Ba8gVQjh6ee13NikqgynftFj6PkHxTvJJZuAQUqH2qez5TtkS4oPVZQ%2B0FxXMI%2F83ZKTPFkJxBv%2FA%2BZU0QlOO2a69%2Bd6d%2B10AnX0VjXHzx8Sgm%2Bn5AmWn7cqJ&acctmode=0&pass_ticket=GZLxqASSBh3FPPPacqSl7oCGUhyoxS3VXuilX2d94ArYVvHvBZKpYf%2B7K6t8lJ37&wx_header=0#rd



# 数据准备
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# 生成合成数据
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 转换为 DataFrame 以便绘图
df_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
df_train['target'] = y_train
df_test = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(X_test.shape[1])])
df_test['target'] = y_test



# 训练模型并进行预测

# 1. Gradient Boosting Decision Tree (GBDT)
gbdt_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbdt_model.fit(X_train, y_train)
y_pred_gbdt = gbdt_model.predict(X_test)

# 2. LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)

# 3. XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# 4. AdaBoost
ada_model = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3), n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)


# 性能评估
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.4f}, R2: {r2:.4f}")

print("Performance Comparison:")
evaluate_model(y_test, y_pred_gbdt, "GBDT")
evaluate_model(y_test, y_pred_lgb, "LightGBM")
evaluate_model(y_test, y_pred_xgb, "XGBoost")
evaluate_model(y_test, y_pred_ada, "AdaBoost")


# 可视化
plt.style.use('ggplot')  # 使用有效的样式
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

def plot_predictions(ax, y_true, y_pred, model_name):
    sns.scatterplot(x=y_true, y=y_pred, ax=ax)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', label='Perfect Prediction')
    ax.set_title(f'{model_name} Predictions vs Actual')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()

# 预测与实际值对比
plot_predictions(axes[0], y_test, y_pred_gbdt, 'GBDT')
plot_predictions(axes[1], y_test, y_pred_lgb, 'LightGBM')
plot_predictions(axes[2], y_test, y_pred_xgb, 'XGBoost')
plot_predictions(axes[3], y_test, y_pred_ada, 'AdaBoost')

plt.tight_layout()
plt.show()

# 预测值分布图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

def plot_prediction_distribution(ax, y_pred, model_name):
    sns.histplot(y_pred, ax=ax, kde=True, bins=30)
    ax.set_title(f'{model_name} Predicted Values Distribution')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Frequency')

# 绘制预测值的分布
plot_prediction_distribution(axes[0], y_pred_gbdt, 'GBDT')
plot_prediction_distribution(axes[1], y_pred_lgb, 'LightGBM')
plot_prediction_distribution(axes[2], y_pred_xgb, 'XGBoost')
plot_prediction_distribution(axes[3], y_pred_ada, 'AdaBoost')

plt.tight_layout()
plt.show()


# 代码中的调参示例
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# 创建模型
gbdt = GradientBoostingRegressor()

# 使用网格搜索进行调参
grid_search = GridSearchCV(estimator=gbdt, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)



















































































