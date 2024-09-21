#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 21:14:48 2024

@author: jack
"""

#%%>>>>>>>>>>>>>>>>>>>>>>> 1. 检测缺失值

import pandas as pd
import numpy as np

# 创建一个带有缺失值的小数据集
data = {'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, np.nan, 8],
        'C': [1.1, 'y', 'z', np.nan]}

df = pd.DataFrame(data)

# 检测缺失值
print("是否存在缺失值：")
print(df.isnull())  # 逐个元素显示是否为缺失值

# 统计每列缺失值的数量
print("\n每列缺失值数量：")
print(df.isnull().sum())




#%%>>>>>>>>>>>>>>>>>>>>>>> 2. 删除缺失值（dropna()）
# 删除含有缺失值的行
df_drop_rows = df.dropna(axis=0)
print("删除缺失值行后的DataFrame：")
print(df_drop_rows)

# 删除含有缺失值的列
df_drop_columns = df.dropna(axis=1)
print("\n删除缺失值列后的DataFrame：")
print(df_drop_columns)

# 只删除全为缺失的行
df_drop_all = df.dropna(how='all')
print("\n删除全为缺失值行后的DataFrame：")
print(df_drop_all)




#%%>>>>>>>>>>>>>>>>>>>>>>> 3. 用均值填充（fillna()）
# 用均值填充A列的缺失值
df_mean_fill = df.copy()
df_mean_fill['A'].fillna(df['A'].mean(), inplace=True)
print("用均值填充A列后的DataFrame：")
print(df_mean_fill)





#%%>>>>>>>>>>>>>>>>>>>>>>> 4. 用中位数填充

# 用中位数填充B列的缺失值
df_median_fill = df.copy()
df_median_fill['B'].fillna(df['B'].median(), inplace=True)
print("用中位数填充B列后的DataFrame：")
print(df_median_fill)



#%%>>>>>>>>>>>>>>>>>>>>>>> 5. 用众数填充

# 用众数填充C列的缺失值
df_mode_fill = df.copy()
df_mode_fill['C'].fillna(df['C'].mode()[0], inplace=True)
print("用众数填充C列后的DataFrame：")
print(df_mode_fill)





#%%>>>>>>>>>>>>>>>>>>>>>>> 6. 前向填充（ffill()）和后向填充（bfill()）
# 前向填充
df_ffill = df.copy()
df_ffill.fillna(method='ffill', inplace=True)
print("前向填充后的DataFrame：")
print(df_ffill)

# 后向填充
df_bfill = df.copy()
df_bfill.fillna(method='bfill', inplace=True)
print("\n后向填充后的DataFrame：")
print(df_bfill)



#%%>>>>>>>>>>>>>>>>>>>>>>> 7. 插值法（interpolate()）

# 线性插值法填充A列
df_interpolate = df.copy()
df_interpolate['A'].interpolate(method='linear', inplace=True)
print("插值法填充A列后的DataFrame：")
print(df_interpolate)





#%%>>>>>>>>>>>>>>>>>>>>>>> 8. 条件填充

# 根据条件填充A列的缺失值，例如根据C列的值填充
df_conditional_fill = df.copy()
df_conditional_fill.loc[df['C'] == 'z', 'A'] = 100
print("根据条件填充A列后的DataFrame：")
print(df_conditional_fill)




#%%>>>>>>>>>>>>>>>>>>>>>>> 9. 通过模型预测填充
from sklearn.linear_model import LinearRegression

# # 用其他列作为特征，训练模型填充缺失值
# df_model_fill = df.copy()

# # 删除含有NaN的行，获取训练数据
# df_train = df_model_fill.dropna()

# # 特征与目标
# X_train = df_train[['A', 'C']].dropna()  # 假设A列与C列有关系
# y_train = df_train['B']

# # 构建线性回归模型
# model = LinearRegression()
# model.fit(X_train, y_train)

# # 预测缺失值
# missing_data = df_model_fill[df_model_fill['B'].isnull()]
# predicted_values = model.predict(missing_data[['A', 'C']])

# # 填充缺失值
# df_model_fill.loc[df_model_fill['B'].isnull(), 'B'] = predicted_values
# print("用模型预测填充B列后的DataFrame：")
# print(df_model_fill)






#%%>>>>>>>>>>>>>>>>>>>>>>> 10. 标记缺失值
# 创建一个新列来标记A列中缺失值
df_mark_missing = df.copy()
df_mark_missing['A_missing'] = df['A'].isnull().astype(int)
print("标记A列缺失值后的DataFrame：")
print(df_mark_missing)





#%%>>>>>>>>>>>>>>>>>>>>>>> 代表案例

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子，保证可重复性
np.random.seed(42)

# 生成虚拟数据集
data = {
    'A': np.random.randint(1, 100, size=50).astype(float),  # 数值型
    'B': np.random.normal(50, 10, size=50),  # 数值型（带有正态分布）
    'C': np.random.choice(['X', 'Y', 'Z'], size=50),  # 类别型
    'D': np.random.choice([np.nan, 1, 0], size=50, p=[0.2, 0.4, 0.4])  # 二进制分类，带有缺失值
}

# 将部分数据设为NaN（人为加入缺失值）
df = pd.DataFrame(data)
df.loc[np.random.choice(df.index, size=10, replace=False), 'A'] = np.nan
df.loc[np.random.choice(df.index, size=15, replace=False), 'B'] = np.nan

# 创建子图布局
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 原始数据（带缺失值）
sns.scatterplot(x=df.index, y=df['A'], ax=axes[0, 0], color='red', marker='o', s=100, label="Column A (Original)")
sns.scatterplot(x=df.index, y=df['B'], ax=axes[0, 0], color='blue', marker='x', s=100, label="Column B (Original)")
axes[0, 0].set_title('Original Data with Missing Values')
axes[0, 0].legend()

# 2. 均值填充
df_mean_filled = df.copy()
df_mean_filled['A'].fillna(df['A'].mean(), inplace=True)
df_mean_filled['B'].fillna(df['B'].mean(), inplace=True)
sns.scatterplot(x=df_mean_filled.index, y=df_mean_filled['A'], ax=axes[0, 1], color='green', marker='o', s=100, label="Column A (Mean Filled)")
sns.scatterplot(x=df_mean_filled.index, y=df_mean_filled['B'], ax=axes[0, 1], color='purple', marker='x', s=100, label="Column B (Mean Filled)")
axes[0, 1].set_title('Mean Filling')
axes[0, 1].legend()

# 3. 中位数填充
df_median_filled = df.copy()
df_median_filled['A'].fillna(df['A'].median(), inplace=True)
df_median_filled['B'].fillna(df['B'].median(), inplace=True)
sns.scatterplot(x=df_median_filled.index, y=df_median_filled['A'], ax=axes[1, 0], color='orange', marker='o', s=100, label="Column A (Median Filled)")
sns.scatterplot(x=df_median_filled.index, y=df_median_filled['B'], ax=axes[1, 0], color='cyan', marker='x', s=100, label="Column B (Median Filled)")
axes[1, 0].set_title('Median Filling')
axes[1, 0].legend()

# 4. 插值填充
df_interpolated = df.copy()
df_interpolated['A'].interpolate(method='linear', inplace=True)
df_interpolated['B'].interpolate(method='linear', inplace=True)
sns.scatterplot(x=df_interpolated.index, y=df_interpolated['A'], ax=axes[1, 1], color='pink', marker='o', s=100, label="Column A (Interpolated)")
sns.scatterplot(x=df_interpolated.index, y=df_interpolated['B'], ax=axes[1, 1], color='yellow', marker='x', s=100, label="Column B (Interpolated)")
axes[1, 1].set_title('Interpolation Filling')
axes[1, 1].legend()

# 调整子图间距
plt.tight_layout()

# 显示图形
plt.show()

















