#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:05:45 2024

@author: jack
"""

#%%>>>>>>>>>>>>>>>>>>>>>>> 1. np.mean() - 算术平均值
# 一维数组：
import numpy as np

data = np.array([1, 2, 3, 4, 5])
mean_value = np.mean(data)
print("Mean value:", mean_value)



# 二维数组：
data_2d = np.array([[1, 2, 3], [4, 5, 6]])

# 沿 axis=0（即列方向）计算均值
mean_axis0 = np.mean(data_2d, axis=0)
print("Mean along axis 0:", mean_axis0)

# 沿 axis=1（即行方向）计算均值
mean_axis1 = np.mean(data_2d, axis=1)
print("Mean along axis 1:", mean_axis1)


#%%>>>>>>>>>>>>>>>>>>>>>>> 2. np.median() - 中位数

# 一维数组：
data = np.array([3, 1, 4, 1, 5, 9])
median_value = np.median(data)
print("Median value:", median_value)

# 二维数组：
data_2d = np.array([[3, 1, 4], [1, 5, 9]])

# 沿 axis=0（列方向）计算中位数
median_axis0 = np.median(data_2d, axis=0)
print("Median along axis 0:", median_axis0)

# 沿 axis=1（行方向）计算中位数
median_axis1 = np.median(data_2d, axis=1)
print("Median along axis 1:", median_axis1)



#%%>>>>>>>>>>>>>>>>>>>>>>> 3. np.std() - 标准差


# 一维数组：
data = np.array([1, 2, 3, 4, 5])
std_value = np.std(data)
print("Standard deviation:", std_value)

# 二维数组：
data_2d = np.array([[1, 2, 3], [4, 5, 6]])

# 沿 axis=0（列方向）计算标准差
std_axis0 = np.std(data_2d, axis=0)
print("Standard deviation along axis 0:", std_axis0)

# 沿 axis=1（行方向）计算标准差
std_axis1 = np.std(data_2d, axis=1)
print("Standard deviation along axis 1:", std_axis1)


#%%>>>>>>>>>>>>>>>>>>>>>>> 4. np.var() - 方差

# 一维数组：
data = np.array([1, 2, 3, 4, 5])
var_value = np.var(data)
print("Variance:", var_value)


# 二维数组：
data_2d = np.array([[1, 2, 3], [4, 5, 6]])

# 沿 axis=0（列方向）计算方差
var_axis0 = np.var(data_2d, axis=0)
print("Variance along axis 0:", var_axis0)

# 沿 axis=1（行方向）计算方差
var_axis1 = np.var(data_2d, axis=1)
print("Variance along axis 1:", var_axis1)


#%%>>>>>>>>>>>>>>>>>>>>>>> 5. np.min() - 最小值

# 一维数组：
import numpy as np

data = np.array([5, 2, 8, 1, 7])
min_value = np.min(data)
print("Minimum value:", min_value)


# 二维数组：
data_2d = np.array([[5, 2, 8], [1, 7, 3]])

# 沿 axis=0（列方向）计算最小值
min_axis0 = np.min(data_2d, axis=0)
print("Minimum along axis 0:", min_axis0)

# 沿 axis=1（行方向）计算最小值
min_axis1 = np.min(data_2d, axis=1)
print("Minimum along axis 1:", min_axis1)


#%%>>>>>>>>>>>>>>>>>>>>>>> 6. np.max() - 最大值


# 一维数组：
data = np.array([5, 2, 8, 1, 7])
max_value = np.max(data)
print("Maximum value:", max_value)

# 二维数组：
data_2d = np.array([[5, 2, 8], [1, 7, 3]])

# 沿 axis=0（列方向）计算最大值
max_axis0 = np.max(data_2d, axis=0)
print("Maximum along axis 0:", max_axis0)

# 沿 axis=1（行方向）计算最大值
max_axis1 = np.max(data_2d, axis=1)
print("Maximum along axis 1:", max_axis1)


#%%>>>>>>>>>>>>>>>>>>>>>>> 7. np.percentile() - 百分位数


# 一维数组：
data = np.array([1, 2, 3, 4, 5])
percentile_25 = np.percentile(data, 25)
percentile_50 = np.percentile(data, 50)  # 50% 百分位数即中位数
percentile_75 = np.percentile(data, 75)
print("25th percentile:", percentile_25)
print("50th percentile (median):", percentile_50)
print("75th percentile:", percentile_75)

# 二维数组：
data_2d = np.array([[10, 20, 30], [40, 50, 60]])

# 沿 axis=0（列方向）计算百分位数
percentile_50_axis0 = np.percentile(data_2d, 50, axis=0)
print("50th percentile along axis 0:", percentile_50_axis0)

# 沿 axis=1（行方向）计算百分位数
percentile_50_axis1 = np.percentile(data_2d, 50, axis=1)
print("50th percentile along axis 1:", percentile_50_axis1)


#%%>>>>>>>>>>>>>>>>>>>>>>> 8. np.quantile() - 分位数

# 一维数组：
data = np.array([1, 2, 3, 4, 5])
quantile_25 = np.quantile(data, 0.25)
quantile_50 = np.quantile(data, 0.50)  # 50% 分位数即中位数
quantile_75 = np.quantile(data, 0.75)
print("25th quantile:", quantile_25)
print("50th quantile (median):", quantile_50)
print("75th quantile:", quantile_75)

# 二维数组：
data_2d = np.array([[10, 20, 30], [40, 50, 60]])

# 沿 axis=0（列方向）计算分位数
quantile_50_axis0 = np.quantile(data_2d, 0.50, axis=0)
print("50th quantile along axis 0:", quantile_50_axis0)

# 沿 axis=1（行方向）计算分位数
quantile_50_axis1 = np.quantile(data_2d, 0.50, axis=1)
print("50th quantile along axis 1:", quantile_50_axis1)



#%%>>>>>>>>>>>>>>>>>>>>>>> 9. np.corrcoef() - 相关系数矩阵

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])
correlation_matrix = np.corrcoef(x, y)
print("Correlation matrix:\n", correlation_matrix)


#%%>>>>>>>>>>>>>>>>>>>>>>> 10. np.cov() - 协方差矩阵

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])
covariance_matrix = np.cov(x, y)
print("Covariance matrix:\n", covariance_matrix)


#%%>>>>>>>>>>>>>>>>>>>>>>> Python 代码

import numpy as np
import matplotlib.pyplot as plt

# 设置随机数种子以确保结果可重复
np.random.seed(42)

# 生成虚拟数据集
x = np.random.normal(loc=50, scale=10, size=1000)  # 平均值为50，标准差为10的正态分布
y = 2 * x + np.random.normal(loc=0, scale=10, size=1000)  # 带有噪声的线性关系 y = 2x + 噪声

# 计算统计量
mean_x = np.mean(x)
mean_y = np.mean(y)
var_x = np.var(x)
var_y = np.var(y)
std_x = np.std(x)
std_y = np.std(y)
min_x = np.min(x)
min_y = np.min(y)
max_x = np.max(x)
max_y = np.max(y)
percentile_25_x = np.percentile(x, 25)
percentile_25_y = np.percentile(y, 25)
percentile_50_x = np.percentile(x, 50)
percentile_50_y = np.percentile(y, 50)
percentile_75_x = np.percentile(x, 75)
percentile_75_y = np.percentile(y, 75)
correlation = np.corrcoef(x, y)[0, 1]
covariance = np.cov(x, y)[0, 1]

# 打印统计结果
print(f"Mean of x: {mean_x:.2f}, Mean of y: {mean_y:.2f}")
print(f"Variance of x: {var_x:.2f}, Variance of y: {var_y:.2f}")
print(f"Standard Deviation of x: {std_x:.2f}, Standard Deviation of y: {std_y:.2f}")
print(f"Min of x: {min_x:.2f}, Max of x: {max_x:.2f}")
print(f"Min of y: {min_y:.2f}, Max of y: {max_y:.2f}")
print(f"25th Percentile of x: {percentile_25_x:.2f}, 50th Percentile (Median) of x: {percentile_50_x:.2f}, 75th Percentile of x: {percentile_75_x:.2f}")
print(f"25th Percentile of y: {percentile_25_y:.2f}, 50th Percentile (Median) of y: {percentile_50_y:.2f}, 75th Percentile of y: {percentile_75_y:.2f}")
print(f"Correlation between x and y: {correlation:.2f}")
print(f"Covariance between x and y: {covariance:.2f}")

# 创建图形
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 子图 1：散点图 (x, y) 显示两者之间的关系
axs[0, 0].scatter(x, y, color='red', alpha=0.6)
axs[0, 0].set_title("Scatter plot of x and y")
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("y")

# 子图 2：x 和 y 的直方图
axs[0, 1].hist(x, bins=30, color='blue', alpha=0.7, label='x')
axs[0, 1].hist(y, bins=30, color='green', alpha=0.7, label='y')
axs[0, 1].set_title("Histograms of x and y")
axs[0, 1].set_xlabel("Value")
axs[0, 1].set_ylabel("Frequency")
axs[0, 1].legend()

# 子图 3：x 的百分位数折线图
percentiles_x = np.percentile(x, [0, 25, 50, 75, 100])
axs[1, 0].plot([0, 25, 50, 75, 100], percentiles_x, marker='o', color='purple', label='x')
axs[1, 0].set_title("Percentiles of x")
axs[1, 0].set_xlabel("Percentile")
axs[1, 0].set_ylabel("Value")
axs[1, 0].legend()

# 子图 4：y 的百分位数折线图
percentiles_y = np.percentile(y, [0, 25, 50, 75, 100])
axs[1, 1].plot([0, 25, 50, 75, 100], percentiles_y, marker='s', color='orange', label='y')
axs[1, 1].set_title("Percentiles of y")
axs[1, 1].set_xlabel("Percentile")
axs[1, 1].set_ylabel("Value")
axs[1, 1].legend()

# 调整布局
plt.tight_layout()
plt.show()












