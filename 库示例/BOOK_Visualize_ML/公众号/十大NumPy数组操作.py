#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 12:12:37 2024

@author: jack
"""

#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 1. 创建数组 (np.array())

import numpy as np

# 创建一维数组
arr1 = np.array([1, 2, 3, 4])
print("一维数组:", arr1)

# 创建二维数组
arr2 = np.array([[1, 2], [3, 4]])
print("二维数组:\n", arr2)

# 创建三维数组
arr3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("三维数组:\n", arr3)

# 指定数据类型为浮点型
arr4 = np.array([1, 2, 3, 4], dtype=float)
print("指定浮点型数组:", arr4)



#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2. 数组形状操作 (reshape()、flatten()、ravel())

# 创建二维数组
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 改变形状为 (3, 2)
reshaped = arr.reshape((3, 2))
print("重塑后的数组:\n", reshaped)

# 展平数组
flattened = arr.flatten()
print("展平后的数组:", flattened)

# 使用 ravel 展平数组
ravelled = arr.ravel()
print("使用 ravel 展平的数组:", ravelled)




#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 3. 数组索引与切片 (array slicing)
# 创建二维数组
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 访问单个元素
print("访问 (0, 1) 元素:", arr[0, 1])

# 切片操作
print("切片得到的子数组:\n", arr[1:, 1:])

# 使用步长进行切片
print("步长为 2 的切片:\n", arr[::2, ::2])



#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 4. 数组广播 (broadcasting)
# 广播标量和数组
arr = np.array([1, 2, 3])
print("数组与标量相加:", arr + 10)

# 广播不同形状的数组
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
arr3 = np.array([1, 2, 3])
print("二维数组与一维数组相加:\n", arr2 + arr3)



#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 5. 数学运算 (array operations)
arr = np.array([1, 2, 3])

# 基本算术运算
print("数组加 5:", arr + 5)
print("数组乘 2:", arr * 2)

# 数组的统计运算
print("数组的和:", arr.sum())
print("数组的均值:", arr.mean())

# 线性代数运算
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])
print("矩阵乘法:\n", np.dot(mat1, mat2))


#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 6. 数组条件筛选 (boolean masking)
arr = np.array([1, 2, 3, 4, 5, 6])

# 布尔掩码筛选
mask = arr > 3
print("大于 3 的元素:", arr[mask])

# 使用 np.where 条件替换
new_arr = np.where(arr > 3, arr * 10, arr)
print("大于 3 的元素乘以 10:", new_arr)


#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 7. 数组合并与分割 (concatenate()、split())

# 合并数组
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
combined = np.concatenate((arr1, arr2))
print("合并后的数组:", combined)

# 堆叠数组
stacked = np.stack((arr1, arr2), axis=0)
print("堆叠后的数组:\n", stacked)

# 分割数组
split_arr = np.split(combined, 2)
print("分割后的数组:", split_arr)



#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 8. 数组维度扩展与压缩 (expand_dims()、squeeze())

# 原始数组
arr = np.array([1, 2, 3])

# 扩展维度
expanded = np.expand_dims(arr, axis=0)
print("扩展维度后的数组:\n", expanded)

# 压缩维度
squeezed = np.squeeze(expanded)
print("压缩维度后的数组:", squeezed)

#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 9. 数组排序与去重 (sort()、unique())

arr = np.array([3, 1, 2, 3, 4])

# 排序
sorted_arr = np.sort(arr)
print("排序后的数组:", sorted_arr)

# 去重
unique_arr = np.unique(arr)
print("去重后的数组:", unique_arr)



#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 10. 随机数生成 (np.random 模块)

# 生成均匀分布的随机数
rand_arr = np.random.rand(3, 3)
print("均匀分布的随机数数组:\n", rand_arr)

# 生成标准正态分布的随机数
randn_arr = np.random.randn(3, 3)
print("标准正态分布的随机数数组:\n", randn_arr)

# 生成随机整数
randint_arr = np.random.randint(0, 10, size=(3, 3))
print("随机整数数组:\n", randint_arr)

# 随机打乱数组
arr = np.array([1, 2, 3, 4, 5])
np.random.shuffle(arr)
print("打乱后的数组:", arr)





#%%>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 完整案例

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 1. 使用 NumPy 生成虚拟数据集
# 生成 X 特征数据，均匀分布在 0 到 10 之间
X = np.random.rand(100, 1) * 10

# 生成 Y 特征数据，假设 Y 与 X 存在线性关系，但有噪声影响
# Y = 2.5 * X + 噪声，噪声服从正态分布
noise = np.random.randn(100, 1) * 5
Y = 2.5 * X + noise

# 2. 数组操作 - 数据标准化 (Z-Score Normalization)
X_mean = X.mean()
X_std = X.std()
X_norm = (X - X_mean) / X_std  # 标准化后的 X

Y_mean = Y.mean()
Y_std = Y.std()
Y_norm = (Y - Y_mean) / Y_std  # 标准化后的 Y

# 3. 数组合并 - 合并标准化后的 X 和 Y
data = np.concatenate((X_norm, Y_norm), axis=1)

# 4. 数组形状变换
# 假设我们需要将一维数组展平处理或者重塑为不同形状
X_flat = X_norm.flatten()  # 展平
Y_flat = Y_norm.flatten()

# 5. 条件筛选 - 筛选 X_norm 中大于 0 的数据
X_positive = X_norm[X_norm > 0]
Y_positive = Y_norm[:X_positive.shape[0]]  # 对应筛选出的 Y 数据

# 6. 统计运算 - 计算均值、方差、最大值、最小值
X_stats = {
    'mean': np.mean(X),
    'var': np.var(X),
    'max': np.max(X),
    'min': np.min(X)
}
Y_stats = {
    'mean': np.mean(Y),
    'var': np.var(Y),
    'max': np.max(Y),
    'min': np.min(Y)
}

# 7. 广播操作 - 对 Y 进行批量运算，将所有元素乘以一个常数
Y_scaled = Y_norm * 2.5

# 8. 线性回归拟合 - 使用标准化后的数据
model = LinearRegression()
model.fit(X_norm, Y_norm)
Y_pred = model.predict(X_norm)

# 设置图形大小
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Advanced Data Analysis: Scatter Plot, Regression Line, Distribution and Operations', fontsize=16)

# 绘制散点图和线性回归线
axs[0, 0].scatter(X_norm, Y_norm, color='red', label='Data Points (Normalized)', alpha=0.7)
axs[0, 0].plot(X_norm, Y_pred, color='blue', label='Regression Line', linewidth=2)
axs[0, 0].set_title('Normalized Scatter Plot with Regression Line')
axs[0, 0].set_xlabel('X (Normalized)')
axs[0, 0].set_ylabel('Y (Normalized)')
axs[0, 0].legend()

# 绘制 X 的直方图
axs[0, 1].hist(X_norm, bins=20, color='green', edgecolor='black', alpha=0.7)
axs[0, 1].set_title('Distribution of X (Normalized)')
axs[0, 1].set_xlabel('X Values')
axs[0, 1].set_ylabel('Frequency')

# 绘制 Y 的直方图
axs[1, 0].hist(Y_norm, bins=20, color='purple', edgecolor='black', alpha=0.7)
axs[1, 0].set_title('Distribution of Y (Normalized)')
axs[1, 0].set_xlabel('Y Values')
axs[1, 0].set_ylabel('Frequency')

# 绘制残差图，展示真实值与预测值之间的差异
residuals = Y_norm - Y_pred
axs[1, 1].scatter(X_norm, residuals, color='orange', label='Residuals', alpha=0.7)
axs[1, 1].axhline(0, color='black', linestyle='--', linewidth=2)
axs[1, 1].set_title('Residuals Plot (Normalized)')
axs[1, 1].set_xlabel('X (Normalized)')
axs[1, 1].set_ylabel('Residuals')
axs[1, 1].legend()

# 调整布局以避免子图重叠
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 展示图形
plt.show()

# 打印统计信息
print("X 的统计信息:", X_stats)
print("Y 的统计信息:", Y_stats)










