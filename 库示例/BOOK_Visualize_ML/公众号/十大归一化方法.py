#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:29:12 2024

@author: jack
"""

# https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247485637&idx=1&sn=4c1baf0c7cf027e622bd77a73afd6af0&chksm=c0e5d203f7925b1591a9a1d0d393dd604cee711f8ceca1f14947a0469db187aa35e060eabe99&cur_album_id=3445855686331105280&scene=190#rd



#%%>>>>>>>>>>>>>>  1. 最小-最大缩放（Min-Max Scaling）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 生成虚拟数据集
np.random.seed(42)
data = np.random.rand(100, 1) * 1000  # 随机生成1000个[0, 100)之间的数

# 使用Min-Max Scaling进行归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 绘制原始数据和缩放后的数据
plt.figure(figsize=(12, 5))

# 原始数据分布
plt.subplot(1, 2, 1)
plt.hist(data, bins=20, color='blue', edgecolor='black', alpha=0.7)
plt.title("Original Data Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")

# 缩放后的数据分布
plt.subplot(1, 2, 2)
plt.hist(data_scaled, bins=20, color='green', edgecolor='black', alpha=0.7)
plt.title("Scaled Data Distribution (Min-Max)")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# 数据分析图：原始数据 vs 缩放后的数据
plt.figure(figsize=(8, 6))
plt.scatter(data, data_scaled, color='purple', alpha=0.7)
plt.title("Original vs Scaled Data")
plt.xlabel("Original Data")
plt.ylabel("Scaled Data")
plt.grid(True)
plt.show()


#%%>>>>>>>>>>>>>> 2. 标准化（Standardization）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 生成虚拟数据
np.random.seed(0)
data = {
    'Feature1': np.random.normal(loc=10, scale=5, size=1000),
    'Feature2': np.random.normal(loc=20, scale=10, size=1000)
}

df = pd.DataFrame(data)

# 标准化数据
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

# 绘制原始数据散点图
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(df['Feature1'], df['Feature2'], alpha=0.7)
plt.title('Original Data')
plt.xlabel('Feature1')
plt.ylabel('Feature2')

# 绘制标准化后的数据散点图
plt.subplot(1, 2, 2)
plt.scatter(df_scaled['Feature1'], df_scaled['Feature2'], alpha=0.7, color='orange')
plt.title('Standardized Data')
plt.xlabel('Feature1')
plt.ylabel('Feature2')

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>  3. 最大绝对值缩放（MaxAbs Scaling）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler

# 生成一个随机数据集（100个样本，2个特征）
np.random.seed(42)
data = np.random.randn(10, 2) * 100  # 放大数据以便展示缩放效果

# 应用MaxAbsScaler进行归一化
scaler = MaxAbsScaler()
data_scaled = scaler.fit_transform(data)

# 绘制原始数据和归一化后数据的对比图
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 原始数据
ax[0].scatter(data[:, 0], data[:, 1], color='blue', label='Original Data')
ax[0].set_title('Original Data')
ax[0].set_xlim(-25, 25)
ax[0].set_ylim(-25, 25)
ax[0].legend()

# 归一化后数据
ax[1].scatter(data_scaled[:, 0], data_scaled[:, 1], color='green', label='MaxAbs Scaled Data')
ax[1].set_title('MaxAbs Scaled Data')
ax[1].set_xlim(-1.1, 1.1)
ax[1].set_ylim(-1.1, 1.1)
ax[1].legend()

plt.show()

# 绘制特征最大绝对值的对比图
fig, ax = plt.subplots(figsize=(8, 4))

max_values_original = np.max(np.abs(data), axis=0)
max_values_scaled = np.max(np.abs(data_scaled), axis=0)

bar_width = 0.35
index = np.arange(len(max_values_original))

bars1 = ax.bar(index, max_values_original, bar_width, color='blue', label='Original')
bars2 = ax.bar(index + bar_width, max_values_scaled, bar_width, color='green', label='Scaled')

ax.set_xlabel('Features')
ax.set_ylabel('Max Absolute Value')
ax.set_title('Comparison of Max Absolute Values Before and After Scaling')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(['Feature 1', 'Feature 2'])
ax.legend()

plt.show()


#%%>>>>>>>>>>>>>>  4. 均值归一化（Mean Normalization）
import numpy as np
import matplotlib.pyplot as plt

# 生成虚拟数据
np.random.seed(0)
data = np.random.rand(100) * 1000  # 生成0到100之间的1000个随机数

# 计算均值、最小值和最大值
mean = np.mean(data)
min_val = np.min(data)
max_val = np.max(data)

# 应用均值归一化
data_normalized = (data - mean) / (max_val - min_val)

# 绘制原始数据和归一化后的数据

# 原始数据图
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(data, bins=20, color='skyblue', edgecolor='black')
plt.title('Original Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 归一化后的数据图
plt.subplot(1, 2, 2)
plt.hist(data_normalized, bins=20, color='salmon', edgecolor='black')
plt.title('Normalized Data Distribution')
plt.xlabel('Normalized Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>  5. Z-score 标准化（Z-score Normalization）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skewnorm
from sklearn.preprocessing import StandardScaler

# 1. 生成虚拟数据集
np.random.seed(42)

# 特征1：高度偏态分布 (skewed distribution)
a = 10  # 控制偏度的参数，越大越偏
feature_1 = skewnorm.rvs(a, size=1000)

# 特征2：正态分布 (normal distribution)
feature_2 = np.random.normal(loc=0, scale=1, size=1000)

# 构建 DataFrame
df = pd.DataFrame({
    'Feature 1 (Skewed)': feature_1,
    'Feature 2 (Normal)': feature_2
})

# 2. 应用 Z-score 标准化
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 3. 绘制图表
plt.figure(figsize=(12, 6))

# 原始数据的分布
plt.subplot(1, 2, 1)
plt.hist(df['Feature 1 (Skewed)'], bins=30, alpha=0.7, label='Feature 1 (Skewed)')
plt.hist(df['Feature 2 (Normal)'], bins=30, alpha=0.7, label='Feature 2 (Normal)')
plt.title('Distribution of Original Data')
plt.legend()

# 标准化后的数据分布
plt.subplot(1, 2, 2)
plt.hist(df_normalized['Feature 1 (Skewed)'], bins=30, alpha=0.7, label='Feature 1 (Skewed)')
plt.hist(df_normalized['Feature 2 (Normal)'], bins=30, alpha=0.7, label='Feature 2 (Normal)')
plt.title('Distribution of Data After Z-score Normalization')
plt.legend()

plt.tight_layout()
plt.show()

# 数据分析的图：标准化前后均值和标准差的对比
summary = pd.DataFrame({
    'Original Mean': df.mean(),
    'Original Std': df.std(),
    'Normalized Mean': df_normalized.mean(),
    'Normalized Std': df_normalized.std()
})

print(summary)


#%%>>>>>>>>>>>>>>  6. 分位数缩放（Quantile Scaling）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer

# Step 1: 生成虚拟数据集
np.random.seed(42)
# 正态分布数据
feature_1 = np.random.normal(loc=0, scale=1, size=1000)
feature_2 = np.random.normal(loc=5, scale=2, size=1000)

# 添加一些极端值
feature_1 = np.append(feature_1, [10, 15, 20])
feature_2 = np.append(feature_2, [-10, -15, -20])

data = pd.DataFrame({
    'Feature 1': feature_1,
    'Feature 2': feature_2
})

# Step 2: 使用分位数缩放
quantile_transformer = QuantileTransformer(output_distribution='uniform', random_state=42)
data_quantile_scaled = quantile_transformer.fit_transform(data)

data_quantile_scaled = pd.DataFrame(data_quantile_scaled, columns=['Feature 1', 'Feature 2'])

# Step 3: 可视化原始数据和缩放后的数据

# 原始数据分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(data['Feature 1'], data['Feature 2'], color='blue', alpha=0.5)
plt.title('Original Data Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 缩放后数据分布
plt.subplot(1, 2, 2)
plt.scatter(data_quantile_scaled['Feature 1'], data_quantile_scaled['Feature 2'], color='green', alpha=0.5)
plt.title('Quantile Scaled Data Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>  7. 二值化（Binarization）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer

# 生成虚拟数据集
np.random.seed(0)
X = np.random.normal(0, 1, (1000, 2))  # 生成1000个二维正态分布数据点

# 设置二值化阈值
threshold = 0.5

# 使用Binarizer进行二值化
binarizer = Binarizer(threshold=threshold)
X_binarized = binarizer.transform(X)

# 画原始数据的散点图
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], color='blue', alpha=0.5, label='Original Data')
plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
plt.axvline(threshold, color='red', linestyle='--')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# 画二值化后的数据散点图
plt.subplot(1, 2, 2)
plt.scatter(X_binarized[:, 0], X_binarized[:, 1], color='green', alpha=0.5, label='Binarized Data')
plt.title('Binarized Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.show()

# 数据分析图：原始数据和二值化数据的直方图对比
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(X.ravel(), bins=20, color='blue', alpha=0.7, label='Original Data')
plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
plt.title('Histogram of Original Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(X_binarized.ravel(), bins=20, color='green', alpha=0.7, label='Binarized Data')
plt.title('Histogram of Binarized Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.show()


#%%>>>>>>>>>>>>>>  8. 对数变换（Log Transformation）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 生成虚拟数据
np.random.seed(42)
data_size = 1000
x = np.linspace(1, 100, data_size)  # 避免对数变换时出现0值
y = x ** 2 + np.random.normal(0, 1000, data_size)  # y = x^2加上一些噪声

# 创建数据框
df = pd.DataFrame({'x': x, 'y': y})

# 对数变换
df['log_y'] = np.log(df['y'] + 1)  # 加1以避免对数变换中出现无穷大值

# 绘制原始数据和对数变换后的数据
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 原始数据图
sns.scatterplot(x='x', y='y', data=df, ax=axes[0])
axes[0].set_title('Original Data')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

# 对数变换后的数据图
sns.scatterplot(x='x', y='log_y', data=df, ax=axes[1])
axes[1].set_title('Data After Log Transformation')
axes[1].set_xlabel('x')
axes[1].set_ylabel('log_y')

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>  9. 平方根变换（Square Root Transformation）
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# 生成具有正偏态分布的虚拟数据集
np.random.seed(42)
data = np.random.chisquare(df=5, size=1000)

# 进行平方根变换
sqrt_transformed_data = np.sqrt(data)

# 绘制原始数据和变换后数据的分布图
plt.figure(figsize=(14, 6))

# 原始数据分布
plt.subplot(1, 2, 1)
sns.histplot(data, kde=True, color='blue', stat="density")
plt.title("Original Data Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, data.mean(), data.std())
plt.plot(x, p, 'k', linewidth=2)

# 变换后数据分布
plt.subplot(1, 2, 2)
sns.histplot(sqrt_transformed_data, kde=True, color='green', stat="density")
plt.title("Square Root Transformed Data Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, sqrt_transformed_data.mean(), sqrt_transformed_data.std())
plt.plot(x, p, 'k', linewidth=2)

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>  10. Box-Cox 变换（Box-Cox Transformation）

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 生成虚拟数据
np.random.seed(0)  # 设定随机种子以保证结果可重复
data = np.random.exponential(scale=2.0, size=1000)  # 指数分布数据

# Box-Cox变换
transformed_data, lambda_best_fit = stats.boxcox(data)

# 绘制原始数据和Box-Cox变换后的数据分布图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 原始数据的直方图
axes[0].hist(data, bins=30, color='skyblue', edgecolor='black')
axes[0].set_title('Original Data Distribution')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box-Cox变换后的数据的直方图
axes[1].hist(transformed_data, bins=30, color='lightgreen', edgecolor='black')
axes[1].set_title('Box-Cox Transformed Data Distribution')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# 输出Box-Cox变换的lambda值
print(f'Optimal lambda value for Box-Cox transformation: {lambda_best_fit}')






































































































































































































































































































