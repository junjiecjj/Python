#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:10:02 2025

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247487452&idx=1&sn=1f13195e905eb052bc5bf43e074bdb1e&chksm=c1c596ed93144a3e2e80560dfd2812101cb82508544c505a9a421f91cf841a132c4d23c16223&mpshare=1&scene=1&srcid=0104OhIzKTvY1X2jLyZdraLU&sharer_shareinfo=c5ddf2dc7192576249f1c09fd3d8f7d5&sharer_shareinfo_first=c5ddf2dc7192576249f1c09fd3d8f7d5&exportkey=n_ChQIAhIQo4CVwoX07ipRaF5GWNl%2BxBKfAgIE97dBBAEAAAAAAIDRIHGmZK8AAAAOpnltbLcz9gKNyK89dVj0uQ8jbrsIRmbTq%2BRq4xINGRGELFP%2BBZWz5mkh181giS14wQRnQKnSMF8faiRKHX%2BTDqCrancuNaRSjzJ91yCimVg6V%2BBZgRyrGVxTilUzD6zn%2Fyd0acFL8SDAkeLMkoFfsPC4hiFyMAdejbrQsxrEz%2BjTVNlmguxcr8U6zBPFhMmVyCwdTLeYbkJhzHFNEtO7UQf211n3wTa2jVc4%2Fe%2FGl%2Bq9p7lwQpdvSfmHjBFw1DyNU1HGgVSUlnS2HQZaioe8eCXFIrgy3R0lBCU4rg9knfyz9gQixn9TqKvaFdTD8nf23Ys3a5VtjM3Tpxq2GrnuBOuP3y7zGHny&acctmode=0&pass_ticket=1o%2BN7WWg6YfcuH%2FvyVcYSr7gPLU5GF3q%2Bv%2FSUGBIwxOtDzDBYselneAeiQqvsqQq&wx_header=0&poc_token=HBg6emej8v1_21hz_B6tbxBvJhuXB_rUqs5hjqto


"""

#%%>>>>>>>>>>>>>>>>>>>>>>>> 1. 标准化 (Standardization)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 生成虚拟数据集
np.random.seed(0)
data = {
    'feature_1': np.random.normal(50, 10, 100),  # 正态分布
    'feature_2': np.random.normal(20, 5, 100),   # 正态分布
    'feature_3': np.random.uniform(30, 60, 100)  # 均匀分布
}

# 创建DataFrame
df = pd.DataFrame(data)

# 标准化
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

plt.figure(figsize=(14, 6))

# 子图1: 标准化前的直方图
plt.subplot(2, 2, 1)
plt.hist(df['feature_1'], bins=15, alpha=0.7, color='dodgerblue', edgecolor='black')
plt.hist(df['feature_2'], bins=15, alpha=0.7, color='salmon', edgecolor='black')
plt.hist(df['feature_3'], bins=15, alpha=0.7, color='limegreen', edgecolor='black')
plt.title('Histogram before Standardization')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend(['feature_1', 'feature_2', 'feature_3'])

# 子图2: 标准化后的直方图
plt.subplot(2, 2, 2)
plt.hist(df_standardized['feature_1'], bins=15, alpha=0.7, color='dodgerblue', edgecolor='black')
plt.hist(df_standardized['feature_2'], bins=15, alpha=0.7, color='salmon', edgecolor='black')
plt.hist(df_standardized['feature_3'], bins=15, alpha=0.7, color='limegreen', edgecolor='black')
plt.title('Histogram after Standardization')
plt.xlabel('Standardized Value')
plt.ylabel('Frequency')
plt.legend(['feature_1', 'feature_2', 'feature_3'])

# 子图3: 标准化前的盒须图
plt.subplot(2, 2, 3)
plt.boxplot([df['feature_1'], df['feature_2'], df['feature_3']], patch_artist=True,
            boxprops=dict(facecolor='skyblue'), medianprops=dict(color='red'))
plt.title('Box Plot before Standardization')
plt.xticks([1, 2, 3], ['feature_1', 'feature_2', 'feature_3'])
plt.ylabel('Value')

# 子图4: 标准化后的盒须图
plt.subplot(2, 2, 4)
plt.boxplot([df_standardized['feature_1'], df_standardized['feature_2'], df_standardized['feature_3']], patch_artist=True,
            boxprops=dict(facecolor='skyblue'), medianprops=dict(color='red'))
plt.title('Box Plot after Standardization')
plt.xticks([1, 2, 3], ['feature_1', 'feature_2', 'feature_3'])
plt.ylabel('Standardized Value')

# 调整布局
plt.tight_layout()
plt.show()






#%%>>>>>>>>>>>>>>>>>>>>>>>> 2. 归一化 (Normalization)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 设置随机种子以便复现
np.random.seed(42)

# 生成虚拟数据集
num_samples = 100
feature1 = np.random.normal(loc=100, scale=20, size=num_samples)
feature2 = np.random.normal(loc=50, scale=10, size=num_samples)
feature3 = np.random.normal(loc=30, scale=5, size=num_samples)

# 创建DataFrame
data = pd.DataFrame({
    'Feature 1': feature1,
    'Feature 2': feature2,
    'Feature 3': feature3
})

# 归一化数据
scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# 设置绘图样式
sns.set(style="whitegrid")

# 创建图形
plt.figure(figsize=(16, 10))

# 散点图（归一化前和归一化后）
plt.subplot(3, 3, 1)
plt.scatter(data['Feature 1'], data['Feature 2'], color='royalblue', alpha=0.6, label='Before Normalization')
plt.scatter(data_normalized['Feature 1'], data_normalized['Feature 2'], color='coral', alpha=0.6, label='After Normalization')
plt.title('Scatter Plot Before and After Normalization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# 箱线图（归一化前和归一化后）
plt.subplot(3, 3, 2)
sns.boxplot(data=data)
plt.title('Box Plot Before Normalization')

plt.subplot(3, 3, 5)
sns.boxplot(data=data_normalized)
plt.title('Box Plot After Normalization')

# 直方图（归一化前和归一化后）
plt.subplot(3, 3, 3)
plt.hist(data['Feature 1'], bins=15, color='cyan', alpha=0.7, label='Feature 1')
plt.hist(data['Feature 2'], bins=15, color='magenta', alpha=0.7, label='Feature 2')
plt.hist(data['Feature 3'], bins=15, color='yellow', alpha=0.7, label='Feature 3')
plt.title('Histogram Before Normalization')
plt.legend()

plt.subplot(3, 3, 6)
plt.hist(data_normalized['Feature 1'], bins=15, color='cyan', alpha=0.7, label='Feature 1')
plt.hist(data_normalized['Feature 2'], bins=15, color='magenta', alpha=0.7, label='Feature 2')
plt.hist(data_normalized['Feature 3'], bins=15, color='yellow', alpha=0.7, label='Feature 3')
plt.title('Histogram After Normalization')
plt.legend()

# 调整布局
plt.tight_layout()
plt.show()






#%%>>>>>>>>>>>>>>>>>>>>>>>> 3. 对数变换 (Log Transformation)


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 生成右偏的伽马分布数据
data = np.random.gamma(shape=2, scale=2, size=1000)

# 对数变换数据
log_data = np.log(data)

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 设置颜色
colors = ['#FF6347', '#4682B4']

# 1. 原始数据的直方图
axes[0, 0].hist(data, bins=30, color=colors[0], alpha=0.7)
axes[0, 0].set_title('Original Data Histogram', fontsize=14)
axes[0, 0].set_xlabel('Value', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)

# 2. 对数变换后数据的直方图
axes[0, 1].hist(log_data, bins=30, color=colors[1], alpha=0.7)
axes[0, 1].set_title('Log Transformed Data Histogram', fontsize=14)
axes[0, 1].set_xlabel('Log(Value)', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)

# 3. 原始数据的QQ图
stats.probplot(data, dist="norm", plot=axes[1, 0])
axes[1, 0].get_lines()[1].set_color('black')  # 设置拟合线的颜色
axes[1, 0].get_lines()[0].set_color(colors[0])  # 设置点的颜色
axes[1, 0].set_title('QQ Plot of Original Data', fontsize=14)

# 4. 对数变换后数据的QQ图
stats.probplot(log_data, dist="norm", plot=axes[1, 1])
axes[1, 1].get_lines()[1].set_color('black')  # 设置拟合线的颜色
axes[1, 1].get_lines()[0].set_color(colors[1])  # 设置点的颜色
axes[1, 1].set_title('QQ Plot of Log Transformed Data', fontsize=14)

# 调整布局
plt.tight_layout()
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>>> 4. 平方根变换 (Square Root Transformation)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# 设置随机种子
np.random.seed(42)

# 生成一个右偏态分布的数据集
data_size = 1000
data = np.random.exponential(scale=2.0, size=data_size)

# 应用平方根变换
transformed_data = np.sqrt(data)

# 创建数据框
df = pd.DataFrame({
    'Original': data,
    'Transformed': transformed_data
})

# 创建绘图区域
plt.figure(figsize=(12, 10))

# 原始数据直方图
plt.subplot(2, 2, 1)
sns.histplot(df['Original'], bins=30, kde=True, color='cyan', stat='density')
plt.title('Original Data Histogram')
plt.xlabel('Value')
plt.ylabel('Density')

# 变换后数据直方图
plt.subplot(2, 2, 2)
sns.histplot(df['Transformed'], bins=30, kde=True, color='magenta', stat='density')
plt.title('Square Root Transformed Data Histogram')
plt.xlabel('Value')
plt.ylabel('Density')

# Q-Q图
plt.subplot(2, 2, 3)
stats.probplot(df['Original'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Original Data')

# 箱线图
plt.subplot(2, 2, 4)
sns.boxplot(data=df[['Original', 'Transformed']], palette='Set2')
plt.title('Boxplot of Original and Transformed Data')
plt.ylabel('Value')

# 调整布局
plt.tight_layout()
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>>> 5. Box-Cox变换 (Box-Cox Transformation)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 生成一个虚拟数据集（右偏分布）
np.random.seed(42)
data = np.random.exponential(scale=2, size=1000)

# Box-Cox 变换
data_transformed, _ = stats.boxcox(data)

# 创建图形和子图
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# 1. 原始数据直方图
sns.histplot(data, kde=True, color='r', bins=30, edgecolor='black', ax=axs[0, 0])
axs[0, 0].set_title('Original Data Histogram')

# 2. Box-Cox 变换后的数据直方图
sns.histplot(data_transformed, kde=True, color='b', bins=30, edgecolor='black', ax=axs[0, 1])
axs[0, 1].set_title('Box-Cox Transformed Data Histogram')

# 3. 原始数据的 Q-Q 图
stats.probplot(data, dist="norm", plot=axs[0, 2])
axs[0, 2].set_title('Original Data Q-Q Plot')

# 4. Box-Cox 变换后的 Q-Q 图
stats.probplot(data_transformed, dist="norm", plot=axs[1, 0])
axs[1, 0].set_title('Box-Cox Transformed Data Q-Q Plot')

# 5. 原始和变换后的数据箱线图对比
sns.boxplot(data=[data, data_transformed], palette="Set2", ax=axs[1, 1])
axs[1, 1].set_xticks([0, 1])
axs[1, 1].set_xticklabels(['Original Data', 'Box-Cox Transformed Data'])
axs[1, 1].set_title('Original vs Box-Cox Transformed Data Boxplot')

# 隐藏最后一个子图的坐标轴
axs[1, 2].axis('off')

# 调整布局
plt.tight_layout()
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>>> 6. 最大绝对值缩放 (MaxAbs Scaling)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler

# 生成虚拟数据集
np.random.seed(0)
data = np.random.randn(100, 3) * 100  # 生成100行3列的随机数据
columns = ['Feature1', 'Feature2', 'Feature3']
df = pd.DataFrame(data, columns=columns)

# 使用最大绝对值缩放
scaler = MaxAbsScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=columns)

# 绘制图形
plt.figure(figsize=(16, 10))

# 散点图
plt.subplot(2, 2, 1)
plt.scatter(scaled_df['Feature1'], scaled_df['Feature2'], c='cyan', alpha=0.6)
plt.title('Scatter Plot: Feature1 vs Feature2')
plt.xlabel('Feature1 (Scaled)')
plt.ylabel('Feature2 (Scaled)')
plt.grid(True)

# 箱形图
plt.subplot(2, 2, 2)
plt.boxplot(scaled_df.values, labels=scaled_df.columns, patch_artist=True,
            boxprops=dict(facecolor='orange', color='blue'),
            medianprops=dict(color='red'))
plt.title('Box Plot of Scaled Features')
plt.ylabel('Values (Scaled)')

# 直方图
plt.subplot(2, 2, 3)
plt.hist(scaled_df['Feature1'], bins=15, color='magenta', alpha=0.7, label='Feature1', edgecolor='black')
plt.hist(scaled_df['Feature2'], bins=15, color='yellow', alpha=0.5, label='Feature2', edgecolor='black')
plt.hist(scaled_df['Feature3'], bins=15, color='green', alpha=0.3, label='Feature3', edgecolor='black')
plt.title('Histogram of Scaled Features')
plt.xlabel('Value (Scaled)')
plt.ylabel('Frequency')
plt.legend()

# 密度图
plt.subplot(2, 2, 4)
for col in scaled_df.columns:
    scaled_df[col].plot(kind='kde', label=col, linewidth=2)
plt.title('Density Plot of Scaled Features')
plt.xlabel('Value (Scaled)')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()






#%%>>>>>>>>>>>>>>>>>>>>>>>> 7. 小数缩放 (Decimal Scaling)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 创建虚拟数据集
np.random.seed(42)
data_size = 100
feature1 = np.random.uniform(-5000, 5000, data_size)
feature2 = np.random.uniform(1000, 20000, data_size)

# 生成DataFrame
data = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2
})

# 定义小数缩放函数
def decimal_scaling(column):
    max_abs_value = np.max(np.abs(column))
    scaling_factor = np.ceil(np.log10(max_abs_value))
    return column / (10**scaling_factor)

# 应用小数缩放
data_scaled = data.copy()
data_scaled['feature1'] = decimal_scaling(data['feature1'])
data_scaled['feature2'] = decimal_scaling(data['feature2'])

# 创建子图
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# 直方图 - 原始数据
axs[0, 0].hist(data['feature1'], bins=20, color='blue', alpha=0.7, label='Feature1 (Original)')
axs[0, 0].set_title('Histogram of Feature1 (Original)', fontsize=12)
axs[0, 0].set_xlabel('Feature1')
axs[0, 0].set_ylabel('Frequency')

axs[0, 1].hist(data['feature2'], bins=20, color='green', alpha=0.7, label='Feature2 (Original)')
axs[0, 1].set_title('Histogram of Feature2 (Original)', fontsize=12)
axs[0, 1].set_xlabel('Feature2')
axs[0, 1].set_ylabel('Frequency')

# 直方图 - 缩放后数据
axs[1, 0].hist(data_scaled['feature1'], bins=20, color='orange', alpha=0.7, label='Feature1 (Scaled)')
axs[1, 0].set_title('Histogram of Feature1 (Scaled)', fontsize=12)
axs[1, 0].set_xlabel('Feature1')
axs[1, 0].set_ylabel('Frequency')

axs[1, 1].hist(data_scaled['feature2'], bins=20, color='red', alpha=0.7, label='Feature2 (Scaled)')
axs[1, 1].set_title('Histogram of Feature2 (Scaled)', fontsize=12)
axs[1, 1].set_xlabel('Feature2')
axs[1, 1].set_ylabel('Frequency')

# 散点图 - 原始 vs 缩放后
axs[0, 2].scatter(data['feature1'], data['feature2'], color='purple', label='Original Data', alpha=0.6)
axs[0, 2].set_title('Scatter Plot (Original Data)', fontsize=12)
axs[0, 2].set_xlabel('Feature1')
axs[0, 2].set_ylabel('Feature2')

axs[1, 2].scatter(data_scaled['feature1'], data_scaled['feature2'], color='cyan', label='Scaled Data', alpha=0.6)
axs[1, 2].set_title('Scatter Plot (Scaled Data)', fontsize=12)
axs[1, 2].set_xlabel('Feature1')
axs[1, 2].set_ylabel('Feature2')

# 调整布局
plt.tight_layout()
plt.show()






#%%>>>>>>>>>>>>>>>>>>>>>>>> 8. 分位数变换 (Quantile Transformation)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer

# 设置随机种子以便结果可复现
np.random.seed(42)

# 生成虚拟数据集
n_samples = 1000
# 正态分布数据
data_normal = np.random.normal(loc=0, scale=1, size=n_samples)
# 指数分布数据
data_exponential = np.random.exponential(scale=1, size=n_samples)

# 将数据整合到 DataFrame
df = pd.DataFrame({
    'Normal': data_normal,
    'Exponential': data_exponential
})

# 创建 QuantileTransformer 实例
quantile_transformer = QuantileTransformer(output_distribution='normal')

# 对数据进行分位数变换
df_transformed = quantile_transformer.fit_transform(df)

# 将转化后的数据放入新的 DataFrame
df_transformed = pd.DataFrame(df_transformed, columns=['Normal_Transformed', 'Exponential_Transformed'])

# 绘制图形
plt.figure(figsize=(16, 10))

# 绘制变换前后的直方图
plt.subplot(2, 2, 1)
sns.histplot(df['Normal'], color='skyblue', bins=30, kde=True, label='Normal Distribution', stat='density')
sns.histplot(df_transformed['Normal_Transformed'], color='orange', bins=30, kde=True, label='Transformed Normal', stat='density')
plt.title('Histogram of Normal Distribution vs Transformed Normal')
plt.legend()

plt.subplot(2, 2, 2)
sns.histplot(df['Exponential'], color='lightgreen', bins=30, kde=True, label='Exponential Distribution', stat='density')
sns.histplot(df_transformed['Exponential_Transformed'], color='red', bins=30, kde=True, label='Transformed Exponential', stat='density')
plt.title('Histogram of Exponential Distribution vs Transformed Exponential')
plt.legend()

# 绘制变换前后的箱线图
plt.subplot(2, 2, 3)
sns.boxplot(data=df[['Normal', 'Exponential']], palette='pastel')
plt.title('Boxplot of Original Distributions')

plt.subplot(2, 2, 4)
sns.boxplot(data=df_transformed[['Normal_Transformed', 'Exponential_Transformed']], palette='pastel')
plt.title('Boxplot of Transformed Distributions')

# 调整图形布局
plt.tight_layout()
plt.show()






#%%>>>>>>>>>>>>>>>>>>>>>>>> 9. 主成分分析 (PCA)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# 生成一个虚拟的三维数据集
np.random.seed(42)
n_samples = 500
# 三个特征分布不同
X = np.dot(np.random.rand(3, 3), np.random.randn(3, n_samples)).T

# 应用PCA将3维数据降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 创建图形窗口
fig = plt.figure(figsize=(16, 8))

# 1. 原始数据的三维散点图
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 2], cmap='viridis', s=50, alpha=0.7)
ax1.set_title('Original 3D Data', fontsize=14)
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('X3')

# 2. PCA降维后的二维散点图
ax2 = fig.add_subplot(122)
scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=X[:, 2], cmap='plasma', s=50, alpha=0.7)
ax2.set_title('2D Data After PCA', fontsize=14)
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')
plt.colorbar(scatter, ax=ax2, label='Original X3 Value')

# 3. PCA解释方差比例图（Scree Plot）
plt.figure(figsize=(8, 4))
plt.bar(range(1, 3), pca.explained_variance_ratio_, color='red', alpha=0.7)
plt.title('Scree Plot: Explained Variance Ratio by Component', fontsize=14)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks([1, 2])

# 显示图形
plt.tight_layout()
plt.show()






#%%>>>>>>>>>>>>>>>>>>>>>>>> 10. 独立成分分析 (Independent Component Analysis, ICA)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 生成两个独立的信号源
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# 源信号1: 正弦波
s1 = np.sin(2 * time)
# 源信号2: 方波
s2 = np.sign(np.sin(3 * time))

# 将源信号放在矩阵 S 中
S = np.c_[s1, s2]

# 将源信号加上噪声
S += 0.2 * np.random.normal(size=S.shape)

# 标准化信号：减去均值，除以标准差
S /= S.std(axis=0)

# 创建一个随机的混合矩阵
A = np.array([[1, 1], [0.5, 2]])  # 混合矩阵
X = np.dot(S, A.T)  # 混合信号

# 应用 ICA 来分离信号
ica = FastICA(n_components=2)
S_ = ica.fit_transform(X)  # 重构的源信号
A_ = ica.mixing_  # 得到的混合矩阵

# 图形显示
plt.figure(figsize=(12, 8))

# 绘制原始的独立信号
plt.subplot(3, 1, 1)
plt.title("Original Independent Signals (Sources)")
colors = ['red', 'blue']
for i, sig in enumerate(S.T):
    plt.plot(time, sig, color=colors[i], label=f"Source {i+1}")
plt.legend()

# 绘制混合信号
plt.subplot(3, 1, 2)
plt.title("Mixed Signals")
for i, sig in enumerate(X.T):
    plt.plot(time, sig, color=colors[i], label=f"Mixed {i+1}")
plt.legend()

# 绘制分离后的信号
plt.subplot(3, 1, 3)
plt.title("Signals after ICA (Separated Signals)")
for i, sig in enumerate(S_.T):
    plt.plot(time, sig, color=colors[i], label=f"ICA Separated {i+1}")
plt.legend()

plt.tight_layout()
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>>>








#%%>>>>>>>>>>>>>>>>>>>>>>>>








#%%>>>>>>>>>>>>>>>>>>>>>>>>








#%%>>>>>>>>>>>>>>>>>>>>>>>>








#%%>>>>>>>>>>>>>>>>>>>>>>>>








#%%>>>>>>>>>>>>>>>>>>>>>>>>








#%%>>>>>>>>>>>>>>>>>>>>>>>>








