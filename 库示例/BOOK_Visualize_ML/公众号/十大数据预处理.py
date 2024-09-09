#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:47:50 2024

@author: jack

https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247486491&idx=1&sn=33c380534fbc6dde8b34bed91d327802&chksm=c10a572d191f7afdd1aa66380b01f0e28aa5f761fe81ee5220255efa889b572499054a694d57&mpshare=1&scene=1&srcid=0909wZEAo2oOaCN023prZgK3&sharer_shareinfo=7c6b764729a3159f1159d2b26bc4eea0&sharer_shareinfo_first=7c6b764729a3159f1159d2b26bc4eea0&exportkey=n_ChQIAhIQUVi9khACDazYqDcx5qfFuhKfAgIE97dBBAEAAAAAAEAlIRHP0KEAAAAOpnltbLcz9gKNyK89dVj046Ldg9gaQMqURm0hJ6YW1Vg%2FcWWeWv9MqX%2BX8L0%2FE3RgKMIZoljVDoXYZZkj%2FWCPXygN0r7hLxuGo2%2BWvFYiF3aYjn%2Fa5WnrnnlAhNFmGu320Ag89cltDXhwvkULSa7dosLnNcvKraPvnJ05FPFvEVwVhisP1LoUN%2BdFw%2B0FMXsVUGUAPM0hIett6XAhNSWy6%2F5NauyzZOfGhdDm2GbBNSjDz%2F8Sop9HV3d3HUgQtbiJU8ON9TKygtet1lWJSGNCyZVldxjDKgM5WrXR%2FFwLF4wInupeGT3%2Bjc3iHc70140lckL805xilAnXuFAHoV2o9HfbmyYc7QgW&acctmode=0&pass_ticket=5oYrzMQ8Va1H%2FUeBo6Erveu4PfcAeP0p9iYc8HLV7D2C1W7BWvwxFIexG66WVzJo&wx_header=0#rd

"""

#%%>>>>>>>>>>>>>>>>>>>>>>> 1. 数据清洗 (Data Cleaning)
# 1.1 处理缺失值 (Handling Missing Values)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 设置随机种子
np.random.seed(42)

# 生成虚拟数据集
data = {
    'price': np.random.normal(300000, 75000, 100).tolist() + [np.nan]*5 + [1e7, 1e8],  # 房屋价格，包含缺失值和异常值
    'area': np.random.normal(2000, 500, 100).tolist() + [np.nan]*5 + [15000, 20000],   # 房屋面积，包含缺失值和异常值
    'bedrooms': np.random.randint(1, 6, 100).tolist() + [np.nan]*5 + [1, 10],          # 卧室数量，包含缺失值和异常值
    'year_built': np.random.randint(1950, 2020, 100).tolist() + [np.nan]*5 + [1800, 2025]  # 建造年份，包含缺失值和异常值
}

df = pd.DataFrame(data)

# 绘制数据清洗前的分布图
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
sns.histplot(df['price'], kde=True, color='red', label='Price')
plt.title('Price Distribution (Before Cleaning)')
plt.legend()

plt.subplot(2, 2, 2)
sns.histplot(df['area'], kde=True, color='blue', label='Area')
plt.title('Area Distribution (Before Cleaning)')
plt.legend()

plt.subplot(2, 2, 3)
sns.histplot(df['bedrooms'], kde=True, color='green', label='Bedrooms')
plt.title('Bedrooms Distribution (Before Cleaning)')
plt.legend()

plt.subplot(2, 2, 4)
sns.histplot(df['year_built'], kde=True, color='purple', label='Year Built')
plt.title('Year Built Distribution (Before Cleaning)')
plt.legend()

plt.tight_layout()
plt.show()

# 数据清洗步骤

# 1. 处理缺失值：使用中位数填充缺失值
df.fillna(df.median(), inplace=True)

# 2. 处理异常值：将超过三倍标准差的值视为异常值并替换为上下限
for col in df.columns:
    upper_limit = df[col].mean() + 3 * df[col].std()
    lower_limit = df[col].mean() - 3 * df[col].std()
    df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
    df[col] = np.where(df[col] < lower_limit, lower_limit, df[col])

# 3. 数据标准化
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 绘制数据清洗后的分布图
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
sns.histplot(df['price'], kde=True, color='red', label='Price')
plt.title('Price Distribution (After Cleaning)')
plt.legend()

plt.subplot(2, 2, 2)
sns.histplot(df['area'], kde=True, color='blue', label='Area')
plt.title('Area Distribution (After Cleaning)')
plt.legend()

plt.subplot(2, 2, 3)
sns.histplot(df['bedrooms'], kde=True, color='green', label='Bedrooms')
plt.title('Bedrooms Distribution (After Cleaning)')
plt.legend()

plt.subplot(2, 2, 4)
sns.histplot(df['year_built'], kde=True, color='purple', label='Year Built')
plt.title('Year Built Distribution (After Cleaning)')
plt.legend()

plt.tight_layout()
plt.show()

# 绘制标准化后的数据分布图
plt.figure(figsize=(16, 12))

plt.subplot(2, 2, 1)
sns.histplot(df_scaled['price'], kde=True, color='red', label='Price (Scaled)')
plt.title('Price Distribution (After Scaling)')
plt.legend()

plt.subplot(2, 2, 2)
sns.histplot(df_scaled['area'], kde=True, color='blue', label='Area (Scaled)')
plt.title('Area Distribution (After Scaling)')
plt.legend()

plt.subplot(2, 2, 3)
sns.histplot(df_scaled['bedrooms'], kde=True, color='green', label='Bedrooms (Scaled)')
plt.title('Bedrooms Distribution (After Scaling)')
plt.legend()

plt.subplot(2, 2, 4)
sns.histplot(df_scaled['year_built'], kde=True, color='purple', label='Year Built (Scaled)')
plt.title('Year Built Distribution (After Scaling)')
plt.legend()

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 2. 数据标准化 (Normalization)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 生成虚拟数据集
np.random.seed(42)
data = {
    'Feature 1': np.random.randint(0, 1000, 100),
    'Feature 2': np.random.randint(0, 50, 100),
    'Feature 3': np.random.randint(-100, 100, 100)
}

df = pd.DataFrame(data)

# 数据标准化
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 创建绘图
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 原始数据分布
axes[0, 0].hist(df['Feature 1'], bins=20, color='red', alpha=0.7)
axes[0, 0].set_title('Original Feature 1 Distribution')
axes[0, 1].hist(df['Feature 2'], bins=20, color='blue', alpha=0.7)
axes[0, 1].set_title('Original Feature 2 Distribution')
axes[0, 2].hist(df['Feature 3'], bins=20, color='green', alpha=0.7)
axes[0, 2].set_title('Original Feature 3 Distribution')

# 标准化后数据分布
axes[1, 0].hist(df_normalized['Feature 1'], bins=20, color='red', alpha=0.7)
axes[1, 0].set_title('Normalized Feature 1 Distribution')
axes[1, 1].hist(df_normalized['Feature 2'], bins=20, color='blue', alpha=0.7)
axes[1, 1].set_title('Normalized Feature 2 Distribution')
axes[1, 2].hist(df_normalized['Feature 3'], bins=20, color='green', alpha=0.7)
axes[1, 2].set_title('Normalized Feature 3 Distribution')

# 调整布局
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 3. 数据编码 (Encoding)

# 3.2 独热编码 (One-Hot Encoding)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 生成虚拟数据集
np.random.seed(42)
data = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C'], 100),
    'Value': np.random.randint(1, 100, 100)
})

# 原始数据分布图
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
sns.countplot(data['Category'], palette="Set2")
plt.title('Original Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
sns.scatterplot(x='Category', y='Value', data=data, palette="Set2", hue='Category')
plt.title('Original Category vs. Value')
plt.xlabel('Category')
plt.ylabel('Value')

# One-Hot Encoding
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(data[['Category']])
onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=onehot_encoder.get_feature_names_out(['Category']))

# 将One-Hot编码后的特征加入原始数据集中
data_onehot = pd.concat([data.drop('Category', axis=1), onehot_encoded_df], axis=1)

plt.subplot(1, 3, 3)
sns.heatmap(data_onehot.corr(), annot=True, cmap="Spectral")
plt.title('Correlation Heatmap After One-Hot Encoding')

plt.tight_layout()
plt.show()

# Label Encoding
label_encoder = LabelEncoder()
data['Category_Label'] = label_encoder.fit_transform(data['Category'])

# 生成新的图表
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.countplot(data['Category_Label'], palette="Set1")
plt.title('Label Encoded Category Distribution')
plt.xlabel('Encoded Category')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.scatterplot(x='Category_Label', y='Value', data=data, hue='Category_Label', palette="Set1")
plt.title('Label Encoded Category vs. Value')
plt.xlabel('Encoded Category')
plt.ylabel('Value')

plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 4. 特征选择 (Feature Selection)

# 4.1 过滤法 (Filter Method)

# 4.2 包裹法 (Wrapper Method)

# 4.3 嵌入法 (Embedded Method)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif

# 设置随机种子
np.random.seed(42)

# 生成虚拟数据集
X, y = make_classification(n_samples=500, n_features=15, n_informative=10, n_redundant=2, n_classes=3, random_state=42)
feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y

# 计算相关性矩阵
correlation_matrix = df.corr()

# 选择与目标变量最相关的前5个特征
selector = SelectKBest(score_func=f_classif, k=5)
selector.fit(df[feature_names], y)
selected_features = selector.get_support(indices=True)
selected_feature_names = [feature_names[i] for i in selected_features]

# 使用随机森林计算特征重要性
model = RandomForestClassifier(random_state=42)
model.fit(df[selected_feature_names], y)
importances = model.feature_importances_

# 创建子图
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# 绘制相关性热图
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axs[0], cbar_kws={'shrink': .8})
axs[0].set_title('Feature Correlation Matrix')

# 绘制特征重要性图
sns.barplot(x=importances, y=selected_feature_names, palette='bright', ax=axs[1])
axs[1].set_title('Feature Importances (Random Forest)')
axs[1].set_xlabel('Importance Score')
axs[1].set_ylabel('Selected Features')

# 显示图形
plt.tight_layout()
plt.show()

#%%>>>>>>>>>>>>>>>>>>>>>>> 5. 特征缩放 (Feature Scaling)
# 5.1 标准化与归一化

# 5.2 对数变换 (Log Transformation)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 生成虚拟数据集
np.random.seed(42)
data = np.random.rand(100, 2) * 100  # 100个样本，2个特征，范围0-100之间

# 创建标准化（Standardization）和最小最大缩放（Min-Max Scaling）对象
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

# 进行特征缩放
data_standardized = scaler_standard.fit_transform(data)
data_minmax_scaled = scaler_minmax.fit_transform(data)

# 绘图
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 原始数据分布
axs[0, 0].scatter(data[:, 0], data[:, 1], color='blue', edgecolor='k', s=50, alpha=0.7)
axs[0, 0].set_title("Original Data", fontsize=14)
axs[0, 0].set_xlabel("Feature 1")
axs[0, 0].set_ylabel("Feature 2")

# 标准化后的数据分布
axs[0, 1].scatter(data_standardized[:, 0], data_standardized[:, 1], color='green', edgecolor='k', s=50, alpha=0.7)
axs[0, 1].set_title("Standardized Data", fontsize=14)
axs[0, 1].set_xlabel("Feature 1 (Standardized)")
axs[0, 1].set_ylabel("Feature 2 (Standardized)")

# 最小最大缩放后的数据分布
axs[1, 0].scatter(data_minmax_scaled[:, 0], data_minmax_scaled[:, 1], color='red', edgecolor='k', s=50, alpha=0.7)
axs[1, 0].set_title("Min-Max Scaled Data", fontsize=14)
axs[1, 0].set_xlabel("Feature 1 (Min-Max Scaled)")
axs[1, 0].set_ylabel("Feature 2 (Min-Max Scaled)")

# 原始数据与缩放后数据对比
axs[1, 1].scatter(data[:, 0], data[:, 1], color='blue', label='Original', edgecolor='k', s=50, alpha=0.5)
axs[1, 1].scatter(data_standardized[:, 0], data_standardized[:, 1], color='green', label='Standardized', edgecolor='k', s=50, alpha=0.5)
axs[1, 1].scatter(data_minmax_scaled[:, 0], data_minmax_scaled[:, 1], color='red', label='Min-Max Scaled', edgecolor='k', s=50, alpha=0.5)
axs[1, 1].set_title("Comparison of All Data", fontsize=14)
axs[1, 1].set_xlabel("Feature 1")
axs[1, 1].set_ylabel("Feature 2")
axs[1, 1].legend(loc='upper right')

# 调整图像布局
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 6. 降维 (Dimensionality Reduction)
# 6.1 主成分分析 (PCA)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 生成虚拟数据集
n_samples = 1000
n_features = 20
n_classes = 4
X, y = make_classification(n_samples=n_samples, n_features=n_features,
                           n_classes=n_classes, random_state=42, n_clusters_per_class=1)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 绘图
plt.figure(figsize=(12, 6))

# PCA图
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='tab10', s=60, edgecolor='k')
plt.title('PCA Dimensionality Reduction')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Classes')

# t-SNE图
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='tab10', s=60, edgecolor='k')
plt.title('t-SNE Dimensionality Reduction')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Classes')

plt.tight_layout()
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>> 7. 数据拆分 (Data Splitting)
# 7.1 训练集/验证集/测试集划分

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 生成虚拟数据集
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 数据拆分：80% 训练集，20% 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 拟合线性回归模型
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# 在训练集和测试集上做预测
y_train_pred = lin_reg.predict(X_train)
y_test_pred = lin_reg.predict(X_test)

# 设置图形大小和颜色
plt.figure(figsize=(10, 6))

# 图1：训练集与测试集数据的散点图
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Training Data', s=50)
plt.scatter(X_test, y_test, color='red', label='Testing Data', s=50)
plt.plot(X_train, y_train_pred, color='green', label='Regression Line', linewidth=2)
plt.title('Training and Testing Data with Regression Line', fontsize=12)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# 图2：测试集的预测结果与实际值对比
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='red', label='Actual Testing Data', s=50)
plt.scatter(X_test, y_test_pred, color='orange', label='Predicted Testing Data', s=50, marker='x')
plt.plot(X_train, y_train_pred, color='green', label='Regression Line', linewidth=2)
plt.title('Testing Data: Actual vs Predicted', fontsize=12)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 8. 数据增强 (Data Augmentation)
# 8.1 图像数据增强 (Image Data Augmentation)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 设置随机种子
np.random.seed(42)

# 生成虚拟数据集：2D 正态分布
mean = [5, 5]
cov = [[1, 0.5], [0.5, 1]]  # 协方差矩阵
data = np.random.multivariate_normal(mean, cov, 500)

# 数据增强：1. 图像翻转（左右翻转）
def flip_horizontal(data):
    flipped_data = data.copy()
    flipped_data[:, 0] = -flipped_data[:, 0]  # 反转x轴
    return flipped_data

# 数据增强：2. 图像旋转（旋转角度为45度）
def rotate(data, angle_degrees):
    angle_radians = np.deg2rad(angle_degrees)
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                [np.sin(angle_radians), np.cos(angle_radians)]])
    rotated_data = np.dot(data, rotation_matrix)
    return rotated_data

# 进行数据增强
flipped_data = flip_horizontal(data)
rotated_data = rotate(data, 45)

# 数据缩放到[0, 1]范围以便可视化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
flipped_data_scaled = scaler.transform(flip_horizontal(data))
rotated_data_scaled = scaler.transform(rotate(data, 45))

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# 原始数据分布
sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], ax=axs[0, 0], s=50, color='blue', label='Original Data')
sns.kdeplot(x=data_scaled[:, 0], y=data_scaled[:, 1], ax=axs[0, 0], cmap='Blues', fill=True, alpha=0.4)
axs[0, 0].set_title('Original Data Distribution', fontsize=14)
axs[0, 0].legend()

# 翻转后的数据分布
sns.scatterplot(x=flipped_data_scaled[:, 0], y=flipped_data_scaled[:, 1], ax=axs[0, 1], s=50, color='green', label='Flipped Data')
sns.kdeplot(x=flipped_data_scaled[:, 0], y=flipped_data_scaled[:, 1], ax=axs[0, 1], cmap='Greens', fill=True, alpha=0.4)
axs[0, 1].set_title('Horizontally Flipped Data', fontsize=14)
axs[0, 1].legend()

# 旋转后的数据分布
sns.scatterplot(x=rotated_data_scaled[:, 0], y=rotated_data_scaled[:, 1], ax=axs[1, 0], s=50, color='red', label='Rotated Data (45°)')
sns.kdeplot(x=rotated_data_scaled[:, 0], y=rotated_data_scaled[:, 1], ax=axs[1, 0], cmap='Reds', fill=True, alpha=0.4)
axs[1, 0].set_title('Rotated Data (45°)', fontsize=14)
axs[1, 0].legend()

# 三种数据的对比
sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], ax=axs[1, 1], s=50, color='blue', label='Original Data', alpha=0.6)
sns.scatterplot(x=flipped_data_scaled[:, 0], y=flipped_data_scaled[:, 1], ax=axs[1, 1], s=50, color='green', label='Flipped Data', alpha=0.6)
sns.scatterplot(x=rotated_data_scaled[:, 0], y=rotated_data_scaled[:, 1], ax=axs[1, 1], s=50, color='red', label='Rotated Data', alpha=0.6)
axs[1, 1].set_title('Comparison of Augmented Data', fontsize=14)
axs[1, 1].legend()

# 设置子图间距
plt.tight_layout()
plt.show()





#%%>>>>>>>>>>>>>>>>>>>>>>> 9. 数据平衡 (Data Balancing)
# 9.1 欠采样 (Under-sampling)
# 原理
# 通过减少多数类样本的数量，使得类别分布更加平衡。

# 方法
# 随机删除多数类样本，或选择性保留代表性样本。

# 9.2 过采样 (Over-sampling)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# 设置随机种子，保证结果可重复
np.random.seed(42)

# 生成虚拟数据集 (不平衡)
# 类别 0 占 90%，类别 1 占 10%
X = np.random.randn(1000, 2)  # 两个特征
y = np.hstack([np.zeros(900), np.ones(100)])

# 打印原始数据类别分布
print("原始数据分布:", Counter(y))

# 欠采样
undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_under, y_under = undersampler.fit_resample(X, y)

# 打印欠采样后类别分布
print("欠采样后数据分布:", Counter(y_under))

# 过采样
oversampler = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_over, y_over = oversampler.fit_resample(X, y)

# 打印过采样后类别分布
print("过采样后数据分布:", Counter(y_over))

# 画图 - 原始数据、欠采样后数据、过采样后数据的对比
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

# 原始数据分布
axes[0].scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0', alpha=0.6)
axes[0].scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1', alpha=0.6)
axes[0].set_title("Original Data")
axes[0].legend()
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# 欠采样后的数据分布
axes[1].scatter(X_under[y_under == 0][:, 0], X_under[y_under == 0][:, 1], color='red', label='Class 0', alpha=0.6)
axes[1].scatter(X_under[y_under == 1][:, 0], X_under[y_under == 1][:, 1], color='blue', label='Class 1', alpha=0.6)
axes[1].set_title("After Undersampling")
axes[1].legend()
axes[1].set_xlabel('Feature 1')

# 过采样后的数据分布
axes[2].scatter(X_over[y_over == 0][:, 0], X_over[y_over == 0][:, 1], color='red', label='Class 0', alpha=0.6)
axes[2].scatter(X_over[y_over == 1][:, 0], X_over[y_over == 1][:, 1], color='blue', label='Class 1', alpha=0.6)
axes[2].set_title("After Oversampling")
axes[2].legend()
axes[2].set_xlabel('Feature 1')

# 设置图形的标题和显示
plt.suptitle("Comparison of Class Distributions Before and After Sampling", fontsize=16)
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 10. 数据转换 (Data Transformation)
# 10.1 特征构造 (Feature Engineering)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 生成虚拟数据集
np.random.seed(42)
data = pd.DataFrame({
    'Feature1': np.random.normal(loc=0, scale=1, size=100),  # 正态分布数据
    'Feature2': np.random.rand(100) * 100  # 随机生成 [0, 100] 区间数据
})

# 创建标准化和归一化的转换器
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

# 对数据进行标准化
data_standardized = scaler_standard.fit_transform(data)
data_standardized = pd.DataFrame(data_standardized, columns=data.columns)

# 对数据进行归一化
data_normalized = scaler_minmax.fit_transform(data)
data_normalized = pd.DataFrame(data_normalized, columns=data.columns)

# 绘图设置
plt.figure(figsize=(12, 8))

# 原始数据分布
plt.subplot(3, 2, 1)
plt.hist(data['Feature1'], bins=20, color='blue', alpha=0.7, label='Feature1')
plt.title('Original Feature1 Distribution', fontsize=12)
plt.legend()

plt.subplot(3, 2, 2)
plt.hist(data['Feature2'], bins=20, color='green', alpha=0.7, label='Feature2')
plt.title('Original Feature2 Distribution', fontsize=12)
plt.legend()

# 标准化后数据分布
plt.subplot(3, 2, 3)
plt.hist(data_standardized['Feature1'], bins=20, color='red', alpha=0.7, label='Standardized Feature1')
plt.title('Standardized Feature1 Distribution', fontsize=12)
plt.legend()

plt.subplot(3, 2, 4)
plt.hist(data_standardized['Feature2'], bins=20, color='orange', alpha=0.7, label='Standardized Feature2')
plt.title('Standardized Feature2 Distribution', fontsize=12)
plt.legend()

# 归一化后数据分布
plt.subplot(3, 2, 5)
plt.hist(data_normalized['Feature1'], bins=20, color='purple', alpha=0.7, label='Normalized Feature1')
plt.title('Normalized Feature1 Distribution', fontsize=12)
plt.legend()

plt.subplot(3, 2, 6)
plt.hist(data_normalized['Feature2'], bins=20, color='pink', alpha=0.7, label='Normalized Feature2')
plt.title('Normalized Feature2 Distribution', fontsize=12)
plt.legend()

# 调整布局
plt.tight_layout()
plt.show()










#%%>>>>>>>>>>>>>>>>>>>>>>>







#%%>>>>>>>>>>>>>>>>>>>>>>>







