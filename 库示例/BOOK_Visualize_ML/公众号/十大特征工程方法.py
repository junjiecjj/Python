#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:24:21 2024

@author: jack


https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247486913&idx=1&sn=2ff73fa70edc290088c486b963b6c948&chksm=c1e9d19570cc44aa72da07d0ddf6d8a8e0a1b844c032be54859238045b084cc3aa613835e336&mpshare=1&scene=1&srcid=0923w25sw4kV4sxU7UcifODC&sharer_shareinfo=1798d1a3fb1f1e82f71c41574be6e0f9&sharer_shareinfo_first=f79150cc7784ccc59070916425bb660b&exportkey=n_ChQIAhIQfLcm6V7ddX7%2FmEdNrS2vlhKfAgIE97dBBAEAAAAAAAAbMO2LW8YAAAAOpnltbLcz9gKNyK89dVj0MMLgW5eGyW%2BK%2FOaMVrycfdC3WK8wvHCXL1fAa7B326xC0dzpyX89FhMjAKJtx80wDpqgcztV%2BTycLfzqVeUWDYHzNYjt3LTdCCPoVd4QGUEmTiUfdzCTtEssMtlRcNEK%2FSgjK8MPiXe%2F3tV%2F38neYrEdG9rUT7WwgZHHnxweNkBOAR97DyVPtCCyYyGWfICqCI5eyJymXLa1RGHkqu3iRr%2FlxfK1VIMXHvc%2FnTwv6y3bk8XLLevqQp6RT2fNJVuL8rUxPQR7z5VG2Fb3T0oy7OuYgBq7F%2BHNeit%2BqvAEjv9%2Flcf3RByL5F6MNmPaBYlwxVzJS6GvFTM6&acctmode=0&pass_ticket=9CBge3pOR%2Fyo3lBbruS1AFLXazZW9bRKUmm96dSnmEoxS4M534%2BvrJUFi%2B%2BzFgk6&wx_header=0#rd


"""

#%%>>>>>>>>>>>>>>>>>>>>>>> 1. 特征缩放
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 生成虚拟数据集
np.random.seed(42)
X1 = np.random.normal(100, 50, 100)  # 生成均值为100，标准差为50的随机数据
X2 = np.random.normal(200, 10, 100)  # 生成均值为200，标准差为10的随机数据

# 原始数据集
X_original = np.column_stack((X1, X2))

# 使用Min-Max缩放
min_max_scaler = MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X_original)

# 使用标准化（Standardization）
standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X_original)

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(12, 10), )

# 原始数据分布图
axes[0, 0].scatter(X_original[:, 0], X_original[:, 1], color='blue', label='Original Data')
axes[0, 0].set_title('Original Data')
axes[0, 0].set_xlabel('Feature 1')
axes[0, 0].set_ylabel('Feature 2')

# Min-Max缩放后数据分布图
axes[0, 1].scatter(X_minmax[:, 0], X_minmax[:, 1], color='red', label='Min-Max Scaled Data')
axes[0, 1].set_title('Min-Max Scaled Data')
axes[0, 1].set_xlabel('Feature 1')
axes[0, 1].set_ylabel('Feature 2')

# 标准化后的数据分布图
axes[1, 0].scatter(X_standard[:, 0], X_standard[:, 1], color='green', label='Standard Scaled Data')
axes[1, 0].set_title('Standard Scaled Data')
axes[1, 0].set_xlabel('Feature 1')
axes[1, 0].set_ylabel('Feature 2')

# 显示各个特征的分布（直方图）
axes[1, 1].hist(X_original[:, 0], color='blue', alpha=0.5, label='Original Feature 1')
axes[1, 1].hist(X_original[:, 1], color='orange', alpha=0.5, label='Original Feature 2')
axes[1, 1].set_title('Original Feature Distribution')
axes[1, 1].set_xlabel('Feature Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend(loc='upper right')

# 调整布局
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 2. 缺失值处理

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子
np.random.seed(42)

# 生成虚拟数据集
data = pd.DataFrame({
    'A': np.random.rand(100),
    'B': np.random.rand(100) * 10,
    'C': np.random.normal(0, 1, 100),
})

# 在数据集中随机引入缺失值
data.loc[np.random.choice(data.index, size=20, replace=False), 'A'] = np.nan
data.loc[np.random.choice(data.index, size=15, replace=False), 'B'] = np.nan
data.loc[np.random.choice(data.index, size=10, replace=False), 'C'] = np.nan

# 可视化函数
def plot_missing_values_analysis(data, title, ax):
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis', ax=ax[0], linewidths=0.5)
    ax[0].set_title(f'Missing Values Heatmap - {title}')

    sns.histplot(data['A'], kde=True, color="orange", ax=ax[1])
    ax[1].set_title(f'Distribution of Column A - {title}')

    sns.scatterplot(x=data.index, y=data['B'], color='blue', ax=ax[2])
    ax[2].set_title(f'Scatterplot of Column B - {title}')

    sns.boxplot(data['C'], color="green", ax=ax[3])
    ax[3].set_title(f'Boxplot of Column C - {title}')

# 处理方法1：插值法处理缺失值
data_interpolated = data.copy()
data_interpolated['A'] = data_interpolated['A'].interpolate()
data_interpolated['B'] = data_interpolated['B'].interpolate()
data_interpolated['C'] = data_interpolated['C'].interpolate()

# 处理方法2：删除缺失值
data_dropped = data.dropna()

# 设置图形大小和布局
fig, axes = plt.subplots(3, 4, figsize=(12, 10))

# 原始数据集的可视化
plot_missing_values_analysis(data, "Original", axes[0])

# 插值法处理后的数据集可视化
plot_missing_values_analysis(data_interpolated, "Interpolated", axes[1])

# 删除缺失值后的数据集可视化
plot_missing_values_analysis(data_dropped, "Dropped Rows", axes[2])

# 调整布局
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 3. 独热编码
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# 设置绘图风格
sns.set(style="whitegrid")

# 1. 创建虚拟数据集
np.random.seed(42)  # 为了可重复性
data = {
    'Category': np.random.choice(['A', 'B', 'C', 'D'], size=100),
    'Value': np.random.randint(1, 100, size=100)
}

df = pd.DataFrame(data)

# 2. 原始数据集的分布可视化
fig, axs = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'hspace': 0.4, 'wspace': 0.3})
plt.suptitle('Analysis of Original Categorical Features and One-Hot Encoded Data', fontsize=18)

# 图1：类别分布条形图
sns.countplot(x='Category', data=df, ax=axs[0, 0], palette="Set2")
axs[0, 0].set_title('Plot 1: Distribution of Original Categorical Features', fontsize=14)
axs[0, 0].set_xlabel('Category', fontsize=12)
axs[0, 0].set_ylabel('Count', fontsize=12)

# 图2：类别与值的关系散点图
sns.scatterplot(x='Category', y='Value', data=df, ax=axs[0, 1], palette="Set1", s=100)
axs[0, 1].set_title('Plot 2: Relationship Between Category and Value', fontsize=14)
axs[0, 1].set_xlabel('Category', fontsize=12)
axs[0, 1].set_ylabel('Value', fontsize=12)

# 3. 对分类特征进行独热编码
encoder = OneHotEncoder( )
one_hot_encoded = encoder.fit_transform(df[['Category']])

# 创建独热编码后的DataFrame
encoded_columns = encoder.get_feature_names_out(['Category'])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoded_columns)

# 将独热编码后的数据与原始数值列合并
encoded_df = pd.concat([df[['Value']], one_hot_df], axis=1)

# 4. 独热编码后数据的可视化
# 图3：热力图显示独热编码后的数据
sns.heatmap(encoded_df.corr(), annot=True, cmap="YlGnBu", ax=axs[1, 0])
axs[1, 0].set_title('Plot 3: Correlation Heatmap of One-Hot Encoded Features', fontsize=14)

# 图4：独热编码后的类别和数值之间的关系（箱线图）
sns.boxplot(data=encoded_df, ax=axs[1, 1], palette="Set3")
axs[1, 1].set_title('Plot 4: Relationship Between Encoded Features and Value', fontsize=14)
axs[1, 1].set_xlabel('Encoded Features', fontsize=12)
axs[1, 1].set_ylabel('Value', fontsize=12)

# 展示图形
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 4. 特征交互

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# 设置随机种子
np.random.seed(42)

# 生成虚拟数据
n_samples = 500
X1 = np.random.uniform(0, 10, n_samples)
X2 = np.random.uniform(0, 10, n_samples)

# 定义目标变量Y，基于X1和X2的复杂交互
Y = np.sin(X1) * np.log(X2 + 1) + np.random.normal(0, 0.2, n_samples)

# 创建数据集
data = pd.DataFrame({'X1': X1, 'X2': X2, 'Y': Y})

# 设置画布
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 图1：X1与X2的交互对Y的影响（散点图）
scatter = axes[0].scatter(data['X1'], data['X2'], c=data['Y'], cmap='plasma', s=50, alpha=0.8)
axes[0].set_xlabel('X1')
axes[0].set_ylabel('X2')
axes[0].set_title('Feature Interaction between X1 and X2')
fig.colorbar(scatter, ax=axes[0], label='Y')

# 图2：3D图形，展示Y随X1和X2变化的趋势
ax = fig.add_subplot(122, projection='3d')
ax.plot_trisurf(data['X1'], data['X2'], data['Y'], cmap='viridis', edgecolor='none', alpha=0.8)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('3D View of Feature Interaction')

# 显示图形
plt.tight_layout()
plt.show()






#%%>>>>>>>>>>>>>>>>>>>>>>> 5. 特征选择
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成虚拟数据集
np.random.seed(42)
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=2, n_classes=2, random_state=42)

# 创建DataFrame
columns = [f'Feature_{i+1}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=columns)
df['Target'] = y

# 1. 特征选择方法一：方差过滤（Variance Threshold）
# 设置方差过滤器，剔除方差小于阈值的特征
selector = VarianceThreshold(threshold=0.1)
X_variance_filtered = selector.fit_transform(X)

# 获取被选中的特征索引
selected_features_var = np.array(columns)[selector.get_support()]
df_filtered_var = pd.DataFrame(X_variance_filtered, columns=selected_features_var)

# 2. 特征选择方法二：相关性矩阵（Correlation Matrix）
# 计算相关性矩阵
correlation_matrix = df.corr()

# 可视化
plt.figure(figsize=(18, 8))

# 图1：原始特征的相关性热力图
plt.subplot(1, 2, 1)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)
plt.title('Original Features Correlation Matrix')

# 基于相关性矩阵筛选出与目标变量高度相关的特征（绝对值大于0.1的特征）
high_corr_features = correlation_matrix['Target'][abs(correlation_matrix['Target']) > 0.1].index.tolist()
df_filtered_corr = df[high_corr_features].drop(columns=['Target'])

# 图2：方差过滤后的特征分布
plt.subplot(1, 2, 2)
df_filtered_var.plot(kind='box', ax=plt.gca(), title="Variance Filtered Features Distribution", patch_artist=True, boxprops=dict(facecolor='lightblue'))

plt.tight_layout()
plt.show()

# 随机森林的特征重要性
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# 特征重要性排序
plt.figure(figsize=(12, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], color='lightgreen', align='center')
plt.xticks(range(X.shape[1]), np.array(columns)[indices], rotation=90)
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 6. 特征构造

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime, timedelta

# 生成虚拟交易数据集
np.random.seed(42)

# 生成随机的交易ID
transaction_id = np.arange(1, 1001)

# 生成随机的客户ID
customer_id = np.random.randint(100, 200, size=1000)

# 生成随机的交易日期（过去一年的随机日期）
start_date = datetime(2023, 1, 1)
transaction_date = [start_date + timedelta(days=random.randint(0, 364)) for _ in range(1000)]

# 生成随机的交易金额
amount = np.random.uniform(5, 500, size=1000)

# 构造DataFrame
data = pd.DataFrame({
    'transaction_id': transaction_id,
    'customer_id': customer_id,
    'transaction_date': transaction_date,
    'amount': amount
})

# 添加特征：交易日期的星期几
data['transaction_weekday'] = data['transaction_date'].dt.weekday  # 0=Monday, 6=Sunday

# 添加特征：基于交易金额分类
def categorize_amount(amount):
    if amount < 100:
        return 'Low'
    elif amount < 300:
        return 'Medium'
    else:
        return 'High'

data['amount_category'] = data['amount'].apply(categorize_amount)

# 绘图

# 设置画布和子图
fig, ax = plt.subplots(2, 1, figsize=(14, 12))

# 图1：每个星期几的交易总金额和交易数量
weekday_amount = data.groupby('transaction_weekday')['amount'].sum().reset_index()
weekday_count = data.groupby('transaction_weekday')['transaction_id'].count().reset_index()

# 合并数据
weekday_data = pd.merge(weekday_amount, weekday_count, on='transaction_weekday')
weekday_data.columns = ['transaction_weekday', 'total_amount', 'transaction_count']

# 绘制交易总金额（条形图）
sns.barplot(x='transaction_weekday', y='total_amount', data=weekday_data, palette='Set2', ax=ax[0])
ax[0].set_title('Total Transaction Amount by Weekday', fontsize=16)
ax[0].set_xlabel('Weekday (0=Monday, 6=Sunday)', fontsize=14)
ax[0].set_ylabel('Total Transaction Amount', fontsize=14)

# 共享x轴绘制交易数量（折线图）
ax2 = ax[0].twinx()
sns.lineplot(x='transaction_weekday', y='transaction_count', data=weekday_data, color='r', marker='o', ax=ax2)
ax2.set_ylabel('Transaction Count', fontsize=14, color='r')
ax2.tick_params(axis='y', labelcolor='r')

# 图2：不同金额分类的交易数量随时间变化
category_count_by_date = data.groupby(['transaction_date', 'amount_category'])['transaction_id'].count().reset_index()

# 绘制金额分类随时间变化的折线图
sns.lineplot(x='transaction_date', y='transaction_id', hue='amount_category', data=category_count_by_date, ax=ax[1], palette='bright', marker='o')
ax[1].set_title('Transaction Count by Amount Category Over Time', fontsize=16)
ax[1].set_xlabel('Transaction Date', fontsize=14)
ax[1].set_ylabel('Transaction Count', fontsize=14)

# 调整布局
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 7. 主成分分析
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# 生成一个虚拟的三维数据集
np.random.seed(42)
mean = [0, 0, 0]
cov = [[1, 0.8, 0.7], [0.8, 1, 0.75], [0.7, 0.75, 1]]  # 协方差矩阵
data = np.random.multivariate_normal(mean, cov, 500)

# 应用PCA，将数据从3D降维到2D
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

# 获取PCA的特征向量
components = pca.components_

# 创建图形，分为1行3列
fig = plt.figure(figsize=(18, 6))

# 第一图：原始数据的三维散点图
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 0], cmap='rainbow', s=40)
ax1.set_title("Original 3D Data", fontsize=14)
ax1.set_xlabel('X Axis')
ax1.set_ylabel('Y Axis')
ax1.set_zlabel('Z Axis')

# 第二图：PCA降维到二维后的数据散点图
ax2 = fig.add_subplot(132)
scatter = ax2.scatter(data_2d[:, 0], data_2d[:, 1], c=data[:, 0], cmap='rainbow', s=40)
ax2.set_title("PCA Reduced to 2D Data", fontsize=14)
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')

# 第三图：PCA主成分在三维空间中的投影
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 0], cmap='rainbow', s=40, alpha=0.6)

# 绘制特征向量（主成分）在三维空间的投影
origin = np.mean(data, axis=0)  # 原点为数据的均值
for i in range(components.shape[0]):
    vec = components[i] * 3  # 将主成分向量放大3倍便于可视化
    ax3.quiver(*origin, vec[0], vec[1], vec[2], color='black', linewidth=2)

ax3.set_title("Projection of Principal Components in 3D Space", fontsize=14)
ax3.set_xlabel('X Axis')
ax3.set_ylabel('Y Axis')
ax3.set_zlabel('Z Axis')

# 显示颜色条
fig.colorbar(scatter, ax=ax2, shrink=0.6, aspect=10)

# 调整布局并显示图像
plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 8. 目标编码

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子
np.random.seed(42)

# 创建虚拟数据集
n = 1000
categories = ['A', 'B', 'C', 'D', 'E', 'F']
data = pd.DataFrame({
    'Product Category': np.random.choice(categories, n),
    'Sales': np.random.normal(loc=100, scale=20, size=n) + np.random.choice([10, 20, 30, 40, 50, 60], n)
})

# 可视化原始数据分布
plt.figure(figsize=(14, 10))

# 1. 分类变量的原始分布（Product Category）
plt.subplot(2, 2, 1)
sns.countplot(x='Product Category', data=data, palette='Set1')
plt.title('Original Product Category Distribution')
plt.xlabel('Product Category')
plt.ylabel('Count')

# 2. 不同类别的Sales分布
plt.subplot(2, 2, 2)
sns.boxplot(x='Product Category', y='Sales', data=data, palette='Set2')
plt.title('Sales Distribution by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Sales')

# 目标编码：基于类别的Sales均值
category_means = data.groupby('Product Category')['Sales'].mean()
data['Category_Target_Encoded'] = data['Product Category'].map(category_means)

# 3. 类别目标编码值分布
plt.subplot(2, 2, 3)
sns.barplot(x=category_means.index, y=category_means.values, palette='Set3')
plt.title('Target Encoded Values for Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Mean Sales (Target Encoded)')

# 4. 编码后数据分布与目标变量的关系
plt.subplot(2, 2, 4)
sns.scatterplot(x='Category_Target_Encoded', y='Sales', data=data, hue='Product Category', palette='Set1', s=100)
plt.title('Sales vs Target Encoded Product Category')
plt.xlabel('Target Encoded Product Category')
plt.ylabel('Sales')

# 调整布局，展示所有图
plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>> 9. 文本处理
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import random

# 确保下载停用词（首次运行时需要）
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 生成虚拟文本数据
np.random.seed(42)
random.seed(42)

# 创建虚拟评论数据集
data = {
    'Review': [
        "This product is excellent, I love it!",
        "Absolutely terrible, would not recommend.",
        "It's okay, not the best but decent quality.",
        "Loved it! Fantastic service and very satisfied.",
        "Not bad, but I expected something better.",
        "Terrible experience, very disappointed.",
        "Highly recommend, great product and service!",
        "Poor quality, waste of money.",
        "Satisfied with the product, will buy again.",
        "Would never buy again, very bad experience."
    ],
    'Sentiment': [1, -1, 0, 1, 0, -1, 1, -1, 1, -1]  # 1 = positive, 0 = neutral, -1 = negative
}

df = pd.DataFrame(data)

# 文本预处理
def preprocess_text(text):
    # 转换为小写
    text = text.lower()
    # 移除停用词和非字母字符
    tokens = [word for word in text.split() if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

df['Processed_Review'] = df['Review'].apply(preprocess_text)

# 统计词频
all_words = " ".join(df['Processed_Review']).split()
word_counts = Counter(all_words)
common_words = word_counts.most_common(10)

# 词云生成
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Set2').generate(" ".join(all_words))

# TF-IDF 特征提取
tfidf = TfidfVectorizer(max_features=10)
tfidf_matrix = tfidf.fit_transform(df['Processed_Review']).toarray()
tfidf_df = pd.DataFrame(tfidf_matrix, columns=tfidf.get_feature_names_out())

# 可视化
fig, axs = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [1, 1]})

# 词频柱状图
sns.barplot(x=[word for word, _ in common_words], y=[count for _, count in common_words], ax=axs[0, 0], palette="Set1")
axs[0, 0].set_title('Top 10 Word Frequency')
axs[0, 0].set_xlabel('Words')
axs[0, 0].set_ylabel('Frequency')

# 词云图
axs[0, 1].imshow(wordcloud, interpolation='bilinear')
axs[0, 1].axis('off')
axs[0, 1].set_title('Word Cloud')

# TF-IDF 特征热力图
sns.heatmap(tfidf_df, annot=True, cmap="YlGnBu", ax=axs[1, 0], cbar=True, linewidths=0.5)
axs[1, 0].set_title('TF-IDF Feature Matrix')
axs[1, 0].set_xlabel('Top Words')
axs[1, 0].set_ylabel('Reviews')

# 情感分布图
sns.histplot(df['Sentiment'], bins=3, kde=False, ax=axs[1, 1], palette="coolwarm", shrink=0.8)
axs[1, 1].set_title('Sentiment Distribution')
axs[1, 1].set_xlabel('Sentiment')
axs[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()



#%%>>>>>>>>>>>>>>>>>>>>>>> 10. 异常值处理

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 生成虚拟数据
# 正常分布数据
data = np.random.normal(loc=50, scale=10, size=1000)
# 添加异常值
outliers = np.array([120, 130, 140, 150, 160])
data_with_outliers = np.concatenate([data, outliers])

# 创建DataFrame
df = pd.DataFrame(data_with_outliers, columns=['Value'])

# 绘制原始数据的直方图和箱线图
plt.figure(figsize=(14, 6))

# 直方图
plt.subplot(1, 2, 1)
sns.histplot(df['Value'], bins=30, color='skyblue', kde=True)
plt.title('Histogram of Data with Outliers')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 箱线图
plt.subplot(1, 2, 2)
sns.boxplot(x=df['Value'], color='salmon')
plt.title('Boxplot of Data with Outliers')
plt.xlabel('Value')

plt.tight_layout()
plt.show()

# 使用Z-score检测异常值
z_scores = np.abs(stats.zscore(df['Value']))
df['Outlier'] = z_scores > 3  # Z-score threshold

# 移除异常值
df_cleaned = df[~df['Outlier']]

# 绘制去除异常值后的直方图和箱线图
plt.figure(figsize=(14, 6))

# 直方图
plt.subplot(1, 2, 1)
sns.histplot(df_cleaned['Value'], bins=30, color='lightgreen', kde=True)
plt.title('Histogram of Cleaned Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 箱线图
plt.subplot(1, 2, 2)
sns.boxplot(x=df_cleaned['Value'], color='orange')
plt.title('Boxplot of Cleaned Data')
plt.xlabel('Value')

plt.tight_layout()
plt.show()


#%%>>>>>>>>>>>>>>>>>>>>>>>








