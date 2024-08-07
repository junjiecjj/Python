
"""

# 最强总结，十大数据预处理方法
https://mp.weixin.qq.com/s?__biz=MzkwNjY5NDU2OQ==&mid=2247484938&idx=1&sn=0d95b88129d7fb3b70060bcf21eb0e87&chksm=c0e5dcccf79255da8d8c74653331af6cbdf52b844e0652788ea95bdee1bd488cee99e721f941&mpshare=1&scene=1&srcid=0726Kkga00BXHGjAT2wRC1nX&sharer_shareinfo=aa6fe33440bbd6d1a964fcccf4ab42ec&sharer_shareinfo_first=aa6fe33440bbd6d1a964fcccf4ab42ec&exportkey=n_ChQIAhIQ8lkrIjM87PsK0nUmvqSCdRKfAgIE97dBBAEAAAAAAPM9ISxQqmcAAAAOpnltbLcz9gKNyK89dVj0uMs4fj15jI9dYCF753JNHZJGcIFn7pBmtL3f52Wy4bB6d3JNl1v%2Bnhh82uQQYJXR3i9JDXjzWo6YJe%2FT%2BNvapS3piVyqIgYYI56ySJX9CbbCKVPh9eTy6VmuvBApqXSO4LFDRoNfoRhNjvYCbckH%2FlYY%2B%2BVaxNsSCCNuUlZRaANNoGSYbjAx3tf2Xrz1hsezZittfvKH1fCiGiz2n%2FybeBrU0Tzoykq9uypDZimsFphRloOUuXKwpMU%2FUhRMxX%2BVUIrRY%2BlChaU4Q2Mn36%2FpXf7Zx7vTqext6YKa8qIjaIbawiN9MwOvzNbJeO14QG7vQ5LL%2BudI7q4t&acctmode=0&pass_ticket=2pvTzPVn6XdCSFN%2BwBXLPvrAKogtSfIyPhQAeU3Lx2Sn9a8cLfpK9UxlOK7Widjj&wx_header=0#rd

"""

#%% 1. 数据清洗（Data Cleaning）

import pandas as pd
import numpy as np

# 创建虚构数据集
np.random.seed(0)
data = {
    'Product_ID': range(1, 101),
    'Price': np.random.normal(loc=50, scale=10, size=100),
    'Sales': np.random.poisson(lam=20, size=100).astype(float),
    'Rating': np.random.normal(loc=3.5, scale=1, size=100)
}

# 插入一些缺失值
data['Price'][np.random.choice(100, 10, replace=False)] = np.nan
data['Sales'][np.random.choice(100, 5, replace=False)] = np.nan
data['Rating'][np.random.choice(100, 8, replace=False)] = np.nan

df = pd.DataFrame(data)

# 使用均值填充缺失值。
# 填充缺失值
df['Price'].fillna(df['Price'].mean(), inplace=True)
df['Sales'].fillna(df['Sales'].mean(), inplace=True)
df['Rating'].fillna(df['Rating'].mean(), inplace=True)

# 处理异常值

## 我们使用IQR（四分位距）方法来处理异常值。
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = remove_outliers(df, 'Price')
df = remove_outliers(df, 'Sales')
df = remove_outliers(df, 'Rating')

# 数据分析和可视化
import seaborn as sns
import matplotlib.pyplot as plt

# 绘制价格分布图
plt.figure(figsize=(12, 6))
sns.histplot(df['Price'], kde=True, bins=20)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# 绘制销量与评分的关系图
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Sales', y='Rating', data=df)
plt.title('Sales vs Rating')
plt.xlabel('Sales')
plt.ylabel('Rating')
plt.show()

# 绘制价格和销量的箱线图
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Price', 'Sales']])
plt.title('Boxplot of Price and Sales')
plt.show()



#%% 2. 数据标准化（Standardization）

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. 创建一个随机数据集
np.random.seed(0)
data = np.random.rand(100, 2) * 1000

# 创建一个DataFrame
df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])

# 2. 对数据进行标准化
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# 创建标准化后的DataFrame
df_standardized = pd.DataFrame(data_standardized, columns=['Feature1', 'Feature2'])

# 3. 绘制标准化前后的数据分布直方图
plt.figure(figsize=(12, 6))

# 标准化前
plt.subplot(1, 2, 1)
plt.hist(df['Feature1'], bins=20, alpha=0.7, label='Feature1', color='blue')
plt.hist(df['Feature2'], bins=20, alpha=0.7, label='Feature2', color='green')
plt.title('Before Standardization')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

# 标准化后
plt.subplot(1, 2, 2)
plt.hist(df_standardized['Feature1'], bins=20, alpha=0.7, label='Feature1', color='blue')
plt.hist(df_standardized['Feature2'], bins=20, alpha=0.7, label='Feature2', color='green')
plt.title('After Standardization')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

# 4. 绘制标准化前后的数据散点图
plt.figure(figsize=(12, 6))

# 标准化前
plt.subplot(1, 2, 1)
plt.scatter(df['Feature1'], df['Feature2'], color='blue', alpha=0.7)
plt.title('Before Standardization')
plt.xlabel('Feature1')
plt.ylabel('Feature2')

# 标准化后
plt.subplot(1, 2, 2)
plt.scatter(df_standardized['Feature1'], df_standardized['Feature2'], color='red', alpha=0.7)
plt.title('After Standardization')
plt.xlabel('Feature1')
plt.ylabel('Feature2')

plt.tight_layout()
plt.show()



#%% 3. 数据归一化（Normalization）

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data  # 特征数据

# 创建MinMaxScaler对象
scaler = MinMaxScaler()

# 对数据集进行归一化
X_normalized = scaler.fit_transform(X)

# 可视化归一化前后的数据分布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 归一化前的数据分布
ax1.scatter(X[:, 0], X[:, 1], c=iris.target)
ax1.set_title('Before Normalization')
ax1.set_xlabel('Sepal Length (cm)')
ax1.set_ylabel('Sepal Width (cm)')

# 归一化后的数据分布
ax2.scatter(X_normalized[:, 0], X_normalized[:, 1], c=iris.target)
ax2.set_title('After Normalization')
ax2.set_xlabel('Sepal Length (normalized)')
ax2.set_ylabel('Sepal Width (normalized)')

plt.tight_layout()
plt.show()



#%% 4. 类别编码（Categorical Encoding）
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# 生成示例数据
data = {
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'New York', 'Los Angeles', 'Chicago'],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male'],
    'Age': [25, 30, 22, 35, 40, 29, 23, 37],
    'Income': [70000, 80000, 60000, 120000, 110000, 75000, 65000, 130000]
}

df = pd.DataFrame(data)

# 原始数据的描述性统计
print("原始数据描述性统计:")
print(df.describe(include='all'))

# 可视化原始数据
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x='City', data=df)
plt.title('City Distribution')

plt.subplot(1, 2, 2)
sns.boxplot(x='Gender', y='Income', data=df)
plt.title('Income by Gender')

plt.tight_layout()
plt.show()

# 类别编码
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(df[['City', 'Gender']])
encoded_feature_names = encoder.get_feature_names_out(['City', 'Gender'])

encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
encoded_df['Age'] = df['Age']
encoded_df['Income'] = df['Income']

# 编码后的数据的描述性统计
print("编码后的数据描述性统计:")
print(encoded_df.describe(include='all'))

# 可视化编码后的数据
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(encoded_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')

plt.subplot(1, 2, 2)
sns.boxplot(x='City_New York', y='Income', data=encoded_df)
plt.title('Income by New York City Category')

plt.tight_layout()
plt.show()




#%% 5. 特征选择（Feature Selection）

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler

# 加载数据集
data = load_iris()
X = data.data
y = data.target

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 特征选择方法
selectors = [
    ('SelectKBest_f_classif', SelectKBest(score_func=f_classif, k=2)),
    ('SelectKBest_mutual_info_classif', SelectKBest(score_func=mutual_info_classif, k=2))
]

# 绘制图形
plt.figure(figsize=(14, 6))

for i, (name, selector) in enumerate(selectors):
    plt.subplot(1, 2, i + 1)
    X_new = selector.fit_transform(X_scaled, y)
    mask = selector.get_support()

    plt.scatter(X_new[:, 0], X_new[:, 1], c=y, edgecolor='k', s=50)
    plt.title(f'{name} Feature Selection')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()



#%% 6. 特征缩放（Feature Scaling）

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 加载Iris数据集
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# 原始数据集的散点图
sns.pairplot(df, hue='species', markers=['o', 's', 'D'])
plt.suptitle('Original Data', y=1.02)
plt.show()

# 标准化
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df.iloc[:, :-1]), columns=iris.feature_names)
df_standardized['species'] = df['species']

# 标准化数据集的散点图
sns.pairplot(df_standardized, hue='species', markers=['o', 's', 'D'])
plt.suptitle('Standardized Data', y=1.02)
plt.show()

# 最小最大缩放
scaler = MinMaxScaler()
df_minmax = pd.DataFrame(scaler.fit_transform(df.iloc[:, :-1]), columns=iris.feature_names)
df_minmax['species'] = df['species']

# 最小最大缩放数据集的散点图
sns.pairplot(df_minmax, hue='species', markers=['o', 's', 'D'])
plt.suptitle('Min-Max Scaled Data', y=1.02)
plt.show()



#%% 7. 特征构造（Feature Engineering）

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 生成示例数据
np.random.seed(42)
data = pd.DataFrame({
    'area': np.random.randint(1000, 3500, 100),
    'bedrooms': np.random.randint(1, 5, 100),
    'bathrooms': np.random.randint(1, 3, 100),
    'price': np.random.randint(100000, 500000, 100)
})

# 构造新特征
data['price_per_sqft'] = data['price'] / data['area']
data['bed_bath_ratio'] = data['bedrooms'] / data['bathrooms']

# 绘制图形
plt.figure(figsize=(14, 6))

# 图形1：价格与每平方英尺价格的关系
plt.subplot(1, 2, 1)
sns.scatterplot(x=data['area'], y=data['price_per_sqft'])
plt.title('Price per Square Foot vs Area')
plt.xlabel('Area (sqft)')
plt.ylabel('Price per Square Foot ($)')

# 图形2：价格与卧室-浴室比例的关系
plt.subplot(1, 2, 2)
sns.scatterplot(x=data['bed_bath_ratio'], y=data['price'])
plt.title('Price vs Bedroom-Bathroom Ratio')
plt.xlabel('Bedroom-Bathroom Ratio')
plt.ylabel('Price ($)')

plt.tight_layout()
plt.show()

# 更多图形
plt.figure(figsize=(14, 6))

# 图形3：面积与价格的关系
plt.subplot(1, 2, 1)
sns.scatterplot(x=data['area'], y=data['price'])
plt.title('Area vs Price')
plt.xlabel('Area (sqft)')
plt.ylabel('Price ($)')

# 图形4：每平方英尺价格的分布
plt.subplot(1, 2, 2)
sns.histplot(data['price_per_sqft'], kde=True)
plt.title('Distribution of Price per Square Foot')
plt.xlabel('Price per Square Foot ($)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()



#%% 8. 降维（Dimensionality Reduction）

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 加载手写数字数据集
digits = load_digits()
data = digits.data
labels = digits.target

# 使用PCA将数据降到2维
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# 使用t-SNE将数据降到2维
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data)

# 绘制PCA降维后的数据
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Digits")
plt.title('PCA of Digits Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# 绘制t-SNE降维后的数据
plt.subplot(1, 2, 2)
scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Digits")
plt.title('t-SNE of Digits Dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

plt.show()

# 计算PCA的前两个主成分的解释方差比
explained_variance = pca.explained_variance_ratio_
print(f'Explained variance by the first two principal components: {explained_variance[0]:.2f}, {explained_variance[1]:.2f}')

# 用Seaborn绘制Pairplot图（可以绘制更多维度，但这里只选前两个维度示范）
import pandas as pd

df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
df_pca['label'] = labels

sns.pairplot(df_pca, hue='label', palette='viridis')
plt.suptitle('Pairplot of PCA Components', y=1.02)
plt.show()



#%% 9. 数据增强（Data Augmentation）

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image_path = 'lenna.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将图像从BGR格式转换为RGB格式

# 定义数据增强函数
def augment_data(image):
    # 随机旋转
    angle = np.random.randint(-30, 30)
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))

    # 随机水平翻转
    flip = np.random.choice([True, False])
    if flip:
        flipped_image = cv2.flip(rotated_image, 1)
    else:
        flipped_image = rotated_image

    return flipped_image

# 数据增强示例
num_samples = 4
plt.figure(figsize=(12, 8))

for i in range(num_samples):
    augmented_image = augment_data(image)
    plt.subplot(2, num_samples//2, i+1)
    plt.imshow(augmented_image)
    plt.axis('off')

plt.tight_layout()
plt.show()



#%% 10. 数据平衡（Data Balancing）
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter

# 生成不平衡数据集
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, weights=[0.95], flip_y=0, random_state=42)

# 查看原始数据最强总结，十大聚类算法 ！！的类别分布
print(f"Original dataset shape: {Counter(y)}")

# 可视化原始数据集
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='viridis', alpha=0.6)
plt.title('Original Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# 进行SMOTE过采样
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# 查看过采样后的数据类别分布
print(f"Resampled dataset shape: {Counter(y_res)}")

# 可视化过采样后的数据集
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_res[:, 0], y=X_res[:, 1], hue=y_res, palette='viridis', alpha=0.6)
plt.title('SMOTE Resampled Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# 数据平衡前后的类别分布直方图
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.histplot(y, ax=axes[0], bins=2, kde=False)
axes[0].set_title('Original Dataset Class Distribution')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Frequency')

sns.histplot(y_res, ax=axes[1], bins=2, kde=False)
axes[1].set_title('SMOTE Resampled Dataset Class Distribution')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()






















